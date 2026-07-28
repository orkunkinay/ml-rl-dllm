import pytest

from eval.dataset_metrics import evaluate_mbpp_generations
from eval.dataset_metrics import xsum_metrics
from eval.pipeline import parse_dataset_selection


def test_mbpp_execution_reports_each_failure_type():
    generations = [
        {
            "generation_sanitized": "def add(a, b):\n    return a + b\n",
            "test_cases": "assert add(2, 3) == 5",
            "length_limit_reached": False,
        },
        {
            "generation_sanitized": "def broken(:\n    pass\n",
            "test_cases": "assert True",
            "length_limit_reached": False,
        },
        {
            "generation_sanitized": "def add(a, b):\n    return a - b\n",
            "test_cases": "assert add(2, 3) == 5",
            "length_limit_reached": True,
        },
        {
            "generation_sanitized": "while True:\n    pass\n",
            "test_cases": "assert True",
            "length_limit_reached": False,
        },
    ]

    metrics = evaluate_mbpp_generations(generations, timeout_seconds=1.0)

    assert metrics == {
        "pass@1": 0.25,
        "syntax_errors": 1,
        "test_failures": 1,
        "timeouts": 1,
        "length_limit_failures": 1,
        "total": 4,
    }
    assert [item["execution"]["status"] for item in generations] == [
        "passed",
        "syntax_error",
        "test_failure",
        "timeout",
    ]


def test_xsum_metrics_include_rouge_and_length_diagnostics():
    metrics = xsum_metrics(
        [
            {
                "generation": "The cat sat.",
                "reference": "The cat sat.",
                "generation_token_count": 4,
                "length_limit_reached": False,
            },
            {
                "generation": "A dog ran.",
                "reference": "A cat slept.",
                "generation_token_count": 256,
                "length_limit_reached": True,
            },
        ]
    )

    assert metrics["rouge1"] == pytest.approx(2 / 3)
    assert metrics["rouge2"] == pytest.approx(0.5)
    assert metrics["rougeL"] == pytest.approx(2 / 3)
    assert metrics["average_output_tokens"] == 130
    assert metrics["average_output_words"] == 3
    assert metrics["length_limit_failures"] == 1


def test_dataset_selection_rejects_unknown_names_and_deduplicates():
    assert parse_dataset_selection("mbpp, xsum,mbpp") == ["mbpp", "xsum"]
    with pytest.raises(ValueError, match="Unknown dataset"):
        parse_dataset_selection("gsm8k,typo")
