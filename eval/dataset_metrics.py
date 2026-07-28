#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Dataset-specific, dependency-light evaluation helpers."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from rouge_score import rouge_scorer
except ImportError:  # Allows lightweight inspection before optional dependencies install.
    rouge_scorer = None


def execution_result(
    code: str, tests: str, timeout_seconds: float = 5.0
) -> dict[str, Any]:
    """Run an MBPP completion and classify its outcome.

    This is intended for benchmark code and is not a security boundary.  The child
    runs in a temporary directory with Python isolated mode enabled.
    """

    with tempfile.TemporaryDirectory(prefix="mbpp_eval_") as temp_dir:
        script_path = Path(temp_dir) / "candidate.py"
        script_path.write_text(f"{code}\n\n{tests}\n", encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "status": "timeout",
                "passed": False,
                "return_code": None,
                "stderr": (exc.stderr or "")[-4000:],
                "stdout": (exc.stdout or "")[-4000:],
            }

    stderr = result.stderr[-4000:]
    return {
        "status": (
            "passed"
            if result.returncode == 0
            else "syntax_error"
            if "SyntaxError" in stderr
            else "test_failure"
        ),
        "passed": result.returncode == 0,
        "return_code": result.returncode,
        "stderr": stderr,
        "stdout": result.stdout[-4000:],
    }


def evaluate_mbpp_generations(
    generations: list[dict[str, Any]], timeout_seconds: float = 5.0
) -> dict[str, Any]:
    """Annotate MBPP records and return execution-based pass@1 diagnostics."""

    counts = Counter()
    for item in generations:
        result = execution_result(
            item["generation_sanitized"], item["test_cases"], timeout_seconds
        )
        item["execution"] = result
        item["pass@1"] = 1.0 if result["passed"] else 0.0
        counts[result["status"]] += 1

    total = len(generations)
    return {
        "pass@1": counts["passed"] / total if total else 0.0,
        "syntax_errors": counts["syntax_error"],
        "test_failures": counts["test_failure"],
        "timeouts": counts["timeout"],
        "length_limit_failures": sum(
            not item["pass@1"] and item.get("length_limit_reached", False)
            for item in generations
        ),
        "total": total,
    }


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def _f1(overlap: int, predicted_count: int, reference_count: int) -> float:
    if not overlap or not predicted_count or not reference_count:
        return 0.0
    precision = overlap / predicted_count
    recall = overlap / reference_count
    return 2 * precision * recall / (precision + recall)


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for right_index, right_token in enumerate(right, start=1):
            current.append(
                previous[right_index - 1] + 1
                if left_token == right_token
                else max(previous[right_index], current[-1])
            )
        previous = current
    return previous[-1]


def xsum_metrics(generations: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute average ROUGE F1 and output diagnostics for XSum."""

    rouge1_scores = []
    rouge2_scores = []
    rouge_l_scores = []
    output_words = []
    output_tokens = []
    scorer = (
        rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        if rouge_scorer is not None
        else None
    )
    for item in generations:
        prediction_text = item["generation"]
        reference_text = item["reference"]
        prediction = prediction_text.split()
        reference = reference_text.split()
        if scorer is not None:
            scores = scorer.score(reference_text, prediction_text)
            rouge1_scores.append(scores["rouge1"].fmeasure)
            rouge2_scores.append(scores["rouge2"].fmeasure)
            rouge_l_scores.append(scores["rougeL"].fmeasure)
        else:
            unigrams_prediction = _ngrams(prediction, 1)
            unigrams_reference = _ngrams(reference, 1)
            bigrams_prediction = _ngrams(prediction, 2)
            bigrams_reference = _ngrams(reference, 2)
            rouge1_scores.append(
                _f1(
                    sum((unigrams_prediction & unigrams_reference).values()),
                    sum(unigrams_prediction.values()),
                    sum(unigrams_reference.values()),
                )
            )
            rouge2_scores.append(
                _f1(
                    sum((bigrams_prediction & bigrams_reference).values()),
                    sum(bigrams_prediction.values()),
                    sum(bigrams_reference.values()),
                )
            )
            rouge_l_scores.append(
                _f1(
                    _lcs_length(prediction, reference), len(prediction), len(reference)
                )
            )
        output_words.append(len(prediction))
        output_tokens.append(item["generation_token_count"])

    total = len(generations)
    mean = lambda values: sum(values) / total if total else 0.0
    return {
        "rouge1": mean(rouge1_scores),
        "rouge2": mean(rouge2_scores),
        "rougeL": mean(rouge_l_scores),
        "average_output_words": mean(output_words),
        "average_output_tokens": mean(output_tokens),
        "length_limit_failures": sum(
            item.get("length_limit_reached", False) for item in generations
        ),
        "total": total,
    }
