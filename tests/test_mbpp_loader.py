from unittest.mock import patch

from data.loaders.mbpp import MBPPDataset


class _Tokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return "prompt"


def test_mbpp_loader_downloads_full_dataset_when_local_copy_is_missing(tmp_path):
    dataset_splits = {
        "test": [{"text": "test", "task_id": 1, "test_list": ["assert True"]}],
        "prompt": [
            {"text": "example", "code": "pass", "test_list": ["assert True"]}
        ],
    }

    with (
        patch("data.loaders.mbpp.DATASETS_PATH", tmp_path),
        patch("data.loaders.mbpp.load_dataset", return_value=dataset_splits) as load,
    ):
        dataset = MBPPDataset(_Tokenizer())

    load.assert_called_once_with("google-research-datasets/mbpp", "full")
    assert len(dataset) == 1
    assert dataset.few_shot_prompt


def test_mbpp_loader_prefers_local_dataset_copy(tmp_path):
    dataset_splits = {
        "test": [{"text": "test", "task_id": 1, "test_list": ["assert True"]}],
        "prompt": [
            {"text": "example", "code": "pass", "test_list": ["assert True"]}
        ],
    }
    (tmp_path / "mbpp").mkdir()

    with (
        patch("data.loaders.mbpp.DATASETS_PATH", tmp_path),
        patch("data.loaders.mbpp.load_from_disk", return_value=dataset_splits) as load,
        patch("data.loaders.mbpp.load_dataset") as download,
    ):
        dataset = MBPPDataset(_Tokenizer())

    load.assert_called_once_with(str(tmp_path / "mbpp"))
    download.assert_not_called()
    assert len(dataset) == 1
