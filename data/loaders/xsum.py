#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""XSum evaluation dataset."""

import numpy as np
import torch
from datasets import load_dataset
from datasets import load_from_disk

from data.loaders.gsm8k import DATASETS_PATH


XSUM_SYSTEM_PROMPT = (
    "Summarize the following news article in one concise sentence. "
    "Return only the summary."
)


class XSumDataset(torch.utils.data.Dataset):
    """XSum test split with prompts suitable for the repository's chat models."""

    def __init__(self, tokenizer, subsample=-1, system_prompt=XSUM_SYSTEM_PROMPT):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.load_test_dataset()
        self.subsample = (
            np.random.choice(len(self.dataset), subsample, replace=False)
            if subsample != -1
            else np.arange(len(self.dataset))
        )
        print(f"evaluating {len(self.subsample)} examples")
        assert subsample <= len(self.dataset), (
            "Subsample size is greater than dataset size"
        )

    def __len__(self):
        return len(self.subsample)

    def load_test_dataset(self):
        local_path = DATASETS_PATH / "xsum"
        if local_path.exists():
            self.dataset = load_from_disk(str(local_path))["test"]
        else:
            self.dataset = load_dataset("EdinburghNLP/xsum", "default")["test"]

    def create_prompt(self, document):
        messages = [
            {
                "role": "user",
                "content": f"{self.system_prompt}\n\nArticle:\n{document}",
            }
        ]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def __getitem__(self, idx):
        item = self.dataset[self.subsample[idx].item()]
        document = item["document"]
        summary = item["summary"]
        sample_id = item.get("id", str(self.subsample[idx].item()))
        return self.create_prompt(document), document, summary, sample_id

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        documents = [item[1] for item in batch]
        summaries = [item[2] for item in batch]
        sample_ids = [item[3] for item in batch]
        encoded = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        )
        return {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "prompts": prompts,
            "documents": documents,
            "summaries": summaries,
            "sample_ids": sample_ids,
        }
