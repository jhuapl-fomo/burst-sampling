# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from typing import Tuple
from datasets import load_dataset

import spacy

from src.generation_datasets.generation_dataset import GenerationDataset


class CNNDailyMail(GenerationDataset):
    def __init__(self, seed: int) -> None:
        super().__init__("cnn_dailymail", seed, "Article")

        self.dataset = load_dataset(
            "cnn_dailymail", "3.0.0", split="train", streaming=True
        )

        self.dataset.shuffle(seed=self.seed)

        self.dataset = iter(self.dataset)

    def __next__(self) -> str:
        item = next(self.dataset)

        return item["article"]

    def get_input_output_pair(self, output_text: str, prefix_percentage: float = 0.1) -> Tuple[str, str]:
        return super().get_input_output_pair(output_text, prefix_percentage)
