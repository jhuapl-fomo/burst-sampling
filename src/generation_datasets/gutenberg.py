# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from typing import Tuple
from datasets import load_dataset

import random

from src.generation_datasets.generation_dataset import GenerationDataset

class Gutenberg(GenerationDataset):
    def __init__(self, seed: int) -> None:
        super().__init__("pg19", seed, "Book")

        self.dataset = load_dataset("pg19", split="train")

        self.dataset.shuffle(seed=self.seed)

        self.dataset.filter(lambda x: len(x) > 500)

        self.dataset = iter(self.dataset)

    def __next__(self) -> str:
        item = next(self.dataset)

        return item["text"]

    def get_input_output_pair(self, output_text: str, prefix_percentage: float = 0.1) -> Tuple[str, str]:
        output_paragraphs = output_text.split("\n\n")

        filtered_paragraphs = [item for item in output_paragraphs if len(item.split("\n")) > 3]

        if len(filtered_paragraphs) == 0:
            output_paragraphs = output_text.split("\n\n\n\n")
            filtered_paragraphs = [item for item in output_paragraphs if len(item.split("\n\n")) > 3]

        print(len(filtered_paragraphs))

        paragraph = random.choice(filtered_paragraphs)

        return super().get_input_output_pair(paragraph, prefix_percentage)

