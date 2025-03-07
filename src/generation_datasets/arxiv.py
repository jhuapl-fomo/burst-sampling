# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from typing import Tuple

from datasets import load_dataset

from src.generation_datasets.generation_dataset import GenerationDataset


class Arxiv(GenerationDataset):
    def __init__(self, path_to_arxiv: str, seed: int) -> None:
        super().__init__("arxiv", seed, "Abstract")

        self.dataset = load_dataset(
            "arxiv_dataset", split="train", data_dir=path_to_arxiv, streaming=True
        )

        self.dataset.shuffle(seed=self.seed)

        self.dataset = iter(self.dataset)

    def __next__(self) -> str:
        item = next(self.dataset)

        return item["abstract"]

    def get_input_output_pair(self, output_text: str, prefix_percentage: float = 0.1) -> Tuple[str, str]:
       return super().get_input_output_pair(output_text, prefix_percentage)



