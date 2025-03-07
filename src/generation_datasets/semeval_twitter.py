# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from typing import Tuple
from datasets import load_dataset

from src.generation_datasets.generation_dataset import GenerationDataset

class SemevalTwitter(GenerationDataset):
    def __init__(self, seed: int) -> None:
        super().__init__("semeval_twitter", seed, "Tweet")

        self.dataset = load_dataset(
            "maxmoynan/SemEval2017-Task4aEnglish", split="train", streaming=True
        )

        self.dataset.shuffle(seed=self.seed)

        self.dataset = iter(self.dataset)

    def __next__(self) -> str:
        item = next(self.dataset)

        return item["tweet"]

    def get_input_output_pair(self, output_text: str, prefix_percentage: float = 0.1) -> Tuple[str, str]:
        output_text = output_text[:2000]

        doc = self.nlp(output_text)

        num_prefix_tokens = max(prefix_percentage * (len(output_text) / 4), 5)

        prefix = doc[:num_prefix_tokens].text

        return prefix, output_text



