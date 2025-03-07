# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from abc import ABC, abstractmethod

from typing import Tuple, List

import spacy

class GenerationDataset(ABC):
    def __init__(self, name: str, seed: int, type: str) -> None:
        self.dataset = None

        self.name = name

        self.seed = seed

        self.nlp = spacy.load("en_core_web_sm")

        self.start = "Please add an example to this list."

        self.type = type

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self) -> str:
        pass

    def get_prompt(self, input_shots: List[str]):
        return self.start + "\n" + '\n'.join([self.type + ": " + x for x in input_shots]) + f'\n{self.type}: '

    def get_input_output_pair(self, output_text: str, prefix_percentage: float = 0.1) -> Tuple[str, str]:
        output_text = output_text[:2000]

        doc = self.nlp(output_text)

        num_prefix_tokens = prefix_percentage * (len(output_text) / 4)

        prefix = doc[:num_prefix_tokens].text

        return prefix, output_text
