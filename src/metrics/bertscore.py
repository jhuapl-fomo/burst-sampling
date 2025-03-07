# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from typing import List

from evaluate import load

class BertScore:

    def __init__(self) -> None:
        self.bertscore = load("bertscore")
    
    def get_score(self, predictions: List[str], references: List[str]):

        results = self.bertscore.compute(predictions=predictions, references=references, lang="en")

        return results