# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from collections import Counter
from typing import List, Any, Tuple

import numpy as np

from src.process.process import Process


class CategoricalProcess(Process):
    def __init__(self, n_bins: int, model_length: int) -> None:
        super().__init__(n_bins, model_length)

        self.p_vector = None 
    
    def train(self, output: List[int]) -> None:
        binned_output = self.bin_top_k(output)

        _, counts = np.unique(binned_output, return_counts=True)

        self.p_vector = counts/sum(counts)

        print(self.p_vector)


    def predict(self, input: List[Any]) -> Tuple[int, int]:
        bin = self.bin_index.dot(np.random.multinomial(1, self.p_vector))

        return (self.bins[bin-1], self.bins[bin])
