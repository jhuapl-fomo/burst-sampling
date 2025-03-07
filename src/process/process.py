# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from abc import ABC, abstractmethod

from typing import List, Tuple

import numpy as np

class Process(ABC):
    def __init__(self, n_bins: int, model_length: int) -> None:
        self.n_bins = n_bins
        self.bins = [0]
        self.bin_index = np.array(range(1,n_bins+1))

        for i in range(n_bins - 1):
            self.bins.append(10 ** (i + 1))
        
        self.bins.append(model_length)

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def predict(self, input: List[int]) -> Tuple[int, int]:
        pass

    def bin_top_k(self, input_examples: List[int]) -> List[int]:
        return list(np.digitize(input_examples, self.bins))

