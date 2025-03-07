# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from collections import Counter

import numpy as np

from src.analyzer.lm_analyzer import LMAnalyzer
from src.detector.metric_based.metric import Metric


class GLTRMetric(Metric):
    def __init__(self) -> None:
        super().__init__()

        self.bins  = [0, 10, 100, 1000]

    def get_score(self, analyzer: LMAnalyzer) -> float:

        self.bins.append(len(analyzer.tokenizer))

        binned_outputs = [list(np.digitize(x, self.bins)) for x in analyzer.total_k_scores]

        counts = [np.unique(x, return_counts=True) for x in binned_outputs]

        counts_filled = []

        for count in counts:
            count_filled = [0]*4

            count = list(np.column_stack(count))
            
            for id, num in count:
                count_filled[id-1] = num
            
            counts_filled.append(count_filled)

        gltr = np.array(counts_filled)

        gltr = gltr/gltr.sum(axis=1, keepdims=True)

        return gltr.tolist()
