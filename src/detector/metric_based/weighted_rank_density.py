# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from typing import List

import numpy as np

from src.analyzer.lm_analyzer import LMAnalyzer

from src.detector.metric_based.metric import Metric

class WeightedRankDensity(Metric):
    def __init__(self) -> None:
        super().__init__()
    
    def get_score(self, analyzer: LMAnalyzer, num_bins: int, weights: List[float]) -> float:
        k_scores_total = []

        bins = [0]

        for i in range(num_bins):
            bins.append(10 ** (i + 1))

        bins.append(self.model.config.n_positions)

        for batch_idx in range(len(analyzer.probs)):
            num_tokens = len(analyzer.probs[batch_idx])
            k_scores = []
            for j in range(1, num_tokens - 1):
                distribution_pairs = analyzer.ordered_probs_at_idx(batch_idx, j)
                ids = [pair[0] for pair in distribution_pairs]
                k = ids.index(analyzer.token_id_map[batch_idx][j][1])
                k_scores.append(k)
            
            k_scores_total(k_scores)
        
        binned_k_scores = np.digitize(k_scores, bins)

        unique, counts = np.unique(binned_k_scores, return_counts=True)

        return np.dot(counts, weights)
