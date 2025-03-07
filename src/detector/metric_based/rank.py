# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import numpy as np

from src.analyzer.lm_analyzer import LMAnalyzer
from src.detector.metric_based.metric import Metric


class RankMetric(Metric):
    def __init__(self) -> None:
        super().__init__()

    def get_score(self, analyzer: LMAnalyzer) -> float:

        ranks = [np.array(x) + 1 for x in analyzer.total_k_scores]

        return [np.mean(x) for x in ranks]
