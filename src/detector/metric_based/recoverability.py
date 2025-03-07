# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import numpy as np

from src.analyzer.lm_analyzer import LMAnalyzer

from src.detector.metric_based.metric import Metric

class RecoverabilityMetric(Metric):

    def __init__(self) -> None:
        super().__init__()
    
    def get_score(self, analyzer: LMAnalyzer, threshold: float, k_threshold: bool) -> float:

        batch_len = len(analyzer.text)

        recoverability_scores = []

        for batch_idx in range(batch_len):
            if k_threshold: 
                recoverability_scores.append(analyzer.recoverability_score(batch_idx, k = threshold))
            else:
                recoverability_scores.append(analyzer.recoverability_score(batch_idx, p = threshold))
        
        return np.array(recoverability_scores).T.tolist()
