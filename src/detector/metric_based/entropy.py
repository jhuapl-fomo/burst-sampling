# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import torch

import torch.nn.functional as F

from src.analyzer.lm_analyzer import LMAnalyzer
from src.detector.metric_based.metric import Metric


class EntropyMetric(Metric):
    def __init__(self) -> None:
        super().__init__()

    def get_score(self, analyzer: LMAnalyzer) -> float:
        logits = analyzer.logits[:, :-1, :]

        softmax = F.softmax(logits, dim=-1) 

        log_softmax = torch.log(softmax)

        neg_entropy = softmax * log_softmax

        neg_entropy = -neg_entropy.sum(-1)

        neg_entropy = neg_entropy * analyzer._text_encoded_pt["attention_mask"][..., 1:]

        neg_entropy = neg_entropy.sum(dim=1)/analyzer._text_encoded_pt["attention_mask"].sum(dim=1)

        return neg_entropy.cpu().tolist()
