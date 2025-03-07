# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import torch

from torch.nn import CrossEntropyLoss

from src.analyzer.lm_analyzer import LMAnalyzer
from src.detector.metric_based.metric import Metric


class PerplexityMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.loss_func = CrossEntropyLoss(reduction="none")
    
    def get_score(self, analyzer: LMAnalyzer) -> float:

        shift_logits = analyzer.logits[:, :-1, :].contiguous()

        shift_labels = analyzer._text_encoded_pt["input_ids"][..., 1:].contiguous()

        shift_attention_mask_batch = analyzer._text_encoded_pt["attention_mask"][..., 1:].contiguous()
        
        perplexity_batch = torch.exp(
            (self.loss_func(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / (shift_attention_mask_batch.sum(1))
        )

        return perplexity_batch.cpu().tolist()
