# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import torch

from src.analyzer.lm_analyzer import LMAnalyzer
from src.detector.metric_based.metric import Metric


class LogLikelihoodMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    def get_score(self, analyzer: LMAnalyzer) -> float:
        shift_logits = analyzer.logits[:, :-1, :].contiguous()
        shift_labels = analyzer._text_encoded_pt["input_ids"][..., 1:].contiguous()
        shift_attention_mask_batch = analyzer._text_encoded_pt["attention_mask"][..., 1:].contiguous()
        
        shift_labels = shift_labels.to(shift_logits.device)
        
        loss = self.loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch

        loss = loss.sum(1)

        loss = loss/shift_attention_mask_batch.sum(1)

        return (-loss).cpu().tolist()
