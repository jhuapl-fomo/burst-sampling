# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from itertools import islice

from typing import List

import numpy as np

import spacy

import torch

from torch.nn import CrossEntropyLoss

from torch.nn.functional import softmax

from src.analyzer.lm_analyzer import LMAnalyzer
from src.detector.metric_based.metric import Metric


class BurstinessMetric(Metric):
    def __init__(self) -> None:
        super().__init__()

        self.nlp = spacy.load("en_core_web_sm")
    
    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]


    def get_score(self, analyzer: LMAnalyzer) -> float:
        sents = [list(self.nlp(x).sents) for x in analyzer.text]

        lengths = [len(sent) for sent in sents]

        sents = [item.text for row in sents for item in row]

        perplexities = []

        for batch in self.batch(sents, 16):
            perplexity = self.get_perplexity(batch, analyzer)
            perplexities.extend(perplexity)

        
        perplexities_iter = iter(perplexities)
        perpleities_sliced = [list(islice(perplexities_iter, elem))
                for elem in lengths]

        return [np.var(x) for x in perpleities_sliced]

    def get_perplexity(self, text: List[str], analyzer: LMAnalyzer) -> float:
        
        text = [''.join([analyzer.tokenizer.bos_token, x]) for x in text]

        tokenized = analyzer.tokenizer(text,return_tensors='pt', padding=True, return_token_type_ids = False, return_attention_mask=True)

        _text_encoded_pt = tokenized["input_ids"]
        attn_mask = tokenized["attention_mask"]

        with torch.inference_mode():
            outputs = analyzer.model(
                _text_encoded_pt.to(analyzer.device),
            )

        logits = outputs.logits.detach().cpu().float()
        
        perplexities = []

        loss_fct = CrossEntropyLoss(reduction="none")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = _text_encoded_pt[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        perplexities.extend(perplexity_batch.tolist())

        return perplexities