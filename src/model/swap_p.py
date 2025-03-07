# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import random

import torch

from transformers.generation.logits_process import LogitsWarper

class SwapPLogitsWarper(LogitsWarper):

    def __init__(self, top_p: float, swap_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        swap_p = float(swap_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if swap_p < 0 or swap_p > 1.0:
            raise ValueError(f"`swap_p` has to be a float > 0 and < 1, but is {swap_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.top_p = top_p
        self.swap_p = swap_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        
        sorted_logits = torch.zeros_like(scores, dtype=torch.float)
        sorted_indices = torch.zeros_like(scores, dtype=torch.int64)

        for i in range(len(scores)):
            sorted_logits[i], sorted_indices[i] = torch.sort(scores[i], descending= random.random() < self.swap_p)
        
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        scores = scores.masked_fill(indices_to_remove, self.filter_value)

        return scores