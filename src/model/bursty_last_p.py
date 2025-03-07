# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import random

import torch

from transformers.generation.logits_process import LogitsWarper

class BurstLastPLogitsWarper(LogitsWarper):

    def __init__(self, top_p: float, last_p_burst: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        last_p_burst = float(last_p_burst)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if last_p_burst < 0 or last_p_burst > 1.0:
            raise ValueError(f"`last_p_burst` has to be a float > 0 and < 1, but is {last_p_burst}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.top_p = top_p
        self.last_p_burst = last_p_burst
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        for i in range(len(scores)):

            if random.random() < self.last_p_burst:
                score = scores[i].masked_fill(indices_to_remove[i], 0)

                argmins = score.argmin(dim=1)

                mask = torch.ones(score.shape, dtype=torch.bool)

                mask[list(range(len(argmins))), argmins] = False

                scores[i] = score.masked_fill(mask, self.filter_value)
            else:
                scores[i] = scores[i].masked_fill(indices_to_remove[i], self.filter_value)

        return scores