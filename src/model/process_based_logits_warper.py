# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import torch

import numpy as np

from transformers.generation.logits_process import LogitsWarper

from src.process.process import Process

import matplotlib.pyplot as plt

class ProcessedBasedLogitsWarper(LogitsWarper):

    def __init__(self, process: Process) -> None:
        self.process = process

    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        for i in range(len(input_ids)):
            bin_range = self.process.predict(input_ids[i])

            top_k = torch.topk(scores[i], bin_range[1])

            indices_to_remove = (scores[i] < top_k[0][-1])  | (scores[i] > top_k[0][bin_range[0]]) 
            
            scores[i] = scores[i].masked_fill(indices_to_remove, -float("Inf"))

        return scores


