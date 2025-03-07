# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import torch

from transformers.generation.logits_process import LogitsProcessor, LogitsWarper

from src.process.process import Process

class ProcessedBasedLogitsProcessor(LogitsProcessor):

    def __init__(self, process: Process) -> None:
        self.process = process
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)

        for i in range(len(input_ids)):
            bin_range = self.process.predict(input_ids[i])

            indicies = sorted_indices[i][bin_range[0]: bin_range[1]]

            scores[i][~indicies] = -float("Inf")
        
        return scores

class ProcessedBasedLogitsWarper(LogitsWarper):

    def __init__(self, process: Process) -> None:
        self.process = process

    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        sorted_logits, sorted_indices = torch.sort(scores, descending=False)

        for i in range(len(input_ids)):
            bin_range = self.process.predict(input_ids[i])

            indicies = sorted_indices[i][bin_range[0]: bin_range[1]]

            scores[i][~indicies] = -float("Inf")
        
        return scores


