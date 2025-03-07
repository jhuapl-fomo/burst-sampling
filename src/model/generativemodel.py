# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from transformers.generation.logits_process import LogitsProcessorList

from src.model.process_based import ProcessedBasedLogitsProcessor
from src.model.process_based import ProcessedBasedLogitsWarper

from src.process.process import Process


class GenerativeModel:
    def __init__(
        self,
        model,
        tokenizer, 
        device,
        sample: bool = False,
        max_length: int = 64,
        top_p: float = None,
        top_k: int = None,
        temperature: float = None,
        process: Process = None,
    ) -> None:
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.top_p = top_p 
        self.top_k = top_k
        self.temperature = temperature
        self.max_length = max_length
        self.sample_ = sample

        self.process = process

        self.logits_processors = None

        if self.process is not None:
            self.logits_processors = LogitsProcessorList()
            if self.sample_: 
                self.logits_processors.append(ProcessedBasedLogitsWarper(self.process))
            else:
                self.logits_processors.append(ProcessedBasedLogitsProcessor(self.process))
            
    def generate(self, input_text: str) -> str:

        inputs = self.tokenizer(input_text, padding=True, return_tensors="pt")

        inputs.to(self.device)

        output = self.model.generate(**inputs, 
                                      logits_processor = self.logits_processors,
                                      do_sample = self.sample_,
                                      max_new_tokens=self.max_length, 
                                      top_p = self.top_p,
                                      top_k = self.top_k,
                                      temperature = self.temperature
                                    )

        return self.tokenizer.batch_decode(output)
        
