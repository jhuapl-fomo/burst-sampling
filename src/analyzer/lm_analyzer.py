# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from typing import List

from numpy import argsort

import torch
from torch.nn.functional import softmax

class LMAnalyzer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logits = None

    def set_text(self, text: List[str]):
        self.text = text
        
        self.analyze()

    def tokenize_text(self):
        """
        Tokenize the input text using the LM's tokenizer. Also,
        store encoded versions of the tokenized text (i.e.,
        each token encoded numerically using it's vocabulary index),
        as well as one pre-tensorized version of the encoded text.
        """
        # Is this always safe, for every LM?
        # Either way, it's the only way to get probabilities
        # for the first word/token of the sequence
        text = [x.replace("<s>", "").replace("<unk>", "").replace("</s>", "") for x in self.text]

        self.text_tokenized = [self.tokenizer.tokenize(x) for x in text]
        self.text_encoded = self.tokenizer(text, return_tensors="np")['input_ids'].tolist()
        self._text_encoded_pt = self.tokenizer(text,return_tensors='pt', padding=True, return_attention_mask = True)

        self.padding_lengths = (self._text_encoded_pt["attention_mask"] == 0).sum(dim=1)

        self.token_id_map = list(zip(
            self.text_tokenized,
            self.text_encoded
        ))

    def collect_outputs(self):
        """
        Runs the neural network (language model) over
        the input text (self._text_encoded_pt -- which contains
        the *tokenized* and *encoded* and *tensorized* text), and
        collects (saves) the softmax distribution and the hidden
        states.
        """
        with torch.inference_mode():
            outputs = self.model(
                **self._text_encoded_pt.to(self.device),
            )
        
        self.logits = outputs.logits
        
        self.probs = softmax(self.logits,dim=2)[:, :-1, :].detach().cpu()


    def preprocess_outputs(self):
        """
        Given the softmax distribution, preprocesses them
        into useful lookup dictionaries.
        """

        self.ordered_probs_for_sentence, self.ordered_idx_for_sentence = torch.sort(self.probs, dim=-1, descending=True)

        self.total_k_scores = []
        self.total_p_scores = []
        self.total_top_p_scores = []

        for batch_idx in range(len(self.probs)):

            num_tokens = len(self.probs[batch_idx])

            k_scores = []
            p_scores = []
            cumulative_p_scores = []

            for j in range(self.padding_lengths[batch_idx] + 1, num_tokens - 1):
                ids, softmax = self.ordered_probs_at_idx(batch_idx, j)
                ids = ids.tolist()
                k = ids.index(self.token_id_map[batch_idx][1][j - self.padding_lengths[batch_idx] + 1])
                cumulative_prob = softmax[: k + 1].sum().item()
                top_p = max(min(cumulative_prob, 1.0), 0.0)

                p_scores.append(softmax[k])
                k_scores.append(k)
                cumulative_p_scores.append(top_p)
            
            self.total_k_scores.append(k_scores)
            self.total_p_scores.append(p_scores)
            self.total_top_p_scores.append(cumulative_p_scores)

    def analyze(self):
        self.tokenize_text()
        self.collect_outputs()
        self.preprocess_outputs()

    def ordered_probs_at_idx(self, batch_idx, token_idx):
        """
        Given an index into the token list/list of probability vectors
        (one per token in the input text), return a map of
           int -> float [as a list of pairs]
        where the keys are the token ids, and the float values are the probability
        mass accorded to that token id.

        The map is sorted by value, so that the keys
        are ordered by descending probability mass.

        Inputs:
           int
        Outputs:
           list of pair of (int, float)
        """
        return self.ordered_idx_for_sentence[batch_idx][token_idx], self.ordered_probs_for_sentence[batch_idx][token_idx]

    def recoverability_score(self, batch_idx, k=None, p=None):
        """
        Returns as a score the percentage of tokens in the
        analysis text that would be recoverable (i.e., fall
        within the nucleus) given the supplied nucleus parameters.
        """
        if (p and k):
            raise Exception("Provided both p and k cutoff.")
        elif not (p or k):
            raise Exception("Provided neither p nor k cutoff.")

        if p is not None:
            return sum([x < p  for x in self.total_top_p_scores[batch_idx]])/float(len(self.total_top_p_scores[batch_idx]))
        elif k is not None:
            return sum([x < k  for x in self.total_k_scores[batch_idx]])/float(len(self.total_k_scores[batch_idx]))