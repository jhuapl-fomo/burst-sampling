# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import sys

import spacy
import math


class DiversityMetric:
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")
    
    def whitespace_split(self, string: str):
        return string.replace('.', '').replace('\n', '').split()

    def lines_to_ngrams(self, lines, n=3):
        """
        Given a list of strings (lines), and an n-value for the n-grams,
        generate a list of n-grams per line.
        """
        ngrams = []

        for line in lines:
            if type(lines) is str:
                # Split line into "words"
                words = [
                    e
                    for e
                    in self.whitespace_split(line)
                    if e != ''
                ]
            else:
                words = line

            ngrams.append([
                tuple(words[i:i + n])
                for i
                in range(len(words) - n + 1)
            ])

        return ngrams
    
    def geometric_mean(self, list_of_nums, weights=None):
        return math.exp(
            sum([
                weight * math.log(num)
                for num, weight
                in zip(list_of_nums, weights)
            ]) / sum(weights)
        )


    
    def normalized_unique_ngrams(self, lines, max_n=4, weights=[0.5, 0.3, 0.15, 0.05]):
        proportion_unique = []

        for n in range(1, max_n+1):
            # Convert each line/sent in a document into a list of ngrams
            ngram_lists = self.lines_to_ngrams(lines, n=n)
            # Flatten the list
            ngrams = [item for sublist in ngram_lists for item in sublist]
            # Record the proprtion of unique ngrams
            if len(ngrams) > 0:
                proportion_unique.append(len(set(ngrams)) / len(ngrams))
            else:
                proportion_unique.append(sys.float_info.min)

        # Average the unique n-gram proportions across all n-gram sizes
        # using a weighted geometric mean that favors smaller ngrams
        # (in other words, mainly penalizes repeating single words multiple times)
        return self.geometric_mean(proportion_unique, weights=weights)

    def chunk_text(self, text, chunk_by_sentence=True, max_chunk_len=10, overlap=0):
        if chunk_by_sentence:
            return [
                [token.text for token in list(sent)]
                for sent
                in self.nlp(text).sents
            ]

        tokens = self.whitespace_split(text)
        sub_seqs = []

        for start_idx in range(0, len(tokens), max_chunk_len - overlap):
            sub_seqs.append(tokens[start_idx:(start_idx+max_chunk_len)])

        return sub_seqs

    def get_score(self, text: str, chunk_by_sentence=False, max_chunk_len=10, overlap=0) -> float:

        self.chunk_by_sentence = chunk_by_sentence
        self.max_chunk_len = max_chunk_len
        self.overlap = overlap

        scores = []

        for example in text:

            chunks = self.chunk_text(
                text=example,
                chunk_by_sentence=self.chunk_by_sentence,
                max_chunk_len=self.max_chunk_len,
                overlap=self.overlap
            )

            scores.append(self.normalized_unique_ngrams(chunks))

        return scores

