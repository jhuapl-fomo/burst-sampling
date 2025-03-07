# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from multiprocessing import Pool

import os

from typing import List

import nltk
from nltk.translate.bleu_score import SmoothingFunction

from tqdm import tqdm


class SelfBLEU:
    def __init__(self) -> None:
        self.gram = None 
        
        self.reference = None

    def get_score(self, test_text: List[str], gram: int = 3) -> List[float]:
        self.gram = gram

        reference = []

        for text in test_text:
            text = nltk.word_tokenize(text)
            reference.append(text)
        
        self.reference = reference

        return self.get_bleu()

    def calc_bleu(self, reference: List[str], hypothesis: str, weight: List[float]) -> float:
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu(self) -> List[float]:
        ngram = self.gram

        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        result = []
        sentence_num = len(self.reference)
        for index in tqdm(range(sentence_num), desc = "Bleu score"):
            hypothesis = self.reference[index]
            other = self.reference[:index] + self.reference[index+1:]
            result.append({f"bleu-{ngram}": pool.apply_async(self.calc_bleu, args=(other, hypothesis, weight))})

        pool.close()

        pool.join()

        for i, item in enumerate(result):
            result[i][f"bleu-{ngram}"] = result[i][f"bleu-{ngram}"].get()

        return result