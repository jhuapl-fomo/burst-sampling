# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from multiprocessing import Pool

import os

from typing import List

from rouge_score import rouge_scorer

from tqdm import tqdm

class SelfROUGE:
    def __init__(self):
        self.type = None 
        
        self.reference = None

        self.scorer = None

    def get_score(self, test_text: List[str], type: str = "3") -> List[float]:
        self.type = type

        self.reference = test_text

        self.scorer = rouge_scorer.RougeScorer([f'rouge{self.type}'], use_stemmer=True)

        return self.get_rouge()

    def calc_rouge(self, references: List[str], hypothesis: str):
        scores = self.scorer.score_multi(references, hypothesis)[f'rouge{self.type}']

        score = dict()

        score[f'rouge{self.type}_prec'] = scores.precision
        score[f'rouge{self.type}_rec'] = scores.recall
        score[f'rouge{self.type}_f1'] = scores.fmeasure

        return score

    def get_rouge(self) -> List[float]:
        pool = Pool(os.cpu_count())

        result = []

        sentence_num = len(self.reference)

        for index in tqdm(range(sentence_num), desc = "rouge score"):
            hypothesis = self.reference[index]
            other = self.reference[:index] + self.reference[index+1:]
            result.append(pool.apply_async(self.calc_rouge, args=(other, hypothesis)))

        pool.close()

        pool.join()

        for i, _ in enumerate(result):
            result[i] = result[i].get()

        return result