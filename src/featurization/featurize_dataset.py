# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import csv

import os

from typing import List, Any

import pandas as pd

import numpy as np

from tqdm import tqdm

from src.analyzer.lm_analyzer import LMAnalyzer

class FeaturizeDataset:
    def __init__(self, dataset_name: str, model_name: str, model, tokenizer, device, output_folder: str) -> None:
        self.dataset = pd.read_csv(dataset_name)

        self.dataset_name = os.path.basename(dataset_name)[:-10]

        self.model_name = model_name.replace("/", "_")

        self.analyzer = LMAnalyzer(model, tokenizer, device)

        self.output_folder = output_folder

    def get_text_features(self, text: str) -> List[List[Any]]:
        examples = []

        self.analyzer.set_text(text)

        for batch_idx in range(len(self.analyzer.probs)):

            num_tokens = len(self.analyzer.probs[batch_idx])

            example = []

            for j in range(1, num_tokens - 1 - self.analyzer.padding_lengths[batch_idx]):
                ids, softmax = self.analyzer.ordered_probs_at_idx(batch_idx, j)
                ids = ids.tolist()
                k = ids.index(self.analyzer.token_id_map[batch_idx][1][j])
                p = max(min(np.sum(softmax[: k + 1]), 1.0), 0.0)

                example.append([self.analyzer.text_tokenized[:j], k, p])
            
            examples.extend(example)

        return examples

    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def featurize(self) -> None:
        with open(
            os.path.join(
                self.output_folder,
                f"{self.model_name}_{self.dataset_name}_tokenized_outputs.csv",
            ),
            "w",
        ) as tokenized_file:
            writer = csv.writer(tokenized_file)
            writer.writerow(["text", "k", "p"])

            for id, row in tqdm(self.dataset.iterrows(), desc=f"featurization of {self.dataset_name} using {self.model_name}"):
                examples = self.get_text_features(row.output)

                writer.writerows(examples)
