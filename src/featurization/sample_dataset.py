# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import csv

import os

import random

from tqdm import tqdm

from src.generation_datasets.generation_dataset import GenerationDataset


class SampleDataset:
    def __init__(self, dataset: GenerationDataset, output_folder: str, n: int) -> None:
        self.dataset = dataset

        self.output_folder = output_folder

        self.n = n

    def sample_dataset(self):
        reservoir = []
        for t, item in tqdm(enumerate(self.dataset)):
            if t < self.n:
                reservoir.append(item)
            else:
                m = random.randint(0,t)
                if m < self.n:
                    reservoir[m] = item
        with open(
            os.path.join(self.output_folder, f"{self.dataset.name}.csv"), "w"
        ) as tokenized_file:
            writer = csv.writer(tokenized_file)
            writer.writerow(["text"])

            for item in reservoir:
                writer.writerow([item])
