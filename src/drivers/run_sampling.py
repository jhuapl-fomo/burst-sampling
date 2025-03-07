# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import logging

from tqdm import tqdm

from src.featurization.sample_dataset import SampleDataset

from src.generation_datasets.arxiv import Arxiv
from src.generation_datasets.cnn_dailymail import CNNDailyMail
from src.generation_datasets.gutenberg import Gutenberg
from src.generation_datasets.openwebtext2 import OpenWebText2
from src.generation_datasets.stackexchange import StackExchange
from src.generation_datasets.semeval_twitter import SemevalTwitter
from src.generation_datasets.wikipedia import Wikipedia
from src.generation_datasets.generation_dataset import GenerationDataset

logger = logging.getLogger(__name__)

def map_dataset(dataset: str) -> GenerationDataset:
    dataset_map = {
        "arxiv": Arxiv,
        "pg19": Gutenberg,
        "stackexchange": StackExchange,
        "cnn_dailymail": CNNDailyMail,
        "openwebtext2": OpenWebText2,
        "semeval_twitter": SemevalTwitter,
        "wikipedia": Wikipedia,
    }

    return dataset_map[dataset]


def sampling(data):
    seed = data["general"]["seed"]

    dataset_names = data["sampling"]["datasets"]

    number_of_items = data["sampling"]["num_sampled"]

    for dataset_name, n in tqdm(zip(dataset_names, number_of_items), desc="sampling datasets"):
        dataset = map_dataset(dataset_name)

        if dataset_name == "arxiv":
            dataset = dataset(data["sampling"]["arxiv_path"], seed)
        else:
            dataset = dataset(seed)

        output_file_path = data["sampling"]["output_folder"]

        logger.info(f"Sampling {number_of_items} items of {dataset_name}")

        sampler = SampleDataset(dataset, output_file_path, n)

        sampler.sample_dataset()
