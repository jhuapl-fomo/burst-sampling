# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import csv

import logging

import os

import pandas as pd

from tqdm import tqdm

from src.generation_datasets.arxiv import Arxiv
from src.generation_datasets.cnn_dailymail import CNNDailyMail
from src.generation_datasets.gutenberg import Gutenberg
from src.generation_datasets.openwebtext2 import OpenWebText2
from src.generation_datasets.stackexchange import StackExchange
from src.generation_datasets.semeval_twitter import SemevalTwitter
from src.generation_datasets.wikipedia import Wikipedia

logger = logging.getLogger(__name__)

def map_dataset(dataset: str):
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

def prompts(data):
    seed = data["general"]["seed"]

    exp_1_input_folder = data['prompts']['exp_1_input_folder']
    exp_1_output_folder = data['prompts']['exp_1_output_folder']
    exp_2_output_folder = data['prompts']['exp_2_output_folder']

    exp_1_input_files = [os.path.join(exp_1_input_folder, x) for x in os.listdir(exp_1_input_folder) if x.find(".csv") != -1]

    logger.info(f"Running prompt creation experiment 1")

    for input_file in tqdm(exp_1_input_files, desc="Experiment 1 Prompt Creation"):
        print(input_file)

        df = pd.read_csv(input_file)

        input_file_name = os.path.basename(input_file)[:-4]

        dataset = map_dataset(input_file_name)

        if input_file_name == "arxiv":
            dataset = dataset(data["sampling"]["arxiv_path"], seed)
        else:
            dataset = dataset(seed)
        
        prompts_outputs = []

        logger.info(f"Running prompt creation experiment 1 for {input_file_name}")

        for id, (text) in tqdm(df.itertuples(), desc=f"Experiment 1: {input_file_name}"):
            prompt, output = dataset.get_input_output_pair(text)
            prompts_outputs.append([prompt, output])
        
        with open(
            os.path.join(exp_1_output_folder, f"{input_file_name}_exp_1.csv"), "w"
        ) as tokenized_file:
            writer = csv.writer(tokenized_file)
            writer.writerow(["prompt", "output"])
            writer.writerows(prompts_outputs)
    
    # exp_2_input_files = [os.path.join(exp_1_output_folder, x) for x in os.listdir(exp_1_output_folder) if x.find(".csv") != -1]

    # logger.info(f"Running prompt creation experiment 2")

    # for input_file in tqdm(exp_2_input_files, desc="Experiment 2 Prompt Creation"):

    #     df = pd.read_csv(input_file)

    #     input_file_name = os.path.basename(input_file)[:-10]

    #     dataset = map_dataset(input_file_name)

    #     if input_file_name == "arxiv":
    #         dataset = dataset(data["sampling"]["arxiv_path"], seed)
    #     else:
    #         dataset = dataset(seed)

    #     demonstrations = []

    #     logger.info(f"Running prompt creation experiment 2 for {input_file_name}")

    #     for i in tqdm(range(len(df)), desc=f"Experiment 2: {input_file_name}"):
    #         samples = df.sample(3).output.tolist()

    #         demonstration = dataset.get_prompt(samples)
        
    #         demonstrations.append([demonstration])

    #     with open(
    #         os.path.join(exp_2_output_folder, f"{input_file_name}_exp_2.csv"), "w"
    #     ) as tokenized_file:
    #         writer = csv.writer(tokenized_file)
    #         writer.writerow(["text"])
    #         writer.writerows(demonstrations)


        



        

