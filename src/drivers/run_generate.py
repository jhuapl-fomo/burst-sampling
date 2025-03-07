# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import csv

import os

import logging

import pandas as pd

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.bettertransformer import BetterTransformer

from tqdm import tqdm

from src.model.generativemodel import GenerativeModel
from src.process.processes.categorical_process import CategoricalProcess

logger = logging.getLogger(__name__)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def generate(data):

    model_name = data["general"]["model"].replace("/", "_")

    exp_1_folder = data["prompts"]["exp_1_output_folder"]

    exp_1_output_folder = data["generate"]["exp_1_output_folder"]

    featurization_files = [x for x in os.listdir(data["featurization"]["output_path"]) if x.find(model_name) != -1]

    featurization_files_names = [os.path.basename(x).replace(model_name + "_", "")[:-22] for x in featurization_files]

    featurization_dict = {key: value for key, value in zip(featurization_files_names, featurization_files)}

    top_k = data["generate"]["top_k"] if "top_k" in data["generate"] else None
    top_p = data["generate"]["top_p"] if "top_p" in data["generate"] else None
    
    temperature = data["generate"]["temperature"] if "temperature" in data["generate"] else None

    max_length = data["generate"]["max_length"]

    sample = data["generate"]["sample"]

    process_name = data["generate"]["process"]

    batch_sizes = data["generate"]["batch_sizes"]

    bins = data["generate"]["bins"] if "bins" in data["generate"] else None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_gpus = torch.cuda.device_count()

    if n_gpus == 1:
        model =  AutoModelForCausalLM.from_pretrained(data["general"]["model"], device_map = "auto", torch_dtype=torch.float16)
    
    else:
        model = AutoModelForCausalLM.from_pretrained(data["general"]["model"], device_map = "balanced", torch_dtype=torch.float16)
    
    model = BetterTransformer.transform(model)

    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(data["general"]["model"], model_max_length=2048)

    tokenizer.pad_token='[PAD]'

    exp_1_input_files = sorted([os.path.join(exp_1_folder, x) for x in os.listdir(exp_1_folder) if x.find(".csv") != -1])

    print(exp_1_input_files)

    for input_file, batch_size in tqdm(zip(exp_1_input_files, batch_sizes), desc="Experiment 1 Generation"):
        
        dataset = os.path.basename(input_file)[:-10]

        if process_name == "burst":
            process = CategoricalProcess(bins, len(tokenizer))

            featurization_data = pd.read_csv(os.path.join(data["featurization"]["output_path"], featurization_dict[dataset]), usecols = ["k"]).k.to_numpy()

            logger.info(f"Using tokenization dataset : {os.path.join(data['featurization']['output_path'], featurization_dict[dataset])}")

            process.train(featurization_data)

        elif process_name == "none":
            process = None
        
        else:
            ValueError(f"Process {process_name} not supported")

        gen_model = GenerativeModel(model, tokenizer, device, sample, max_length, top_p, top_k, temperature, process)

        df = pd.read_csv(input_file)

        prompts_outputs = []

        logger.info(f"Running generation experiment 1 for {dataset} with {model_name}")

        for prompts in tqdm(batch(df.prompt.tolist(), batch_size), desc=f"Experiment 1: {dataset}"):
            generated_text = gen_model.generate(prompts)
            prompts_outputs.extend(generated_text)
        
        with open(
            os.path.join(exp_1_output_folder, f"{dataset}_{model_name}_top_k_{top_k}_top_p_{top_p}_temp_{temperature}_process_{process_name}_bins_{bins}_generated_exp_1.csv"), "w"
        ) as tokenized_file:
            writer = csv.writer(tokenized_file, escapechar='\\')
            writer.writerow(["prompt", "output", "generated_text"])
            for prompt, output, generated in zip(df.prompt.tolist(), df.output.tolist(), prompts_outputs):
                writer.writerow([prompt, output, generated])