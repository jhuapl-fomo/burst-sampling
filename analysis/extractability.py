# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import argparse

import os 

import tomli

import pandas as pd

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def extractability(args):

    config_path = args.config

    with open(config_path, "rb") as f:
        data = tomli.load(f)

    model_name = data["general"]["model"]

    model_name = data["general"]["model"]

    dataset_path = data["sampling"]["output_folder"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_gpus = torch.cuda.device_count()

    if n_gpus == 1:
        model =  AutoModelForCausalLM.from_pretrained(model_name, device_map = "auto", torch_dtype=torch.float16)

    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map = "balanced", torch_dtype=torch.float16)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_name = model_name.replace("/", "_")

    files = [os.path.join(dataset_path, x) for x in os.listdir(dataset_path)]

    for file in files:

        items = pd.read_csv(file).text.tolist()

        batches = batch(items, 16)

        total_extractable = 0

        for bth in batches:

            inputs = tokenizer(bth, padding=True, return_tensors="pt").input_ids

            real_continuations = inputs[:, 100:150]

            inputs.to(device)

            outputs = model.generate(inputs[:, :100], max_new_tokens=50)

            generated_continuations = outputs[:, 100:150]

            for real_continuation, generated_continuation in zip(real_continuations, generated_continuations):

                if torch.equal(real_continuation, generated_continuation):
                    total_extractable +=1
            
        print(f"For {model_name}, {file} is extractable at {total_extractable/len(items)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)

    args = parser.parse_args()

    extractability(args)
