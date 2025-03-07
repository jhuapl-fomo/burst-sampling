# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import logging

import os

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.featurization.featurize_dataset import FeaturizeDataset

logger = logging.getLogger(__name__)

def featurization(data):
    input_path = data["featurization"]["input_path"]
    output_path = data["featurization"]["output_path"]

    model_name = data["general"]["model"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_gpus = torch.cuda.device_count()

    if n_gpus == 1:
        model =  AutoModelForCausalLM.from_pretrained(model_name, device_map = "auto", torch_dtype=torch.float16)
    
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map = "balanced", torch_dtype=torch.float16)

    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=2048)

    tokenizer.pad_token='[PAD]'

    input_files = [
        os.path.join(input_path, x)
        for x in os.listdir(input_path)
        if x.find(".csv") != -1
    ]

    for input_file in input_files:

        logger.info(f"Featurizing {input_file} with {model_name}")

        featurizer = FeaturizeDataset(input_file, model_name, model, tokenizer, device, output_path)

        featurizer.featurize()
