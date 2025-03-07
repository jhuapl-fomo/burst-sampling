# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import logging

import os 

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import torch

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from tqdm import tqdm

from src.detector.detect_gpt import DetectGPT, DetectGPTConfig
from src.detector.gpt_zero import GPTZero
from src.detector.thresholding_model import ThresholdingModel
from src.detector.supervised_model import SupervisedModel

from src.detector.metric_based import *

logger = logging.getLogger(__name__)

def map_metrics(name: str) -> Metric:

    metric_map = {
        "burst": BurstinessMetric,
        "entropy" : EntropyMetric,
        "gltr" : GLTRMetric,
        "ll" : LogLikelihoodMetric,
        "log_rank": LogRankMetric,
        "rank" : RankMetric,
        "perplexity" : PerplexityMetric,
        "recoverability" : RecoverabilityMetric,
        "kburst": PerTokenKBurstiness,
        "pburst": PerTokenPBurstiness,
        "weighted": WeightedRankDensity,
    }

    return metric_map[name]


def run_thresholding_model(dataset: str, data, x_train, y_train, x_test, y_test):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_name = data["general"]["model"]

    metrics = data['detectors']['thresholding']

    stride = data['detectors']['perplexity']['stride']

    k = data['detectors']['recoverability']['k']
    p = data['detectors']['recoverability']['p']

    n_gpus = torch.cuda.device_count()

    if n_gpus == 1:
        model =  AutoModelForCausalLM.from_pretrained(model_name, device_map = "auto", torch_dtype=torch.float16)
    
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map = "balanced", torch_dtype=torch.float16)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    results = []

    for metric_name in metrics:
        metric = map_metrics(metric_name)(model, tokenizer, device)

        logger.info(f"Running thresholding model using {metric_name} on {dataset}")

        x_train_metrics = []

        for item in tqdm(x_train, desc=f"train thresholding data {metric_name}"):
            if metric_name=="burst" or metric_name=="perplexity":
                x_train_metrics.append(metric.get_score(item, stride))
            elif metric_name=="recoverability":
                x_train_metrics.append(metric.get_score(item, k,p ))
            else:
                x_train_metrics.append(metric.get_score(item))

        x_test_metrics = []

        for item in tqdm(x_test, desc=f"test thresholding data {metric_name}"):
            if metric_name=="burst" or metric_name=="perplexity":
                x_test_metrics.append(metric.get_score(item, stride))
            elif metric_name=="recoverability":
                x_train_metrics.append(metric.get_score(item, k,p ))
            else:
                x_test_metrics.append(metric.get_score(item))

        clf_model = ThresholdingModel()

        if metric_name != "gltr":
            x_train_metrics = np.array(x_train_metrics).reshape(-1, 1)
            x_test_metrics = np.array(x_test_metrics).reshape(-1, 1)

        result = clf_model.train_model(x_train_metrics, y_train, x_test_metrics, y_test, f"supervised_{model_name}_{metric_name}")

        result['dataset'] = dataset

        logger.info(f"Finishing thresholding model using {metric_name} on {dataset}")

        results.append(result)

    return results
    
def run_gptzero(dataset: str, x_train, y_train, x_test, y_test):

    gptzero = GPTZero()

    logger.info(f"Running GPTZero on {dataset}")

    result = gptzero.train_model(x_train, y_train, x_test, y_test)

    result['dataset'] = dataset

    logger.info(f"Finishing GPTZero model on {dataset}")

    return [result]

def run_detectgpt(dataset: str, data, x_train, y_train, x_test, y_test):

    detectgpt_config = data['detectors']['detectgpt']

    base_model_name = data["general"]["model"]

    mask_model_name = detectgpt_config['mask_model']

    del detectgpt_config['mask_model']

    base_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map = "balanced", torch_dtype=torch.float16)

    base_model.eval()

    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    mask_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    mask_model = AutoModelForSeq2SeqLM.from_pretrained(mask_model_name, torch_dtype=torch.float16).to(mask_device)

    mask_model.eval()

    mask_tokenizer = AutoTokenizer.from_pretrained(mask_model_name)

    logger.info(f"Running DetectGPT with mask model: {mask_model_name} and on base_model: {base_model_name} on {dataset}")

    config = DetectGPTConfig(**detectgpt_config)

    detectgpt = DetectGPT(base_model, base_tokenizer, base_device, mask_model, mask_tokenizer, mask_device, config)

    result = detectgpt.train_model(x_train, y_train, x_test, y_test)

    result['name'] = f"{base_model_name}_{mask_model_name}_detect_gpt"

    result['dataset'] = dataset
    
    logger.info(f"Finishing DetectGPT with mask model: {mask_model_name} and on base_model: {base_model_name} on {dataset}")

    return [result]

def run_supervised_model(dataset: str, data, x_train, y_train, x_test, y_test):
    supervised_model_config = data['detectors']['supervised']

    model_name = supervised_model_config["model"]

    finetune = supervised_model_config["finetune"]

    epochs = supervised_model_config["epochs"]

    batch_size = supervised_model_config["batch_size"]

    logger.info(f"Running supervised {model_name} on {dataset} with {epochs} and {batch_size}")

    supervised_model = SupervisedModel(model_name, finetune, batch_size, epochs)

    result = supervised_model.train_model(x_train, y_train, x_test, y_test)

    result['dataset'] = dataset

    logger.info(f"Finishing supervised {model_name} on {dataset} with {epochs} and {batch_size}")

    return [result]

def detectors(data):
    detectors = data['detectors']['detectors']

    input_path = data['detectors']['input_path']

    output_path = data['detectors']['output_path']

    input_files = [os.path.join(input_path, x) for x in os.listdir(input_path) if x.find(".csv") != -1]

    total_results = []
    

    for input_file in input_files:

        df = pd.read_csv(input_file)

        df_train, df_test = train_test_split(df, test_size = 0.2)

        x_train, y_train = df_train['text'].tolist(), df_train['label'].tolist() 

        x_test, y_test = df_test['text'].tolist(), df_test['label'].tolist() 
        
        for detector in detectors: 
            if detector == "thresholding":
                results = run_thresholding_model(input_file, data, x_train, y_train, x_test, y_test)
            elif detector == "gptzero":
                results = run_gptzero(input_file, x_train, y_train, x_test, y_test)
            elif detector == "detectgpt":
                results = run_detectgpt(input_file, data, x_train, y_train, x_test, y_test)
            elif detector == "supervised":
                results = run_supervised_model(input_file, data, x_train, y_train, x_test, y_test)
            else:
                raise ValueError(f"Detector {detector} does not exist")
            
            total_results.extend(results)
    
    results_df = pd.DataFrame(total_results)

    results_df.to_csv(os.path.join(output_path, "results.csv"), index=False)


