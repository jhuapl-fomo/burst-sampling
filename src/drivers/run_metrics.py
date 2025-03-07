# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import csv

from collections import defaultdict

import os

import logging

import time

from typing import List

import pandas as pd

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm 

from src.analyzer.lm_analyzer import LMAnalyzer

from src.metrics.diversity import DiversityMetric
from src.metrics.selfbleu import SelfBLEU
from src.metrics.selfrouge import SelfROUGE
from src.detector.metric_based import *

logger = logging.getLogger(__name__)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def map_model_based_metrics(name: str) -> Metric:

    metric_map = {
        "burst": BurstinessMetric,
        "kburst": PerTokenKBurstiness,
        "pburst": PerTokenPBurstiness,
        "toppburst" : PerTokenTopPBurstiness,
        "weighted": WeightedRankDensity,
        "entropy" : EntropyMetric,
        "gltr" : GLTRMetric,
        "ll" : LogLikelihoodMetric,
        "log_rank": LogRankMetric,
        "rank" : RankMetric,
        "perplexity" : PerplexityMetric,
        "recoverability" : RecoverabilityMetric,
        "diversity" : DiversityMetric
    }

    return metric_map[name]

def run_model_based_metrics(data, metric_name: str, lm_analyzer: LMAnalyzer) -> List[float]:
    k_thresholds = data['metrics']['recoverability']['k_thresholds']
    p_thresholds = data['metrics']['recoverability']['p_thresholds']

    metric = map_model_based_metrics(metric_name)()

    scores = []

    if metric_name == "recoverability":
        scores = []
        for k_threshold in k_thresholds:
            scores.extend(metric.get_score(lm_analyzer, k_threshold, True))
        for p_threshold in p_thresholds:
            scores.extend(metric.get_score(lm_analyzer, p_threshold, False))
    
    elif metric_name == "diversity":
        scores.extend(metric.get_score(lm_analyzer.text))

    else:
        scores.extend(metric.get_score(lm_analyzer))

    return scores


def metrics(data):

    input_path = data["metrics"]["input_path"]

    output_path = data["metrics"]["output_path"]

    model_name = data["general"]["model"]

    metrics_to_run = data['metrics']['metrics']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_gpus = torch.cuda.device_count()

    if n_gpus == 1:
        model =  AutoModelForCausalLM.from_pretrained(model_name, device_map = "auto", torch_dtype=torch.float16)
    
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map = "balanced", torch_dtype=torch.float16)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = '[PAD]'

    model_name = model_name.replace("/", "_")

    lm_analyzer = LMAnalyzer(model, tokenizer, device)

    input_files = [
        os.path.join(input_path, x)
        for x in os.listdir(input_path)
        if x.find(".csv") != -1
    ]

    bleu_grams = data['metrics']['bleu']['ngrams']
    rouge_types = data['metrics']['rouge']['types']
    for input_file in tqdm(input_files, desc="Metrics"):

        dataset_name = os.path.basename(input_file)[:-4]

        df = pd.read_csv(input_file)

        examples = df.output.tolist()

        if "bleu" in metrics_to_run:
            bleu = SelfBLEU()
            for bleu_gram in bleu_grams:
                scores = bleu.get_score(examples, bleu_gram)

                for i, example in enumerate(examples): 
                    scores[i]["text"] = example

                bleu_df = pd.DataFrame(scores)

                bleu_df.to_csv(os.path.join(output_path, f"{dataset_name}_bleu-{bleu_gram}.csv"))

        if "rouge" in metrics_to_run:
            rouge = SelfROUGE()
            for rouge_type in rouge_types:
                scores = rouge.get_score(examples, rouge_type)

                for i, example in enumerate(examples): 
                    scores[i]["text"] = example
                
                rouge_df = pd.DataFrame(scores)
                
                rouge_df.to_csv(os.path.join(output_path, f"{dataset_name}_rouge-{rouge_type}.csv"))

        scores = dict()

        scores["text"] = []

        for metric_name in metrics_to_run:
            scores[metric_name] = []

        for i, example in tqdm(enumerate(batch(examples, 10)), desc=f"Metrics for {dataset_name} for {model_name}"):

            lm_analyzer.set_text(example)

            scores["text"].extend(example)

            for metric_name in metrics_to_run:

                score = run_model_based_metrics(data, metric_name, lm_analyzer)

                scores[metric_name].extend(score)

        pd.DataFrame.from_dict(scores).to_csv(os.path.join(output_path, f"{dataset_name}_{model_name}_metrics.csv"), index=False)
