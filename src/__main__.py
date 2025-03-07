# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import argparse

import logging

import tomli

from src.utils import set_randomness
from src.drivers.run_sampling import sampling
from src.drivers.run_featurization import featurization
from src.drivers.run_detectors import detectors
from src.drivers.run_metrics import metrics
from src.drivers.run_prompt_creation import prompts
from src.drivers.run_generate import generate

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def main(args):
    config_path = args.config

    with open(config_path, "rb") as f:
        data = tomli.load(f)

    mode = args.mode

    seed = data["general"]["seed"]

    set_randomness(seed)

    logging.info(f"Running {mode} with config at {config_path}")

    if mode == "featurization":
        featurization(data)
    elif mode == "sampling":
        sampling(data)
    elif mode == "detectors":
        detectors(data)
    elif mode == "metrics":
        metrics(data)
    elif mode == "prompts":
        prompts(data)
    elif mode == "generate":
        generate(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True)

    args = parser.parse_args()

    main(args)
