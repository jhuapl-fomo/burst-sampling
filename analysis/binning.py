# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import os

import numpy as np
import pandas as pd

model_length_llama = 32000
model_length_gpt = 50257

path = "/projects/LTSG/data/tokenization/llama"

files = [os.path.join(path, x) for x in os.listdir(path)]

for file in files:

    df = pd.read_csv(file, usecols=["k"])

    for n_bins in range(2,6):
        bins = [0]
        bin_index = np.array(range(1,n_bins+1))

        for i in range(n_bins - 1):
            bins.append(10 ** (i + 1))
        
        model_length = 50257 if "gpt" in file else 32000

        bins.append(model_length)

        binned = np.digitize(df.k, bins)

        unique, counts = np.unique(binned, return_counts=True)

        print(file)
        print(n_bins)
        print(np.asarray((unique, counts)).T)

        
