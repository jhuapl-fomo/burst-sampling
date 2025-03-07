# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import ast

import os

import pandas as pd

path = "/projects/LTSG/data/metrics"


files = [os.path.join(path, x) for x in os.listdir(path) if "llama" in x or "vicuna" in x]

print(files)

for file in files:

    df = pd.read_csv(file)

    cols = list(df.columns)

    cols.remove("text")

    for col in cols:
        if col == "gltr":
            df = df.join(pd.DataFrame(pd.json_normalize(df.pop('gltr').str.replace("array", "").apply(ast.literal_eval))['gltr'].tolist(), columns=["gltr_1", "gltr_2", "gltr_3", "gltr_4"]))
            
        else:
            df = df.join(pd.json_normalize(df.pop(col).apply(ast.literal_eval)))

    df.describe().to_csv(os.path.basename(file) + "_metrics_analysis.csv", index=False)
