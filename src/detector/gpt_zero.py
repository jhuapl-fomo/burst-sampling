# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

import os
from typing import Any, Dict
import requests
from tqdm import tqdm

from src.detector.detector_model import DetectorModel
from src.detector.utils import metrics


class GPTZero(DetectorModel):
    def __init__(self) -> None:
        super().__init__()

        self.api_key = os.environ["GPTZERO_API_KEY"]
        self.base_url = "https://api.gptzero.me/v2/predict"

    def text_predict(self, document):
        url = f"{self.base_url}/text"
        headers = {
            "accept": "application/json",
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json",
        }
        data = {"document": document}
        response = requests.post(url, headers=headers, json=data)
        return response.json()

    def train_model(self, x_train, y_train, x_test, y_test) -> Dict[str, Any]:
        train_pred_prob = [
            self.text_predict(_)["documents"][0]["completely_generated_prob"]
            for _ in tqdm(x_train)
        ]

        test_pred_prob = [
            self.text_predict(_)["documents"][0]["completely_generated_prob"]
            for _ in tqdm(x_test)
        ]

        train_pred = [round(_) for _ in train_pred_prob]
        test_pred = [round(_) for _ in test_pred_prob]

        acc_train, precision_train, recall_train, f1_train, auc_train = metrics(
            y_train, train_pred, train_pred_prob
        )
        acc_test, precision_test, recall_test, f1_test, auc_test = metrics(
            y_test, test_pred, test_pred_prob
        )

        return {
            "name": "GPTZero",
            "acc_train": acc_train,
            "precision_train": precision_train,
            "recall_train": recall_train,
            "f1_train": f1_train,
            "auc_train": auc_train,
            "acc_test": acc_test,
            "precision_test": precision_test,
            "recall_test": recall_test,
            "f1_test": f1_test,
            "auc_test": auc_test,
        }
