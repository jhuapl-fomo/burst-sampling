# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from typing import Dict, Any

from sklearn.linear_model import LogisticRegression

from src.detector.utils import metrics
from src.detector.detector_model import DetectorModel


class ThresholdingModel(DetectorModel):
    def __init__(self) -> None:
        super().__init__()

        self.model = LogisticRegression()

    def train_model(
        self, x_train, y_train, x_test, y_test, name: str
    ) -> Dict[str, Any]:

        self.model.fit(x_train, y_train)

        y_train_pred = self.model.predict(x_train)

        y_train_pred_prob = self.model.predict_proba(x_train)

        y_train_pred_prob = [_[1] for _ in y_train_pred_prob]

        acc_train, precision_train, recall_train, f1_train, auc_train = metrics(
            y_train, y_train_pred, y_train_pred_prob
        )

        y_test_pred = self.model.predict(x_test)

        y_test_pred_prob = self.model.predict_proba(x_test)

        y_test_pred_prob = [_[1] for _ in y_test_pred_prob]

        acc_test, precision_test, recall_test, f1_test, auc_test = metrics(
            y_test, y_test_pred, y_test_pred_prob
        )

        return {
            "name": name,
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
