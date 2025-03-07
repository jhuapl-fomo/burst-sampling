# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def metrics(label, pred_label, pred_posteriors):
    if len(set(label)) < 3:
        acc = accuracy_score(label, pred_label)
        precision = precision_score(label, pred_label)
        recall = recall_score(label, pred_label)
        f1 = f1_score(label, pred_label)
        auc = roc_auc_score(label, pred_posteriors)
    else:
        acc = accuracy_score(label, pred_label)
        precision = precision_score(label, pred_label, average="weighted")
        recall = recall_score(label, pred_label, average="weighted")
        f1 = f1_score(label, pred_label, average="weighted")
        auc = -1.0
        conf_m = confusion_matrix(label, pred_label)
    return acc, precision, recall, f1, auc
