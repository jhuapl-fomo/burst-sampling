# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from abc import ABC
from typing import Dict, Any


class DetectorModel(ABC):
    def __init__(self) -> None:
        pass

    def train_model(self, x_train, y_train, x_test, y_test) -> Dict[str, Any]:
        pass
