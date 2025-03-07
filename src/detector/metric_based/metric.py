# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from abc import ABC, abstractmethod

from src.analyzer.lm_analyzer import LMAnalyzer

class Metric(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_score(self, analyzer: LMAnalyzer) -> float:
        pass
