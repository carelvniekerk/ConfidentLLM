# coding=utf-8
# --------------------------------------------------------------------------------
# Project: ConfidentLLM
# Author: Carel van Niekerk
# Year: 2024
# Group: Dialogue Systems and Machine Learning Group
# Institution: Heinrich Heine University Düsseldorf
# --------------------------------------------------------------------------------
#
# This code was generated with the help of AI writing assistants
# including GitHub Copilot, ChatGPT, Bing Chat.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License."
"""Module to evaluate the accuracy of the model on a dataset."""

from dataclasses import dataclass

import numpy as np

from confidentllm.evaluation.types import BaseResults, Evaluator
from confidentllm.hydra_tools import builds

__all__ = ["accuracy_config"]


@dataclass
class AccuracyResults(BaseResults):
    """Results specific to the calibration evaluator."""

    accuracy: float = 0.0

    def to_dict(self) -> dict[str, str | float | dict[str, dict[str, float | int]]]:
        """Convert results to a dictionary."""
        base_dict = super().to_dict()
        base_dict.update({"accuracy": self.accuracy})
        return base_dict

    def __str__(self) -> str:
        """Convert results to a string."""
        return f"{super().__str__()}, Accuracy: {self.accuracy}"


class AccuracyEvaluator(Evaluator):
    """Class to evaluate the accuracy of the model on a dataset."""

    def __init__(self, padding_value: int = -1) -> None:
        """Initialize the evaluator."""
        super().__init__(padding_value=padding_value)

    def evaluate(self) -> AccuracyResults:
        """Evaluate the model."""
        predictions = np.array(self.buffer["predictions"])
        labels = np.array(self.buffer["labels"])

        predictions = predictions[labels != self.padding_value]
        labels = labels[labels != self.padding_value]

        acc = np.mean(predictions == labels)

        return AccuracyResults(evaluator_name="accuracy", accuracy=acc * 100.0)


accuracy_config = builds(AccuracyEvaluator, padding_value=-1)
