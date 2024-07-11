# coding=utf-8
# --------------------------------------------------------------------------------
# Project: ConfidentLLM
# Author: Carel van Niekerk
# Year: 2024
# Group: Dialogue Systems and Machine Learning Group
# Institution: Heinrich Heine University Düsseldorf
# --------------------------------------------------------------------------------
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

import numpy as np
from hydra_zen import store

from confidentllm.evaluation.types import Evaluator

__all__ = []


class AccuracyEvaluator(Evaluator):
    """Class to evaluate the accuracy of the model on a dataset."""

    def __init__(self) -> None:
        """Initialize the evaluator."""
        super().__init__()

    def evaluate(self) -> tuple:
        """Evaluate the model."""
        predictions = np.array(self.buffer["predictions"])
        labels = np.array(self.buffer["labels"])

        acc = np.mean(predictions == labels)

        return (acc * 100.0,)


evaluator_store = store(group="evaluator")
evaluator_store(AccuracyEvaluator, name="accuracy")
