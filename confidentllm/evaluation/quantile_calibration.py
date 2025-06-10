# coding=utf-8
# --------------------------------------------------------------------------------
# Project: ConfidentLLM
# Author: Carel van Niekerk
# Year: 2025
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
# limitations under the License.
"""Module to evaluate the calibration of a quantile regression model on a dataset."""

from dataclasses import dataclass, field

import numpy as np
import torch

from confidentllm.evaluation.types import BaseResults, Evaluator
from confidentllm.hydra_tools import builds

__all__ = ["calibration_config"]


@dataclass
class QuantileCalibrationResults(BaseResults):
    """Results specific to the calibration evaluator."""

    mad: float = 0.0

    def to_dict(self) -> dict[str, str | float | dict[str, dict[str, float | int]]]:
        """Convert results to a dictionary."""
        base_dict = super().to_dict()
        base_dict.update({"maximum_absolute_deviation": self.mad})
        return base_dict

    def __str__(self) -> str:
        """Convert results to a string."""
        return f"{super().__str__()}, Maximum Absolute Deviation: {self.mad}"


class QuantileCalibrationEvaluator(Evaluator):
    """Class to evaluate the calibration of the model on a dataset."""

    def evaluate(self) -> QuantileCalibrationResults:
        """Evaluate the model."""
        predictions = np.array(self.buffer.predictions)
        labels = np.array(self.buffer.labels)

        valid_mask = labels != self.padding_value
        predictions = predictions[valid_mask]
        confidences = confidences[valid_mask]
        labels = labels[valid_mask]

        bins = self.split_into_bins(predictions, confidences, labels)

        bin_stats = {}
        for i, bin_data in enumerate(bins):
            bin_stats[f"bin_{i}"] = self.compute_bin_stats(bin_data)

        ece = self.compute_ece(bin_stats)

        return CalibrationResults(
            evaluator_name="Calibration",
            ece=ece * 100.0,
            bin_stats=bin_stats,
        )


calibration_config = builds(CalibrationEvaluator)
