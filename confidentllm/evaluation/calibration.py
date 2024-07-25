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

from dataclasses import dataclass, field

import torch

from confidentllm.evaluation.types import BaseResults, Evaluator
from confidentllm.hydra_tools import builds

__all__ = ["calibration_config"]


@dataclass
class CalibrationResults(BaseResults):
    """Results specific to the calibration evaluator."""

    ece: float = 0.0
    bin_stats: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str | float | dict[str, dict[str, float | int]]]:
        """Convert results to a dictionary."""
        base_dict = super().to_dict()
        base_dict.update({"ece": self.ece, "bin_stats": self.bin_stats})
        return base_dict

    def __str__(self) -> str:
        """Convert results to a string."""
        return f"{super().__str__()}, ECE: {self.ece}"


class CalibrationEvaluator(Evaluator):
    """Class to evaluate the calibration of the model on a dataset."""

    def __init__(self, padding_value: int = -1, number_of_bins: int = 10) -> None:
        """Initialize the evaluator."""
        super().__init__(padding_value=padding_value)
        self.number_of_bins = number_of_bins

    def split_into_bins(
        self,
        predictions: torch.Tensor,
        confidences: torch.Tensor,
        labels: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Split data into bins for calibration evaluation.

        Args:
        ----
            predictions: Model predictions.
            confidences: Model confidence scores.
            labels: Ground truth labels.

        Returns:
        -------
            A list of dictionaries, each representing a bin.

        """
        bin_boundaries = torch.linspace(0, 1, self.number_of_bins + 1)
        bins: list[dict[str, torch.Tensor]] = []

        for i in range(self.number_of_bins):
            lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (confidences > lower) & (confidences <= upper)
            bin_data: dict[str, torch.Tensor] = {
                "predictions": predictions[in_bin],
                "confidences": confidences[in_bin],
                "labels": labels[in_bin],
            }
            bins.append(bin_data)

        return bins

    def compute_bin_stats(
        self,
        bin_data: dict[str, torch.Tensor],
    ) -> dict[str, float | int]:
        """Compute accuracy and average confidence for a bin.

        Args:
        ----
            bin_data: Data for a specific bin.

        Returns:
        -------
            A dictionary containing accuracy, average confidence, and bin size.

        """
        bin_size: int = bin_data["labels"].size(0)
        if bin_size == 0:
            accuracy = -1
            avg_confidence = -1
        else:
            accuracy = (
                (bin_data["predictions"] == bin_data["labels"]).float().mean().item()
            )
            avg_confidence = bin_data["confidences"].mean().item()

        return {
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "bin_size": bin_size,
        }

    def compute_ece(
        self,
        bins: list[dict[str, torch.Tensor]],
        bin_stats: dict[str, dict[str, float | int]],
    ) -> float:
        """Compute the Expected Calibration Error (ECE).

        Args:
        ----
            bins: List of bins containing the data.
            bin_stats: Dictionary containing statistics for each bin.

        Returns:
        -------
            The computed ECE.

        """
        total_samples: int = 0
        ece: float = 0.0

        for i, _ in enumerate(bins):
            bin_size: int = bin_stats[f"bin_{i}"]["bin_size"]  # type: ignore  # noqa: PGH003 - Bin size is an integer.
            accuracy = bin_stats[f"bin_{i}"]["accuracy"]
            avg_confidence = bin_stats[f"bin_{i}"]["avg_confidence"]

            if bin_size == 0:
                continue

            ece += bin_size * abs(accuracy - avg_confidence)
            total_samples += bin_size

        ece /= total_samples
        return ece

    def evaluate(self) -> CalibrationResults:
        """Evaluate the model."""
        if (
            "predictions" not in self.buffer
            or "confidences" not in self.buffer
            or "labels" not in self.buffer
        ):
            msg = "Predictions, confidences, and labels must be present in the buffer."
            raise ValueError(msg)

        predictions = torch.Tensor(self.buffer["predictions"])
        confidences = torch.Tensor(self.buffer["confidences"])
        labels = torch.Tensor(self.buffer["labels"])

        valid_mask = labels != self.padding_value
        predictions = predictions[valid_mask]
        confidences = confidences[valid_mask]
        labels = labels[valid_mask]

        bins = self.split_into_bins(predictions, confidences, labels)

        bin_stats = {}
        for i, bin_data in enumerate(bins):
            bin_stats[f"bin_{i}"] = self.compute_bin_stats(bin_data)

        ece = self.compute_ece(bins, bin_stats)

        return CalibrationResults(
            evaluator_name="Calibration",
            ece=ece * 100.0,
            bin_stats=bin_stats,
        )


calibration_config = builds(CalibrationEvaluator, padding_value=-1, number_of_bins=10)
