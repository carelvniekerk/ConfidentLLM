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
# limitations under the License.
"""Module to evaluate the AUARC of the model on a dataset."""

from dataclasses import dataclass, field

import numpy as np
import torch

from confidentllm.evaluation.types import BaseResults, Evaluator
from confidentllm.hydra_tools import builds

__all__ = ["AUARCEvaluator", "AUARCResults", "auarc_config"]


@dataclass
class AUARCResults(BaseResults):
    """Results specific to the AUARC evaluator."""

    auarc: float = 0.0
    ar_stats: tuple[list[float], list[float]] = field(default_factory=lambda: ([], []))

    def to_dict(self) -> dict[str, str | float | dict[str, dict[str, float | int]]]:
        """Convert results to a dictionary."""
        base_dict = super().to_dict()
        base_dict.update({"auarc": self.auarc, "ar_stats": self.ar_stats})
        return base_dict

    def __str__(self) -> str:
        """Convert results to a string."""
        return f"{super().__str__()}, AUARC: {self.auarc}"


class AUARCEvaluator(Evaluator):
    """Class to evaluate the calibration of the model on a dataset."""

    def __init__(
        self,
        padding_value: int | str = "-1",
        number_of_bins: int = 10,
    ) -> None:
        """Initialize the evaluator."""
        super().__init__(padding_value=padding_value)
        self.number_of_bins = number_of_bins

    def _eval_and_sort(
        self,
        predictions: np.ndarray,
        confidences: torch.Tensor,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        correct: np.ndarray = (predictions == labels).astype(int)
        sorted_indices = np.argsort(confidences.numpy())
        correct = correct[sorted_indices]
        confidences_sorted = confidences.numpy()[sorted_indices]

        return correct, confidences_sorted

    def _compute_ar_stats(
        self,
        correct: np.ndarray,
    ) -> tuple[list[float], list[float]]:
        """Compute the AR stats."""
        accuracies: list[float] = []
        rejection_rates: list[float] = []
        num_observations: int = len(correct)  # Total number of predictions

        # Iterate over possible rejection thresholds
        for i in range(num_observations + 1):
            # Number of predictions to reject
            num_rejected = i
            # Number of predictions to accept
            num_accepted = num_observations - num_rejected
            if num_accepted == 0:
                break
            # Calculate accuracy on the accepted predictions
            num_correct = sum(correct[num_rejected:])
            accuracy = num_correct / num_accepted
            # Calculate rejection rate
            rejection_rate = num_rejected / num_observations
            # Store the results
            rejection_rates.append(rejection_rate)
            accuracies.append(accuracy)

        return accuracies, rejection_rates

    def evaluate(self) -> AUARCResults:
        """Evaluate the model."""
        if not self.buffer.confidences:
            msg = "The buffer does not contain confidence scores."
            raise ValueError(msg)

        predictions = np.array(self.buffer.predictions)
        confidences = torch.Tensor(self.buffer.confidences)
        labels = np.array(self.buffer.labels)

        valid_mask = labels != self.padding_value
        predictions = predictions[valid_mask]
        confidences = confidences[valid_mask]
        labels = labels[valid_mask]

        correct, confidences_sorted = self._eval_and_sort(
            predictions=predictions,
            confidences=confidences,
            labels=labels,
        )

        accuracies, rejection_rates = self._compute_ar_stats(
            correct=correct,
        )

        # Calculate the area under the accuracy-rejection curve
        auarc = np.trapezoid(accuracies, rejection_rates)

        return AUARCResults(
            evaluator_name="AUARC",
            auarc=auarc,
            ar_stats=(accuracies, rejection_rates),
        )


auarc_config = builds(AUARCEvaluator)
