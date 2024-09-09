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
"""Types for the evaluation module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

__all__ = ["BaseResults", "EvaluationBatch", "Evaluator"]


@dataclass
class BaseResults:
    """Base class for evaluator results."""

    evaluator_name: str

    def to_dict(self) -> dict[str, Any]:
        """Convert results to a dictionary."""
        return {"evaluator_name": self.evaluator_name}

    def __str__(self) -> str:
        """Convert results to a string."""
        return f"Evaluator: {self.evaluator_name}"


@dataclass
class EvaluationBatch:
    """Class for evaluation batch."""

    labels: list[str | float] = field(default_factory=list)
    predictions: list[str | float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)

    def __add__(self, other: "EvaluationBatch") -> "EvaluationBatch":
        """Add values from another batch to this batch."""
        self.labels += other.labels
        self.predictions += other.predictions
        self.confidences += other.confidences
        return self


class Evaluator(ABC):
    """Abstract base class for evaluators."""

    def __init__(  # noqa: D107
        self,
        padding_value: int = -1,
    ) -> None:
        self.buffer: EvaluationBatch = EvaluationBatch()
        self.padding_value = padding_value

    def add_batch(self, batch: EvaluationBatch) -> None:
        """Add a batch of data to the buffer.

        Args:
        ----
            batch: The data to add to the buffer.

        """
        self.buffer += batch

    @abstractmethod
    def evaluate(self) -> BaseResults:
        """Evaluate the model."""
        ...
