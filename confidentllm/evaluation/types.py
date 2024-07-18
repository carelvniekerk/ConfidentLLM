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
"""Types for the evaluation module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


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


class Evaluator(ABC):
    """Abstract base class for evaluators."""

    def __init__(  # noqa: D107
        self,
        padding_value: int = -1,
    ) -> None:
        self.buffer: dict = {}
        self.padding_value = padding_value

    def add_batch(self, batch: dict) -> None:
        """Add a batch of data to the buffer.

        Args:
        ----
            batch: The data to add to the buffer.

        """
        for key, value in batch.items():
            if key not in self.buffer:
                self.buffer[key] = []
            self.buffer[key] += value

    @abstractmethod
    def evaluate(self) -> BaseResults:
        """Evaluate the model."""
        ...
