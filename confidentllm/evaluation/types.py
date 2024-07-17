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
"""Types for the evaluation module."""

from abc import ABC, abstractmethod


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
    def evaluate(self) -> tuple:
        """Evaluate the model."""
        ...
