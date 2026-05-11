# coding=utf-8
# --------------------------------------------------------------------------------
# Project: ConfidentLLM
# Author: Carel van Niekerk, Renato Vukovic
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
"""Types for the generation module."""

from abc import ABC, abstractmethod

from torch import Tensor

__all__ = ["ConfidenceMetric"]


class ConfidenceMetric(ABC):
    """Method for extracting confidence scores from the logits."""

    @abstractmethod
    def __call__(
        self,
        next_token_ids: Tensor,
        scores: Tensor,
    ) -> Tensor:
        """Extract the confidence scores from the logits."""
        ...  # pragma: no cover
