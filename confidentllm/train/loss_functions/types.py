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
#     http: //www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Loss function types."""

from abc import ABC, abstractmethod
from enum import StrEnum, auto

import torch
from transformers.modeling_outputs import CausalLMOutput

__all__ = ["ComputeLossFunction", "LossFunction"]


class LossFunction(StrEnum):
    """Enum class to store the loss functions of the models."""

    DEFAULT = auto()
    UA_CLM = auto()


class ComputeLossFunction(ABC):
    """Abstract class for computing the loss function."""

    @abstractmethod
    def __call__(
        self,
        outputs: CausalLMOutput,
        labels: torch.Tensor,
        num_items_in_batch: int | None = None,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Compute the loss.

        Args:
        ----
            outputs (CausalLMOutput): The model outputs.
            labels (Tensor): The labels.
            num_items_in_batch (int, optional): The number of items in the batch.
                Default is None.
            ignore_index (int, optional): The index to ignore.
                Default is -100.

        Returns:
        -------
            Tensor: The loss.

        """
        ...
