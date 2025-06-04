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
"""Huber quantile regression loss function."""

import torch
from transformers.modeling_outputs import SequenceClassifierOutput

from confidentllm.train.loss_functions.types import ComputeLossFunction

__all__ = ["HuberQuantileLoss"]


class HuberQuantileLoss(ComputeLossFunction):
    """Huber quantile regression loss."""

    def __init__(self, huber_k: float = 0.001, quantiles: list[float] | None = None):
        """Initialize the Huber quantile loss function.

        Args:
        ----
            huber_k (float): The Huber k value.
                Default is 0.001.

        """
        super().__init__()
        self.huber_k = huber_k

        if quantiles is None:
            quantiles = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        if not isinstance(quantiles, list):
            raise TypeError(f"Expected a list of quantiles, got {type(quantiles)}")  # noqa: EM102, TRY003
        if not all(isinstance(q, float) for q in quantiles):
            raise TypeError("All quantiles must be of type float")  # noqa: EM101, TRY003
        if not all(0 < q < 1 for q in quantiles):
            raise ValueError("All quantiles must be in the range (0, 1)")  # noqa: EM101, TRY003
        self.quantiles: torch.Tensor = torch.tensor(quantiles, dtype=torch.float32)

    def __call__(
        self,
        outputs: SequenceClassifierOutput,  # type: ignore[override]
        labels: torch.Tensor,
        num_items_in_batch: int | None = None,  # noqa: ARG002
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Huber quantile regression loss.

        Args:
        ----
            outputs (SequenceClassifierOutput): The model outputs.
            labels (Tensor): The labels.
            num_items_in_batch (int, optional): The number of items in the batch.
                Default is None.
            ignore_index (int, optional): The index to ignore.
                Default is -100.

        Returns:
        -------
            Tensor: The loss.

        """
        if outputs.logits is None:
            raise ValueError("The outputs must contain logits.")  # noqa: EM101, TRY003
        # Repeat labels to match the number of quantiles and the sequence length
        num_quantiles: int = outputs.logits.size(-1)
        sequence_length: int = outputs.logits.size(-2)

        labels = labels.reshape(-1, 1).repeat(1, sequence_length)
        labels = labels.unsqueeze(-1).repeat(1, 1, num_quantiles)

        # Calculate the Huber quantile loss
        errors: torch.Tensor = outputs.logits - labels
        huber_loss: torch.Tensor = torch.where(
            condition=torch.abs(errors) <= self.huber_k,
            input=0.5 * errors**2,
            other=self.huber_k * (torch.abs(errors) - 0.5 * self.huber_k),
        )
        rho: torch.Tensor = torch.abs(
            (errors < 0).float()
            - self.quantiles.unsqueeze(0).unsqueeze(0).to(outputs.logits.device),  # type: ignore[union-attr]
        )
        loss: torch.Tensor = rho * huber_loss

        # Padding
        loss[labels == ignore_index] = 0.0
        loss = loss.sum(dim=-1) / (labels != ignore_index).float().sum(dim=-1)
        # Average over the batch
        loss = loss.mean(dim=-1).mean()
        return loss
