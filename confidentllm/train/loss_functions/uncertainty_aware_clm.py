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
"""Uncertainty-aware CLM loss function."""

import torch
from transformers.modeling_outputs import CausalLMOutput

from confidentllm.train.loss_functions.types import ComputeLossFunction

__all__ = ["UncertaintyAwareCLMLoss"]


class UncertaintyAwareCLMLoss(ComputeLossFunction):
    """Uncertainty-aware loss for causal language modeling."""

    def __call__(
        self,
        outputs: CausalLMOutput,
        labels: torch.Tensor,
        num_items_in_batch: int | None = None,  # noqa: ARG002
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Uncertainty-aware loss for causal language modeling.

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
        # Shift labels and logits to align
        labels = labels[:, 1:]
        logits: torch.Tensor = outputs.logits[:, :-1, :]  # type: ignore[index]
        ignore_indices: torch.Tensor = labels == ignore_index
        labels[ignore_indices] = 0

        greedy_predictions: torch.Tensor = torch.argmax(logits, dim=-1)
        predictive_distributions: torch.Tensor = torch.softmax(
            input=logits,
            dim=-1,
        )
        prediction_probabilities: torch.Tensor = predictive_distributions.max(
            dim=-1,
        ).values

        correct_predictions: tuple[torch.Tensor, ...] | list[torch.Tensor] = (
            torch.where(
                condition=greedy_predictions == labels,
            )
        )
        incorrect_predictions: tuple[torch.Tensor, ...] | list[torch.Tensor] = (
            torch.where(
                condition=greedy_predictions != labels,
            )
        )

        log_probabilities: torch.Tensor = torch.log(
            input=predictive_distributions + 1e-8,
        )
        entropy: torch.Tensor = -torch.sum(
            input=predictive_distributions * log_probabilities,
            dim=-1,
        )

        correct_prediction_loss_term: torch.Tensor = 1 - prediction_probabilities
        correct_prediction_loss_term *= (1 - entropy.tanh() + 1e-8).log()
        correct_prediction_loss_term[
            incorrect_predictions[0],
            incorrect_predictions[1],
        ] = 0
        correct_prediction_loss_term[ignore_indices] = 0
        num_correct_predictions: torch.Tensor = correct_prediction_loss_term != 0
        num_correct_predictions = num_correct_predictions.sum(dim=-1)
        num_correct_predictions[num_correct_predictions == 0] = 1
        correct_prediction_loss_term = -correct_prediction_loss_term.sum(dim=-1)
        correct_prediction_loss_term /= num_correct_predictions

        incorrect_prediction_loss_term: torch.Tensor = prediction_probabilities
        incorrect_prediction_loss_term *= (entropy.tanh() + 1e-8).log()
        incorrect_prediction_loss_term[
            correct_predictions[0],
            correct_predictions[1],
        ] = 0
        incorrect_prediction_loss_term[ignore_indices] = 0
        num_incorrect_predictions: torch.Tensor = incorrect_prediction_loss_term != 0
        num_incorrect_predictions = num_incorrect_predictions.sum(dim=-1)
        num_incorrect_predictions[num_incorrect_predictions == 0] = 1
        incorrect_prediction_loss_term = -incorrect_prediction_loss_term.sum(dim=-1)
        incorrect_prediction_loss_term /= num_incorrect_predictions

        loss: torch.Tensor = (
            correct_prediction_loss_term + incorrect_prediction_loss_term
        )

        return loss.mean()
