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
"""Entropy based confidence metrics."""

import torch
from hydra_zen import store
from torch import Tensor

from confidentllm.confidence_metrics.probability import PredictiveProbability
from confidentllm.confidence_metrics.types import ConfidenceMetric
from confidentllm.hydra_tools import builds

__all__: list[str] = []


class Entropy(ConfidenceMetric):
    """Entropy as a confidence metric."""

    def __call__(
        self,
        next_token_ids: Tensor,  # noqa: ARG002
        scores: Tensor,
    ) -> Tensor:
        """Entropy as a confidence metric."""
        log_probability: Tensor = torch.log(scores + 1e-8)
        entropy: Tensor = -torch.sum(scores * log_probability, dim=-1)

        # Normalise the entropy
        normalisation_factor: Tensor = torch.log(torch.tensor(scores.size(-1)))
        entropy /= normalisation_factor

        return entropy


class MeanTokenEntropy(Entropy):
    """Confidence extraction method based on mean token entropy."""

    def __call__(
        self,
        next_token_ids: Tensor,
        scores: Tensor,
    ) -> Tensor:
        """Extract confidence based on mean token entropy."""
        entropy: Tensor = super().__call__(next_token_ids, scores)

        mean_token_entropy: Tensor = torch.mean(entropy, dim=-1)
        mean_token_entropy = mean_token_entropy.unsqueeze(-1).repeat(1, scores.size(-2))

        return mean_token_entropy


class PredictiveEntropy(PredictiveProbability):
    """Confidence extraction method based on mean token entropy."""

    def __call__(
        self,
        next_token_ids: Tensor,
        scores: Tensor,
    ) -> Tensor:
        """Extract confidence based on mean token entropy."""
        if scores.size(0) == 1:
            raise ValueError("Predictive entropy requires num beams > 1.")  # noqa: EM101, TRY003

        next_token_probs: Tensor = super().__call__(next_token_ids, scores)

        predictive_probability: Tensor = torch.exp(
            torch.log(next_token_probs + 1e-8).sum(-1),
        )

        predictive_entropy: Tensor = -torch.sum(
            input=predictive_probability * torch.log(predictive_probability + 1e-8),
            dim=-1,
        )

        normalising_factor: Tensor = torch.log(torch.tensor(scores.size(0)))
        predictive_entropy /= normalising_factor

        predictive_entropy = predictive_entropy.repeat(scores.size(0))
        predictive_entropy = predictive_entropy.unsqueeze(-1).repeat(1, scores.size(-2))

        return predictive_entropy


EntropyConfig = builds(Entropy)
MeanTokenEntropyConfig = builds(MeanTokenEntropy)
PredictiveEntropyConfig = builds(PredictiveEntropy)

generation_method_store = store(group="generation_method/confidence_metric")

generation_method_store(
    EntropyConfig,
    name="entropy",
)

generation_method_store(
    MeanTokenEntropyConfig,
    name="mean_token_entropy",
)

generation_method_store(
    PredictiveEntropyConfig,
    name="predictive_entropy",
)
