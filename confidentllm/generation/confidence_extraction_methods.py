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
"""Confidence extraction methods for language models."""

import torch
from hydra_zen import store
from torch import Tensor

from confidentllm.generation.types import ConfidenceExtractionMethod
from confidentllm.hydra_tools import builds

__all__ = ["ProbabilityDisparityConfig"]


class PredictiveProbability(ConfidenceExtractionMethod):
    """Confidence extraction method based on predictive probability."""

    def __call__(
        self,
        next_token_ids: Tensor,
        scores: Tensor,
    ) -> Tensor:
        """Extract confidence based on predictive probability."""
        # Extract the token indices corresponding to the generated tokens
        next_token_ids = next_token_ids[:, -scores.size(-2) :]

        # Use torch.gather to extract the probabilities of the next tokens
        next_token_probs: Tensor = torch.gather(
            scores,  # tensor of shape (batch_size, gen_length, vocab_size)
            dim=-1,  # we are selecting along the vocab dimension
            index=next_token_ids.unsqueeze(-1),  # shape (batch_size, gen_length, 1)
        ).squeeze(-1)  # shape (batch_size, gen_length)

        return next_token_probs


class ProbabilityDisparity(PredictiveProbability):
    """Confidence extraction method based on probability disparity."""

    def __call__(
        self,
        next_token_ids: Tensor,
        scores: Tensor,
    ) -> Tensor:
        """Extract confidence based on probability disparity."""
        next_token_probs: Tensor = super().__call__(next_token_ids, scores)

        # Calculate the probability of the second most likely token
        second_highest_probs: torch.return_types.topk = torch.topk(
            input=scores,
            k=2,
            dim=-1,
        )

        disparity: Tensor = (
            next_token_probs - second_highest_probs.values[:, :, 1]  # noqa: PD011 - .values is a property
        )

        return disparity


class Entropy(ConfidenceExtractionMethod):
    """Confidence extraction method based on entropy."""

    def __call__(
        self,
        next_token_ids: Tensor,  # noqa: ARG002
        scores: Tensor,
    ) -> Tensor:
        """Extract confidence based on entropy."""
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
        next_token_ids: Tensor,  # noqa: ARG002
        scores: Tensor,
    ) -> Tensor:
        """Extract confidence based on mean token entropy."""
        if scores.size(0) == 1:
            raise ValueError("Predictive entropy requires num beams > 1.")  # noqa: EM101, TRY003

        next_token_probs: Tensor = super().__call__(next_token_ids, scores)

        predictive_probability: Tensor = torch.exp(
            torch.log(next_token_probs + 1e-8).sum(-1)
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


PredictiveProbabilityConfig = builds(PredictiveProbability)
ProbabilityDisparityConfig = builds(ProbabilityDisparity)
EntropyConfig = builds(Entropy)
MeanTokenEntropyConfig = builds(MeanTokenEntropy)
PredictiveEntropyConfig = builds(PredictiveEntropy)

generation_method_store = store(group="generation_method/confidence_extraction_method")
generation_method_store(
    PredictiveProbabilityConfig,
    name="predictive_probability",
)
generation_method_store(
    ProbabilityDisparityConfig,
    name="probability_disparity",
)

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
