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


PredictiveProbabilityConfig = builds(PredictiveProbability)
ProbabilityDisparityConfig = builds(ProbabilityDisparity)

generation_method_store = store(group="generation_method/confidence_extraction_method")
generation_method_store(
    PredictiveProbabilityConfig,
    name="predictive_probability",
)
generation_method_store(
    ProbabilityDisparityConfig,
    name="probability_disparity",
)
