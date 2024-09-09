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
"""Tests for the confidence extractions methods."""

import pytest
import torch

from confidentllm.generation.confidence_extraction_methods import (
    PredictiveProbability,
    ProbabilityDisparity,
)
from tests.generation.mock_transformers import (
    MOCK_LOGITS_MULTI_BEAM,
    MOCK_SEQUENCE_MULTI_BEAM,
)


def test_predictive_probability() -> None:
    """Test the PredictiveProbability confidence extraction method."""
    method = PredictiveProbability()

    logits: torch.Tensor = torch.cat(MOCK_LOGITS_MULTI_BEAM, dim=0)
    logits = logits.reshape(-1, len(MOCK_LOGITS_MULTI_BEAM), logits.size(-1)).transpose(
        dim0=0,
        dim1=1,
    )
    logits = torch.softmax(logits, dim=-1)
    confidence: torch.Tensor = method(MOCK_SEQUENCE_MULTI_BEAM, logits)

    target_confidences = torch.gather(
        logits,  # tensor of shape (batch_size, gen_length, vocab_size)
        dim=-1,  # we are selecting along the vocab dimension
        index=MOCK_SEQUENCE_MULTI_BEAM[:, -logits.size(1) :].unsqueeze(
            dim=-1,
        ),  # shape (batch_size, gen_length, 1)
    ).squeeze(-1)

    if (confidence != target_confidences).any():
        msg = f"Confidences do not match: {confidence} != {target_confidences}"
        raise AssertionError(msg)
    if confidence.min() < 0 or confidence.max() > 1:
        msg = f"Confidences should be between 0 and 1: {confidence}"
        raise AssertionError(msg)


def test_probability_disparity() -> None:
    """Test the ProbabilityDisparity confidence extraction method."""
    method = ProbabilityDisparity()

    logits: torch.Tensor = torch.cat(MOCK_LOGITS_MULTI_BEAM, dim=0)
    logits = logits.reshape(-1, len(MOCK_LOGITS_MULTI_BEAM), logits.size(-1)).transpose(
        dim0=0,
        dim1=1,
    )
    logits = torch.softmax(logits, dim=-1)
    confidence: torch.Tensor = method(MOCK_SEQUENCE_MULTI_BEAM, logits)

    next_token_confidences = torch.gather(
        logits,  # tensor of shape (batch_size, gen_length, vocab_size)
        dim=-1,  # we are selecting along the vocab dimension
        index=MOCK_SEQUENCE_MULTI_BEAM[:, -logits.size(1) :].unsqueeze(
            dim=-1,
        ),  # shape (batch_size, gen_length, 1)
    ).squeeze(-1)

    second_highest_probs: torch.return_types.topk = torch.topk(
        input=logits,
        k=2,
        dim=-1,
    )

    target_confidences: torch.Tensor = (
        next_token_confidences - second_highest_probs.values[:, :, 1]  # noqa: PD011 - .values is a property
    )

    if (confidence != target_confidences).any():
        msg = f"Confidences do not match: {confidence} != {target_confidences}"
        raise AssertionError(msg)
