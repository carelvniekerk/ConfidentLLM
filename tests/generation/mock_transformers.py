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
"""Mock tokenizer for testing."""

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizer
from transformers.generation import GenerateDecoderOnlyOutput

__all__ = ["MOCK_SEQUENCE", "MOCK_LOGITS", "mock_tokenizer", "mock_model"]

MOCK_INPUTS: torch.Tensor = torch.tensor([[1, 2, 3]])
MOCK_SEQUENCE_SINGLE_BEAM: torch.Tensor = torch.tensor([[1, 2, 3, 4, 5, 6]])
MOCK_SEQUENCE_MULTI_BEAM: torch.Tensor = torch.tensor(
    [
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 6, 0, 6],
        [1, 2, 3, 0, 1, 0],
    ],
)
MOCK_LOGITS_MULTI_BEAM: tuple[torch.Tensor, ...] = (
    torch.tensor(
        [
            [
                [0.1, 0.0, 0.0, 0.0, 0.6, 0.0, 0.3],
            ],
            [
                [0.1, 0.0, 0.0, 0.0, 0.6, 0.0, 0.3],
            ],
            [
                [0.1, 0.0, 0.0, 0.0, 0.6, 0.0, 0.3],
            ],
        ],
    ).log(),
    torch.tensor(
        [
            [
                [0.1, 0.1, 0.0, 0.0, 0.0, 0.8, 0.0],
            ],
            [
                [0.1, 0.1, 0.0, 0.0, 0.0, 0.8, 0.0],
            ],
            [
                [0.1, 0.1, 0.0, 0.0, 0.0, 0.8, 0.0],
            ],
        ],
    ).log(),
    torch.tensor(
        [
            [
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
            ],
            [
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
            ],
            [
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
            ],
        ],
    ).log(),
)
MOCK_LOGITS_SINGLE_BEAM: tuple[torch.Tensor, ...] = tuple(
    logits[0] for logits in MOCK_LOGITS_MULTI_BEAM
)


@pytest.fixture
def mock_tokenizer() -> PreTrainedTokenizer:
    """Fixture to mock the tokenizer."""
    tokenizer = MagicMock(spec=PreTrainedTokenizer)
    tokenizer.apply_chat_template.return_value = BatchEncoding(
        {
            "input_ids": MOCK_INPUTS,
            "attention_mask": torch.ones_like(MOCK_INPUTS),
        },
    )
    return tokenizer


@pytest.fixture
def mock_model() -> PreTrainedModel:
    """Fixture to mock the model."""
    model = MagicMock(spec=PreTrainedModel)
    model.device = torch.device("cpu")
    model.output_logits = True

    def generate_side_effect(
        *args: tuple[Any, ...],  # noqa: ARG001
        **kwargs: dict[str, Any],
    ) -> GenerateDecoderOnlyOutput:
        """Customize behavior for the generate method."""
        # Greedy decoding
        if kwargs.get("num_beams", 1) == 1:
            if not model.output_logits:
                return GenerateDecoderOnlyOutput(
                    sequences=MOCK_SEQUENCE_SINGLE_BEAM,  # type: ignore[argument]
                )
            if kwargs.get("max_new_tokens", 3) == 3:  # noqa: PLR2004
                return GenerateDecoderOnlyOutput(
                    sequences=MOCK_SEQUENCE_SINGLE_BEAM,  # type: ignore[argument]
                    logits=MOCK_LOGITS_SINGLE_BEAM,  # type: ignore[argument]
                )
            if kwargs.get("max_new_tokens", 3) == 2:  # noqa: PLR2004
                return GenerateDecoderOnlyOutput(
                    sequences=MOCK_SEQUENCE_MULTI_BEAM,  # type: ignore[argument]
                    logits=MOCK_LOGITS_MULTI_BEAM[1:],  # type: ignore[argument]
                )
        # Beam search decoding
        if kwargs.get("num_beams", 1) == len(MOCK_LOGITS_MULTI_BEAM):
            # Single token sampling for CoT Decoding
            if kwargs.get("max_new_tokens", 3) == 1:
                return GenerateDecoderOnlyOutput(
                    sequences=MOCK_SEQUENCE_MULTI_BEAM[:, : MOCK_INPUTS.size(1) + 1],  # type: ignore[argument]
                    logits=MOCK_LOGITS_MULTI_BEAM[:1],  # type: ignore[argument]
                )
            if kwargs.get("max_new_tokens", 3) == 3:  # noqa: PLR2004
                return GenerateDecoderOnlyOutput(
                    sequences=MOCK_SEQUENCE_MULTI_BEAM,  # type: ignore[argument]
                    logits=MOCK_LOGITS_MULTI_BEAM,  # type: ignore[argument]
                )

        return GenerateDecoderOnlyOutput()

    # Set the side_effect to the generate method
    model.generate.side_effect = generate_side_effect

    return model
