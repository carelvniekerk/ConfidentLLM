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
"""Tests for the text generation function."""

from typing import TYPE_CHECKING

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from confidentllm.generation.confidence_extraction_methods import PredictiveProbability
from confidentllm.generation.greedy_causal_lm_generation import (
    GreedyCausalLMGenerationMethod,
)
from tests.generation.mock_transformers import (
    MOCK_LOGITS_SINGLE_BEAM,
    mock_model,  # noqa: F401
    mock_tokenizer,  # noqa: F401 - pytest fixtures
)

if TYPE_CHECKING:
    from confidentllm.generation.types import CausalLMGenerationMethod, GenerationOutput


@pytest.fixture
def target_scores() -> torch.Tensor:
    """Get the target scores."""
    logits: torch.Tensor = torch.cat(MOCK_LOGITS_SINGLE_BEAM, dim=0)
    logits = torch.softmax(logits, dim=-1)

    return logits.max(-1).values  # noqa: PD011


def test_generate(
    mock_tokenizer: PreTrainedTokenizer,  # noqa: F811 - pytest fixtures
    mock_model: PreTrainedModel,  # noqa: F811
    target_scores: torch.Tensor,
) -> None:
    """Test the generation method."""
    method: CausalLMGenerationMethod = GreedyCausalLMGenerationMethod(
        confidence_extraction_method=PredictiveProbability(),
    )
    method.set_tokenizer(mock_tokenizer)
    method.set_model(mock_model)

    generation_output: GenerationOutput = method("This is a test.")

    if (generation_output.generated_ids != mock_model.generate().sequences).all():  # type: ignore[attrtibute]
        msg = (
            f"Generated IDs are incorrect, expected {mock_model.generate().sequences}, "  # type: ignore[attrtibute]
            f"got {generation_output.generated_ids}"
        )
        raise AssertionError(msg)

    if (generation_output.generation_scores != target_scores).all():
        msg = (
            f"Generation scores are incorrect, expected {target_scores}, "
            f"got {generation_output.generation_scores}"
        )
        raise AssertionError(msg)
