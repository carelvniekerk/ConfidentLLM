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
"""Types for the generation module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer

from confidentllm.confidence_metrics import ConfidenceMetric

__all__ = [
    "CausalLMGenerationMethod",
    "GenerationOutput",
    "ModelNotSetError",
    "TokenizerNotSetError",
]


@dataclass
class GenerationOutput:
    """Dataclass for the output of the generation method."""

    generated_ids: Tensor
    generation_scores: Tensor


class ModelNotSetError(Exception):
    """Exception raised when the model is not set."""

    def __init__(self, model: PreTrainedModel | None) -> None:  # noqa: D107
        self.model = model
        self.message = (
            "The model is not set. Please set the model before generating text."
        )
        super().__init__(self.message)


class TokenizerNotSetError(Exception):
    """Exception raised when the model is not set."""

    def __init__(self, tokenizer: PreTrainedTokenizer | None) -> None:  # noqa: D107
        self.tokenizer = tokenizer
        self.message = (
            "The tokenizer is not set. Please set the tokenizer before generating text."
        )
        super().__init__(self.message)


class CausalLMGenerationMethod(ABC):
    """Protocol for the generation method."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        confidence_metric: ConfidenceMetric,
        tokenizer: PreTrainedTokenizer | None = None,
        model: PreTrainedModel | None = None,
        max_length: int = 256,
        sampling: bool = False,
        temperature: float = 1.0,
        num_beams: int = 1,
        zero_shot_prompt: str = "",
    ) -> None:
        """Initialize the generation method."""
        self.tokenizer = tokenizer
        self.model = model

        self.confidence_metric = confidence_metric

        self.sampling = sampling
        self.temperature = temperature
        self.num_beams = num_beams
        self.max_length = max_length

        self.zero_shot_prompt = zero_shot_prompt

    def set_tokenizer(self, tokenizer: PreTrainedTokenizer) -> None:
        """Set the tokenizer for the generation method."""
        self.tokenizer = tokenizer

    def set_model(self, model: PreTrainedModel) -> None:
        """Set the model for the generation method."""
        self.model = model

    @abstractmethod
    def __call__(
        self,
        prompt: str | None = None,
        responses: list[str] | None = None,
    ) -> GenerationOutput:
        """Generate text based on the given prompt.

        Args:
        ----
            prompt (str | None): The prompt for text generation.
            responses (list[str] | None): Optional list of responses to consider.

        Returns:
        -------
            GenerationOutput: The generated text and their scores.

        """
        ...  # pragma: no cover
