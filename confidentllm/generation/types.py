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
from typing import Iterator

from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer

__all__ = [
    "GenerationOutput",
    "ChatUserMessage",
    "ChatAssistantMessage",
    "ChatConversation",
    "ModelNotSetError",
    "TokenizerNotSetError",
    "CausalLMGenerationMethod",
]


@dataclass
class GenerationOutput:
    """Dataclass for the output of the generation method."""

    generated_ids: Tensor
    generation_scores: Tensor


@dataclass
class ChatUserMessage:
    """Dataclass for the chat user message."""

    content: str
    role: str = "user"


@dataclass
class ChatAssistantMessage:
    """Dataclass for the chat user message."""

    content: str
    role: str = "assistant"


@dataclass
class ChatConversation:
    """Dataclass for the chat conversation."""

    messages: list[list[ChatUserMessage | ChatAssistantMessage]]

    def __iter__(self) -> Iterator[list[dict[str, str]]]:
        """Convert the chat conversation to a iterator."""
        for chat in self.messages:
            yield [message.__dict__ for message in chat]


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


class ConfidenceExtractionMethod(ABC):
    """Method for extracting confidence scores from the logits."""

    @abstractmethod
    def __call__(
        self,
        next_token_ids: Tensor,
        scores: Tensor,
    ) -> Tensor:
        """Extract the confidence scores from the logits."""
        ...


class CausalLMGenerationMethod(ABC):
    """Protocol for the generation method."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        confidence_extraction_method: ConfidenceExtractionMethod,
        tokenizer: PreTrainedTokenizer | None = None,
        model: PreTrainedModel | None = None,
        max_length: int = 256,
        sampling: bool = True,
        temperature: float = 1.0,
        num_beams: int = 1,
    ) -> None:
        """Initialize the generation method."""
        self.tokenizer = tokenizer
        self.model = model

        self.confidence_extraction_method = confidence_extraction_method

        if not isinstance(sampling, bool):
            msg = "Sampling must be a boolean value."
            raise TypeError(msg)
        self.sampling = sampling
        self.temperature = temperature
        self.num_beams = num_beams
        self.max_length = max_length

    def set_tokenizer(self, tokenizer: PreTrainedTokenizer) -> None:
        """Set the tokenizer for the generation method."""
        self.tokenizer = tokenizer

    def set_model(self, model: PreTrainedModel) -> None:
        """Set the model for the generation method."""
        self.model = model

    @abstractmethod
    def __call__(
        self,
        prompt: str,
    ) -> GenerationOutput:
        """Generate text based on the given prompt.

        Args:
        ----
            prompt (str): The prompt for text generation.
            max_length (int, optional): The maximum length of the generated text.
            temperature (float, optional): The temperature for sampling. Default is 1.0.
            num_beams (int, optional): The number of beams for beam search.

        Returns:
        -------
            GenerationOutput: The generated text and their scores.

        """
        ...
