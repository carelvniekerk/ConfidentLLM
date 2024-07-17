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
# limitations under the License."
"""Types for the generation module."""

from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module
from transformers.tokenization_utils import PreTrainedTokenizer

from confidentllm.decoding_strategies.types import DecodingStrategy

__all__ = [
    "GenerateFunction",
    "CausalLMGenerationMethod",
    "OutputProcessor",
    "ModelNotSetError",
    "TokenizerNotSetError",
]


class GenerateFunction(ABC):
    """Interface for the generator used in the uncertainllm package."""

    def __init__(self, decoding_strategy: DecodingStrategy) -> None:
        """Initialize the generate function.

        Args:
        ----
            decoding_strategy (DecodingStrategy): The decoding strategy to use.

        """
        self.decoding_strategy = decoding_strategy
        self.model: Module | None = None

    def set_model(self, model: Module) -> None:
        """Set the model for the generation method."""
        self.model = model

    @abstractmethod
    def __call__(  # noqa: PLR0913
        self,
        input_ids: Tensor,
        max_length: int,
        min_length: int,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        bad_words_ids: list[list[int]] | None,
        pad_token_id: int,
        eos_token_id: int | None,
        batch_size: int,
        attention_mask: Tensor | None,
        model_specific_kwargs: dict | None,
    ) -> tuple[Tensor, Tensor]:
        """Generate sequences based on the given input.

        Args:
        ----
            input_ids (torch.Tensor): The input tensor.
            attention_mask (torch.Tensor | None): The attention mask tensor.
            max_length (int): The maximum length of the generated sequences.
            min_length (int): The minimum length of the generated sequences.
            repetition_penalty (float): The repetition penalty.
            no_repeat_ngram_size (int): The size of the n-grams to avoid repetition.
            bad_words_ids (list[list[int]] | None): The list of bad word IDs to avoid.
            pad_token_id (int): The ID of the padding token.
            eos_token_id (int | None): The ID of the end-of-sequence token.
            batch_size (int): The batch size.
            decoding_kwargs (dict | None): Additional decoding arguments.
            model_specific_kwargs (dict | None): Additional model-specific arguments.

        Returns:
        -------
            tuple[torch.Tensor, torch.Tensor]: The generated sequences and their scores.

        """
        ...


class CausalLMGenerationMethod(ABC):
    """Protocol for the generation method."""

    def __init__(self, generator: GenerateFunction) -> None:  # noqa: D107
        self.tokenizer: PreTrainedTokenizer | None = None
        self.generator = generator

    def set_tokenizer(self, tokenizer: PreTrainedTokenizer) -> None:
        """Set the tokenizer for the generation method."""
        self.tokenizer = tokenizer

    def set_model(self, model: Module) -> None:
        """Set the model for the generation method."""
        self.generator.set_model(model)

    @abstractmethod
    def __call__(self, prompt: str, max_length: int = 256) -> tuple[Tensor, Tensor]:
        """Generate text based on the given prompt.

        Args:
        ----
            prompt (str): The prompt for text generation.
            max_length (int, optional): The maximum length of the generated text.

        Returns:
        -------
            generated_ids (torch.Tensor): The generated token IDs.
            generation_probs (torch.Tensor): The generation probabilities.

        """
        ...


class OutputProcessor(ABC):
    """Class for processing the generated answers."""

    def __init__(self) -> None:
        """Initialize the processor."""
        self.model: Module | None = None
        self.tokenizer: PreTrainedTokenizer | None = None
        self.generator: GenerateFunction | None = None

    def set_model(self, model: Module) -> None:
        """Set the model for the generation method."""
        if isinstance(self.generator, type(None)):
            msg = "The generator is not set."
            raise TypeError(msg)
        self.generator.set_model(model)

    def set_tokenizer(self, tokenizer: PreTrainedTokenizer) -> None:
        """Set the tokenizer for the generation method."""
        self.tokenizer = tokenizer

    @abstractmethod
    def __call__(
        self,
        output: Tensor,
        output_probs: Tensor,
        **kwargs: dict | None,
    ) -> tuple[int | str | None, Tensor]:
        """Process the generated answer."""
        ...


class ModelNotSetError(Exception):
    """Exception raised when the model is not set."""

    def __init__(self, model: Module | None) -> None:  # noqa: D107
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
