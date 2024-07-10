# coding=utf-8
# --------------------------------------------------------------------------------
# Project: ConfidentLLM
# Author: Carel van Niekerk
# Year: 2024
# Group: Dialogue Systems and Machine Learning Group
# Institution: Heinrich Heine University Düsseldorf
# --------------------------------------------------------------------------------
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

from typing import Protocol

from torch import Tensor


class GenerateFunction(Protocol):
    """Interface for the generator used in the uncertainllm package."""

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


class CausalLMGenerationMethod(Protocol):
    """Protocol for the generation method."""

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


class OutputProcessor(Protocol):
    """Class for processing the generated answers."""

    def __call__(
        self,
        output: Tensor,
        output_probs: Tensor,
        **kwargs: dict | None,
    ) -> tuple[int | str | None, Tensor]:
        """Process the generated answer."""
        ...
