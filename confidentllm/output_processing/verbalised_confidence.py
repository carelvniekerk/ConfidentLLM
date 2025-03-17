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
"""Model output answer processor."""

import re
from dataclasses import dataclass
from typing import Callable, Protocol, Unpack

import torch
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizer, TensorType
from transformers.generation import GenerateDecoderOnlyOutput
from transformers.tokenization_utils_base import PaddingStrategy

from confidentllm.generation.types import GenerationOutput
from confidentllm.output_processing.answer_processor import Answer, AnswerProcessor
from confidentllm.output_processing.numeric_answer_processor import (
    NumericAnswerProcessor,
)
from confidentllm.output_processing.types import OutputProcessorKwargs

__all__ = [
    "VerbalisedConfidenceAnswerProcessor",
    "VerbalisedConfidenceNumericAnswerProcessor",
]

MAX_CONTEXT_LENGTH: int = 2048


class VerbalisedConfidenceAnswerProcessorProtocol(Protocol):
    """Protocol for verbalised confidence answer processor."""

    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel

    confidence_prompt: str
    max_answer_generation_length: int
    max_context_length: int

    _extract_confidence: Callable[[str], float]


@dataclass
class VerbalisedConfidences:
    """Dataclass for verbalised confidences."""

    confidence_terms: list[str]
    confidences: torch.Tensor


class VerbalisedConfidenceGenerator:
    """Class for generating verbalised confidence."""

    def _generate_verbalised_confidences(
        self: VerbalisedConfidenceAnswerProcessorProtocol,
        generation_output: GenerationOutput,
    ) -> VerbalisedConfidences:
        """Generate the verbalised confidence."""
        output_text: list[str] = self.tokenizer.batch_decode(
            generation_output.generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        # Remove new lines and trailing spaces
        output_text = [text_item.replace("\n", "").strip() for text_item in output_text]
        # Add the answer extraction prompt
        output_text = [
            f"{text_item}. {self.confidence_prompt}" for text_item in output_text
        ]

        max_context_length = min(
            self.tokenizer.model_max_length - self.max_answer_generation_length,
            self.max_context_length,
        )
        inputs: BatchEncoding = self.tokenizer.batch_encode_plus(
            output_text,
            return_tensors=TensorType.PYTORCH,
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=True,
            max_length=max_context_length,
        )

        with torch.no_grad():
            confidence_output: GenerateDecoderOnlyOutput = self.model.generate(
                input_ids=inputs.input_ids.to(self.model.device),
                attention_mask=inputs.attention_mask.to(self.model.device),
                max_new_tokens=self.max_answer_generation_length,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )  # type: ignore[reportAssignmentType]

        confidence_term_token_ids: torch.Tensor = confidence_output.sequences[
            :,
            inputs.input_ids.size(-1) :,
        ]

        confidence_terms: list[str] = self.tokenizer.batch_decode(  # type: ignore  # noqa: PGH003
            confidence_term_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        confidence_terms = [term.replace("\n", "").strip() for term in confidence_terms]
        confidences: torch.Tensor = torch.Tensor(
            [self._extract_confidence(term) for term in confidence_terms],
        )

        return VerbalisedConfidences(
            confidence_terms=confidence_terms,
            confidences=confidences,
        )

    @staticmethod
    def _extract_confidence(generated_text: str) -> float:
        # Define the regex pattern to match the confidence statement
        pattern = r"\s*(\d+)"

        # Search for the pattern in the generated text
        match = re.search(pattern, generated_text)

        # If a match is found, extract the confidence value
        if match:
            confidence_value = int(match.group(1))
            # Ensure the confidence value is within the range 0-100
            if 0 <= confidence_value <= 100:  # noqa: PLR2004
                return float(confidence_value) / 100.0

        # Return a default value if no valid confidence value is found
        return 1.0  # Default confidence value (strict 100%)


class VerbalisedConfidenceAnswerProcessor(
    AnswerProcessor,
    VerbalisedConfidenceGenerator,
):
    """Answer processor with verbalised confidence generation."""

    def __init__(
        self,
        prompt: str = (
            "So the answer is/My confidence that this answer is correct (0-100) is"
        ),
        max_answer_generation_length: int = 20,
    ) -> None:
        """Initialize the processor."""
        answer_prompt, confidence_prompt = prompt.split("/", 1)

        super().__init__(
            prompt=answer_prompt,
            max_answer_generation_length=max_answer_generation_length,
        )

        self.confidence_prompt = confidence_prompt

    def __call__(
        self,
        generation_output: GenerationOutput,
        **kwargs: Unpack[OutputProcessorKwargs],
    ) -> Answer:
        """Process the output.

        Args:
        ----
            generation_output (GenerationOutput): The generation output.
            **kwargs (dict, optional): Additional keyword arguments.

        Returns:
        -------
            str: The answer.
            torch.Tensor: The confidence of the answer.

        """
        answer_object = super().__call__(
            generation_output,
            **kwargs,
        )

        confidences: VerbalisedConfidences = self._generate_verbalised_confidences(  # type: ignore[misc]
            generation_output,
        )

        confidence: torch.Tensor = confidences.confidences[
            answer_object.best_answer_idx
        ]

        confidence_term: str = confidences.confidence_terms[
            answer_object.best_answer_idx
        ]
        reasoning: str = (
            f"{answer_object.reasoning}. {self.confidence_prompt} {confidence_term}"
        )

        return Answer(
            answer=answer_object.answer,
            confidence=confidence,
            reasoning=reasoning,
        )


class VerbalisedConfidenceNumericAnswerProcessor(
    NumericAnswerProcessor,
    VerbalisedConfidenceGenerator,
):
    """Numeric answer processor with verbalised confidence generation."""

    def __init__(
        self,
        prompt: str = (
            "So the answer as a number is/"
            "My confidence that this answer is correct (0-100) is"
        ),
        max_answer_generation_length: int = 20,
    ) -> None:
        """Initialize the processor."""
        answer_prompt, confidence_prompt = prompt.split(":", 1)
        answer_prompt += ":"

        super().__init__(
            prompt=answer_prompt,
            max_answer_generation_length=max_answer_generation_length,
        )

        self.confidence_prompt = confidence_prompt

    def __call__(
        self,
        generation_output: GenerationOutput,
        **kwargs: Unpack[OutputProcessorKwargs],
    ) -> Answer:
        """Process the output.

        Args:
        ----
            generation_output (GenerationOutput): The generation output.
            **kwargs (dict, optional): Additional keyword arguments.

        Returns:
        -------
            str: The answer.
            torch.Tensor: The confidence of the answer.

        """
        answer_object = super().__call__(
            generation_output,
            **kwargs,
        )

        confidences: VerbalisedConfidences = self._generate_verbalised_confidences(  # type: ignore[misc]
            generation_output,
        )

        confidence: torch.Tensor = confidences.confidences[
            answer_object.best_answer_idx
        ]

        confidence_term: str = confidences.confidence_terms[
            answer_object.best_answer_idx
        ]
        reasoning: str = (
            f"{answer_object.reasoning} {self.confidence_prompt} {confidence_term}"
        )

        return Answer(
            answer=answer_object.answer,
            confidence=confidence,
            reasoning=reasoning,
        )
