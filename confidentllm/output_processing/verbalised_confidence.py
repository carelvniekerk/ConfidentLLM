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
from typing import Callable, Protocol

import torch
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizer
from transformers.generation import GenerateDecoderOnlyOutput

from confidentllm.generation.types import GenerationOutput
from confidentllm.output_processing.answer_processor import Answer, AnswerProcessor
from confidentllm.output_processing.numeric_answer_processor import (
    NumericAnswerProcessor,
)

__all__ = [
    "VerbalisedConfidenceAnswerProcessor",
    "VerbalisedConfidenceNumericAnswerProcessor",
]


class VerbalisedConfidenceAnswerProcessorProtocol(Protocol):
    """Protocol for verbalised confidence answer processor."""

    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel

    confidence_prompt: str
    max_answer_generation_length: int

    _extract_confidence: Callable[[str], float]


class VerbalisedConfidenceGenerator:
    """Class for generating verbalised confidence."""

    def _generate_verbalised_confidence(
        self: VerbalisedConfidenceAnswerProcessorProtocol,
        generation_output: GenerationOutput,
    ) -> torch.Tensor:
        """Generate the verbalised confidence."""
        output_text = self.tokenizer.decode(
            generation_output.generated_ids[0],
            skip_special_tokens=True,
        )
        output_text = f"{output_text} {self.confidence_prompt}"

        inputs: BatchEncoding = self.tokenizer(
            output_text,
            return_tensors="pt",
        )

        confidence_output: GenerateDecoderOnlyOutput = self.model.generate(
            input_ids=inputs.input_ids.to(self.model.device),
            attention_mask=inputs.attention_mask.to(self.model.device),
            max_length=self.max_answer_generation_length + inputs.input_ids.shape[-1],
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )  # type: ignore[reportAssignmentType]

        confidence_term_ids: torch.Tensor = confidence_output.sequences[0][
            inputs.input_ids[0].shape[1] :
        ]

        confidence_term: str = self.tokenizer.decode(  # type: ignore  # noqa: PGH003
            confidence_term_ids,
            skip_special_tokens=True,
        )
        confidence_term = confidence_term.replace("\n", "").strip()
        confidence: torch.Tensor = torch.Tensor(
            [self._extract_confidence(confidence_term)],
        )

        return confidence

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
            "So the answer is: My confidence that this answer is correct (0-100):"
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
        **kwargs: dict | None,
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
        answer_object = super().__call__(generation_output, **kwargs)

        confidence = self._generate_verbalised_confidence(generation_output)  # type: ignore[reportAttributeAccessIssue]

        return Answer(
            answer=answer_object.answer,
            confidence=confidence,
            reasoning=answer_object.reasoning,
        )


class VerbalisedConfidenceNumericAnswerProcessor(
    NumericAnswerProcessor,
    VerbalisedConfidenceGenerator,
):
    """Numeric answer processor with verbalised confidence generation."""

    def __init__(
        self,
        prompt: str = (
            "So the answer (in numbers) is: "
            "My confidence that this answer is correct (0-100):"
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
        **kwargs: dict | None,
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
        answer_object = super().__call__(generation_output, **kwargs)

        confidence = self._generate_verbalised_confidence(generation_output)  # type: ignore[reportAttributeAccessIssue]

        return Answer(
            answer=answer_object.answer,
            confidence=confidence,
            reasoning=answer_object.reasoning,
        )
