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
"""Model output answer processor."""

import re

import torch

from confidentllm.generation.types import (
    GenerateFunction,
    GenerationOutput,
    TokenizerNotSetError,
)
from confidentllm.output_processing.answer_processor import Answer, AnswerProcessor

__all__ = []


class VerbalisedConfidenceProcessor(AnswerProcessor):
    """Class for processing the generated answers, providing verbalised confidence."""

    def __init__(
        self,
        generator: GenerateFunction,
        prompt: str = "My confidence that this answer is correct (0-100):",
        max_answer_generation_length: int = 20,
    ) -> None:
        """Initialize the processor."""
        super().__init__(generator, prompt, max_answer_generation_length)  # type: ignore  # noqa: PGH003

    def __call__(
        self,
        generation_output: GenerationOutput,
        **kwargs: dict | None,  # noqa: ARG002
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
        if isinstance(self.tokenizer, type(None)):
            raise TokenizerNotSetError(self.tokenizer)

        output_text = self.tokenizer.decode(
            generation_output.generated_ids[0],
            skip_special_tokens=True,
        )

        output_text = f"{output_text} So the answer is: "

        inputs = self.tokenizer(
            output_text,
            return_tensors="pt",
        )

        answer_output = self.generator(
            input_ids=inputs["input_ids"],  # type: ignore  # noqa: PGH003
            max_length=self.max_answer_generation_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )  # type: ignore  # noqa: PGH003

        answer_term = answer_output.generated_ids[0][
            answer_output.generation_scores[0] >= 0.0
        ]

        answer = self.tokenizer.decode(answer_term, skip_special_tokens=True)
        answer = answer.replace("\n", "").strip()
        answer = self._extract_numeric_answer(answer)
        answer = answer[0] if answer else -1

        output_text = self.tokenizer.decode(
            generation_output.generated_ids[0],
            skip_special_tokens=True,
        )
        output_text = f"{output_text} {self.prompt} "

        inputs = self.tokenizer(
            output_text,
            return_tensors="pt",
        )

        confidence_output = self.generator(
            input_ids=inputs["input_ids"],  # type: ignore  # noqa: PGH003
            max_length=self.max_answer_generation_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )  # type: ignore  # noqa: PGH003

        confidence_term = confidence_output.generated_ids[0][
            confidence_output.generation_scores[0] >= 0.0
        ]

        confidence_term = self.tokenizer.decode(
            confidence_term,
            skip_special_tokens=True,
        )
        confidence_term = confidence_term.replace("\n", "").strip()
        confidence = self._extract_confidence(confidence_term)
        confidence = torch.Tensor([confidence])

        return Answer(answer=str(answer), confidence=confidence)

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

    @staticmethod
    def _extract_numeric_answer(text: str) -> list:
        """Extract numbers from the text."""
        # Use a regex pattern that matches numbers with optional commas
        number_pattern = r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b"
        numbers = re.findall(number_pattern, text)
        # Remove commas from the numbers and convert to integers or floats
        return [
            float(num.replace(",", "")) if "." in num else int(num.replace(",", ""))
            for num in numbers
        ]
