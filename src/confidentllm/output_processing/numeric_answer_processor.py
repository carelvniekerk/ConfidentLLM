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
from typing import Unpack

from confidentllm.generation.types import GenerationOutput
from confidentllm.output_processing.answer_processor import Answer, AnswerProcessor
from confidentllm.output_processing.types import OutputProcessorKwargs

__all__ = ["NumericAnswerProcessor"]


class NumericAnswerProcessor(AnswerProcessor):
    """Class for processing the generated answers."""

    def __init__(
        self,
        prompt: str = "So the answer as a number is",
        max_answer_generation_length: int = 20,
    ) -> None:
        """Initialize the processor."""
        super().__init__(
            prompt=prompt,
            max_answer_generation_length=max_answer_generation_length,
        )

    def __call__(
        self,
        generation_output: GenerationOutput,
        **kwargs: Unpack[OutputProcessorKwargs],
    ) -> Answer:
        """Process the output.

        Args:
        ----
            generation_output (GenerationOutput): The generation output.
            return_best_answer_idx (bool, optional): Return most confident answer index.
            **kwargs (dict, optional): Additional keyword arguments.

        Returns:
        -------
            Answer: The processed answer and confidence.

        """
        answer_object = super().__call__(
            generation_output,
            **kwargs,
        )
        possible_answer: list[float] = self._extract_numbers(answer_object.answer)
        answer: float = possible_answer[-1] if possible_answer else -1

        return Answer(
            answer=str(answer),
            confidence=answer_object.confidence,
            reasoning=answer_object.reasoning,
        )

    @staticmethod
    def _extract_numbers(text: str) -> list[float]:
        """Extract numbers from the text."""
        # Use a regex pattern that matches numbers with optional commas
        number_pattern = r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b"
        numbers = re.findall(number_pattern, text)
        # Remove commas from the numbers and convert to integers or floats
        return [
            float(num.replace(",", "")) if "." in num else int(num.replace(",", ""))
            for num in numbers
        ]
