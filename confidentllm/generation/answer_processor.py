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
"""Model output answer processor."""

import re

import torch
from hydra_zen import make_custom_builds_fn, store

from confidentllm.generation.generation_function import greedy_generate_function
from confidentllm.generation.types import (
    GenerateFunction,
    OutputProcessor,
    TokenizerNotSetError,
)

__all__ = []

builds = make_custom_builds_fn(populate_full_signature=True)


class NoOverlappingSpanFoundError(Exception):
    """Exception raised when no overlapping span is found."""

    def __init__(
        self,
        search_term: str,
        search_space: str,
    ) -> None:
        """Initialize the exception."""
        self.search_term = search_term
        self.search_space = search_space
        self.message = (
            f"No overlapping span found for search term '{search_term}' "
            f"in search space '{search_space}'."
        )
        super().__init__(self.message)


class AnswerProcessor(OutputProcessor):
    """Class for processing the generated answers."""

    def __init__(
        self,
        generator: GenerateFunction,
        prompt: str = "So the answer is:",
    ) -> None:
        """Initialize the processor."""
        super().__init__()
        self.generator = generator
        self.prompt = prompt

    def __call__(
        self,
        output: torch.Tensor,
        output_probs: torch.Tensor,
        **kwargs: dict | None,  # noqa: ARG002
    ) -> tuple[int | None, torch.Tensor]:
        """Process the output.

        Args:
        ----
            output (torch.Tensor): The output to process.
            output_probs (torch.Tensor, optional): The output probabilities.
            **kwargs (dict, optional): Additional keyword arguments.

        Returns:
        -------
            str: The answer.
            torch.Tensor: The confidence of the answer.

        """
        if isinstance(self.tokenizer, type(None)):
            raise TokenizerNotSetError(self.tokenizer)
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        output_text = f"{output_text} {self.prompt} "

        inputs = self.tokenizer(
            output_text,
            return_tensors="pt",
        )

        generated_ids, generation_probs = self.generator(
            input_ids=inputs["input_ids"],  # type: ignore  # noqa: PGH003
            max_length=5,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )  # type: ignore  # noqa: PGH003

        search_term = generated_ids[0][generation_probs[0] >= 0.0]

        best_span = self._find_largest_overlap_span(
            output[0][output_probs[0] >= 0.0],
            search_term,
        )

        conf = output_probs[0][output_probs[0] >= 0.0][best_span[0] : best_span[1] + 1]

        answer = self.tokenizer.decode(search_term, skip_special_tokens=True)
        answer = answer.replace("\n", "").strip()
        answer = self.extract_numbers(answer)
        answer = answer[0] if answer else None

        return answer, conf

    def _find_largest_overlap_span(
        self,
        search_space: torch.Tensor,
        search_span: torch.Tensor,
    ) -> tuple[int, int]:
        """Find the largest overlapping span in the search space."""
        max_overlap = 0
        best_span = None

        search_length = len(search_span)

        for i in range(len(search_space) - search_length + 1):
            current_span = search_space[i : i + search_length]
            overlap = torch.sum(current_span == search_span)

            if overlap > max_overlap:
                max_overlap = overlap
                best_span = (i, i + search_length - 1)

        if not best_span:
            raise NoOverlappingSpanFoundError(
                search_term=self.tokenizer.decode(  # type: ignore  # noqa: PGH003
                    search_span,
                    skip_special_tokens=True,
                ),
                search_space=self.tokenizer.decode(  # type: ignore  # noqa: PGH003
                    search_space,
                    skip_special_tokens=True,
                ),
            )
        return best_span

    @staticmethod
    def extract_numbers(text: str) -> list:
        """Extract numbers from the text."""
        return [int(num) for num in re.findall(r"\b\d+\b", text)]


AnswerProcessorConfig = builds(AnswerProcessor)

answer_processor = AnswerProcessorConfig(
    generator=greedy_generate_function,  # type: ignore  # noqa: PGH003
)

output_processor_store = store(group="output_processor")
output_processor_store(answer_processor, name="answer_processor")
