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

import difflib
import logging
from dataclasses import dataclass, field

import torch
from transformers import BatchEncoding
from transformers.generation import GenerateDecoderOnlyOutput

from confidentllm.generation.types import (
    GenerationOutput,
    TokenizerNotSetError,
)
from confidentllm.output_processing.types import OutputProcessor, ProcessedOutput

__all__ = ["Answer", "AnswerProcessor"]

logger = logging.getLogger("__main__")


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


@dataclass
class Answer(ProcessedOutput):
    """Dataclass for the processed output of the generation method."""

    answer: str = "-1"
    confidence: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    reasoning: str = field(default_factory=str)


class AnswerProcessor(OutputProcessor):
    """Class for processing the generated answers."""

    def __init__(
        self,
        prompt: str = "So the answer is:",
        max_answer_generation_length: int = 20,
    ) -> None:
        """Initialize the processor."""
        super().__init__()
        self.prompt = prompt
        self.max_answer_generation_length = max_answer_generation_length

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
            Answer: The answer and confidence.

        """
        if isinstance(self.tokenizer, type(None)):
            raise TokenizerNotSetError(self.tokenizer)

        output_text: str = self.tokenizer.decode(
            generation_output.generated_ids[0],
            skip_special_tokens=True,
        )
        # Remove new lines and trailing spaces
        output_text = output_text.replace("\n", "").strip()
        # Add the answer extraction prompt
        output_text = f"{output_text} {self.prompt}"

        inputs: BatchEncoding = self.tokenizer(
            output_text,
            return_tensors="pt",
        )

        answer_output: GenerateDecoderOnlyOutput = self.model.generate(
            input_ids=inputs.input_ids.to(self.model.device),
            attention_mask=inputs.attention_mask.to(self.model.device),
            max_new_tokens=self.max_answer_generation_length,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )  # type: ignore[reportAssignmentType]

        search_term: list[int] = (
            answer_output.sequences[0][inputs.input_ids[0].size(-1) :]
            .detach()
            .cpu()
            .tolist()
        )

        search_space: list[int] = (
            generation_output.generated_ids[0][
                -generation_output.generation_scores.size(-1) :
            ]
            .detach()
            .cpu()
            .tolist()
        )
        search_probs: list[float] = (
            generation_output.generation_scores[0].detach().cpu().tolist()
        )

        clean_search_space: list[int] = []
        clean_search_probs: list[float] = []
        for token, prob in zip(search_space, search_probs, strict=True):
            if token not in self._ignore_tokens:
                clean_search_space.append(token)
                clean_search_probs.append(prob)
        search_term = [
            token for token in search_term if token not in self._ignore_tokens
        ]

        try:
            best_span: tuple[int, int] = self._find_largest_overlap_span(
                clean_search_space,
                search_term,
            )
        except NoOverlappingSpanFoundError as err:
            logger.warning(err.message)
            return Answer()

        conf: torch.Tensor = torch.tensor(
            clean_search_probs[best_span[0] : best_span[1] + 1],
        ).mean()

        answer: str = self.tokenizer.decode(search_term, skip_special_tokens=True)
        answer = answer.replace("\n", "").strip()

        reasoning: str = self.tokenizer.decode(search_space, skip_special_tokens=True)
        reasoning = reasoning.replace("\n", "").strip()
        reasoning = f"{reasoning} {self.prompt} {answer}"

        return Answer(answer=answer, confidence=conf, reasoning=reasoning)

    @property
    def _ignore_tokens(self) -> list[int]:
        """Get the tokens to ignore during answer extraction and mathing."""
        if isinstance(self.tokenizer, type(None)):
            raise TokenizerNotSetError(self.tokenizer)

        ignore_tokens: list[int] = []
        if self.tokenizer.pad_token_id is not None:
            ignore_tokens.append(self.tokenizer.pad_token_id)
        if self.tokenizer.eos_token_id is not None:
            ignore_tokens.append(self.tokenizer.eos_token_id)
        if self.tokenizer.bos_token_id is not None:
            ignore_tokens.append(self.tokenizer.bos_token_id)

        new_line_token_id: int = self.tokenizer.convert_tokens_to_ids("\n")  # type: ignore[reportAssignmentType]
        ignore_tokens.append(new_line_token_id)

        return ignore_tokens

    def _find_largest_overlap_span(
        self,
        search_space: list[int],
        search_span: list[int],
    ) -> tuple[int, int]:
        """Find the largest overlapping span in the search space."""
        search_space_size: int = len(search_space)
        search_span_size: int = len(search_span)

        seq_matcher = difflib.SequenceMatcher(
            isjunk=None,
            a=search_space,
            b=search_span,
        )  # type: ignore  # noqa: PGH003 - numpy arrays can be dealt with as sequences in this method.
        match = seq_matcher.find_longest_match(
            alo=0,
            ahi=search_space_size,
            blo=0,
            bhi=search_span_size,
        )
        match = difflib.Match(
            a=search_space_size - match.a - match.size,
            b=search_span_size - match.b - match.size,
            size=match.size,
        )

        best_span = (
            (match.a, match.a + match.size) if match.a < search_space_size else ()
        )

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
