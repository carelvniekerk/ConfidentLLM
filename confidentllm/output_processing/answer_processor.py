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
import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Unpack

import torch
from transformers import BatchEncoding, TensorType
from transformers.generation import GenerateDecoderOnlyOutput

from confidentllm.generation.types import (
    GenerationOutput,
    TokenizerNotSetError,
)
from confidentllm.output_processing.types import (
    OutputProcessor,
    OutputProcessorKwargs,
    ProcessedOutput,
)

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
    best_answer_idx: int = -1


class AnswerProcessor(OutputProcessor):
    """Class for processing the generated answers."""

    def __init__(
        self,
        prompt: str = "So the answer is",
        max_answer_generation_length: int = 20,
    ) -> None:
        """Initialize the processor."""
        super().__init__()
        self.prompt = prompt
        self.max_answer_generation_length = max_answer_generation_length

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
            Answer: The answer and confidence.

        """
        return_best_answer_idx: bool = kwargs.get("return_best_answer_idx", False)

        if isinstance(self.tokenizer, type(None)):
            raise TokenizerNotSetError(self.tokenizer)

        output_text: list[str] = self._decode_tokens(generation_output.generated_ids)
        output_text = [
            self._concatenate_prompt(beam_text, self.prompt)
            for beam_text in output_text
        ]

        inputs: BatchEncoding = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=output_text,
            return_tensors=TensorType.PYTORCH,
            padding=True,
        )
        inputs.to(self.model.device)

        answer_output: GenerateDecoderOnlyOutput = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=self.max_answer_generation_length,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )  # type: ignore[reportAssignmentType]

        answer_tokens: list[list[int]] = (
            answer_output.sequences[:, inputs.input_ids[0].size(-1) :]
            .detach()
            .cpu()
            .tolist()
        )

        generated_response_tokens: list[list[int]] = (
            generation_output.generated_ids[
                :,
                -generation_output.generation_scores.size(-1) :,
            ]
            .detach()
            .cpu()
            .tolist()
        )
        generated_response_token_probabilities: list[list[float]] = (
            generation_output.generation_scores.detach().cpu().tolist()
        )

        # Remove stop tokens from the generated repsonse/reasoning and the answer tokens
        # for improved answer span extraction
        clean_generated_response_tokens: list[list[int]] = []
        clean_generated_response_token_probabilities: list[list[float]] = []
        for response, probs in zip(
            generated_response_tokens,
            generated_response_token_probabilities,
            strict=True,
        ):
            clean_response: list[int] = []
            clean_response_probs: list[float] = []
            for token, prob in zip(response, probs, strict=True):
                if token not in self._ignore_tokens:
                    clean_response.append(token)
                    clean_response_probs.append(prob)
            clean_generated_response_tokens.append(clean_response)
            clean_generated_response_token_probabilities.append(clean_response_probs)

        answer_tokens = [
            [token for token in search_term if token not in self._ignore_tokens]
            for search_term in answer_tokens
        ]

        best_spans: list[tuple[int, int]] = []
        for search_space, search_span in zip(
            clean_generated_response_tokens,
            answer_tokens,
            strict=True,
        ):
            try:
                span: tuple[int, int] = self._find_largest_overlap_span(
                    search_space=search_space,
                    search_span=search_span,
                )
                best_spans.append(span)
            except NoOverlappingSpanFoundError as err:
                logger.warning(err.message)
                span = (-1, -1)
                best_spans.append(span)

        # Extract the answer span confidence
        token_confidences_list: list[torch.Tensor] = []
        reasoning_with_highlighted_answer: list[list[int]] = []
        for span, tokens, probs in zip(
            best_spans,
            clean_generated_response_tokens,
            clean_generated_response_token_probabilities,
            strict=True,
        ):
            if span[0] == -1:
                # No overlapping span found use confidence of 0
                token_confidences_list.append(torch.tensor(0.0))
                reasoning_with_highlighted_answer.append(tokens)
                continue
            # Average of the answer span tokens used to reduce answer confidence
            token_confidences_list.append(
                torch.tensor(probs[span[0] : span[1] + 1]).mean(),
            )
            highlighted_answer_tokens: list[int] = self._highlight_answer_span(
                response_tokens=tokens,
                span=span,
            )
            reasoning_with_highlighted_answer.append(highlighted_answer_tokens)
        answer_confidences: torch.Tensor = torch.tensor(token_confidences_list)

        # Select the answer with the highest confidence
        best_answer_id: int = int(torch.argmax(answer_confidences).item())

        answer: str = self._decode_tokens([answer_tokens[best_answer_id]])[0]
        answer_confidence: torch.Tensor = answer_confidences[best_answer_id]

        reasoning: str = self._decode_tokens(
            tokens=[reasoning_with_highlighted_answer[best_answer_id]],
        )[0]
        reasoning = self._concatenate_prompt(reasoning, f"{self.prompt} {answer}")
        reasoning = self._postprocess_reasoning_text(reasoning)

        if return_best_answer_idx:
            return Answer(
                answer=answer,
                confidence=answer_confidence,
                reasoning=reasoning,
                best_answer_idx=best_answer_id,
            )
        return Answer(answer=answer, confidence=answer_confidence, reasoning=reasoning)

    @cached_property
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

        new_line_token_id: list[int] = self.tokenizer.convert_tokens_to_ids(["\n", "Ċ"])  # type: ignore[assignment]
        ignore_tokens.extend(new_line_token_id)

        full_stop_token_id: int = self.tokenizer.convert_tokens_to_ids(".")  # type: ignore[assignment]
        ignore_tokens.append(full_stop_token_id)

        return ignore_tokens

    @cached_property
    def _bracket_token_ids(self) -> list[int]:
        """Get the tokens ids of the brackets `[]`."""
        bracket_tokens: list[str] = ["[", "]"]
        bracket_token_ids: list[int] = self.tokenizer.convert_tokens_to_ids(
            bracket_tokens,
        )  # type: ignore[assignment]

        return bracket_token_ids

    def _decode_tokens(self, tokens: list[list[int]] | torch.Tensor) -> list[str]:
        """Decode the tokens to a string."""
        if isinstance(self.tokenizer, type(None)):
            raise TokenizerNotSetError(self.tokenizer)

        text: list[str] = self.tokenizer.batch_decode(
            sequences=tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        text = [txt.replace("\n", " ").strip() for txt in text]

        return text

    @staticmethod
    def _concatenate_prompt(text: str, prompt: str) -> str:
        """Concatenate the text with the prompt."""
        text += ". " if text.endswith(".") else " "
        text += f"{prompt}"

        return text

    def _highlight_answer_span(
        self,
        response_tokens: list[int],
        span: tuple[int, int],
    ) -> list[int]:
        """Highlight the answer span in the tokens."""
        highlighted_answer_tokens: list[int] = response_tokens[: span[0]]
        highlighted_answer_tokens.append(self._bracket_token_ids[0])
        highlighted_answer_tokens.extend(response_tokens[span[0] : span[1] + 1])
        highlighted_answer_tokens.append(self._bracket_token_ids[1])
        highlighted_answer_tokens.extend(response_tokens[span[1] + 1 :])

        return highlighted_answer_tokens

    @staticmethod
    def _postprocess_reasoning_text(reasoning: str) -> str:
        """Postprocess the reasoning text."""
        # Step 1: Split lowercase-uppercase, e.g., "sentenceAnd" -> "sentence. And"
        reasoning = re.sub(r"([a-z])([A-Z])", r"\1. \2", reasoning)

        # Step 2: Split word-number pairs, e.g., "reasoning12" -> "reasoning 12"
        reasoning = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", reasoning)

        # Step 3: Ensure space after text and before brackets
        reasoning = re.sub(r"(\w)(\[) ", r"\1 [", reasoning)
        reasoning = re.sub(r"(=)(\[) ", r"\1 [", reasoning)
        reasoning = re.sub(r"^\[ ", r"[", reasoning)

        # Step 4: Add full stop at the end if missing
        if not reasoning.endswith("."):
            reasoning += "."

        # Step 5: Capitalize the first letter of each sentence
        reasoning = re.sub(r"([a-z]) ([A-Z])", r"\1. \2", reasoning)
        reasoning = re.sub(
            pattern=r"(^|(?<=\.\s))([a-z])",
            repl=lambda match: match.group(1) + match.group(2).upper(),
            string=reasoning,
        )

        return reasoning

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
        )
        match = seq_matcher.find_longest_match(
            alo=0,
            ahi=search_space_size,
            blo=0,
            bhi=search_span_size,
        )

        best_span: tuple[int, int] | tuple[()] = (
            (match.a, match.a + match.size - 1) if match.size > 0 else ()
        )

        if not best_span:
            raise NoOverlappingSpanFoundError(
                search_term=self.tokenizer.decode(
                    search_span,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ),
                search_space=self.tokenizer.decode(
                    search_space,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ),
            )

        return best_span
