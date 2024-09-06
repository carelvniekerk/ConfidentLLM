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
from transformers import BatchEncoding, TensorType
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
        *,
        return_best_answer_idx: bool = False,
        **kwargs: dict | None,  # noqa: ARG002
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
        if isinstance(self.tokenizer, type(None)):
            raise TokenizerNotSetError(self.tokenizer)

        output_text: list[str] = self.tokenizer.batch_decode(
            generation_output.generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        # Remove new lines and trailing spaces
        output_text = [text_item.replace("\n", "").strip() for text_item in output_text]
        # Add the answer extraction prompt
        output_text = [f"{text_item}. {self.prompt}" for text_item in output_text]

        inputs: BatchEncoding = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=output_text,
            return_tensors=TensorType.PYTORCH,
            padding=True,
        )

        answer_output: GenerateDecoderOnlyOutput = self.model.generate(
            input_ids=inputs.input_ids.to(self.model.device),
            attention_mask=inputs.attention_mask.to(self.model.device),
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

        try:
            best_spans: list[tuple[int, int]] = []
            for search_space, search_span in zip(
                clean_generated_response_tokens,
                answer_tokens,
                strict=True,
            ):
                span: tuple[int, int] = self._find_largest_overlap_span(
                    search_space=search_space,
                    search_span=search_span,
                )
                best_spans.append(span)
        except NoOverlappingSpanFoundError as err:
            logger.warning(err.message)
            return Answer()

        # Extract the answer span confidence
        token_confidences_list: list[torch.Tensor] = []
        for span, probs in zip(
            best_spans,
            clean_generated_response_token_probabilities,
            strict=True,
        ):
            # Average of the answer span tokens used to reduce answer confidence
            token_confidences_list.append(
                torch.tensor(probs[span[0] : span[1] + 1]).mean(),
            )
        answer_confidences: torch.Tensor = torch.tensor(token_confidences_list)

        # Select the answer with the highest confidence
        best_answer_id: int = torch.argmax(answer_confidences).item()  # type: ignore[assignment] # This number is a integer id

        answer: str = self.tokenizer.decode(
            answer_tokens[best_answer_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        answer = answer.replace("\n", "").strip()
        answer_confidence: torch.Tensor = answer_confidences[best_answer_id]

        reasoning: str = self.tokenizer.decode(
            generated_response_tokens[best_answer_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        reasoning = reasoning.replace("\n", "").strip()
        reasoning = f"{reasoning}. {self.prompt} {answer}"

        if return_best_answer_idx:
            return Answer(
                answer=answer,
                confidence=answer_confidence,
                reasoning=reasoning,
                best_answer_idx=best_answer_id,
            )
        return Answer(answer=answer, confidence=answer_confidence, reasoning=reasoning)

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
                    clean_up_tokenization_spaces=True,
                ),
                search_space=self.tokenizer.decode(  # type: ignore  # noqa: PGH003
                    search_space,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ),
            )

        return best_span
