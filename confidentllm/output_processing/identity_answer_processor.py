# coding=utf-8
# --------------------------------------------------------------------------------
# Project: ConfidentLLM
# Author: Carel van Niekerk
# Year: 2025
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

from typing import Unpack

import torch

from confidentllm.generation.types import GenerationOutput, TokenizerNotSetError
from confidentllm.output_processing.answer_processor import Answer, AnswerProcessor
from confidentllm.output_processing.types import OutputProcessorKwargs

__all__ = ["IdentityAnswerProcessor"]


class IdentityAnswerProcessor(AnswerProcessor):
    """Class for processing the generated answers."""

    def __init__(
        self,
        prompt: str = "",
        max_answer_generation_length: int = -1,
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
        return_best_answer_idx: bool = kwargs.get("return_best_answer_idx", False)

        if isinstance(generation_output.generated_ids, list):
            answer_object = Answer(
                answer=generation_output.generated_ids[0],  # type: ignore[arg-type]
                confidence=torch.tensor(1.0),
            )
            return answer_object

        if self.tokenizer is None:
            raise TokenizerNotSetError(self.tokenizer)

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

        answer_confidences: torch.Tensor = torch.tensor(
            clean_generated_response_token_probabilities,
        )
        answer_confidences = answer_confidences.mean(-1)

        # Select the answer with the highest confidence
        best_answer_id: int = int(torch.argmax(answer_confidences).item())

        answer: str = self._decode_tokens(
            tokens=[generated_response_tokens[best_answer_id]],
        )[0]
        answer_confidence: torch.Tensor = answer_confidences[best_answer_id]

        if return_best_answer_idx:
            return Answer(
                answer=answer,
                confidence=answer_confidence,
                best_answer_idx=best_answer_id,
            )
        return Answer(answer=answer, confidence=answer_confidence)
