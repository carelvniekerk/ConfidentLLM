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
"""Greedy generation method for API based causal language models."""

from typing import TYPE_CHECKING

import torch
from hydra_zen import store
from transformers import BatchEncoding, TensorType
from transformers.generation import GenerateDecoderOnlyOutput

from confidentllm.conversations.create_chat import create_conversation
from confidentllm.generation.types import (
    CausalLMGenerationMethod,
    GenerationOutput,
    ModelNotSetError,
    TokenizerNotSetError,
)
from confidentllm.hydra_tools import builds

if TYPE_CHECKING:
    from confidentllm.conversations.types import ChatConversation

__all__: list[str] = []


class GreedyAPILMGenerationMethod(CausalLMGenerationMethod):
    """Greedy generation method for causal language models."""

    def __call__(
        self,
        prompt: str | None = None,
        responses: list[str] | None = None,
    ) -> GenerationOutput:
        """Generate text based on the given prompt."""
        if self.model is None:
            raise ModelNotSetError(self.model)

        if self.zero_shot_prompt and prompt is not None:
            prompt = f"{prompt}\n\n{self.zero_shot_prompt}"
        elif self.zero_shot_prompt and responses is not None:
            responses[0] = f"{responses[0]}\n\n{self.zero_shot_prompt}"

        conversation: ChatConversation = create_conversation(
            responses=[responses] if responses is not None else [""],  # type: ignore[list-item]
            questions=[prompt] if prompt is not None else None,
        )

        if responses is None:
            conversation.messages[0] = conversation.messages[0][:-1]

        generation_output: list[str] = self.model.generate(
            conversation=conversation,
            max_new_tokens=self.max_length,
            temperature=self.temperature,
        )  # type: ignore[reportAssignmentType]
        generation_scores = torch.zeros(
            size=(len(generation_output), 1),
            dtype=torch.float32,
        )

        return GenerationOutput(
            generated_ids=generation_output,  # type: ignore[arg-type]
            generation_scores=generation_scores,
        )


GreedyAPILMGenerationConfig = builds(
    GreedyAPILMGenerationMethod,
    confidence_metric=None,
)

generation_method_store = store(group="generation_method")
generation_method_store(
    GreedyAPILMGenerationConfig,
    name="api_greedy_decoding",
)
