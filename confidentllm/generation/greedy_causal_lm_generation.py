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
"""Main execution file for the project."""

from typing import TYPE_CHECKING

import torch
from hydra_zen import store
from transformers import BatchEncoding, TensorType
from transformers.generation import GenerateDecoderOnlyOutput
from transformers.tokenization_utils_base import PaddingStrategy

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

MAX_CONTEXT_LENGTH: int = 2048


class GreedyCausalLMGenerationMethod(CausalLMGenerationMethod):
    """Greedy generation method for causal language models."""

    def __call__(
        self,
        prompt: str | None = None,
        responses: list[str] | None = None,
    ) -> GenerationOutput:
        """Generate text based on the given prompt."""
        if self.tokenizer is None:
            raise TokenizerNotSetError(self.tokenizer)
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

        max_context_length = min(
            self.tokenizer.model_max_length - self.max_generation_length,
            self.max_context_length,
        )
        inputs: BatchEncoding = self.tokenizer.apply_chat_template(
            list(conversation),
            add_generation_prompt=True,
            return_tensors=TensorType.PYTORCH,
            return_dict=True,
            # padding=PaddingStrategy.MAX_LENGTH,
            truncation=True,
            max_length=max_context_length,
        )  # type: ignore[reportAssignmentType] # When using return dict a type BatchEncoding is returned

        with torch.no_grad():
            generation_output: GenerateDecoderOnlyOutput = self.model.generate(
                input_ids=inputs.input_ids.to(self.model.device),
                attention_mask=inputs.attention_mask.to(self.model.device),
                max_new_tokens=self.max_generation_length,
                do_sample=self.sampling,
                temperature=self.temperature,
                output_logits=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )  # type: ignore[reportAssignmentType] # When using return dict a type GenerateDecoderOnlyOutput is returned

        if generation_output.logits is None:
            msg = "The logits are not set."
            raise ValueError(msg)

        generation_scores: torch.Tensor = torch.cat(
            generation_output.logits,
            dim=0,
        ).unsqueeze(0)
        generation_scores = torch.softmax(generation_scores, dim=-1)

        generation_scores = self.confidence_metric(
            next_token_ids=generation_output.sequences,
            scores=generation_scores,
        )

        return GenerationOutput(
            generated_ids=generation_output.sequences,
            generation_scores=generation_scores,
        )


GreedyCausalLMGenerationConfig = builds(
    GreedyCausalLMGenerationMethod,
    confidence_metric=None,
)

generation_method_store = store(group="generation_method")
generation_method_store(
    GreedyCausalLMGenerationConfig,
    name="greedy_decoding",
)
