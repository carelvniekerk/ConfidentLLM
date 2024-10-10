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

import torch
from hydra_zen import store
from transformers import BatchEncoding, TensorType
from transformers.generation import GenerateDecoderOnlyOutput

from confidentllm.generation.confidence_extraction_methods import (
    ProbabilityDisparityConfig,
)
from confidentllm.generation.types import (
    CausalLMGenerationMethod,
    ChatConversation,
    ChatUserMessage,
    GenerationOutput,
    ModelNotSetError,
    TokenizerNotSetError,
)
from confidentllm.hydra_tools import builds

__all__: list[str] = []


class CoTDecodingCausalLMGenerationMethod(CausalLMGenerationMethod):
    """Greedy generation method for causal language models."""

    def __call__(
        self,
        prompt: str,
    ) -> GenerationOutput:
        """Generate text based on the given prompt."""
        if isinstance(self.tokenizer, type(None)):
            raise TokenizerNotSetError(self.tokenizer)
        if isinstance(self.model, type(None)):
            raise ModelNotSetError(self.model)

        conversation: ChatConversation = ChatConversation(
            messages=[[ChatUserMessage(content=prompt)]],
        )

        inputs: BatchEncoding = self.tokenizer.apply_chat_template(
            list(conversation),
            add_generation_prompt=True,
            return_tensors=TensorType.PYTORCH,
            return_dict=True,
        )  # type: ignore[reportAssignmentType] # When using return dict a type BatchEncoding is returned
        inputs = inputs.to(self.model.device)

        first_token_generation_output: GenerateDecoderOnlyOutput = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1,
            do_sample=self.sampling,
            num_beams=self.num_beams,
            num_return_sequences=self.num_beams,
            temperature=self.temperature,
            output_logits=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )  # type: ignore[reportAssignmentType] # When using return dict a type GenerateDecoderOnlyOutput is returned

        # Add an extra row of ones to the attention mask for the first tokens
        attention_mask: torch.Tensor = torch.cat(
            [inputs.attention_mask] * self.num_beams,
            dim=0,
        )
        first_token_attention_mask: torch.Tensor = torch.ones(
            self.num_beams,
            1,
        ).to(self.model.device)
        attention_mask = torch.cat(
            [attention_mask, first_token_attention_mask],
            dim=1,
        )

        generation_output: GenerateDecoderOnlyOutput = self.model.generate(
            input_ids=first_token_generation_output.sequences,
            attention_mask=attention_mask,
            max_new_tokens=self.max_length - 1,
            do_sample=self.sampling,
            temperature=self.temperature,
            output_logits=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )  # type: ignore[reportAssignmentType] # When using return dict a type GenerateDecoderOnlyOutput is returned

        if (
            generation_output.logits is None
            or first_token_generation_output.logits is None
        ):
            msg = "The logits are not set."
            raise ValueError(msg)

        generation_scores: torch.Tensor = torch.cat(
            first_token_generation_output.logits + generation_output.logits,
            dim=0,
        )
        generation_scores = generation_scores.reshape(
            -1,
            self.num_beams,
            generation_scores.size(-1),
        ).transpose(0, 1)
        generation_scores = torch.softmax(generation_scores, dim=-1)

        generation_scores = self.confidence_extraction_method(
            next_token_ids=generation_output.sequences,
            scores=generation_scores,
        )

        return GenerationOutput(
            generated_ids=generation_output.sequences,
            generation_scores=generation_scores,
        )


CoTDecodingCausalLMGenerationConfig = builds(
    CoTDecodingCausalLMGenerationMethod,
    confidence_extraction_method=ProbabilityDisparityConfig,
    num_beams=5,
)

generation_method_store = store(group="generation_method")
generation_method_store(
    CoTDecodingCausalLMGenerationConfig,
    name="cot_decoding",
)
