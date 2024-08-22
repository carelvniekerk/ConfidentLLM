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

from hydra_zen import store

from confidentllm.generation.generation_function import dynamic_generate_function
from confidentllm.generation.types import (
    CausalLMGenerationMethod,
    GenerateFunction,
    GenerationOutput,
    TokenizerNotSetError,
)
from confidentllm.hydra_tools import builds

__all__ = []


class GreedyCausalLMGenerationMethod(CausalLMGenerationMethod):
    """Greedy generation method for causal language models."""

    def __init__(self, generator: GenerateFunction) -> None:
        super().__init__(generator)

    def __call__(
        self,
        prompt: str,
        max_length: int = 256,
    ) -> GenerationOutput:
        """Generate text based on the given prompt."""
        if isinstance(self.tokenizer, type(None)):
            raise TokenizerNotSetError(self.tokenizer)

        conversation: list[list[dict[str, str]]] = [
            [
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        ]

        inputs = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        return self.generator(
            input_ids=inputs["input_ids"],  # type: ignore  # noqa: PGH003
            max_length=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )  # type: ignore  # noqa: PGH003


CausalLMGenerationConfig = builds(GreedyCausalLMGenerationMethod)

causal_lm_generation_method = CausalLMGenerationConfig(
    generator=dynamic_generate_function,  # type: ignore  # noqa: PGH003
)

generation_method_store = store(group="generation_method")
generation_method_store(
    causal_lm_generation_method,
    name="causal_lm_generation_method",
)
