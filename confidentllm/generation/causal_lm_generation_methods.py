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
"""Main execution file for the project."""

import torch
from transformers.tokenization_utils import PreTrainedTokenizer

from confidentllm.generation.types import CausalLMGenerationMethod, GenerateFunction


def create_generation_method(
    tokenizer: PreTrainedTokenizer,
    generator: GenerateFunction,
) -> CausalLMGenerationMethod:
    """Create a generation method using the given tokenizer and generator.

    Args:
    ----
        tokenizer (PreTrainedTokenizer): The tokenizer to use for text generation.
        generator (Generator): The generator to use for text generation.

    Returns:
    -------
        GenerationMethod: The created generation method.

    """

    def generate(
        prompt: str,
        max_length: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate text based on the given prompt."""
        conversation: list[list[dict[str, str]]] = [
            [
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        ]

        inputs = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        generated_ids, generation_probs = generator(
            input_ids=inputs["input_ids"],  # type: ignore  # noqa: PGH003
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )  # type: ignore  # noqa: PGH003

        return generated_ids, generation_probs

    return generate
