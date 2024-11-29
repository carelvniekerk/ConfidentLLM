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
#     http: //www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization and data preparation for dpo training."""

import torch
from transformers import BatchEncoding, PreTrainedTokenizer, TensorType
from transformers.tokenization_utils_base import PaddingStrategy

from confidentllm.generation.types import (
    ChatConversation,
    ChatUserMessage,
)

__all__ = ["prepare_dpo_data"]


def _cleanup_response(response: str) -> str:
    """Clean up the response string."""
    response = ".".join(response.split(".")[:-1])
    response = response.replace("[", "").replace("]", "").strip()

    return response


def prepare_dpo_data(
    data: dict[str, list[str]],
    tokenizer: PreTrainedTokenizer,
    max_prompt_length: int,
) -> dict[str, list[str]]:
    """Tokenize the input strings and return the tokenized data."""
    prompt_convrsations: ChatConversation = ChatConversation(
        messages=[[ChatUserMessage(question)] for question in data["questions"]],
    )

    prompt_inputs: BatchEncoding = tokenizer.apply_chat_template(
        conversation=list(prompt_convrsations),
        add_generation_prompt=True,
        return_tensors=TensorType.PYTORCH,
        return_dict=True,
        padding=PaddingStrategy.MAX_LENGTH,  # type: ignore[arg-type] # PaddingStrategy is a valid type
        truncation=True,
        max_length=max_prompt_length,
    )  # type: ignore[assignment]
    prompts: list[str] = tokenizer.batch_decode(
        sequences=prompt_inputs.input_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    preferred_responses: list[str] = [
        _cleanup_response(response) for response in data["preferred_responses"]
    ]

    rejected_responses: list[str] = [
        _cleanup_response(response) for response in data["rejected_responses"]
    ]

    output_data: dict[str, list[str]] = {
        "prompt": prompts,
        "chosen": preferred_responses,
        "rejected": rejected_responses,
    }

    return output_data
