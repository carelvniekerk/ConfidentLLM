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
#     http: //www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization and data preparation for training a reward model."""

import torch
from transformers import BatchEncoding, PreTrainedTokenizer, TensorType
from transformers.tokenization_utils_base import PaddingStrategy

from confidentllm.generation.types import (
    ChatConversation,
    ChatUserMessage,
)

__all__ = ["rl_preprocessing"]


def rl_preprocessing(
    data: dict[str, list[str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Tokenize the input strings and return the tokenized data."""
    conversations: ChatConversation = ChatConversation(
        messages=[[ChatUserMessage(question)] for question in data["question"]],
    )

    inputs: BatchEncoding = tokenizer.apply_chat_template(
        conversation=list(conversations),
        add_generation_prompt=True,
        return_tensors=TensorType.PYTORCH,
        return_dict=True,
        padding=PaddingStrategy.MAX_LENGTH,  # type: ignore[arg-type] # PaddingStrategy is a valid type
        truncation=True,
        max_length=max_length,
    )  # type: ignore[assignment]

    output_data: dict[str, torch.Tensor] = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
    }

    return output_data
