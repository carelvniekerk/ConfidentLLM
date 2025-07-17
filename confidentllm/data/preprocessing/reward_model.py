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

from typing import TYPE_CHECKING

import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils.generic import PaddingStrategy, TensorType

from confidentllm.conversations.create_chat import create_conversation

if TYPE_CHECKING:
    from confidentllm.conversations.types import ChatConversation

__all__ = ["reward_model_preprocessing"]


def reward_model_preprocessing(
    data: dict[str, list[str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    *,
    include_preference_margin: bool = True,
) -> dict[str, torch.Tensor]:
    """Tokenize the input strings and return the tokenized data."""
    preferred_conversations: ChatConversation = create_conversation(
        responses=data["preferred_response"],
        questions=data.get("question"),
    )

    rejected_conversations: ChatConversation = create_conversation(
        responses=data["rejected_response"],
        questions=data.get("question"),
    )

    preferred_inputs: BatchEncoding = tokenizer.apply_chat_template(  # type: ignore[assignment]
        conversation=list(preferred_conversations),
        add_generation_prompt=False,
        return_tensors=TensorType.PYTORCH,
        return_dict=True,
        padding=PaddingStrategy.MAX_LENGTH,  # type: ignore[arg-type] # PaddingStrategy is a valid type
        truncation=True,
        max_length=max_length,
    )

    rejected_inputs: BatchEncoding = tokenizer.apply_chat_template(  # type: ignore[assignment]
        conversation=list(rejected_conversations),
        add_generation_prompt=False,
        return_tensors=TensorType.PYTORCH,
        return_dict=True,
        padding=PaddingStrategy.MAX_LENGTH,  # type: ignore[arg-type] # PaddingStrategy is a valid type
        truncation=True,
        max_length=max_length,
    )

    output_data: dict[str, torch.Tensor] = {
        "input_ids_chosen": preferred_inputs.input_ids,
        "attention_mask_chosen": preferred_inputs.attention_mask,
        "input_ids_rejected": rejected_inputs.input_ids,
        "attention_mask_rejected": rejected_inputs.attention_mask,
    }
    if include_preference_margin:
        output_data["margin"] = torch.tensor(data["margin"])

    return output_data
