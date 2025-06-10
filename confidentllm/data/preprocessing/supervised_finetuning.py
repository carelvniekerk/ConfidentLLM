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
"""Tokenization and data preparation for supervised finetuning."""

from typing import TYPE_CHECKING

import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils.generic import PaddingStrategy, TensorType

from confidentllm.conversations.create_chat import create_conversation

if TYPE_CHECKING:
    from confidentllm.conversations.types import ChatConversation

__all__ = ["sft_preprocessing"]


def sft_preprocessing(
    data: dict[str, list[str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    ignore_index: int = -100,
) -> dict[str, torch.Tensor]:
    """Tokenize the input strings and return the tokenized data."""
    answer_key: str = "preferred_response"
    answer_key = "long_format_answer" if answer_key not in data else answer_key
    answer_key = "answer" if answer_key not in data else answer_key

    conversations: ChatConversation = create_conversation(
        responses=data[answer_key],
        questions=data.get("question"),
    )

    inputs: BatchEncoding = tokenizer.apply_chat_template(
        conversation=list(conversations),
        add_generation_prompt=False,
        return_tensors=TensorType.PYTORCH,
        return_dict=True,
        padding=PaddingStrategy.MAX_LENGTH,  # type: ignore[arg-type] # PaddingStrategy is a valid type
        truncation=True,
        max_length=max_length,
    )  # type: ignore[assignment]

    output_dict: dict[str, torch.Tensor] = {}

    output_dict["input_ids"] = inputs.input_ids
    output_dict["attention_mask"] = inputs.attention_mask

    labels: torch.Tensor = torch.tensor(
        data=data["preferred_response_confidence"],
        dtype=torch.float32,
    )

    labels = labels.reshape(-1, 1).repeat(1, inputs.input_ids.shape[1])
    labels[inputs.attention_mask == 0] = ignore_index

    output_dict["label"] = labels

    return output_dict
