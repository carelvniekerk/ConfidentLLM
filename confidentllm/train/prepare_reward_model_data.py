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
"""Tokenization and data preparation for training a reward model."""

import torch
from transformers import BatchEncoding, PreTrainedTokenizer, TensorType

from confidentllm.generation.types import (
    ChatAssistantMessage,
    ChatConversation,
    ChatUserMessage,
)

__all__ = ["prepare_reward_model_data"]


def _cleanup_response(response: str) -> str:
    """Clean up the response string."""
    response = ".".join(response.split(".")[:-1])
    response = response.replace("[", "").replace("]", "").strip()

    return response


def prepare_reward_model_data(
    data: dict[str, list[str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    *,
    include_preference_margin: bool = True,
) -> dict[str, torch.Tensor]:
    """Tokenize the input strings and return the tokenized data."""
    preferred_conversations: ChatConversation = ChatConversation(
        messages=[
            [
                ChatUserMessage(question),
                ChatAssistantMessage(_cleanup_response(response)),
            ]
            for question, response in zip(
                data["questions"],
                data["preferred_responses"],
                strict=True,
            )
        ],
    )

    rejected_conversations: ChatConversation = ChatConversation(
        messages=[
            [
                ChatUserMessage(question),
                ChatAssistantMessage(_cleanup_response(response)),
            ]
            for question, response in zip(
                data["questions"],
                data["rejected_responses"],
                strict=True,
            )
        ],
    )

    preferred_inputs: BatchEncoding = tokenizer.apply_chat_template(
        conversation=list(preferred_conversations),
        add_generation_prompt=False,
        return_tensors=TensorType.PYTORCH,
        return_dict=True,
        padding=True,
        max_length=max_length,
    )  # type: ignore[assignment]

    rejected_inputs: BatchEncoding = tokenizer.apply_chat_template(
        conversation=list(rejected_conversations),
        add_generation_prompt=False,
        return_tensors=TensorType.PYTORCH,
        return_dict=True,
        padding=True,
        max_length=max_length,
    )  # type: ignore[assignment]

    output_data: dict[str, torch.Tensor] = {
        "input_ids_chosen": preferred_inputs.input_ids,
        "attention_mask_chosen": preferred_inputs.attention_mask,
        "input_ids_rejected": rejected_inputs.input_ids,
        "attention_mask_rejected": rejected_inputs.attention_mask,
    }
    if include_preference_margin:
        output_data["margin"] = torch.tensor(data["margin"])

    return output_data
