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
"""Tokenization and data preparation for supervised finetuning."""

import torch
from transformers import BatchEncoding, PreTrainedTokenizer, TensorType
from transformers.tokenization_utils_base import PaddingStrategy

from confidentllm.generation.types import (
    ChatAssistantMessage,
    ChatConversation,
    ChatUserMessage,
)

__all__ = ["prepare_supervised_data"]


def _cleanup_response(response: str) -> str:
    """Clean up the response string."""
    response = ".".join(response.split(".")[:-1])
    response = response.replace("[", "").replace("]", "").strip()

    return response


def prepare_supervised_data(
    data: dict[str, list[str]],
) -> dict[str, list[list[dict[str, str]]]]:
    """Tokenize the input strings and return the tokenized data."""
    conversations: ChatConversation = ChatConversation(
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

    return {"messages": list(conversations)}
