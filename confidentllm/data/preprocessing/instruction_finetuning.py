# coding=utf-8
# --------------------------------------------------------------------------------
# Project: ConfidentLLM
# Author: Carel van Niekerk, Renato Vukovic
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

from confidentllm.conversations.create_chat import create_conversation

if TYPE_CHECKING:
    from confidentllm.conversations.types import ChatConversation

__all__ = ["instruction_preprocessing"]


def instruction_preprocessing(
    data: dict[str, list[str]],
) -> dict[str, list[list[dict[str, str]]]]:
    """Tokenize the input strings and return the tokenized data."""
    answer_key: str = "preferred_response"
    answer_key = "long_format_answer" if answer_key not in data else answer_key
    answer_key = "answer" if answer_key not in data else answer_key

    conversations: ChatConversation = create_conversation(
        responses=data[answer_key],
        questions=data.get("question"),
    )

    return {"messages": list(conversations)}
