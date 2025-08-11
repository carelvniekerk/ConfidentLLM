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
"""Data cleaning tools for preprocessing data."""

from itertools import cycle

from confidentllm.conversations.types import (
    ChatAssistantMessage,
    ChatConversation,
    ChatUserMessage,
)

__all__ = ["cleanup_response", "create_conversation"]


def cleanup_response(response: str) -> str:
    """Clean up the response string."""
    response = response if response.endswith(".") else response + "."
    response = ".".join(response.split(".")[:-1])
    response = response.replace("[", "").replace("]", "").strip()

    return response


def create_conversation(
    responses: list[str] | list[list[str]],
    questions: list[str] | None = None,
) -> ChatConversation:
    """Create a ChatConversation from a list of questions and responses."""
    if questions is not None:
        conversation: ChatConversation = ChatConversation(
            messages=[
                [
                    ChatUserMessage(question),
                    ChatAssistantMessage(cleanup_response(response)),  # type: ignore[arg-type]
                ]
                for question, response in zip(questions, responses, strict=True)
            ],
        )
    else:
        conversation = ChatConversation(
            messages=[
                [
                    message_cls(utterance)  # type: ignore[misc]
                    for message_cls, utterance in zip(
                        cycle([ChatUserMessage, ChatAssistantMessage]),
                        utterances,
                        strict=False,
                    )
                ]
                for utterances in responses
            ],
        )

    return conversation
