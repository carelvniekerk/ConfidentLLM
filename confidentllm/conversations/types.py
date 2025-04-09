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
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Types for the conversations module."""

from dataclasses import dataclass
from typing import Iterator

__all__ = [
    "ChatAssistantMessage",
    "ChatConversation",
    "ChatUserMessage",
]


@dataclass
class ChatUserMessage:
    """Dataclass for the chat user message."""

    content: str
    role: str = "user"


@dataclass
class ChatAssistantMessage:
    """Dataclass for the chat user message."""

    content: str
    role: str = "assistant"


@dataclass
class ChatConversation:
    """Dataclass for the chat conversation."""

    messages: list[list[ChatUserMessage | ChatAssistantMessage]]

    def __iter__(self) -> Iterator[list[dict[str, str]]]:
        """Convert the chat conversation to a iterator."""
        for chat in self.messages:
            yield [message.__dict__ for message in chat]
