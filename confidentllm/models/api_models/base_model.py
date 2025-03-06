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
"""Base model class for API Models."""

from abc import ABC, abstractmethod
from typing import Any

from confidentllm.conversations.types import ChatConversation

__all__ = ["BaseAPIModel"]


class BaseAPIModel(ABC):
    """Base class for API Models."""

    def __init__(
        self,
        model_name: str,
        system_prompt: str = "You are a helpful assistant.",
        api_key: str | None = None,
    ) -> None:
        """Initialize the API Model.

        Args:
        ----
            system_prompt (str): The system prompt to use.
            model_name (ModelName): The model name to use.
            api_key (str): The API key to use for the model.

        """
        self.system_prompt: Any = self._create_system_prompt(system_prompt)
        self.model_name = model_name

        if api_key is None:
            raise ValueError("API Key is required for API Models.")  # noqa: EM101, TRY003

        self.api = self._init_api(api_key)

    @abstractmethod
    def _init_api(self, api_key: str) -> Any:  # noqa: ANN401
        """Initialize the API.

        Args:
        ----
            api_key (str): The API key to use.

        Returns:
        -------
            Any: The initialized API. Type depends on the API.

        """

    @abstractmethod
    def _create_system_prompt(self, prompt: str) -> Any:  # noqa: ANN401 # Different types for different APIs
        """Create the system prompt.

        Args:
        ----
            prompt (str): The prompt to use.

        Returns:
        -------
            Any: The system prompt to use. Type depends on the API.

        """

    @abstractmethod
    def generate(
        self,
        conversation: ChatConversation,
        max_new_tokens: int,
        num_beams: int,
        temperature: float,
    ) -> list[str]:
        """Generate a response to a conversation.

        Args:
        ----
            conversation (ChatConversation): The conversation to generate a response.
            max_new_tokens (int): The maximum number of tokens to generate.
            num_beams (int): The number of beams to use.
            temperature (float): The temperature to use for sampling.

        Returns:
        -------
            list[str]: The generated responses.

        """
