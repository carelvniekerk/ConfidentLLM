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
"""OpenAI API Model class for API Models."""

from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from confidentllm.conversations.types import ChatConversation
from confidentllm.models.api_models.base_model import BaseAPIModel

__all__ = ["ChatGPTModel"]


class ChatGPTModel(BaseAPIModel):
    """OpenAI ChatGPT API Model class for API Models."""

    def _init_api(self, api_key: str) -> OpenAI:
        """Initialize the OpenAI API.

        Args:
        ----
            api_key (str): The API key to use for the model.

        Returns:
        -------
            OpenAI: The OpenAI API object.

        """
        return OpenAI(api_key=api_key)

    def _create_system_prompt(
        self,
        system_prompt: str,
    ) -> ChatCompletionSystemMessageParam:
        """Create the system prompt.

        Args:
        ----
            system_prompt (str): The system prompt to use.

        Returns:
        -------
            ChatCompletionSystemMessageParam: The system prompt.

        """
        return ChatCompletionSystemMessageParam(
            content=system_prompt,
            role="system",
        )

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
        conversation_context: list[list[ChatCompletionMessageParam]] = []
        for chat in conversation.messages:
            chat_context: list[ChatCompletionMessageParam] = []
            for message in chat:
                if message.role == "user":
                    chat_context.append(
                        ChatCompletionUserMessageParam(
                            content=message.content,
                            role="user",
                        ),
                    )
                elif message.role == "assistant":
                    chat_context.append(
                        ChatCompletionAssistantMessageParam(
                            content=message.content,
                            role="assistant",
                        ),
                    )
            conversation_context.append(chat_context)

        responses: list[ChatCompletion] = [
            self.api.chat.completions.create(
                model=self.model_name,
                messages=conv,
                max_completion_tokens=max_new_tokens,
                n=num_beams,
                temperature=temperature,
            )
            for conv in conversation_context
        ]

        response_text: list[str] = []
        for response in responses:
            if not response.choices or not response.choices[0].message.content:
                response_text.append("I'm sorry, I don't have a response for that.")
            else:
                response_text.append(response.choices[0].message.content)

        return response_text
