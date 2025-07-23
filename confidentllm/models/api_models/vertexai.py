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
"""Google Vertex AI API Model Class."""

import os

import vertexai
from dotenv import load_dotenv
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    SafetySetting,
)

from confidentllm.conversations.types import ChatConversation
from confidentllm.models.api_models.base_model import BaseAPIModel

__all__ = ["GeminiModel"]


class GeminiModel(BaseAPIModel):
    """OpenAI ChatGPT API Model class for API Models."""

    def _init_api(self, api_key: str) -> GenerativeModel:  # noqa: ARG002
        """Initialize the OpenAI API.

        Args:
        ----
            api_key (str): The API key to use for the model.

        """
        load_dotenv()
        project_id: str = os.environ.get("GCP_PROJECT_ID", "")
        location: str = os.environ.get("GCP_LOCATION", "")
        vertexai.init(project=project_id, location=location)

        config: GenerationConfig = GenerationConfig(
            temperature=1.0,
            max_output_tokens=256,
            seed=0,
        )
        safety_settings: list[SafetySetting] = [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
        ]
        model: GenerativeModel = GenerativeModel(
            model_name=self.model_name,
            generation_config=config,
            safety_settings=safety_settings,
        )
        return model

    def _create_system_prompt(
        self,
        system_prompt: str,
    ) -> Content:
        """Create the system prompt.

        Args:
        ----
            system_prompt (str): The system prompt to use.

        Returns:
        -------
            Content: The system prompt.

        """
        prompt: Part = Part.from_text(system_prompt)
        return Content(
            parts=[prompt],
            role="system",
        )

    def generate(
        self,
        conversation: ChatConversation,
        max_new_tokens: int,
        temperature: float,
        num_beams: int = 1,  # noqa: ARG002
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
        conversation_context: list[list[Content]] = []
        for chat in conversation.messages:
            chat_context: list[Content] = [
                Content(
                    parts=[Part.from_text(message.content)],
                    role=message.role,
                )
                for message in chat
            ]
            conversation_context.append(chat_context)

        generation_config: GenerationConfig = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            seed=0,
        )
        self.api._generation_config = generation_config  # noqa: SLF001

        responses: list[GenerationResponse] = [
            self.api.generate_content(
                contents=contents,
            )
            for contents in conversation_context
        ]

        response_text: list[str] = [response.text for response in responses]

        return response_text
