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
"""API model loader for ConfidentLLM."""

from confidentllm.models.api_models.base_model import BaseAPIModel
from confidentllm.models.api_models.configuration import MODEL_CLASS, MODEL_NAME
from confidentllm.models.model_name_and_type import ModelName


class APIModelLoader:
    """API model loader for ConfidentLLM."""

    def __init__(
        self,
        model_name: ModelName,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 1.0,
        max_tokens: int = 512,
        api_key: str | None = None,
    ) -> None:
        """Initialize the API model loader.

        Args:
        ----
            model_name (ModelName): The model name to use.
            system_prompt (str): The system prompt to use.
            temperature (float): The temperature for the model.
            max_tokens (int): The maximum number of tokens to generate.
            api_key (str): The API key to use for the model.

        """
        self.model_name: str = MODEL_NAME.get(model_name, "")
        self.model_class: BaseAPIModel = MODEL_CLASS.get(model_name, BaseAPIModel)  # type: ignore[arg-type]
        self.system_prompt: str = system_prompt
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self.api_key: str | None = api_key

    def load_model(self) -> BaseAPIModel:
        """Load the model."""
        model_instance = self.model_class(
            system_prompt=self.system_prompt,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
        )  # type: ignore[operator]

        return model_instance
