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
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License."
"""Module to load pretrained models and tokenizers from Hugging Face's model hub."""

from pathlib import Path

import torch
from hydra_zen import store
from hydra_zen.third_party.pydantic import pydantic_parser
from torch.nn import Module
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from confidentllm.hydra_tools import builds
from confidentllm.models.configuration import (
    get_chat_template,
    get_pretrained_model_name_or_path,
)
from confidentllm.models.model_name import ModelName

__all__ = ["ModelLoader"]


# Default device to load the model on (Always use MPS or CUDA if available)
DEFAULT_DEVICE: str = "mps" if torch.backends.mps.is_available() else "cpu"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else DEFAULT_DEVICE


class ModelLoader:
    """Abstract class to load a pretrained model and tokenizer."""

    def __init__(
        self,
        pretrained_model_name_or_path: ModelName | Path,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        """Initialize the model loader."""
        if (
            isinstance(pretrained_model_name_or_path, Path)
            and not pretrained_model_name_or_path.exists()
        ):
            if pretrained_model_name_or_path.name not in ModelName.__members__:
                msg = (
                    f"The specified path {pretrained_model_name_or_path} does not "
                    "exist and is not a valid model name."
                )
                raise FileNotFoundError(msg)
            pretrained_model_name_or_path = ModelName[
                pretrained_model_name_or_path.name
            ]

        self.pretrained_model_name_or_path: str | Path = (
            get_pretrained_model_name_or_path(
                pretrained_model_name_or_path,
            )
        )
        self.device: torch.device = torch.device(device)
        self.chat_template: str | None = get_chat_template(
            pretrained_model_name_or_path,  # type: ignore - Path will return None as expected
        )

    def load(self) -> tuple[Module, PreTrainedTokenizer]:
        """Load a pretrained model from Hugging Face's model hub.

        Args:
        ----
            pretrained_model_name_or_path (str): The name of or the path to the model.
            device (str): The device to load the model on.

        Returns:
        -------
            Module: The model loaded on the specified device.

        """
        model: Module = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model_name_or_path,
        )

        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,  # type: ignore - Tokenizer is a PreTrainedTokenizer
        )

        if self.chat_template:
            tokenizer.chat_template = self.chat_template

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token

        return model.to(self.device), tokenizer


# Add default model loader to the store
ModelConfig = builds(
    ModelLoader,
    pretrained_model_name_or_path=ModelName.GPT2,
    zen_wrappers=[pydantic_parser],
)
store(ModelConfig, name="default", group="model")
