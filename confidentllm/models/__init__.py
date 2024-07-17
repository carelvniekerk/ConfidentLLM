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

from enum import StrEnum

import torch
from hydra_zen import make_custom_builds_fn, store
from torch.nn import Module
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

builds = make_custom_builds_fn(populate_full_signature=True)

__all__ = ["ModelName", "ModelLoader"]


class ModelName(StrEnum):
    """Enum class to store the names of the models."""

    GEMMA_11_2B_IT = "google/gemma-1.1-2b-it"
    GPT2 = "gpt2"


class ModelLoader:
    """Abstract class to load a pretrained model and tokenizer."""

    def __init__(self, pretrained_model_name_or_path: ModelName, device: str) -> None:
        """Initialize the model loader."""
        self.pretrained_model_name_or_path: str = pretrained_model_name_or_path.value
        self.device: torch.device = torch.device(device)

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
            self.pretrained_model_name_or_path,  # type: ignore  # noqa: PGH003 - Tokenizer is a PreTrainedTokenizer
        )

        return model.to(self.device), tokenizer


CausalLMConfig = builds(ModelLoader)

gemma_11_2b_it = CausalLMConfig(
    pretrained_model_name_or_path=ModelName.GEMMA_11_2B_IT,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

gpt2 = CausalLMConfig(
    pretrained_model_name_or_path=ModelName.GPT2,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

model_store = store(group="model")
model_store(gemma_11_2b_it, name="gemma_11_2b_it")
model_store(gpt2, name="gpt2")
