# coding=utf-8
# --------------------------------------------------------------------------------
# Project: Training Project
# Author: Carel van Niekerk
# Year: 2024
# Group: Dialogue Systems and Machine Learning Group
# Institution: Heinrich Heine University Düsseldorf
# --------------------------------------------------------------------------------
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

__all__ = []


class ModelName(StrEnum):
    """Enum class to store the names of the models."""

    GEMMA_11_2B_IT = "google/gemma-1.1-2b-it"
    GPT2 = "gpt2"


def load_model(
    pretrained_model_name_or_path: str,  # type: ignore  # noqa: PGH003
    device: str,  # type: ignore  # noqa: PGH003 # Workaround for omegaconf primitives
) -> Module:
    """Load a pretrained model from Hugging Face's model hub.

    Args:
    ----
        pretrained_model_name_or_path (str): The name of or the path to the model.
        device (str): The device to load the model on.

    Returns:
    -------
        Module: The model loaded on the specified device.

    """
    device: torch.device = torch.device(device)

    model: Module = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
    )

    return model.to(device)


CausalLMConfig = builds(load_model)

gemma_11_2b_it = CausalLMConfig(
    pretrained_model_name_or_path=ModelName.GEMMA_11_2B_IT.value,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

model_store = store(group="model")
model_store(gemma_11_2b_it, name="gemma_11_2b_it")


def load_tokenizer(
    pretrained_model_name_or_path: ModelName,
) -> PreTrainedTokenizer:
    """Load a pretrained tokenizer from Hugging Face's model hub.

    Args:
    ----
        pretrained_model_name_or_path (str): The name of or the path to the tokenizer.

    Returns:
    -------
        Module: The tokenizer.

    """
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path.value,
    )  # type: ignore  # noqa: PGH003

    return tokenizer


TokenizerConfig = builds(load_tokenizer)

gemma_11_2b_it = TokenizerConfig(
    pretrained_model_name_or_path=ModelName.GEMMA_11_2B_IT,
)

tokenizer_store = store(group="tokenizer")
tokenizer_store(gemma_11_2b_it, name="gemma_11_2b_it")
