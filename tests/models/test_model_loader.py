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
# limitations under the License.
"""Tests for the model loader module."""

from pathlib import Path

import pytest
import torch

from confidentllm.models.model_loader import ModelLoader
from confidentllm.models.model_name import ModelName


def test_model_loader_with_name() -> None:
    """Test the model loader class."""
    loader = ModelLoader(pretrained_model_name_or_path=ModelName.GPT2, device="meta")

    model, tokenizer = loader.load()

    if loader.pretrained_model_name_or_path != "gpt2":
        msg = "Pretrained model name or path is not set correctly."
        raise AssertionError(msg)
    if model.device != torch.device("meta"):
        msg = "Model device is not set to 'meta'."
        raise AssertionError(msg)
    if tokenizer.padding_side != "left":
        msg = "Tokenizer padding side is not set to 'left'."
        raise AssertionError(msg)
    if not tokenizer.clean_up_tokenization_spaces:
        msg = "Tokenizer clean up tokenization spaces is not set."
        raise AssertionError(msg)
    if tokenizer.pad_token_id is None:
        msg = "Tokenizer pad token id is not set."
        raise AssertionError(msg)
    if tokenizer.chat_template is None:
        msg = "Tokenizer chat template is not set."
        raise AssertionError(msg)


def test_model_loader_with_path() -> None:
    """Test the model loader class."""
    with pytest.raises(FileNotFoundError):
        _ = ModelLoader(pretrained_model_name_or_path=Path("gpt2"), device="meta")

    loader = ModelLoader(pretrained_model_name_or_path=Path("GPT2"), device="meta")

    if loader.pretrained_model_name_or_path != "gpt2":
        msg = "Pretrained model name or path is not set correctly."
        raise AssertionError(msg)
