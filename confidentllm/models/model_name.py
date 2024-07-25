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
"""Modul to store all supported models."""

from enum import StrEnum

__all__ = ["ModelName"]


class ModelName(StrEnum):
    """Enum class to store the names of the models."""

    # Default small GPT2 model for testing
    GPT2 = "gpt2"

    # Google GEMMA Models
    GEMMA_2B = "google/gemma-1.1-2b"
    GEMMA_2B_IT = "google/gemma-1.1-2b-it"
    GEMMA2_9B_IT = "google/gemma-2-9b-it"
    GEMMA2_9B = "google/gemma-2-9b"

    # Microsoft PHI Models
    PHI2 = "microsoft/phi-2"
    PHI3_MINI_INSTRUCT = "microsoft/Phi-3-mini-128k-instruct"

    # Meta-Llama Models
    LLAMA2_7B = "meta-llama/Llama-2-7b-hf"
    LLAMA2_7B_CHAT = "meta-llama/Llama-2-7b-chat-hf"
    LLAMA3_8B_INSTRUCT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    LLAMA3_8B = "meta-llama/Meta-Llama-3.1-8B"

    # Mistral Models
    MISTRAL_7B = "mistralai/Mistral-7B-v0.3"
    MISTRAL_7B_INSTRUCT = "mistralai/Mistral-7B-Instruct-v0.3"
    MISTRAL_NEMO_12B = "mistralai/Mistral-Nemo-Base-2407"
    MISTRAL_NEMO_12B_INSTRUCT = "mistralai/Mistral-Nemo-Instruct-2407"
