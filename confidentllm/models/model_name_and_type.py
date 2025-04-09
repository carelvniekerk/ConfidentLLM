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
"""Modul to store all supported models."""

from enum import StrEnum, auto

__all__ = [
    "ModelDataTypes",
    "ModelDevice",
    "ModelMode",
    "ModelName",
    "ModelType",
]


class ModelName(StrEnum):
    """Enum class to store the names of the models."""

    # GPT2 based Models
    GPT2_137M = auto()
    GPT2_380M = auto()
    GPT2_812M = auto()
    GPT2_2B = auto()
    RM_GPT2_HARMLESS_774M = auto()

    # Google GEMMA Models
    GEMMA_2B_IT = auto()
    GEMMA2_2B_IT = auto()
    GEMMA2_9B_IT = auto()
    GEMMA2_9B = auto()
    RM_GEMMA_2B = auto()
    GEMMA3_4B_IT = auto()

    # Microsoft PHI Models
    PHI2_3B = auto()
    PHI3_MINI_INSTRUCT_4B = auto()
    PHI4_14B = auto()

    # Meta-Llama Models
    LLAMA2_7B = auto()
    LLAMA2_7B_CHAT = auto()
    LLAMA3_8B_INSTRUCT = auto()
    LLAMA3_8B = auto()
    LLAMA32_3B_INSTRUCT = auto()
    URM_LLAMA31_8B = auto()

    # Mistral Models
    MISTRAL_7B = auto()
    MISTRAL_7B_INSTRUCT = auto()
    MISTRAL_NEMO_12B = auto()
    MISTRAL_NEMO_12B_INSTRUCT = auto()

    # QWEN Models
    QWEN25_7B_INSTRUCT = auto()

    # Deepseek Models
    DEEPSEEK_R1_DISTILL_QWEN_7B = auto()

    # OpenAI Models
    GPT4O_MINI = auto()

    # VertexAI Models
    GEMINI15_FLASH = auto()


class ModelType(StrEnum):
    """Enum class to store the types of the models."""

    CAUSAL_LM = auto()
    SEQUENCE_CLS = auto()
    OPENAI = auto()
    VERTEXAI = auto()


class ModelMode(StrEnum):
    """Enum class to store the modes of the models."""

    TRAIN = auto()
    EVAL = auto()


class ModelDataTypes(StrEnum):
    """Enum class to store the data types of the models."""

    FLOAT64 = auto()
    FLOAT32 = auto()
    FLOAT16 = auto()
    BFLOAT16 = auto()


class ModelDevice(StrEnum):
    """Enum class to store the devices of the models."""

    CPU = auto()
    CUDA = auto()
    MPS = auto()
    META = auto()
