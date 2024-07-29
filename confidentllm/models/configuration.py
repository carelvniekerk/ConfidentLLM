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
"""Module to store chat templates for different models."""

from pathlib import Path

from confidentllm.models.model_name import ModelName

__all__ = ["get_chat_template", "get_pretrained_model_name_or_path"]

CHAT_TEMPLATES: dict[ModelName, str] = {
    ModelName.GPT2: (
        "{{ bos_token }}"
        "{% if messages[0]['role'] == 'system' %}"
        "{{ raise_exception('System role not supported') }}{% endif %}"
        "{% for message in messages %}"
        "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
        "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"  # noqa: E501
        "{% endif %}{% if (message['role'] == 'assistant') %}"
        "{{ '\nOutput: ' + message['content'] | trim }}{% else %}"
        "{{ 'Instruct: ' + message['content'] | trim + '\n' }}"
        "{% set role = message['role'] %}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '\nOutput: ' }}{% endif %}"
    ),
    ModelName.PHI2: (
        "{{ bos_token }}"
        "{% if messages[0]['role'] == 'system' %}"
        "{{ raise_exception('System role not supported') }}{% endif %}"
        "{% for message in messages %}"
        "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
        "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"  # noqa: E501
        "{% endif %}{% if (message['role'] == 'assistant') %}"
        "{{ '\nOutput: ' + message['content'] | trim }}{% else %}"
        "{{ 'Instruct: ' + message['content'] | trim + '\n' }}"
        "{% set role = message['role'] %}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '\nOutput: ' }}{% endif %}"
    ),
}


PRETRAINED_MODEL_NAME_OR_PATH = {
    ModelName.GPT2: "gpt2",
    ModelName.GEMMA_2B: "google/gemma-1.1-2b",
    ModelName.GEMMA_2B_IT: "google/gemma-1.1-2b-it",
    ModelName.GEMMA2_9B_IT: "google/gemma-2-9b-it",
    ModelName.GEMMA2_9B: "google/gemma-2-9b",
    ModelName.PHI2: "microsoft/phi-2",
    ModelName.PHI3_MINI_INSTRUCT: "microsoft/Phi-3-mini-128k-instruct",
    ModelName.LLAMA2_7B: "meta-llama/Llama-2-7b-hf",
    ModelName.LLAMA2_7B_CHAT: "meta-llama/Llama-2-7b-chat-hf",
    ModelName.LLAMA3_8B_INSTRUCT: "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ModelName.LLAMA3_8B: "meta-llama/Meta-Llama-3.1-8B",
    ModelName.MISTRAL_7B: "mistralai/Mistral-7B-v0.3",
    ModelName.MISTRAL_7B_INSTRUCT: "mistralai/Mistral-7B-Instruct-v0.3",
    ModelName.MISTRAL_NEMO_12B: "mistralai/Mistral-Nemo-Base-2407",
    ModelName.MISTRAL_NEMO_12B_INSTRUCT: "mistralai/Mistral-Nemo-Instruct-2407",
}


def get_chat_template(name: ModelName) -> str | None:
    """Get the chat template for the given model name."""
    return CHAT_TEMPLATES.get(name)


def get_pretrained_model_name_or_path(name: ModelName | Path) -> str | Path:
    """Get the pretrained model name or path for the given model name."""
    return PRETRAINED_MODEL_NAME_OR_PATH.get(name, name)  # type: ignore - Paths will be returned as is
