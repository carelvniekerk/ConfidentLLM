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

from confidentllm.models.model_name import ModelName

__all__ = ["get_chat_template"]

CHAT_TEMPLATES: dict[ModelName, str] = {
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


def get_chat_template(name: ModelName) -> str | None:
    """Get the chat template for the given model name."""
    return CHAT_TEMPLATES.get(name)
