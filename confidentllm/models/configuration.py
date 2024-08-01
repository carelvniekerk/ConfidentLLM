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


def get_pretrained_model_name_or_path(name: ModelName | Path) -> str | Path:
    """Get the pretrained model name or path for the given model name."""
    return PRETRAINED_MODEL_NAME_OR_PATH.get(name, name)  # type: ignore - Paths will be returned as is


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
    ModelName.GEMMA2_9B: (
        "{{ bos_token }}{% if messages[0]['role'] == 'system' %}"
        "{{ raise_exception('System role not supported') }}{% endif %}"
        "{% for message in messages %}"
        "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
        "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"  # noqa: E501
        "{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}"
        "{% else %}{% set role = message['role'] %}{% endif %}"
        "{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}"  # noqa: E501
        "{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}"
        "{% endif %}"
    ),
    ModelName.LLAMA3_8B: (
        "{{- bos_token }}\n{%- if custom_tools is defined %}\n"
        "    {%- set tools = custom_tools %}\n{%- endif %}\n"
        "{%- if not tools_in_user_message is defined %}\n"
        "    {%- set tools_in_user_message = true %}\n{%- endif %}\n"
        "{%- if not date_string is defined %}\n"
        '    {%- set date_string = "26 Jul 2024" %}'  # noqa: E501
        "\n{%- endif %}\n{%- if not tools is defined %}\n"
        "    {%- set tools = none %}\n{%- endif %}\n\n"
        "{#- This block extracts the system message, so we can slot it into the right place. #}\n"
        "{%- if messages[0]['role'] == 'system' %}\n"
        "    {%- set system_message = messages[0]['content']|trim %}\n"
        "    {%- set messages = messages[1:] %}\n{%- else %}\n"
        '    {%- set system_message = "" %}\n{%- endif %}\n\n'
        '{#- System message + builtin tools #}\n{{- "system\\n\\n" }}\n'
        "{%- if builtin_tools is defined or tools is not none %}\n"
        '    {{- "Environment: ipython\\n" }}\n{%- endif %}\n'
        "{%- if builtin_tools is defined %}\n"
        "    {{- \"Tools: \" + builtin_tools | reject('equalto', "
        '\'code_interpreter\') | join(", ") + "\\n\\n"}}\n{%- endif %}\n'
        '{{- "Cutting Knowledge Date: December 2023\\n" }}\n'
        '{{- "Today Date: " + date_string + "\\n\\n" }}\n'
        "{%- if tools is not none and not tools_in_user_message %}\n"
        '    {{- "You have access to the following functions. To call a function, '
        'please respond with JSON for a function call." }}\n'
        '    {{- \'Respond in the format {"name": function name, "parameters": '
        "dictionary of argument name and its value}.' }}\n"
        '    {{- "Do not use variables.\\n\\n" }}\n'
        "    {%- for t in tools %}\n"
        "        {{- t | tojson(indent=4) }}\n"
        '        {{- "\\n\\n" }}\n'
        "    {%- endfor %}\n{%- endif %}\n"
        "{{- system_message }}\n{{- '' }}\n\n"
        "{#- Custom tools are passed in a user message with some extra guidance #}\n"
        "{%- if tools_in_user_message and not tools is none %}\n"
        "{#- Extract the first user message so we can plug it in here #}\n"
        "    {%- if messages | length != 0 %}\n"
        "        {%- set first_user_message = messages[0]['content']|trim %}\n"
        "        {%- set messages = messages[1:] %}\n"
        "    {%- else %}\n"
        '        {{- raise_exception("Cannot put tools in the first user message '
        "when there's no first user message!\") }}\n"
        "{%- endif %}\n    {{- 'user\\n\\n' -}}\n"
        '    {{- "Given the following functions, please respond with a JSON for '
        'a function call " }}\n'
        '    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n'
        '    {{- \'Respond in the format {"name": function name, "parameters": '
        "dictionary of argument name and its value}.' }}\n"
        '    {{- "Do not use variables.\\n\\n" }}\n'
        "    {%- for t in tools %}\n"
        "        {{- t | tojson(indent=4) }}\n"
        '        {{- "\\n\\n" }}\n'
        "    {%- endfor %}\n"
        "    {{- first_user_message + ''}}\n{%- endif %}\n\n"
        "{%- for message in messages %}\n"
        "{%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n"
        "        {{- '' + message['role'] + '\\n\\n'+ message['content'] | trim + '' }}\n"
        "{%- elif 'tool_calls' in message %}\n"
        "        {%- if not message.tool_calls|length == 1 %}\n"
        '            {{- raise_exception("This model only supports single tool-calls '
        'at once!") }}\n'
        "        {%- endif %}\n"
        "        {%- set tool_call = message.tool_calls[0].function %}\n"
        "        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n"
        "            {{- 'assistant\\n\\n' -}}\n"
        "            {{- '' + tool_call.name + '.call(' }}\n"
        "            {%- for arg_name, arg_val in tool_call.arguments | items %}\n"
        "                {{- arg_name + '=\"' + arg_val + '\"' }}\n"
        "                {%- if not loop.last %}\n"
        "                    {{- ', ' }}\n"
        "                {%- endif %}\n"
        "            {%- endfor %}\n"
        "            {{- ')' }}\n"
        "        {%- else  %}\n"
        "            {{- 'assistant\\n\\n' -}}\n"
        "            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n"
        "            {{- '\"parameters\": ' }}\n"
        "            {{- tool_call.arguments | tojson }}\n"
        "            {{- '}' }}\n"
        "        {%- endif %}\n"
        "        {%- if builtin_tools is defined %}\n"
        "            {#- This means we're in ipython mode #}\n"
        "            {{- '' }}\n"
        "        {%- else %}\n"
        "            {{- '' }}\n"
        "        {%- endif %}\n"
        "{%- elif message.role == 'tool' or message.role == 'ipython' %}\n"
        '        {{- "ipython\\n\\n" }}\n'
        "        {%- if message.content is mapping or message.content is iterable %}\n"
        "            {{- message.content | tojson }}\n"
        "        {%- else %}\n"
        "            {{- message.content }}\n"
        "        {%- endif %}\n"
        "        {{- '' }}\n"
        "{%- endif %}\n"
        "{%- endfor %}\n"
        "{%- if add_generation_prompt %}\n"
        "    {{- 'assistant\\n\\n' }}\n"
        "{%- endif %}\n"
    ),
}


def get_chat_template(name: ModelName) -> str | None:
    """Get the chat template for the given model name."""
    return CHAT_TEMPLATES.get(name)
