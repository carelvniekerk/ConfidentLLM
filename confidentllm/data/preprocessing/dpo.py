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
"""Tokenization and data preparation for dpo training."""

from transformers import PreTrainedTokenizer, TensorType
from transformers.tokenization_utils_base import PaddingStrategy

from confidentllm.data.preprocessing.data_cleaning_tools import create_conversation
from confidentllm.generation.types import ChatConversation

__all__ = ["dpo_preprocessing"]


def dpo_preprocessing(
    data: dict[str, list[str]],
    tokenizer: PreTrainedTokenizer,
    max_prompt_length: int,
) -> dict[str, list[str]]:
    """Tokenize the input strings and return the tokenized data."""
    preferred_conversations: ChatConversation = create_conversation(
        responses=data["preferred_response"],
        questions=data.get("question"),
    )

    prompt_conversations: ChatConversation = ChatConversation(
        messages=[
            conversation[:-1] for conversation in preferred_conversations.messages
        ],
    )

    prompts: list[str] = tokenizer.apply_chat_template(
        conversation=list(prompt_conversations),
        add_generation_prompt=True,
        return_tensors=TensorType.PYTORCH,
        tokenize=False,
        padding=PaddingStrategy.MAX_LENGTH,  # type: ignore[arg-type] # PaddingStrategy is a valid type
        truncation=True,
        max_length=max_prompt_length,
    )

    preferred_responses: list[str] = [
        conversation[-1].content for conversation in preferred_conversations.messages
    ]

    rejected_conversations: ChatConversation = create_conversation(
        responses=data["rejected_response"],
        questions=data.get("question"),
    )
    rejected_responses: list[str] = [
        conversation[-1].content for conversation in rejected_conversations.messages
    ]

    output_data: dict[str, list[str]] = {
        "prompt": prompts,
        "chosen": preferred_responses,
        "rejected": rejected_responses,
    }

    return output_data
