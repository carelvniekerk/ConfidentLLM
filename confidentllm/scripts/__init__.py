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
"""Execcution scripts for different tasks."""

from confidentllm.scripts.run_download_and_sync import main as download_and_sync
from confidentllm.scripts.run_mathematical_reasoning import (
    main as mathematical_reasoning,
)
from confidentllm.scripts.run_multiple_choice_qa import main as multiple_choice_qa
from confidentllm.scripts.run_reporting import main as reporting
from confidentllm.scripts.run_reward_model_training import main as reward_model_training

__all__ = [
    "download_and_sync",
    "mathematical_reasoning",
    "multiple_choice_qa",
    "reporting",
    "reward_model_training",
]
