# coding=utf-8
# --------------------------------------------------------------------------------
# Project: ConfidenceLLM
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
#     http: //www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Init file for training module."""

from confidentllm.train.dpo_trainer import DPOTrainer
from confidentllm.train.ppo_trainer import PPORLTrainer
from confidentllm.train.prepare_dpo_data import prepare_dpo_data
from confidentllm.train.prepare_reward_model_data import prepare_reward_model_data
from confidentllm.train.prepare_rl_data import prepare_rl_data
from confidentllm.train.prepare_supervised_data import prepare_supervised_data
from confidentllm.train.reward_model_trainer import RewardModelTrainer
from confidentllm.train.supervised_trainer import SupervisedFinetuningTrainer

__all__ = [
    "DPOTrainer",
    "PPORLTrainer",
    "RewardModelTrainer",
    "SupervisedFinetuningTrainer",
    "prepare_dpo_data",
    "prepare_reward_model_data",
    "prepare_rl_data",
    "prepare_supervised_data",
]
