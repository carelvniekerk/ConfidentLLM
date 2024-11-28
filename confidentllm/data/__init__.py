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
"""Data module for ConfidentLLM."""

from hydra_zen import store

from confidentllm.data.commonsense_qa import load_commonsense_qa_data
from confidentllm.data.cot_preference_data import load_cot_preference_data
from confidentllm.data.gsm8k import load_gsm8k_data
from confidentllm.data.mmlu import load_mmlu_data
from confidentllm.data.multi_arith import load_multi_arith_data
from confidentllm.hydra_tools import builds

__all__: list[str] = []

COTPreferenceDataConfig = builds(load_cot_preference_data)
CommonsenseQAConfig = builds(load_commonsense_qa_data)
GSM8KConfig = builds(load_gsm8k_data)
MMLUConfig = builds(load_mmlu_data)
MultiArithConfig = builds(load_multi_arith_data)

data_store = store(group="data")
data_store(GSM8KConfig, name="gsm8k")
data_store(MMLUConfig, name="mmlu")
data_store(MultiArithConfig, name="multiarith")
data_store(CommonsenseQAConfig, name="commonsense_qa")

cot_preference_train_data = COTPreferenceDataConfig(
    run_path="dialgroup-hhu/ConfidentLLM",
    run_name="ancient-universe-243",
    table_name="generation_path_data",
)

train_data_store = store(group="train_data")
eval_data_store = store(group="eval_data")

train_data_store(cot_preference_train_data, name="cot_preference")
eval_data_store(cot_preference_train_data, name="cot_preference")
train_data_store(GSM8KConfig, name="gsm8k")
eval_data_store(GSM8KConfig, name="gsm8k")
train_data_store(MMLUConfig, name="mmlu")
eval_data_store(MMLUConfig, name="mmlu")
train_data_store(MultiArithConfig, name="multiarith")
eval_data_store(MultiArithConfig, name="multiarith")
train_data_store(CommonsenseQAConfig, name="commonsense_qa")
eval_data_store(CommonsenseQAConfig, name="commonsense_qa")
