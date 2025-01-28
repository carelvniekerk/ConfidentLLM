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
"""Data module for ConfidentLLM."""

from hydra_zen import store

from confidentllm.data.datasets.arc import load_arc_data
from confidentllm.data.datasets.commonsense_qa import load_commonsense_qa_data
from confidentllm.data.datasets.cot_preference_data import load_cot_preference_data
from confidentllm.data.datasets.gsm8k import load_gsm8k_data
from confidentllm.data.datasets.mmlu import load_mmlu_data
from confidentllm.data.datasets.multi_arith import load_multi_arith_data
from confidentllm.data.datasets.openbook_qa import load_openbook_qa_data
from confidentllm.data.datasets.union import load_union_data
from confidentllm.data.types import DatasetSplit
from confidentllm.hydra_tools import builds

__all__: list[str] = []

COTPreferenceDataConfig = builds(load_cot_preference_data)
CommonsenseQAConfig = builds(load_commonsense_qa_data)
GSM8KConfig = builds(load_gsm8k_data)
MMLUConfig = builds(load_mmlu_data)
MultiArithConfig = builds(load_multi_arith_data)
ARCConfig = builds(load_arc_data)
OpenbookQAConfig = builds(load_openbook_qa_data)
UnionConfig = builds(load_union_data)

data_store = store(group="data")
data_store(GSM8KConfig(split=DatasetSplit.TEST), name="gsm8k")
data_store(MMLUConfig(split=DatasetSplit.TEST), name="mmlu")
data_store(MultiArithConfig(split=DatasetSplit.TEST), name="multiarith")
data_store(CommonsenseQAConfig(split=DatasetSplit.VALIDATION), name="commonsense_qa")
data_store(ARCConfig(split=DatasetSplit.TEST), name="arc")
data_store(OpenbookQAConfig(split=DatasetSplit.TEST), name="openbook_qa")
data_store(UnionConfig(split=DatasetSplit.TEST), name="union")

cot_preference_train_data = COTPreferenceDataConfig(
    run_path="dialgroup-hhu/ConfidentLLM",
    run_name="ancient-universe-243",
    table_name="generation_path_data",
)

train_data_store = store(group="train_data")
eval_data_store = store(group="eval_data")

train_data_store(cot_preference_train_data, name="cot_preference")
eval_data_store(cot_preference_train_data, name="cot_preference")
train_data_store(GSM8KConfig(split=DatasetSplit.TRAIN), name="gsm8k")
eval_data_store(GSM8KConfig(split=DatasetSplit.VALIDATION), name="gsm8k")
train_data_store(MMLUConfig(split=DatasetSplit.TRAIN), name="mmlu")
eval_data_store(MMLUConfig(split=DatasetSplit.VALIDATION), name="mmlu")
train_data_store(MultiArithConfig(split=DatasetSplit.TRAIN), name="multiarith")
eval_data_store(MultiArithConfig(split=DatasetSplit.TEST), name="multiarith")
train_data_store(CommonsenseQAConfig(split=DatasetSplit.TRAIN), name="commonsense_qa")
eval_data_store(
    CommonsenseQAConfig(split=DatasetSplit.VALIDATION),
    name="commonsense_qa",
)
train_data_store(ARCConfig(split=DatasetSplit.TRAIN), name="arc")
eval_data_store(ARCConfig(split=DatasetSplit.VALIDATION), name="arc")
train_data_store(OpenbookQAConfig(split=DatasetSplit.TRAIN), name="openbook_qa")
eval_data_store(OpenbookQAConfig(split=DatasetSplit.VALIDATION), name="openbook_qa")
train_data_store(UnionConfig(split=DatasetSplit.TRAIN), name="union")
eval_data_store(UnionConfig(split=DatasetSplit.VALIDATION), name="union")
