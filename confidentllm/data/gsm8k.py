# coding=utf-8
# --------------------------------------------------------------------------------
# Project: ConfidentLLM
# Author: Carel van Niekerk
# Year: 2024
# Group: Dialogue Systems and Machine Learning Group
# Institution: Heinrich Heine University Düsseldorf
# --------------------------------------------------------------------------------
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
"""Module containing functions for loading the GSM-8K dataset."""

from functools import partial

from datasets import Dataset, load_dataset
from hydra_zen import store

from confidentllm.data.extract_answers import extract_answers
from confidentllm.data.types import DatasetSplit

GSM8K_ANSWER_PATTERN = r"\n#### (.+)"


def load_gsm8k_data(
    split: DatasetSplit = DatasetSplit.TEST,
    transformation_batch_size: int = 512,
) -> Dataset:
    """Load the GSM-8K dataset.

    Args:
    ----
        split: The split of the dataset to load.
        transformation_batch_size: The batch size to use for the transformation.

    Returns:
    -------
        The dataset.

    """
    data: Dataset = load_dataset("openai/gsm8k", "main", split=split)  # type: ignore  # noqa: PGH003

    data = data.map(
        partial(extract_answers, pattern=GSM8K_ANSWER_PATTERN),
        batched=True,
        batch_size=transformation_batch_size,
    )

    return data


data_store = store(group="data")
data_store(load_gsm8k_data, name="gsm8k")
