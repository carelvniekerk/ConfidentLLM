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
"""Module containing functions for loading the GSM-8K dataset."""

from functools import partial

from datasets import Dataset, load_dataset

from confidentllm.data.extract_answers import extract_answers
from confidentllm.data.types import DatasetSplit

__all__ = ["load_gsm8k_data"]

GSM8K_ANSWER_PATTERN = r"\n#### (.+)"


def load_gsm8k_data(
    split: DatasetSplit = DatasetSplit.TEST,
    transformation_batch_size: int = 512,
    name: str = "GSM8K",  # noqa: ARG001
) -> Dataset:
    """Load the GSM-8K dataset.

    Args:
    ----
        split: The split of the dataset to load.
        transformation_batch_size: The batch size to use for the transformation.
        name: The name of the dataset.

    Returns:
    -------
        The dataset.

    """
    data: Dataset = load_dataset(
        path="openai/gsm8k",
        name="main",
        split=split,
    )  # type: ignore[reportAssignmentType]

    data = data.map(
        partial(extract_answers, pattern=GSM8K_ANSWER_PATTERN),
        batched=True,
        batch_size=transformation_batch_size,
    )

    return data
