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

    # Add metadata to the dataset
    data._info.description = (  # noqa: SLF001 # Adding description to dataset
        "GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically "
        "diverse grade school math word problems. The dataset was created to support "
        "the task of question answering on basic mathematical problems that require "
        "multi-step reasoning."
    )
    data._info.citation = (  # noqa: SLF001 # Adding citation to dataset
        "@article{cobbe2021gsm8k,\n  title={Training Verifiers to Solve Math Word "
        "Problems},\n  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, "
        "Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, "
        "Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and "
        "Hesse, Christopher and Schulman, John},\n  journal={arXiv preprint "
        "arXiv:2110.14168},\n  year={2021}\n}"
    )
    data._info.homepage = "https://huggingface.co/datasets/openai/gsm8k"  # noqa: SLF001
    data._info.license = "MIT License"  # noqa: SLF001

    data = data.map(
        partial(extract_answers, pattern=GSM8K_ANSWER_PATTERN),
        batched=True,
        batch_size=transformation_batch_size,
    )

    return data
