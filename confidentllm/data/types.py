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
"""Module containing enumerations for the datasets used in the project."""

from enum import StrEnum, auto
from typing import Protocol

from datasets import Dataset

__all__ = ["DatasetSplit", "LoadDatasetFunction"]


class DatasetSplit(StrEnum):
    """Split of the dataset."""

    TRAIN = auto()
    TEST = auto()
    VALIDATION = auto()


class LoadDatasetFunction(Protocol):
    """Function type for loading a dataset."""

    def __call__(  # noqa: PLR0913
        self,
        split: DatasetSplit,
        transformation_batch_size: int,
        name: str,
        *,
        use_cache: bool,
        run_path: str | None = None,
        run_name: str | None = None,
        table_name: str | None = None,
        ranking_threshold: float | None = None,
    ) -> Dataset:
        """Load the dataset.

        Args:
        ----
            split: The split of the dataset to load.
            transformation_batch_size: The batch size to use for the transformation.
            name: The name of the dataset.
            use_cache: Whether to use the cache.
            run_path: The path to the wandb run.
            run_name: The name of the wandb run.
            table_name: The name of the table.
            ranking_threshold: The ranking threshold for creating ranked pairs.

        Returns:
        -------
            The dataset.

        """
        ...
