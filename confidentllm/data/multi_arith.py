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
"""Module containing functions for loading the MultiArith dataset."""

from datasets import Dataset, load_dataset

from confidentllm.data.types import DatasetSplit

__all__ = ["load_multi_arith_data"]


def multi_arith_key_mapping(data: dict[str, list[str]]) -> dict[str, list[str]]:
    """Map the keys in the data dictionary to the correct keys.

    Args:
    ----
        data: The data dictionary.

    Returns:
    -------
        The data dictionary with the correct keys.

    """
    data["answer"] = data.pop("final_ans")

    return data


def load_multi_arith_data(
    split: DatasetSplit = DatasetSplit.TEST,
    transformation_batch_size: int = 512,
    name: str = "MultiArith",  # noqa: ARG001 - Used for creating the experiments path
    *,
    use_cache: bool = True,
    **kwargs: dict,  # noqa: ARG001
) -> Dataset:
    """Load the MultiArith dataset.

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
        path="ChilleD/MultiArith",
        name="default",
        split=split,
    )  # type: ignore[reportAssignmentType]

    # Add metadata to the dataset
    data._info.description = (  # noqa: SLF001 # Provide description for the dataset
        "The Multi-Arith dataset contains elementary arithmetic problems with multiple"
        " operations, such as addition, subtraction, multiplication, and division, "
        "requiring sequential reasoning steps to solve."
    )
    data._info.citation = (  # noqa: SLF001 # Provide citation for the dataset
        '@inproceedings{roy-roth-2015-solving,\n  title = "Solving General Arithmetic '
        'Word Problems",\n  author = "Roy, Subhro and Roth, Dan",\n   booktitle = '
        '"Proceedings of the 2015 Conference on Empirical Methods in Natural Language '
        'Processing",\n  year = 2015,\n  address = "Lisbon, Portugal",\n  publisher '
        '= "Association for Computational Linguistics",\n  url = '
        '"https://aclanthology.org/D15-1202",\n  doi = "10.18653/v1/D15-1202",\n  '
        'pages = "1743--1752",\n  }'
    )
    data._info.homepage = "https://huggingface.co/datasets/ChilleD/MultiArith"  # noqa: SLF001
    data._info.license = "CC BY 4.0"  # noqa: SLF001

    data = data.map(
        multi_arith_key_mapping,
        batched=True,
        batch_size=transformation_batch_size,
        load_from_cache_file=use_cache,
    )

    data.cached_version = use_cache  # type: ignore[attr-defined]

    return data
