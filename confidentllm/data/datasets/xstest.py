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
"""Module containing functions for loading the XSTest dataset."""

from datasets import Dataset, Features, Value, load_dataset

from confidentllm.data.types import DatasetSplit

__all__ = ["load_xstest_data"]


def _xstest_map(
    examples: dict[str, list[str | dict[str, list[str]]]],
) -> dict[str, list[str]]:
    question: list[str] = examples.get("prompt", [])  # type: ignore[assignment]

    answer: list[str] = examples.get("label", [])  # type: ignore[assignment]
    # TODO: Add long format answer.

    return {
        "id": examples["id"],  # type: ignore[dict-item]
        "question": question,
        "answer": answer,
        "long_format_answer": answer,
    }


def load_xstest_data(
    split: DatasetSplit = DatasetSplit.TEST,
    transformation_batch_size: int = 512,
    name: str = "XSTest",
    *,
    use_cache: bool = True,
    **kwargs: dict,  # noqa: ARG001
) -> Dataset:
    """Load the XSTest dataset.

    Args:
    ----
        split: The split of the dataset to load.
        transformation_batch_size: The batch size to use for the transformation.
        name: The name of the dataset.
        use_cache: Whether to use the cache.
        kwargs: Additional keyword arguments.

    Returns:
    -------
        The dataset.

    """
    data: Dataset = load_dataset(  # type: ignore[invalid-assignment]
        path="Paul/XSTest",
        split=split,
    )

    # Add metadata to the dataset
    data._info.dataset_name = name  # noqa: SLF001 # Adding dataset name
    data._info.description = (  # noqa: SLF001 # Adding description to dataset
        "XSTest is a test suite designed to identify exaggerated safety / false refusal"
        " in Large Language Models (LLMs). It comprises 250 safe prompts across 10 "
        "different prompt types, along with 200 unsafe prompts as contrasts. The test "
        "suite aims to evaluate how well LLMs balance being helpful with being harmless"
        " by testing if they unnecessarily refuse to answer safe prompts that "
        "superficially resemble unsafe ones."
    )
    data._info.citation = (  # noqa: SLF001 # Adding citation to dataset
        "@inproceedings{rottger2024xstest,\n\ttitle={XSTest: A Test Suite for "
        "Identifying Exaggerated Safety Behaviours in Large Language Models},\n"
        '\tauthor={R{"o}ttger, Paul and Kirk, Hannah and Vidgen, Bertie and '
        "Attanasio, Giuseppe and Bianchi, Federico and Hovy, Dirk},\n"
        "\tbooktitle={Proceedings of the 2024 Conference of the North American "
        "Chapter of the Association for Computational Linguistics: Human Language "
        "Technologies (Volume 1: Long Papers)},\n"
        "\tpages={5377--5400},\n"
        "\tyear={2024}\n"
        "}"
    )
    data._info.homepage = "https://github.com/paul-rottger/xstest"  # noqa: SLF001
    data._info.license = "CC-BY-4.0"  # noqa: SLF001

    data = data.map(
        function=_xstest_map,
        batched=True,
        batch_size=transformation_batch_size,
        load_from_cache_file=use_cache,
        remove_columns=data.column_names,
        features=Features(
            {
                "id": Value("string"),
                "question": Value("string"),
                "answer": Value("string"),
                "long_format_answer": Value("string"),
            },
        ),
    )

    data.cached_version = use_cache  # type: ignore[attr-defined]

    return data
