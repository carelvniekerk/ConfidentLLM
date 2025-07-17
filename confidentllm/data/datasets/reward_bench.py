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
"""Module containing functions for loading the Reward Bench dataset."""

from enum import StrEnum
from functools import partial
from typing import Callable

from datasets import Dataset, Features, Value, load_dataset

from confidentllm.data.types import DatasetSplit

__all__ = ["load_reward_bench_data"]


class RewardBenchSubset(StrEnum):
    """Reward Bench tasks."""

    ALPACA_EVAL_EASY = "alpacaeval-easy"
    ALPACA_EVAL_LENGTH = "alpacaeval-length"
    ALPACA_EVAL_HARD = "alpacaeval-hard"
    MATH_PRM = "math-prm"
    XSTEST_SHOULD_RESPOND = "xstest-should-respond"
    XSTEST_SHOULD_REFUSE = "xstest-should-refuse"
    HEP_GO = "hep-go"
    HEP_CPP = "hep-cpp"
    HEP_JS = "hep-js"
    HEP_RUST = "hep-rust"
    HEP_PYTHON = "hep-python"
    HEP_JAVA = "hep-java"


def _get_reward_bench_task(dataset_name: str) -> RewardBenchSubset:
    """Get the Reward Bench task from the dataset name.

    Args:
    ----
        dataset_name: The name of the dataset.

    Returns:
    -------
        The Reward Bench task.

    """
    subset_name: str = dataset_name.split("_", 2)[-1].lower()
    if subset_name == "bench":
        return RewardBenchSubset.MATH_PRM
    return RewardBenchSubset(subset_name)


def _filter_by_subset(
    subset_name: RewardBenchSubset,
    example: dict[str, str],
) -> bool:
    """Filter the dataset by the subset name.

    Args:
    ----
        subset_name: The subset name.
        example: The example.

    Returns:
    -------
        Whether the example is part of the subset.

    """
    return subset_name.value.lower() in example["subset"]


def _map_reward_bench_data(examples: dict[str, list[str]]) -> dict[str, list[str]]:
    """Map the keys of RewardBench to the standard keys."""
    mapped_examples: dict[str, list[str]] = {
        "question": examples["prompt"],
        "preferred_response": examples["chosen"],
        "rejected_response": examples["rejected"],
    }

    return mapped_examples


def load_reward_bench_data(
    split: DatasetSplit = DatasetSplit.TEST,  # noqa: ARG001
    transformation_batch_size: int = 512,
    name: str = "reward_bench",
    *,
    use_cache: bool = True,
    **kwargs: dict,  # noqa: ARG001
) -> Dataset:
    """Load the MMLU dataset.

    Args:
    ----
        split: The split of the dataset to load.
        transformation_batch_size: The batch size to use for the transformation.
        name: The name of the dataset.
        use_cache: Whether to use the cache.
        kwargs: Additional keyword arguments

    Returns:
    -------
        The dataset.

    """
    data: Dataset = load_dataset(  # type: ignore[invalid-assignment]
        path="allenai/reward-bench",
        split="filtered",
    )

    # Add metadata to the dataset
    data._info.dataset_name = name  # noqa: SLF001 # Adding dataset name
    data._info.description = (  # noqa: SLF001 # Adding description to dataset
        "The RewardBench evaluation dataset assesses reward models across five "
        "categories: Chat, which includes both easy (e.g., AlpacaEval-easy, "
        "MT-Bench-easy) and hard chat tasks (e.g., MT-Bench-hard, LLMBar-"
        "adversarial sets); Safety, covering refusal and ethical response scenarios; "
        "Reasoning, testing code and math comprehension; and a final category "
        "incorporating prior preference datasets such as Anthropic Helpful, "
        "Stanford Human Preferences (SHP), and OpenAI's Learning to Summarize data. "
        "The leaderboard averages performance across these subsets, scoring models "
        "based on whether they assign higher rewards to preferred responses over "
        "rejected ones."
    )
    data._info.citation = (  # noqa: SLF001 # Adding citation to dataset
        "@misc{RewardBench,\n\ttitle={RewardBench: Evaluating Reward Models for "
        "Language Modeling},\n\tauthor={Lambert, Nathan and Pyatkin, Valentina and "
        "Morrison, Jacob and Miranda, LJ and Lin, Bill Yuchen and Chandu, Khyathi "
        "and Dziri, Nouha and Kumar, Sachin and Zick, Tom and Choi, Yejin and Smith, "
        "Noah A. and Hajishirzi, Hannaneh},\n\tyear={2024},\n\thowpublished={"
        "\\url{https://huggingface.co/spaces/allenai/reward-bench}"
    )
    data._info.homepage = "https://huggingface.co/datasets/allenai/reward-bench"  # noqa: SLF001
    data._info.license = "ODC-BY"  # noqa: SLF001

    # Filter the dataset by the subset
    filter_function: Callable[[dict[str, str]], bool] = partial(
        _filter_by_subset,
        _get_reward_bench_task(name),
    )
    data = data.filter(function=filter_function)

    data = data.map(
        function=_map_reward_bench_data,
        batched=True,
        batch_size=transformation_batch_size,
        load_from_cache_file=use_cache,
        remove_columns=data.column_names,
        features=Features(
            {
                "question": Value("string"),
                "preferred_response": Value("string"),
                "rejected_response": Value("string"),
            },
        ),
    )

    data.cached_version = use_cache  # type: ignore[attr-defined]

    return data
