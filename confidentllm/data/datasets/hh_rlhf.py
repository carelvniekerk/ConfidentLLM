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
#     http: //www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HH-RLHF dataset loading functions."""

import re
from enum import StrEnum
from pathlib import Path

from datasets import (
    Dataset,
    Features,
    Sequence,
    Value,
    load_dataset,
)

from confidentllm.data.types import DatasetSplit

__all__ = ["load_hh_rlhf_data"]


class HHRLHFTask(StrEnum):
    """The HH-RLHF dataset tasks."""

    HELPFUL = "helpful-base"
    HARMLESS = "harmless-base"


def _extract_utterances(
    text: str,
    pattern: str = r"(Human|Assistant): (.*?)(?=\s*(?:Human|Assistant):|$)",
) -> list[str]:
    # Remove escape characters
    text = text.replace('"', "").replace("\n", "\n")
    text = re.sub(r"\n\n+", " ", text)

    # Use regex to capture all user and assistant utterances
    regex_matches: list[tuple[str, str]] = re.findall(
        pattern=pattern,
        string=text,
        flags=re.DOTALL,
    )
    # Format as a list of alternating turns
    utterances: list[str] = [utterance.strip() for _, utterance in regex_matches]

    return utterances


def _find_project_root() -> Path:
    """Find the root of the project."""
    directory: Path = Path.cwd()
    while not (directory / "pyproject.toml").exists():
        directory = directory.parent

    return directory / ".data_cache"


def _map_hh_rlhf_data(examples: dict[str, list[str]]) -> dict[str, list[list[str]]]:
    """Map the feature keys in the HH-RLHF to the standard keys."""
    mapped_examples: dict[str, list[str]] = {
        "preferred_response": examples["chosen"],
        "rejected_response": examples["rejected"],
    }

    extracted_examples: dict[str, list[list[str]]] = {}
    extracted_examples["preferred_response"] = [
        _extract_utterances(response)
        for response in mapped_examples["preferred_response"]
    ]
    extracted_examples["rejected_response"] = [
        _extract_utterances(response)
        for response in mapped_examples["rejected_response"]
    ]

    return extracted_examples


def load_hh_rlhf_data(
    split: DatasetSplit = DatasetSplit.TRAIN,
    transformation_batch_size: int = 2048,
    name: str = "helpful-RLHF",
    *,
    use_cache: bool = True,
    **kwargs: dict,  # noqa: ARG001
) -> Dataset:
    """Load the HH-RLHF dataset from Anthropic.

    Args:
    ----
        split: The dataset split to load.
        transformation_batch_size: The batch size for the transformation.
        name: The name of the dataset.
        use_cache: Whether to use the cache.
        kwargs: Additional keyword arguments.

    Returns:
    -------
        The HH-RLHF dataset.

    """
    task_name: str = name.split("-")[0].upper()
    if task_name not in HHRLHFTask.__members__:
        raise ValueError(  # noqa: TRY003
            f"Invalid task name '{task_name}'. "  # noqa: EM102
            f"Valid task names are: {', '.join(HHRLHFTask.__members__)}",
        )
    task: HHRLHFTask = HHRLHFTask[task_name]

    cache_path: Path = (
        _find_project_root() / "hh_rlhf" / task.value / split.value.lower()
    )
    if use_cache and cache_path.exists():
        data = Dataset.load_from_disk(cache_path)
        data.cached_version = use_cache  # type: ignore[attr-defined]
        return data

    data: Dataset = load_dataset(  # type: ignore[assignment]
        path="Anthropic/hh-rlhf",
        data_dir=task.value,
        split=split.value,
    )
    data = data.map(
        function=_map_hh_rlhf_data,
        batched=True,
        batch_size=transformation_batch_size,
        load_from_cache_file=use_cache,
        remove_columns=data.column_names,
        features=Features(
            {
                "preferred_response": Sequence(Value("string")),
                "rejected_response": Sequence(Value("string")),
            },
        ),
    )

    # Add metadata to the dataset
    data._info.dataset_name = name  # noqa: SLF001 # Adding dataset name
    data._info.description = (  # noqa: SLF001 # Adding description to dataset
        "The Anthropic HH RLHF dataset consists of two main components: human "
        "preference data on helpfulness and harmlessness, and red teaming dialogue "
        "data. The preference data, sourced from Training a Helpful and Harmless "
        "Assistant with Reinforcement Learning from Human Feedback, pairs chosen and "
        "rejected responses to train reward models for RLHF but is explicitly not"
        "intended for supervised fine-tuning of dialogue agents due to potential "
        "risks. The red teaming data, from Red Teaming Language Models to Reduce Harms,"
        " contains transcripts of adversarial interactions where human testers attempt "
        "to elicit harmful behavior from AI assistants, annotated with success ratings"
        " and harmlessness scores. This dataset aims to aid research in reducing AI "
        "harm, though it includes sensitive content that may be distressing."
    )
    data._info.citation = (  # noqa: SLF001 # Adding citation to dataset
        "@misc{bai2022traininghelpfulharmlessassistant,\n\ttitle={Training a Helpful "
        "and Harmless Assistant with Reinforcement Learning from Human Feedback}, \n\t"
        "author={Yuntao Bai and Andy Jones and Kamal Ndousse and Amanda Askell and Anna"
        " Chen and Nova DasSarma and Dawn Drain and Stanislav Fort and Deep Ganguli and"
        " Tom Henighan and Nicholas Joseph and Saurav Kadavath and Jackson Kernion and "
        "Tom Conerly and Sheer El-Showk and Nelson Elhage and Zac Hatfield-Dodds and "
        "Danny Hernandez and Tristan Hume and Scott Johnston and Shauna Kravec and "
        "Liane Lovitt and Neel Nanda and Catherine Olsson and Dario Amodei and Tom "
        "Brown and Jack Clark and Sam McCandlish and Chris Olah and Ben Mann and "
        "Jared Kaplan},\n\tyear={2022},\n\turl={https://arxiv.org/abs/2204.05862},\n}"
    )
    data._info.homepage = "https://github.com/anthropics/hh-rlhf?tab=readme-ov-file"  # noqa: SLF001
    data._info.license = "MIT License"  # noqa: SLF001

    data.cached_version = use_cache  # type: ignore[attr-defined]

    cache_path.mkdir(parents=True, exist_ok=True)
    data.save_to_disk(cache_path)

    return data
