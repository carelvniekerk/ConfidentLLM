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
"""Module containing functions for loading the Commonsense QA dataset."""

from enum import Enum, StrEnum, auto
from functools import partial

from datasets import Dataset, Features, Value, load_dataset

from confidentllm.data.extract_answers import create_long_format_answer, format_choices
from confidentllm.data.types import DatasetSplit

__all__ = ["load_arc_data"]


class ARCAnswers(Enum):
    """ARC answers."""

    A = auto()
    B = auto()
    C = auto()
    D = auto()
    E = auto()


class ARCQuestionType(StrEnum):
    """ARC question types."""

    EASY = auto()
    CHALLENGE = auto()


def _get_arc_question_type(dataset_name: str) -> ARCQuestionType:
    """Get the ARC question difficulty from the dataset name.

    Args:
    ----
        dataset_name: The name of the dataset.

    Returns:
    -------
        The ARC question difficulty.

    """
    task_name = dataset_name.split("_", 1)[-1].lower()
    return ARCQuestionType(task_name)


def _arc_map(
    examples: dict[str, list[str | dict[str, list[str]]]],
) -> dict[str, list[str]]:
    _format_choices = partial(format_choices, choices_enum=ARCAnswers)  # type: ignore[arg-type]
    question: list[str] = examples.get("question", [])  # type: ignore[assignment]
    question = [
        f"{question_str}\n{_format_choices(choices['text'])}"  # type: ignore[arg-type, index]
        for question_str, choices in zip(
            question,
            examples["choices"],  # type: ignore[call-overload]
            strict=True,
        )
    ]

    answer: list[str] = [
        ARCAnswers(int(raw_answer)).name  # type: ignore[arg-type]
        if raw_answer.isdigit()  # type: ignore[union-attr]
        else ARCAnswers[raw_answer.upper()].name  # type: ignore[union-attr]
        if raw_answer
        else "-1"
        for raw_answer in examples["answerKey"]
    ]

    long_format_answer: list[str] = [
        create_long_format_answer(
            choices=choices["text"],  # type: ignore[arg-type, index]
            answer=ARCAnswers(int(raw_answer)).name  # type: ignore[arg-type]
            if raw_answer.isdigit()  # type: ignore[union-attr]
            else raw_answer,
            choices_enum=ARCAnswers,  # type: ignore[arg-type]
        )
        for choices, raw_answer in zip(
            examples["choices"],
            examples["answerKey"],
            strict=True,
        )
    ]

    return {
        "id": examples["id"],  # type: ignore[dict-item]
        "question": question,
        "answer": answer,
        "long_format_answer": long_format_answer,
    }


def load_arc_data(
    split: DatasetSplit = DatasetSplit.TEST,
    transformation_batch_size: int = 512,
    name: str = "ARC_EASY",
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
        kwargs: Additional keyword arguments.

    Returns:
    -------
        The dataset.

    """
    data: Dataset = load_dataset(
        path="allenai/ai2_arc",
        name=f"ARC-{_get_arc_question_type(name).name.lower().title()}",
        split=split,
    )  # type: ignore[reportAssignmentType]

    # Add metadata to the dataset
    data._info.dataset_name = name  # noqa: SLF001 # Adding dataset name
    data._info.description = (  # noqa: SLF001 # Adding description to dataset
        "A new dataset of 7,787 genuine grade-school level, multiple-choice science "
        "questions, assembled to encourage research in advanced question-answering. "
        "The dataset is partitioned into a Challenge Set and an Easy Set, where the "
        "former contains only questions answered incorrectly by both a retrieval-based "
        "algorithm and a word co-occurrence algorithm. We are also including a corpus "
        "of over 14 million science sentences relevant to the task, and an "
        "implementation of three neural baseline models for this dataset. We pose ARC "
        "as a challenge to the community."
    )
    data._info.citation = (  # noqa: SLF001 # Adding citation to dataset
        "@article{allenai:arc,\n\tauthor= {Peter Clark  and Isaac Cowhey and Oren "
        "Etzioni and Tushar Khot and Ashish Sabharwal and Carissa Schoenick and Oyvind "
        "Tafjord},\n\ttitle= {Think you have Solved Question Answering? Try ARC, "
        "the AI2 Reasoning Challenge},\n\tjournal= {arXiv:1803.05457v1},\n\tyear= "
        "{2018},\n}"
    )
    data._info.homepage = "https://huggingface.co/datasets/allenai/ai2_arc?row=2"  # noqa: SLF001
    data._info.license = "MIT License"  # noqa: SLF001

    data = data.map(
        function=_arc_map,
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

    data.choices = [ARCAnswers(value=i + 1).name for i in range(5)]  # type: ignore[attr-defined]
    data.cached_version = use_cache  # type: ignore[attr-defined]

    return data
