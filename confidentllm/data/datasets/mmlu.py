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
"""Module containing functions for loading the MMLU dataset."""

from enum import Enum, StrEnum, auto

from datasets import Dataset, Features, Value, load_dataset

from confidentllm.data.types import DatasetSplit

__all__ = ["load_mmlu_data"]


class MMLUTask(StrEnum):
    """MMLU tasks."""

    ALL = auto()
    ABSTRACT_ALGEBRA = auto()
    ANATOMY = auto()
    ASTRONOMY = auto()
    BUSINESS_ETHICS = auto()
    CLINICAL_KNOWLEDGE = auto()
    COLLEGE_BIOLOGY = auto()
    COLLEGE_CHEMISTRY = auto()
    COLLEGE_COMPUTER_SCIENCE = auto()
    COLLEGE_MATHEMATICS = auto()
    COLLEGE_MEDICINE = auto()
    COLLEGE_PHYSICS = auto()
    COMPUTER_SECURITY = auto()
    CONCEPTUAL_PHYSICS = auto()
    ECONOMETRICS = auto()
    ELECTRICAL_ENGINEERING = auto()
    ELEMENTARY_MATHEMATICS = auto()
    FORMAL_LOGIC = auto()
    GLOBAL_FACTS = auto()
    HIGH_SCHOOL_BIOLOGY = auto()
    HIGH_SCHOOL_CHEMISTRY = auto()
    HIGH_SCHOOL_COMPUTER_SCIENCE = auto()
    HIGH_SCHOOL_EUROPEAN_HISTORY = auto()
    HIGH_SCHOOL_GEOGRAPHY = auto()
    HIGH_SCHOOL_GOVERNMENT_AND_POLITICS = auto()
    HIGH_SCHOOL_MACROECONOMICS = auto()
    HIGH_SCHOOL_MATHEMATICS = auto()
    HIGH_SCHOOL_MICROECONOMICS = auto()
    HIGH_SCHOOL_PHYSICS = auto()
    HIGH_SCHOOL_PSYCHOLOGY = auto()
    HIGH_SCHOOL_STATISTICS = auto()
    HIGH_SCHOOL_US_HISTORY = auto()
    HIGH_SCHOOL_WORLD_HISTORY = auto()
    HUMAN_AGING = auto()
    HUMAN_SEXUALITY = auto()
    INTERNATIONAL_LAW = auto()
    JURISPRUDENCE = auto()
    LOGICAL_FALLACIES = auto()
    MACHINE_LEARNING = auto()
    MANAGEMENT = auto()
    MARKETING = auto()
    MEDICAL_GENETICS = auto()
    MISCELLANEOUS = auto()
    MORAL_DISPUTES = auto()
    MORAL_SCENARIOS = auto()
    NUTRITION = auto()
    PHILOSOPHY = auto()
    PREHISTORY = auto()
    PROFESSIONAL_ACCOUNTING = auto()
    PROFESSIONAL_LAW = auto()
    PROFESSIONAL_MEDICINE = auto()
    PROFESSIONAL_PSYCHOLOGY = auto()
    PUBLIC_RELATIONS = auto()
    SECURITY_STUDIES = auto()
    SOCIOLOGY = auto()
    US_FOREIGN_POLICY = auto()
    VIROLOGY = auto()
    WORLD_RELIGIONS = auto()


class MMLUAnswers(Enum):
    """MMLU answers."""

    A = auto()
    B = auto()
    C = auto()
    D = auto()


def _get_mmlu_task(dataset_name: str) -> MMLUTask:
    """Get the MMLU task from the dataset name.

    Args:
    ----
        dataset_name: The name of the dataset.

    Returns:
    -------
        The MMLU task.

    """
    task_name = dataset_name.split("_", 1)[-1].lower()
    if task_name == "mmlu":
        return MMLUTask.ALL
    return MMLUTask(task_name)


def _format_choices(choices: list[str]) -> str:
    """Format the choices for the MMLU dataset.

    Args:
    ----
        choices: The choices to format.

    Returns:
    -------
        The formatted choices.

    """
    choices_str: str = "Select one of the following:"
    for i, choice in enumerate(choices):
        choices_str += f"\n{MMLUAnswers(value=i + 1).name}. {choice}"
    return choices_str


def _mmlu_map(examples: dict[str, list[str | list[str]]]) -> dict[str, list[str]]:
    question: list[str] = examples.get("question", [])  # type: ignore[assignment]
    question = [
        f"{question_str}\n{_format_choices(choices)}"  # type: ignore[arg-type]
        for question_str, choices in zip(question, examples["choices"], strict=True)
    ]

    answer: list[str] = [
        MMLUAnswers(value=answer_index + 1).name  # type: ignore[operator]
        for answer_index in examples["answer"]
    ]

    return {"question": question, "answer": answer}


def load_mmlu_data(
    split: DatasetSplit = DatasetSplit.TEST,
    transformation_batch_size: int = 512,
    name: str = "MMLU",
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
    data: Dataset = load_dataset(
        path="cais/mmlu",
        name=_get_mmlu_task(name).value,
        split=split,
    )  # type: ignore[reportAssignmentType]

    # Add metadata to the dataset
    data._info.description = (  # noqa: SLF001 # Adding description to dataset
        "MMLU is a Massive Multitask test consisting of multiple-choice questions from "
        "various branches of knowledge. The test spans subjects in the humanities, "
        "social sciences, hard sciences, and other areas that are important for some "
        "people to learn. This covers 57 tasks including elementary mathematics, US "
        "history, computer science, law, and more. To attain high accuracy on this "
        "test, models must possess extensive world knowledge and problem solving "
        "ability."
    )
    data._info.citation = (  # noqa: SLF001 # Adding citation to dataset
        "@article{hendryckstest2021,\n\ttitle={Measuring Massive Multitask Language "
        "Understanding},\n\tauthor={Dan Hendrycks and Collin Burns and Steven Basart "
        "and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},\n\t"
        "journal={Proceedings of the International Conference on Learning "
        "Representations (ICLR)},\n\tyear={2021}\n}"
    )
    data._info.homepage = "https://huggingface.co/datasets/cais/mmlu"  # noqa: SLF001
    data._info.license = "MIT License"  # noqa: SLF001

    data = data.map(
        function=_mmlu_map,
        batched=True,
        batch_size=transformation_batch_size,
        load_from_cache_file=use_cache,
        remove_columns=data.column_names,
        features=Features(
            {
                "question": Value("string"),
                "answer": Value("string"),
            },
        ),
    )

    data.choices = [MMLUAnswers(value=i + 1).name for i in range(4)]  # type: ignore[attr-defined]
    data.cached_version = use_cache  # type: ignore[attr-defined]

    return data
