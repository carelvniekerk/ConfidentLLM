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
"""Module containing functions for loading the OpenBook QA dataset."""

from enum import Enum, auto

from datasets import Dataset, Features, Value, load_dataset

from confidentllm.data.types import DatasetSplit

__all__ = ["load_openbook_qa_data"]


class OpenBookQAAnswers(Enum):
    """Open Book QA answers."""

    A = auto()
    B = auto()
    C = auto()
    D = auto()


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
        choices_str += f"\n{OpenBookQAAnswers(value=i + 1).name}. {choice}"
    return choices_str


def _openbook_qa_map(
    examples: dict[str, list[str | dict[str, list[str]]]],
) -> dict[str, list[str]]:
    question: list[str] = examples.get("question_stem", [])  # type: ignore[assignment]
    question = [
        f"{question_str}\n{_format_choices(choices["text"])}"  # type: ignore[arg-type, index]
        for question_str, choices in zip(
            question,
            examples["choices"],  # type: ignore[call-overload]
            strict=True,
        )
    ]

    answer: list[str] = [
        OpenBookQAAnswers[raw_answer.upper()].name if raw_answer else "-1"  # type: ignore[union-attr]
        for raw_answer in examples["answerKey"]
    ]

    return {"id": examples["id"], "question": question, "answer": answer}  # type: ignore[dict-item]


def load_openbook_qa_data(
    split: DatasetSplit = DatasetSplit.TEST,
    transformation_batch_size: int = 512,
    name: str = "OpenBookQA",  # noqa: ARG001
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

    Returns:
    -------
        The dataset.

    """
    data: Dataset = load_dataset(
        path="allenai/openbookqa",
        name="main",
        split=split,
    )  # type: ignore[assignment]

    # Add metadata to the dataset
    data._info.description = (  # noqa: SLF001 # Adding description to dataset
        "OpenBookQA aims to promote research in advanced question-answering, probing a "
        "deeper understanding of both the topic (with salient facts summarized as an "
        "open book, also provided with the dataset) and the language it is expressed in"
        ". In particular, it contains questions that require multi-step reasoning, use "
        "of additional common and commonsense knowledge, and rich text comprehension. "
        "OpenBookQA is a new kind of question-answering dataset modeled after open "
        "book exams for assessing human understanding of a subject."
    )
    data._info.citation = (  # noqa: SLF001 # Adding citation to dataset
        "@inproceedings{OpenBookQA2018,\n\ttitle={Can a Suit of Armor Conduct "
        "Electricity? A New Dataset for Open Book Question Answering},\n\tauthor={Todor"
        " Mihaylov and Peter Clark and Tushar Khot and Ashish Sabharwal},\n\tbooktitle="
        "{EMNLP},\n\tyear={2018}\n}"
    )
    data._info.homepage = "https://huggingface.co/datasets/allenai/openbookqa"  # noqa: SLF001
    data._info.license = "MIT License"  # noqa: SLF001

    data = data.map(
        function=_openbook_qa_map,
        batched=True,
        batch_size=transformation_batch_size,
        load_from_cache_file=use_cache,
        remove_columns=data.column_names,
        features=Features(
            {
                "id": Value("string"),
                "question": Value("string"),
                "answer": Value("string"),
            },
        ),
    )

    data.choices = [OpenBookQAAnswers(value=i + 1).name for i in range(4)]  # type: ignore[attr-defined]
    data.cached_version = use_cache  # type: ignore[attr-defined]

    return data
