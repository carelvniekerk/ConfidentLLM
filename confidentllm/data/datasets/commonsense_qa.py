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

from enum import Enum, auto
from functools import partial

from datasets import Dataset, Features, Value, load_dataset

from confidentllm.data.extract_answers import create_long_format_answer, format_choices
from confidentllm.data.types import DatasetSplit

__all__ = ["load_commonsense_qa_data"]


class CommonsenseQAAnswers(Enum):
    """Commonsense QA answers."""

    A = auto()
    B = auto()
    C = auto()
    D = auto()
    E = auto()


def _commonsense_qa_map(
    examples: dict[str, list[str | dict[str, list[str]]]],
) -> dict[str, list[str]]:
    _format_choices = partial(format_choices, choices_enum=CommonsenseQAAnswers)  # type: ignore[arg-type]
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
        CommonsenseQAAnswers[raw_answer.upper()].name if raw_answer else "-1"  # type: ignore[union-attr]
        for raw_answer in examples["answerKey"]
    ]

    long_format_answer: list[str] = [
        create_long_format_answer(
            choices=choices["text"],  # type: ignore[arg-type, index]
            answer=raw_answer,  # type: ignore[arg-type]
            choices_enum=CommonsenseQAAnswers,  # type: ignore[arg-type]
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


def load_commonsense_qa_data(
    split: DatasetSplit = DatasetSplit.TEST,
    transformation_batch_size: int = 512,
    name: str = "CommonsenseQA",
    *,
    use_cache: bool = True,
    **kwargs: dict,  # noqa: ARG001
) -> Dataset:
    """Load the Commonsense QA dataset.

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
    data: Dataset = load_dataset(  # type: ignore[assignment]
        path="tau/commonsense_qa",
        split=split,
    )

    # Add metadata to the dataset
    data._info.dataset_name = name  # noqa: SLF001 # Adding dataset name
    data._info.description = (  # noqa: SLF001 # Adding description to dataset
        "CommonsenseQA is a new multiple-choice question answering dataset that "
        "requires different types of commonsense knowledge to predict the correct "
        "answers . It contains 12,102 questions with one correct answer and four "
        "distractor answers. The dataset is provided in two major training/validation/"
        "testing set splits: 'Random split' which is the main evaluation split, "
        "and 'Question token split', see paper for details."
    )
    data._info.citation = (  # noqa: SLF001 # Adding citation to dataset
        '@inproceedings{talmor-etal-2019-commonsenseqa,\n\ttitle = "{C}ommonsense{QA}:'
        ' A Question Answering Challenge Targeting Commonsense Knowledge",\n\tauthor '
        '= "Talmor, Alon  and Herzig, Jonathan  and Lourie, Nicholas  and Berant, '
        'Jonathan",\n\tbooktitle = "Proceedings of the 2019 Conference of the North '
        "{A}merican Chapter of the Association for Computational Linguistics: Human "
        'Language Technologies, Volume 1 (Long and Short Papers)",\n\tmonth = jun,'
        '\n\tyear = "2019",\n\taddress = "Minneapolis, Minnesota",\n\tpublisher '
        '= "Association for Computational Linguistics",\n\turl = '
        '"https://aclanthology.org/N19-1421",\n\tdoi = "10.18653/v1/N19-1421",'
        '\n\tpages = "4149--4158",\n\tarchivePrefix = "arXiv",\n\teprint='
        '"1811.00937",\n\tprimaryClass="cs",\n}'
    )
    data._info.homepage = "https://huggingface.co/datasets/tau/commonsense_qa"  # noqa: SLF001
    data._info.license = "MIT License"  # noqa: SLF001

    data = data.map(
        function=_commonsense_qa_map,
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

    data.choices = [CommonsenseQAAnswers(value=i + 1).name for i in range(5)]  # type: ignore[attr-defined]
    data.cached_version = use_cache  # type: ignore[attr-defined]

    return data
