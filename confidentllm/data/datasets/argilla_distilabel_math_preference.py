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
"""Module containing functions for loading the Argilla Math Preference dataset."""

from datasets import Dataset, Features, Value, load_dataset

from confidentllm.data.types import DatasetSplit

__all__ = ["load_argilla_math_data"]


def _preprocess_argilla_math_data(
    examples: dict[str, list[str]],
) -> dict[str, list[str] | list[float]]:
    questions: list[str] = examples.get("instruction", [])
    preferred_responses: list[str] = examples.get("chosen_response", [])
    rejected_responses: list[str] = examples.get("rejected_response", [])

    # Compute the margin
    preferred_scores: list[float] = [
        float(rating) for rating in examples.get("chosen_rating", [])
    ]
    rejected_scores: list[float] = [
        float(rating) for rating in examples.get("rejected_rating", [])
    ]
    # TODO: Update the scale of the margins if needed for training.
    margins: list[float] = [
        preferred_score - rejected_score
        for preferred_score, rejected_score in zip(
            preferred_scores,
            rejected_scores,
            strict=True,
        )
    ]

    # TODO: Extract answers for evaluation
    return {
        "question": questions,
        "preferred_response": preferred_responses,
        "rejected_response": rejected_responses,
        "margin": margins,
    }


def load_argilla_math_data(
    split: DatasetSplit = DatasetSplit.TRAIN,
    transformation_batch_size: int = 2048,
    name: str = "ArgillaMathPreference",
    *,
    use_cache: bool = True,
    **kwargs: dict,  # noqa: ARG001
) -> Dataset:
    """Load the Argilla Maths Preference dataset.

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
        path="argilla/distilabel-math-preference-dpo",
        split=split,
    )

    # Add metadata to the dataset
    data._info.dataset_name = name  # noqa: SLF001 # Adding dataset name
    data._info.description = (  # noqa: SLF001 # Adding description to dataset
        "The Argilla distilabel-math-preference-dpo dataset, developed by Argilla "
        "using the Distilabel framework, comprises approximately 2,418 entries. Each "
        "entry includes a math-related instruction, two model-generated responses, and "
        "corresponding quality ratings, facilitating the enhancement of mathematical "
        "reasoning in language models."
    )
    data._info.homepage = (  # noqa: SLF001
        "https://huggingface.co/datasets/argilla/distilabel-math-preference-dpo"
    )
    data._info.license = "MIT License"  # noqa: SLF001

    data = data.map(
        function=_preprocess_argilla_math_data,
        batched=True,
        batch_size=transformation_batch_size,
        load_from_cache_file=use_cache,
        remove_columns=data.column_names,
        features=Features(
            {
                "question": Value("string"),
                "preferred_response": Value("string"),
                "rejected_response": Value("string"),
                "margin": Value("float"),
            },
        ),
    )

    data.cached_version = use_cache  # type: ignore[attr-defined]

    return data
