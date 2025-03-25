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
"""Dataset containing the generated answer data."""

import re
from typing import TYPE_CHECKING

from datasets import Dataset

from confidentllm.data.datasets.cot_preference_data import (
    Table,
    _cleanup_text,
    _find_project_root,
    _load_run,
    _load_table,
)

if TYPE_CHECKING:
    from pathlib import Path

    from wandb.apis.public.runs import Run

__all__ = ["_extract_utterances", "load_question_answering_data"]


def _reformat_table(
    table: Table,
) -> dict[str, str]:
    """Reformat the table data into a dictionary."""
    answer_columns: list[int] = [
        idx
        for idx, column_name in enumerate(table["columns"])
        if column_name.lower() == "reasoning"
    ]
    answer_column: int = answer_columns[0] if answer_columns else 1

    responses_data: dict[str, str] = {}
    for row in table["data"]:
        question: str = row[0]  # type: ignore[assignment]
        responses_data[question] = row[answer_column]  # type: ignore[assignment]

    return responses_data


def _extract_utterances(
    text: str,
    pattern: str = r"(User|Assistant): (.*?)(?=\s*(?:User|Assistant):|$)",
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


def _process_data(table: Table) -> Dataset:
    """Process the table data into a dataset."""
    preference_data: dict[str, list[str | list[str]]] = {
        "question": [],
        "preferred_response": [],
        "rejected_response": [],
    }
    for question, response in _reformat_table(table).items():
        question = _cleanup_text(text=question)  # noqa: PLW2901
        if not question:
            continue
        response = _cleanup_text(text=response)  # type: ignore[arg-type]  # noqa: PLW2901
        if not response:
            continue

        if "User: " in question:
            responses: list[str] = _extract_utterances(text=question)
            responses.append(response)

            preference_data["preferred_response"].append(responses)
            preference_data["rejected_response"].append(responses)
        else:
            preference_data["question"].append(question)
            preference_data["preferred_response"].append(response)
            preference_data["rejected_response"].append(response)

    if not preference_data["question"]:
        preference_data.pop("question")

    return Dataset.from_dict(preference_data)


def load_question_answering_data(
    run_path: str,
    run_name: str,
    table_name: str,
    name: str = "question_answering",  # noqa: ARG001
    *,
    use_cache: bool = True,
    **kwargs: dict,  # noqa: ARG001
) -> Dataset:
    """Load the question answering data from a Weights and Biases run."""
    data_caching_path: Path = _find_project_root() / run_name
    if data_caching_path.exists() and use_cache:
        dataset: Dataset = Dataset.load_from_disk(data_caching_path)
        dataset.cached_version = use_cache  # type: ignore[attr-defined]
        return dataset

    run: Run = _load_run(path=run_path, run_name=run_name)

    initial_dataset_name: str = (
        run.config.get("run", {}).get("data", {}).get("name", "")
    )
    generation_method_config: dict[str, str | float] = run.config.get("run", {}).get(
        "generation_method",
        {},
    )

    confidence_method: str = (
        generation_method_config.get(
            "confidence_extraction_method",
            {},
        )
        .get("_target_", "")  # type: ignore[union-attr, call-overload]
        .split(".")[-1]
    )

    generation_method_description: str = (
        f"Answer generation with {generation_method_config.get('num_beams')} beams each"
        f" with a maximum length of {generation_method_config.get('max_length')}. "
        f"During decoding sampling was set to {generation_method_config.get('sampling')}"  # noqa: E501
        f" with a temperature of {generation_method_config.get('temperature')}. "
        f"The answers were ranked based on the answer token {confidence_method}."
    )

    table: Table = _load_table(run=run, table_name=table_name)
    text_dataset: Dataset = _process_data(table=table)

    text_dataset._info.description = (  # noqa: SLF001 # Adding description to dataset
        f"Answer generation data for {initial_dataset_name}. "
        f"The data was obtained using {generation_method_description}."
    )
    text_dataset._info.citation = f"See original dataset {initial_dataset_name}."  # noqa: SLF001
    text_dataset._info.license = f"See original dataset {initial_dataset_name}."  # noqa: SLF001
    text_dataset._info.homepage = run.url  # noqa: SLF001

    text_dataset.cached_version = use_cache  # type: ignore[attr-defined]
    text_dataset.save_to_disk(data_caching_path)
    return text_dataset
