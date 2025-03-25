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
    _load_run,
    _load_table,
)
from confidentllm.data.datasets.qa_data import _extract_utterances

if TYPE_CHECKING:
    from wandb.apis.public.runs import Run

__all__ = ["load_predictions_table"]


def _reformat_table(
    table: Table,
) -> dict[str, tuple[str, str, str]]:
    """Reformat the table data into a dictionary."""
    prediction_columns: list[int] = [
        idx
        for idx, column_name in enumerate(table["columns"])
        if column_name.lower() == "answer"
    ]
    prediction_column: int = prediction_columns[0] if prediction_columns else 1

    answer_columns: list[int] = [
        idx
        for idx, column_name in enumerate(table["columns"])
        if column_name.lower() == "true answer"
    ]
    answer_column: int = answer_columns[0] if answer_columns else 2

    confidence_columns: list[int] = [
        idx
        for idx, column_name in enumerate(table["columns"])
        if column_name.lower() == "confidence"
    ]
    confidence_column: int = confidence_columns[0] if confidence_columns else 3

    responses_data: dict[str, tuple[str, str, str]] = {}
    for row in table["data"]:
        question: str = row[0]  # type: ignore[assignment]
        responses_data[question] = (  # type: ignore[assignment]
            row[prediction_column],
            row[answer_column],
            row[confidence_column],
        )

    return responses_data


def _process_data(table: Table) -> Dataset:
    """Process the table data into a dataset."""
    preference_data: dict[str, list[str | list[str] | float]] = {
        "question": [],
        "prediction": [],
        "answer": [],
        "confidence": [],
    }
    for question, response in _reformat_table(table).items():
        prediction, answer, confidence = response
        question = _cleanup_text(text=question)  # noqa: PLW2901
        if not question:
            continue

        if "User: " in question:
            responses: list[str] = _extract_utterances(text=question)

            preference_data["question"].append(responses)
        else:
            preference_data["question"].append(question)

        preference_data["prediction"].append(prediction)
        preference_data["answer"].append(answer)
        preference_data["confidence"].append(float(confidence))

    return Dataset.from_dict(preference_data)


def load_predictions_table(
    run_path: str,
    run_name: str,
    table_name: str,
) -> Dataset:
    """Load the predictions table data from a Weights and Biases run."""
    run: Run = _load_run(path=run_path, run_name=run_name)
    table: Table = _load_table(run=run, table_name=table_name)
    text_dataset: Dataset = _process_data(table=table)

    return text_dataset
