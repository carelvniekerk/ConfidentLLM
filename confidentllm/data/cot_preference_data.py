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
#     http: //www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dataset containing the CoT responses ranked based on confidence scores."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from datasets import Dataset
from numpy import exp

import wandb
from wandb.apis.public.runs import Run, Runs

if TYPE_CHECKING:
    from wandb.apis.public.files import File, Files

__all__ = ["load_cot_preference_data"]


class Table(TypedDict):
    """Table type definition."""

    columns: list[str]
    data: list[list[str | float]]


def _load_run(path: str, run_name: str) -> Run:
    """Load a run from the Weights and Biases API."""
    api = wandb.Api()
    runs: Runs = api.runs(path)

    try:
        run: Run = next(run for run in runs if run.name == run_name)
    except StopIteration as err:
        raise KeyError(f"Run {run_name} not found in project {path}") from err  # noqa: EM102, TRY003

    return run


def _load_table(
    run: Run,
    table_name: str,
    root: Path = Path("wandb_downloads"),
) -> Table:
    """Load a table from a Weights and Biases run."""
    files: Files = run.files()
    table_file: File = next(file for file in files if table_name in file.name)

    table_file_root: Path = root / f"{run.name}/{table_name}"
    if not table_file_root.exists():
        table_file.download(root=table_file_root)  # type: ignore[arg-type]

    table_file_dir: Path = table_file_root / "media" / "table"
    table_file_path: Path = next(table_file_dir.glob("*.json"))

    with table_file_path.open("r") as reader:
        table: Table = json.load(reader)

    return table


def _reformat_table(
    table: Table,
) -> dict[str, dict[str, str | list[dict[str, str | float]]]]:
    """Reformat the table data into a dictionary."""
    decoding_path_columns: list[int] = [
        idx
        for idx, column_name in enumerate(table["columns"])
        if "decoded" in column_name.lower()
    ]
    confidence_columns: list[int] = [
        idx
        for idx, column_name in enumerate(table["columns"])
        if "confidence" in column_name.lower()
    ]

    responses_data: dict[str, dict[str, str | list[dict[str, str | float]]]] = {}
    for row in table["data"]:
        question: str = row[0]  # type: ignore[assignment]
        responses_data[question] = [  # type: ignore[assignment]
            {
                "answer": row[answer_idx],
                "confidence": row[conf_idx],
            }
            for answer_idx, conf_idx in zip(
                decoding_path_columns,
                confidence_columns,
                strict=True,
            )
        ]

    return responses_data


def _rank_data(
    question: str,
    response_data: list[dict[str, str | float]],
    *,
    threshold: float,
) -> dict[str, list[str]]:
    """Rank the response data based on the confidence scores."""
    ranked_data: dict[str, list[str]] = {
        "questions": [],
        "preferred_responses": [],
        "rejected_responses": [],
        "margin": [],
    }
    for response_1 in response_data:
        if response_1["confidence"] < threshold:  # type: ignore[operator]
            continue
        for response_2 in response_data:
            if response_1["confidence"] <= response_2["confidence"]:  # type: ignore[operator]
                continue
            ranked_data["questions"].append(question)
            ranked_data["preferred_responses"].append(response_1["answer"])  # type: ignore[arg-type]
            ranked_data["rejected_responses"].append(response_2["answer"])  # type: ignore[arg-type]
            # Margin the the exponential of the difference in confidence scores. This
            ranked_data["margin"].append(
                exp(response_1["confidence"] - response_2["confidence"]),  # type: ignore[arg-type,operator]
            )

    return ranked_data


def _process_data(table: Table, ranking_threshold: float) -> Dataset:
    """Process the table data into a dataset."""
    preference_data: dict[str, list[str]] = {
        "questions": [],
        "preferred_responses": [],
        "rejected_responses": [],
        "margin": [],
    }
    for question, response_data in _reformat_table(table).items():
        ranked_data: dict[str, list[str]] = _rank_data(
            question=question,
            response_data=response_data,  # type: ignore[arg-type]
            threshold=ranking_threshold,
        )
        preference_data["questions"].extend(ranked_data["questions"])
        preference_data["preferred_responses"].extend(
            ranked_data["preferred_responses"],
        )
        preference_data["rejected_responses"].extend(ranked_data["rejected_responses"])
        preference_data["margin"].extend(ranked_data["margin"])

    return Dataset.from_dict(preference_data)


def load_cot_preference_data(
    run_path: str,
    run_name: str,
    table_name: str,
    ranking_threshold: float = 0.9,
    name: str = "cot_preference",  # noqa: ARG001
) -> Dataset:
    """Load the CoT preference data from a Weights and Biases run."""
    run: Run = _load_run(path=run_path, run_name=run_name)

    initial_dataset_name: str = (
        run.config.get("run", {}).get("data", {}).get("name", "")
    )
    generation_method_config: dict[str, str | float] = run.config.get("run", {}).get(
        "generation_method",
        {},
    )

    if "cotdecoding" not in generation_method_config.get("_target_", "").lower():  # type: ignore[union-attr]
        raise ValueError(  # noqa: TRY003
            "Expected generation method to be CoTDecoding, got "  # noqa: EM102
            f"{generation_method_config.get('_target_', '')}",
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
        f"CoTDecoding with {generation_method_config.get('num_beams')} beams each with "
        f"a maximum length of {generation_method_config.get('max_length')}. During "
        f"decoding sampling was set to {generation_method_config.get('sampling')} "
        f"with a temperature of {generation_method_config.get('temperature')}. "
        f"The answers were ranked based on the answer token {confidence_method}."
    )

    table: Table = _load_table(run=run, table_name=table_name)
    text_dataset: Dataset = _process_data(
        table=table,
        ranking_threshold=ranking_threshold,
    )

    text_dataset._info.description = (  # noqa: SLF001 # Adding description to dataset
        f"CoT decoding based preference data for {initial_dataset_name}. "
        f"The data was obtained using {generation_method_description}."
    )
    text_dataset._info.citation = f"See original dataset {initial_dataset_name}."  # noqa: SLF001
    text_dataset._info.license = f"See original dataset {initial_dataset_name}."  # noqa: SLF001
    text_dataset._info.homepage = run.url  # noqa: SLF001

    return text_dataset
