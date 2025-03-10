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

import json
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from datasets import Dataset

import wandb
from wandb.apis.public.runs import Run, Runs

if TYPE_CHECKING:
    from wandb.apis.public.files import File, Files

__all__ = ["load_question_answering_data"]


class Table(TypedDict):
    """Table type definition."""

    columns: list[str]
    data: list[list[str | float]]


def _remove_prompt(text: str) -> str:
    """Remove the prompt from the text."""
    sentences: list[str] = [sentence for sentence in text.split(".") if sentence]
    sentences = sentences[:-1]

    text = ""
    for sentence in sentences:
        if sentence[0] == " ":
            sentence = sentence[1:]  # noqa: PLW2901
        if sentence[-1] == " ":
            sentence = sentence[:-1]  # noqa: PLW2901
        sentence = sentence.strip()  # noqa: PLW2901
        text += sentence + ". " if sentence else ""

    if not text:
        return text
    if text[-1] == " ":
        text = text[:-1]
    text += "." if text[-1] not in [".", "?", "!"] else ""

    return text


def _cleanup_text(text: str, *, remove_prompt: bool = False) -> str:
    """Clean up the text by removing the prompt."""
    if remove_prompt:
        text = _remove_prompt(text)

    if not text:
        return text

    if text[0] == " ":
        text = text[1:]
    if text[-1] == " ":
        text = text[:-1]
    return text.strip()


def _load_run(path: str, run_name: str) -> Run:
    """Load a run from the Weights and Biases API."""
    api = wandb.Api()
    runs: Runs = api.runs(path)

    try:
        run: Run = next(run for run in runs if run.name == run_name)
    except StopIteration as err:
        raise KeyError(f"Run {run_name} not found in project {path}") from err  # noqa: EM102, TRY003

    return run


def _find_project_root() -> Path:
    """Find the root of the project."""
    directory: Path = Path.cwd()
    while not (directory / "pyproject.toml").exists():
        directory = directory.parent

    return directory / ".data_cache"


def _load_table(
    run: Run,
    table_name: str,
    root: Path | None = None,
) -> Table:
    """Load a table from a Weights and Biases run."""
    if root is None:
        root = _find_project_root()

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
) -> dict[str, dict[str, str]]:
    """Reformat the table data into a dictionary."""
    answer_columns: list[int] = [
        idx
        for idx, column_name in enumerate(table["columns"])
        if column_name.lower() == "answer"
    ]
    answer_column: int = answer_columns[0] if answer_columns else 1

    responses_data: dict[str, dict[str, str]] = {}
    for row in table["data"]:
        question: str = row[0]  # type: ignore[assignment]
        responses_data[question] = row[answer_column]  # type: ignore[assignment]

    return responses_data


def _process_data(table: Table) -> Dataset:
    """Process the table data into a dataset."""
    preference_data: dict[str, list[str]] = {
        "question": [],
        "preferred_response": [],
        "rejected_response": [],
    }
    for question, response in _reformat_table(table).items():
        question = _cleanup_text(text=question)  # noqa: PLW2901
        if not question:
            continue
        response = _cleanup_text(text=response)  # type: ignore[assignment,arg-type]  # noqa: PLW2901
        if not response:
            continue
        preference_data["question"].append(question)
        preference_data["preferred_response"].append(response)  # type: ignore[arg-type]
        preference_data["rejected_response"].append(response)  # type: ignore[arg-type]

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
