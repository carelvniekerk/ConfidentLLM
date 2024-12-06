# coding=utf-8
# --------------------------------------------------------------------------------
# Project: ConfidenceLLM
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
"""Load and merge multiple datasets into a single dataset."""

from datasets import Dataset, concatenate_datasets

from confidentllm.data.arc import load_arc_data
from confidentllm.data.commonsense_qa import load_commonsense_qa_data
from confidentllm.data.cot_preference_data import load_cot_preference_data
from confidentllm.data.gsm8k import load_gsm8k_data
from confidentllm.data.mmlu import load_mmlu_data
from confidentllm.data.multi_arith import load_multi_arith_data
from confidentllm.data.openbook_qa import load_openbook_qa_data
from confidentllm.data.types import DatasetSplit, LoadDatasetFunction

__all__ = ["load_union_data"]

DATASETS: dict[str, LoadDatasetFunction] = {
    "arc": load_arc_data,  # type: ignore [dict-item]
    "commonsense_qa": load_commonsense_qa_data,  # type: ignore [dict-item]
    "cot_preference": load_cot_preference_data,  # type: ignore [dict-item]
    "gsm8k": load_gsm8k_data,  # type: ignore [dict-item]
    "multiarith": load_multi_arith_data,  # type: ignore [dict-item]
    "openbook_qa": load_openbook_qa_data,  # type: ignore [dict-item]
    "mmlu": load_mmlu_data,  # type: ignore [dict-item]
}


def get_dataset_loaders(names: str) -> dict[str, LoadDatasetFunction]:
    """Get the loaders for the datasets.

    Args:
    ----
        names: The names of the datasets split by "+".

    Returns:
    -------
        The loaders for the datasets.

    """
    dataset_names: list[str] = names.split("+")
    dataset_loaders: dict[str, LoadDatasetFunction] = {}
    for dataset_name in dataset_names:
        found: bool = False
        for name, loader in DATASETS.items():
            if name in dataset_name:
                dataset_loaders[name] = loader
                found = True
                break
        if not found:
            msg: str = f"Unknown dataset: {dataset_name}"
            raise ValueError(msg)

    return dataset_loaders


def load_union_data(  # noqa: PLR0913
    split: DatasetSplit = DatasetSplit.TEST,
    transformation_batch_size: int = 512,
    name: str = "union",
    *,
    use_cache: bool = False,
    run_path: str | None = None,
    run_name: str | None = None,
    table_name: str | None = None,
    ranking_threshold: float | None = None,
) -> Dataset:
    """Load the union of multiple datasets.

    Args:
    ----
        split: The split of the datasets to load.
        transformation_batch_size: The batch size to use for the transformation.
        name: The names of the datasets split by "+".
        use_cache: Whether to use the cache.
        run_path: The path to the wandb run.
        run_name: The names of the wand runs split by "+".
        table_name: The name of the table.
        ranking_threshold: The ranking threshold for creating preference data.

    Returns:
    -------
        The merged dataset.

    """
    dataset_loaders: dict[str, LoadDatasetFunction] = get_dataset_loaders(name)
    run_names: list[str | None] = (
        run_name.split("+") if run_name else [None] * len(dataset_loaders)  # type: ignore [assignment, list-item]
    )

    datasets: list[Dataset] = []
    for dataset_loader, dataset_run_name in zip(
        dataset_loaders.items(),
        run_names,
        strict=True,
    ):
        dataset_name, loader = dataset_loader
        dataset: Dataset = loader(
            split=split,
            transformation_batch_size=transformation_batch_size,
            name=dataset_name,
            use_cache=use_cache,
            run_path=run_path,
            run_name=dataset_run_name,
            table_name=table_name,
            ranking_threshold=ranking_threshold,
        )

        if datasets and (dataset.column_names != datasets[0].column_names):
            msg = (
                f"Columns of dataset {dataset_name}: {dataset.column_names} do not "
                f"match columns of first dataset: {datasets[0].column_names}."
            )
            raise KeyError(msg)

        datasets.append(dataset)

    union_dataset = concatenate_datasets(datasets)
    union_dataset.cached_version = datasets[0].cached_version  # type: ignore[attr-defined]

    return union_dataset
