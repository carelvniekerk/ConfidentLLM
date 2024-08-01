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
# limitations under the License."
"""Reporting script for ConfidentLLM."""

import pandas as pd

from confidentllm.reporting import load_wandb_data

__all__ = ["main"]


def tabulate_results(data_df: pd.DataFrame) -> pd.DataFrame:
    """Plot the results of the experiment.

    Args:
    ----
        data_df (pd.DataFrame): The data frame containing the experiment results.

    """
    column_name_mapping: dict[str, str] = {
        "model.pretrained_model_name_or_path": "Model",
        "output_processor.name": "Confidence Estimation",
        "data.name": "Dataset",
        "data.split": "Split",
        "accuracy": "Accuracy",
        "ece": "ECE",
    }

    # Create the pivot table
    pivot_table: pd.DataFrame = data_df.rename(columns=column_name_mapping).pivot_table(
        index=["Model", "Confidence Estimation"],
        columns=["Dataset", "Split"],
        values=["Accuracy", "ECE"],
    )

    # Swap the levels of the columns for better readability
    pivot_table = pivot_table.swaplevel(axis=1)

    # Sort the columns for better organization
    pivot_table = pivot_table.sort_index(axis=1, level=[0, 1])

    # Adjust the column levels to achieve the desired hierarchy
    pivot_table.columns = pivot_table.columns.reorder_levels(  # type: ignore  # noqa: PGH003
        ["Dataset", "Split", None],
    )

    # Sort the columns again to ensure proper order
    pivot_table = pivot_table.sort_index(axis=1)

    return pivot_table


def main() -> None:
    """Run the reporting process."""
    data_df: pd.DataFrame = load_wandb_data("dialgroup-hhu/ConfidentLLM")

    results_df: pd.DataFrame = tabulate_results(data_df)
    print(results_df)


if __name__ == "__main__":
    main()
