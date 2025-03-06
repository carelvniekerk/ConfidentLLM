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
"""Reporting script for ConfidentLLM."""

from pathlib import Path

import numpy as np
import pandas as pd
from hydra_zen import store, zen
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

import wandb
from confidentllm.logging.init_wandb import initialize_wandb
from confidentllm.reporting import load_wandb_data
from confidentllm.scripts.setup_tools import setup_hydra_config_and_logging

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
    pivot_table.columns = pivot_table.columns.reorder_levels(  # type: ignore[attr-defined]
        ["Dataset", "Split", None],
    )

    # Sort the columns again to ensure proper order
    pivot_table = pivot_table.sort_index(axis=1)

    return pivot_table


def create_calibration_plot(data_df: pd.DataFrame) -> dict[str, Figure]:
    """Create a calibration plot for the experiment results.

    Args:
    ----
        data_df (pd.DataFrame): The data frame containing the experiment results.

    Returns:
    -------
        dict[str, Figure]: A dictionary of figures for each dataset.

    """
    # Extract bin accuracies and confidences
    bins: set[str] = {
        ".".join(_bin.split(".")[:-1])
        for _bin in data_df.columns
        if _bin.startswith("bin_stats.")
    }
    bin_ids: list[int] = sorted([int(bin_key.split("_")[-1]) for bin_key in bins])

    accuracy_columns: list[str] = [
        f"bin_stats.bin_{bin_id}.accuracy" for bin_id in bin_ids
    ]
    confidence_columns: list[str] = [
        f"bin_stats.bin_{bin_id}.avg_confidence" for bin_id in bin_ids
    ]

    # Extract the relevant columns
    model_column = "model.pretrained_model_name_or_path"
    confidence_column = "output_processor.name"
    dataset_column = "data.name"
    split_column = "data.split"
    columns: list[str] = [
        model_column,
        confidence_column,
        dataset_column,
        split_column,
        *accuracy_columns,
        *confidence_columns,
    ]

    # Filter the data frame
    calibration_df: pd.DataFrame = data_df[columns]

    # Set up the plotting environment
    plt.style.use("dracula")
    plt.rcParams["font.family"] = "Inconsolata Nerd Font"

    plot_colors: list[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_line_styles: list[str] = ["-", "--", "-.", ":"]

    figs: dict[str, Figure] = {}
    dataset_names: np.ndarray = calibration_df[dataset_column].unique()
    for dataset_name in dataset_names:
        # Create a new figure for each dataset
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.grid(linestyle="solid", color="gray", alpha=0.2)

        dataset_df: pd.DataFrame = calibration_df[
            calibration_df[dataset_column] == dataset_name
        ]
        model_names: np.ndarray = dataset_df[model_column].unique()
        for model_name, model_color in zip(model_names, plot_colors, strict=False):
            model_df: pd.DataFrame = dataset_df[dataset_df[model_column] == model_name]
            confidence_methods: np.ndarray = model_df[confidence_column].unique()
            for confidence_method, confidence_line_style in zip(
                confidence_methods,
                plot_line_styles,
                strict=False,  # Use strict=False to allow for different lengths
            ):
                confidence_df: pd.DataFrame = model_df[
                    model_df[confidence_column] == confidence_method
                ]
                accuracy: np.ndarray = (
                    confidence_df[accuracy_columns].to_numpy().flatten()
                )
                confidence: np.ndarray = (
                    confidence_df[confidence_columns].to_numpy().flatten()
                )

                # Remove empty bins
                mask: np.ndarray = accuracy != -1
                accuracy = accuracy[mask]
                confidence = confidence[mask]

                ax.plot(
                    confidence,
                    accuracy,
                    label=f"{model_name}_{confidence_method}",
                    color=model_color,
                    linestyle=confidence_line_style,
                )

        # Add the perfect calibration line
        ax.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            color="white",
            linestyle="-",
            label="Perfect Calibration",
        )

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Calibration Plot: {dataset_name}")
        ax.legend()
        fig.tight_layout()

        figs[dataset_name] = fig

    return figs


@store(
    name="reporting",
)
def run_reporting() -> None:
    """Run the reporting process."""
    data_df: pd.DataFrame = load_wandb_data("dialgroup-hhu/ConfidentLLM")

    initialize_wandb(project_name="ConfidentLLM_Reporting")
    wandb_log: dict = {}

    # Tabulate the results
    results_df: pd.DataFrame = tabulate_results(data_df)
    table_dir = Path("tables")
    table_dir.mkdir(exist_ok=True)
    results_df.to_csv(table_dir / "results.csv")

    wandb_results_df: pd.DataFrame = pd.read_csv(table_dir / "results.csv")
    wandb_log["results"] = wandb.Table(dataframe=wandb_results_df)

    # Create the calibration plots
    figures = create_calibration_plot(data_df)

    figure_dir = Path("figures")
    figure_dir.mkdir(exist_ok=True)
    for dataset_name, fig in figures.items():
        _path = figure_dir / f"{dataset_name}.pdf"
        fig.savefig(
            _path,
            dpi=400,
            bbox_inches="tight",
            pad_inches=0,
        )
        wandb_log[f"calibration_curve_{dataset_name}"] = wandb.Image(fig)

    wandb.log(wandb_log)
    wandb.finish()


def main() -> None:
    """Run the reporting script."""
    setup_hydra_config_and_logging(
        job_name="reporting",
        config_keys=[],
    )

    # Generate the CLI for run_extraction
    zen(
        run_reporting,
    ).hydra_main(
        config_name="reporting",
        version_base="1.3",
    )


if __name__ == "__main__":
    main()
