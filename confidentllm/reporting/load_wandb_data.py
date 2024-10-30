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
"""Modules for accessing the weights and biases API and loading run data."""

import pandas as pd
import wandb
from wandb.apis.public.runs import Run, Runs
from wandb.old.summary import HTTPSummary, SummarySubDict

__all__ = ["load_wandb_data"]


def _flatten_config(
    config: dict[str, str | int | dict | list[dict]],
) -> dict[str, str | int]:
    """Flatten the configuration dictionary."""
    flat_config: dict[str, str | int] = {}

    for key, value in config.items():
        if "hydra" in key or "zen" in key or key.startswith("_") or key == "path":
            continue
        if isinstance(value, dict):
            for sub_key, sub_value in _flatten_config(value).items():
                flat_config[f"{key}.{sub_key}"] = sub_value
            continue
        if isinstance(value, list):
            for idx, item in enumerate(value):
                for sub_key, sub_value in _flatten_config({str(idx): item}).items():
                    flat_config[f"{key}.{sub_key}"] = sub_value
            continue
        flat_config[key] = value

    if not [key for key in config if "name" in key] and "path" in config:
        if not isinstance(config["path"], str):
            msg = f"Expected path to be a string, got {type(config['path'])}"
            raise TypeError(msg)
        flat_config["name"] = config["path"].split(".")[-1]
        return flat_config

    if not [key for key in config if "name" in key] and "_target_" in config:
        if not isinstance(config["_target_"], str):
            msg = f"Expected path to be a string, got {type(config['_target_'])}"
            raise TypeError(msg)
        flat_config["name"] = config["_target_"].split(".")[-1]
    return flat_config


def _flatten_summary(
    config: dict[str, str | float | SummarySubDict] | SummarySubDict | HTTPSummary,
) -> dict[str, str | float]:
    """Flatten the summary dictionary."""
    flat_summary: dict = {}

    for key, value in config.items():
        if "wandb" in key or key.startswith("_"):
            continue
        if isinstance(value, SummarySubDict | dict):
            for sub_key, sub_value in _flatten_summary(value).items():
                flat_summary[f"{key}.{sub_key}"] = sub_value
            continue
        flat_summary[key] = value

    return flat_summary


def _flatten_run(run: Run) -> dict[str, str | float]:
    """Flatten the run."""
    return {
        **_flatten_config(run.config["run"]),
        **_flatten_summary(run.summary),
    }


def load_wandb_data(path: str) -> pd.DataFrame:
    """Load the wandb data."""
    wandb_api: wandb.Api = wandb.Api()
    runs: Runs = wandb_api.runs(path)

    completed_runs: list[Run] = [
        run for run in runs if run.state == "finished" and run.config.get("run")
    ]

    data: dict[str, list[str | float]] = {}
    for run in completed_runs:
        flat_run: dict[str, str | float] = _flatten_run(run)
        for key, value in flat_run.items():
            if key not in data:
                data[key] = []
            data[key].append(value)

    return pd.DataFrame(data)
