# coding=utf-8
# --------------------------------------------------------------------------------
# Project: ConfidentLLM
# Author: Carel van Niekerk, Benjamin Ruppik
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
"""Initialise the weights and biases logging."""

from pathlib import Path

import wandb
from omegaconf import DictConfig, OmegaConf

__all__ = ["initialize_wandb"]


def find_project_root(current_path: Path, marker: str = "pyproject.toml") -> Path:
    """Traverse up the directory structure to find the root directory.

    Args:
    ----
        current_path (Path): The current file path.
        marker (str): A marker file or directory indicating the root.

    Returns:
    -------
        Path: The root directory path.

    """
    # Remove the filename if it ends with .py
    current_path = current_path.resolve()
    if current_path.suffix == ".py":
        current_path = current_path.parent

    if (current_path / marker).exists():
        return current_path

    while current_path != current_path.parent:
        # Check if any marker exists in the current directory
        if (current_path / marker).exists():
            return current_path

        # Move one directory up
        current_path = current_path.parent

    # If we reach the filesystem root and haven't found the marker, raise an error
    msg = f"{marker} not found in any parent directories."
    raise FileNotFoundError(msg)


def initialize_wandb(
    config: DictConfig | None = None,
    project_name: str = "ConfidentLLM",
) -> None:
    """Initialize wandb."""
    wandb_path = find_project_root(Path(__file__))

    wandb.init(
        project=project_name,
        dir=str(wandb_path),
        settings=wandb.Settings(
            start_method="thread",  # Note: https://docs.wandb.ai/guides/integrations/hydra#troubleshooting-multiprocessing
            x_service_wait=300,
            init_timeout=300,
        ),
    )

    if config is None:
        return
    cfg_dict = OmegaConf.to_container(
        config,
        resolve=True,
    )

    # Add the run config to the wandb config
    # (so that they are tracked in the wandb run)
    wandb.config.run = cfg_dict  # type: ignore[attr-defined]
