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
"""Runner for question answering using the ConfidentLLM package."""

import logging

from hydra.conf import HydraConf, JobConf
from hydra.core.hydra_config import HydraConfig
from hydra_zen import store

from confidentllm.logging import (
    create_logging_config,
    initialize_wandb,
    setup_exception_logging,
)
from hydra_plugins.hpc_submission_launcher.launcher import (
    HPCSubmissionLauncher,  # noqa: F401 - Import Launcher to make it available in the Zen Store
)

__all__ = ["setup_hydra_config_and_logging", "init_wandb"]
logger = logging.getLogger("__main__")


def setup_hydra_config_and_logging(
    job_name: str,
    *,
    change_to_output_dir: bool = True,
) -> None:
    """Set up Hydra configuration and logging."""
    setup_exception_logging(logger)

    store(
        HydraConf(
            job=JobConf(name=job_name, chdir=change_to_output_dir),
            job_logging=create_logging_config(),
        ),
        name="config",
        group="hydra",
    )
    store.add_to_hydra_store()


def init_wandb() -> None:
    """Initialize Weights and Biases."""
    initialize_wandb(config=HydraConfig.get())  # type: ignore - HydraConfig.get() is not in the HydraConf schema
