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
import random

import numpy as np
import torch
from hydra.conf import HydraConf, JobConf, RunDir
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
    add_hpc_launcher: bool = False,
) -> None:
    """Set up Hydra configuration and logging."""
    setup_exception_logging(logger)

    job_config: JobConf = JobConf(name=job_name, chdir=change_to_output_dir)
    logging_config: dict = create_logging_config()
    run_dir: RunDir = RunDir(
        "outputs/${hydra:job.name}/${hydra:runtime.choices.data}/${model.pretrained_model_name_or_path}/${now:%Y-%m-%d_%H-%M-%S}",
    )

    if add_hpc_launcher:
        hydra_defaults: list[str | dict[str, str | None]] = [
            # Standard defaults
            "_self_",
            {"output": "default"},
            {"sweeper": "basic"},
            {"help": "default"},
            {"hydra_help": "default"},
            {"hydra_logging": "default"},
            {"job_logging": "default"},
            {"callbacks": None},
            # Set launcher
            {"launcher": "hpc_submission"},
        ]

        hydra_config: HydraConf = HydraConf(
            defaults=hydra_defaults,
            job=job_config,
            job_logging=logging_config,
            run=run_dir,
        )
    else:
        hydra_config: HydraConf = HydraConf(
            job=job_config,
            job_logging=logging_config,
            run=run_dir,
        )

    store(
        hydra_config,
        name="config",
        group="hydra",
    )
    store.add_to_hydra_store()


def init_wandb() -> None:
    """Initialize Weights and Biases."""
    initialize_wandb(config=HydraConfig.get())  # type: ignore - HydraConfig.get() is not in the HydraConf schema


def set_seed(seed: int) -> None:
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.default_rng(seed)
