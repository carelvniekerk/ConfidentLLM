# coding=utf-8
# --------------------------------------------------------------------------------
# Project: ConfidentLLM
# Author: Carel van Niekerk, Benjamin Ruppik
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
"""Runner for question answering using the ConfidentLLM package."""

import logging
import random
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from git import Repo
from hydra.conf import HydraConf, JobConf, RunDir, SweepDir
from hydra_zen import store
from omegaconf import DictConfig, OmegaConf

import wandb
from confidentllm.hydra_tools import resolve_generation_method, resolve_output_processor
from confidentllm.logging import (
    create_logging_config,
    initialize_wandb,
    setup_exception_logging,
)
from hydra_plugins.hpc_submission_launcher import (
    register_plugin as register_hpc_submission_launcher_plugin,
)

__all__ = [
    "setup_hydra_config_and_logging",
    "init_wandb",
    "set_seed",
    "get_logger",
    "log_system_info",
]
logger = logging.getLogger("__main__")

register_hpc_submission_launcher_plugin()

OmegaConf.register_new_resolver("resolve_generation_method", resolve_generation_method)
OmegaConf.register_new_resolver("resolve_output_processor", resolve_output_processor)


def create_run_dir(
    root_dir: Path,
    config_keys: list[str],
    *,
    is_sweep: bool = False,
) -> RunDir | SweepDir:
    """Create a run directory."""
    run_dir: Path = root_dir / "${hydra.job.name}"

    if is_sweep:
        sub_dir: Path | None = (
            Path("${" + config_keys[0] + "}") if config_keys else None
        )
        for key in config_keys[1:]:
            _key = "${" + key + "}"
            sub_dir = sub_dir / _key  # type: ignore[operator]

        sub_dir = (
            sub_dir / "${now:%Y-%m-%d_%H-%M-%S}"
            if sub_dir
            else Path("${now:%Y-%m-%d_%H-%M-%S}")
        )

        return SweepDir(
            dir=str(run_dir),
            subdir=str(sub_dir),
        )

    for key in config_keys:
        _key = "${" + key + "}"
        run_dir = run_dir / _key

    run_dir = run_dir / "${now:%Y-%m-%d_%H-%M-%S}"

    return RunDir(str(run_dir))


def setup_hydra_config_and_logging(
    job_name: str,
    *,
    config_keys: list[str] | None = None,
    change_to_output_dir: bool = True,
    add_hpc_launcher: bool = False,
) -> None:
    """Set up Hydra configuration and logging."""
    setup_exception_logging(logger)

    job_config: JobConf = JobConf(name=job_name, chdir=change_to_output_dir)
    logging_config: dict = create_logging_config()

    if config_keys is None:
        config_keys = [
            "data.name",
            "data.split",
            "model.pretrained_model_name_or_path",
            "resolve_generation_method:${generation_method}",
            "resolve_output_processor:${output_processor}",
            "run_config.seed",
        ]

    run_dir: RunDir = create_run_dir(
        root_dir=Path("outputs"),
        config_keys=config_keys,
    )  # type: ignore  # noqa: PGH003
    sweep_dir: SweepDir = create_run_dir(
        root_dir=Path("multirun") if not add_hpc_launcher else Path("hpc_jobs"),
        config_keys=config_keys,
        is_sweep=True,
    )  # type: ignore  # noqa: PGH003

    if add_hpc_launcher:
        hydra_defaults: list[str | dict[str, str | None]] = [
            # Standard defaults
            "_self_",
            {"sweeper": "basic"},
            {"help": "default"},
            {"hydra_help": "default"},
            {"hydra_logging": "default"},
            {"callbacks": None},
            # Set launcher
            {"launcher": "hpc_submission"},
        ]

        hydra_config: HydraConf = HydraConf(
            defaults=hydra_defaults,
            job=job_config,
            job_logging=logging_config,
            run=run_dir,
            sweep=sweep_dir,
        )
    else:
        hydra_config = HydraConf(
            job=job_config,
            job_logging=logging_config,
            run=run_dir,
            sweep=sweep_dir,
        )

    store(
        hydra_config,
        name="config",
        group="hydra",
    )
    store.add_to_hydra_store()


def init_wandb() -> None:
    """Initialize Weights and Biases."""
    config_path: Path = Path(".hydra") / "config.yaml"
    config: DictConfig = OmegaConf.load(config_path)  # type: ignore  # noqa: PGH003
    initialize_wandb(config=config)


def set_seed(seed: int) -> None:
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.default_rng(seed)


def get_logger() -> logging.Logger:
    """Get the logger."""
    logger: logging.Logger = logging.getLogger("__main__")

    # Get the transformer logger and propagate its logs to the Hydra root.
    transformers_logger: logging.Logger = transformers.logging.get_logger()
    transformers_logger.handlers = []
    transformers_logger.propagate = True

    wandb.init()

    return logger


def log_system_info() -> None:
    """Log system hostname and environment information."""
    try:
        hostname = socket.gethostname()
    except Exception:  # noqa: BLE001 - We want to proceed no matter what the error is
        hostname = "unknown"

    logging.info(
        msg=f"Running on {hostname = }",  # noqa: G004 - low overhead
    )

    _log_python_env_info()
    _log_git_info()


def _log_git_info() -> None:
    """Get the git info of the current branch and commit hash."""
    try:
        repo = Repo(
            path=Path(__file__).resolve().parent,
            search_parent_directories=True,
        )
        branch_name: str = repo.active_branch.name
        commit_hex: str = repo.head.object.hexsha
        logging.info(
            msg=f"Git {branch_name = }",  # noqa: G004 - low overhead
        )
        logging.info(
            msg=f"Git {commit_hex = }",  # noqa: G004 - low overhead
        )
    except Exception:  # noqa: BLE001 - We want to proceed no matter what the error is
        logging.info(
            msg="Unable to determine git branch/commit",
        )


def _log_python_env_info() -> None:
    """Log Python environment and Poetry information."""
    # Log Python version and executable
    try:
        python_version: str = sys.version.split()[0]
        python_path: str = sys.executable
        logging.info(
            msg=f"Python version: {python_version = }",  # noqa: G004 - low overhead
        )
        logging.info(
            msg=f"Python executable: {python_path = }",  # noqa: G004 - low overhead
        )
    except Exception:  # noqa: BLE001 - We want to proceed no matter what the error is
        logger.info(msg="Unable to determine Python version/path")

    # Check Poetry environment
    try:
        result = subprocess.run(
            args=[
                "poetry",
                "env",
                "info",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            env_info: str = result.stdout.strip()
            logging.info(
                msg=f"Poetry environment:\n{env_info}",  # noqa: G004 - low overhead
            )
        else:
            logging.info(
                msg="Not running in a Poetry environment",
            )
    except Exception:  # noqa: BLE001 - We want to proceed no matter what the error is
        logging.info(
            msg="Unable to determine Poetry environment",
        )
