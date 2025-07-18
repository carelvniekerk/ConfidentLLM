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
"""Runner for question answering using the ConfidentLLM package."""

import logging
import os
import platform
import random
import socket
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from git import Repo
from hydra.conf import HydraConf, JobConf, RunDir, SweepDir
from hydra_zen import store
from omegaconf import DictConfig, OmegaConf

from confidentllm.hydra_tools import (
    resolve_generation_method,
    resolve_output_processor,
    resolve_target_name,
)
from confidentllm.logging import (
    create_logging_config,
    initialize_wandb,
    setup_exception_logging,
)
from hydra_plugins.hpc_submission_launcher import (
    register_plugin as register_hpc_submission_launcher_plugin,
)

__all__ = [
    "get_logger",
    "init_wandb",
    "log_system_info",
    "set_seed",
    "setup_hydra_config_and_logging",
]
logger = logging.getLogger("__main__")

register_hpc_submission_launcher_plugin()

OmegaConf.register_new_resolver("resolve_generation_method", resolve_generation_method)
OmegaConf.register_new_resolver("resolve_output_processor", resolve_output_processor)
OmegaConf.register_new_resolver("resolve_target_name", resolve_target_name)


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
    config_keys: list[str],
    change_to_output_dir: bool = True,
    add_hpc_launcher: bool = False,
) -> None:
    """Set up Hydra configuration and logging."""
    setup_exception_logging(logger)

    job_config: JobConf = JobConf(name=job_name, chdir=change_to_output_dir)
    logging_config: dict = create_logging_config()

    run_dir: RunDir = create_run_dir(  # type: ignore[assignment]
        root_dir=Path("outputs"),
        config_keys=config_keys,
    )
    sweep_dir: SweepDir = create_run_dir(  # type: ignore[assignment]
        root_dir=Path("multirun") if not add_hpc_launcher else Path("hpc_jobs"),
        config_keys=config_keys,
        is_sweep=True,
    )

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


def init_wandb(task_name: str) -> None:
    """Initialize Weights and Biases.

    Args:
    ----
        task_name(str): The name of the task.

    """
    hydra_config_path: Path = Path(".hydra") / "hydra.yaml"
    hydra_config: DictConfig = OmegaConf.load(hydra_config_path)  # type: ignore  # noqa: PGH003

    # When submitting HPC jobs, we don't want to initialize wandb
    if (
        hydra_config.hydra.mode.lower() == "multirun"
        and "submission" in hydra_config.hydra.launcher.__target__
    ):
        return

    config_path: Path = Path(".hydra") / "config.yaml"
    config: DictConfig = OmegaConf.load(config_path)  # type: ignore  # noqa: PGH003
    project_name: str = f"ConfidentLLM-{task_name}"
    initialize_wandb(config=config, project_name=project_name)


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
    transformers_logger: logging.Logger = transformers.utils.logging.get_logger()  # type: ignore[unresolved-attribute] # TODO: Remove when ty bug is fixed
    transformers_logger.handlers = []
    transformers_logger.propagate = True

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

    try:
        # Virtualenv Section
        python_version = sys.version.split()[0]
        implementation = platform.python_implementation()
        executable = Path(sys.executable)
        venv_path = executable.parent.parent if "VIRTUAL_ENV" in os.environ else None
        valid_venv = bool(venv_path and (venv_path / "bin" / "python").exists())

        logging.info("Virtualenv")
        logging.info(f"Python:         {python_version}")  # noqa: G004
        logging.info(f"Implementation: {implementation}")  # noqa: G004
        logging.info(
            f"Path:           {venv_path if venv_path else 'Not in a virtual environment'}",  # noqa: E501, G004
        )
        logging.info(f"Executable:     {executable}")  # noqa: G004
        logging.info(f"Valid:          {valid_venv}")  # noqa: G004

        # Base Section
        base_path = Path(sys.base_prefix)
        platform_name = platform.system().lower()
        os_type = os.name
        base_executable = base_path / "bin" / f"python{python_version[:3]}"
        if not base_executable.exists():
            base_executable = Path(sys.base_exec_prefix)

        logging.info("\nBase")
        logging.info(f"Platform:   {platform_name}")  # noqa: G004
        logging.info(f"OS:         {os_type}")  # noqa: G004
        logging.info(f"Python:     {python_version}")  # noqa: G004
        logging.info(f"Path:       {base_path}")  # noqa: G004
        logging.info(f"Executable: {base_executable}")  # noqa: G004

    except Exception:
        logging.exception("Error logging environment info.")
