# coding=utf-8
# --------------------------------------------------------------------------------
# Project: ConfidentLLM
# Author: Carel van Niekerk, Renato Vukovic
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
"""Set up logging configuration for project."""

from pathlib import Path

from hydra_zen import make_custom_builds_fn
from hydra_zen.typing._builds_overloads import FullBuilds  # type: ignore[access]
from omegaconf.dictconfig import DictConfig

__all__ = [
    "builds",
    "resolve_generation_method",
    "resolve_output_processor",
    "resolve_target_name",
]

builds: FullBuilds = make_custom_builds_fn(populate_full_signature=True)


def function_path_to_name(function_path: str) -> str:
    """Convert a function path to a name."""
    return function_path.split(".")[-1]


def resolve_output_processor(output_processor: dict[str, str | int | float]) -> str:
    """Resolve the output processor."""
    name: str = output_processor.get("_target_", "")  # type: ignore[assignment]

    other_params: list[str] = [
        key for key in output_processor if key not in ["_target_", "generator"]
    ]

    op_str: Path = Path(function_path_to_name(name))
    for param in other_params:
        val: str | int | float = output_processor[param]
        if isinstance(val, str):
            val = val.replace(" ", "_")
        op_str = op_str / f"{param}_{val}"

    return str(op_str)


def resolve_generation_method(generation_method: dict[str, str | int | float]) -> str:
    """Resolve the generation method."""
    name: str = generation_method.get("_target_", "")  # type: ignore[assignment]

    other_params: list[str] = [
        key for key in generation_method if key not in ["_target_", "generator"]
    ]

    gm_str: Path = Path(function_path_to_name(name))
    for param in other_params:
        val: str | float | DictConfig | None = generation_method[param]
        if not val:
            continue
        if isinstance(val, DictConfig):
            val = function_path_to_name(val.get("_target_", ""))
        if isinstance(val, str):
            val = val.replace(" ", "_")
        gm_str = gm_str / f"{param}_{val}"

    return str(gm_str)


def resolve_target_name(method: dict[str, str | int | float]) -> str:
    """Resolve the target name."""
    return function_path_to_name(method.get("_target_", ""))  # type: ignore[arg-type]
