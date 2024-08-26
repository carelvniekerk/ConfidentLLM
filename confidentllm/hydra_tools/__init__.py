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
"""Set up logging configuration for project."""

from pathlib import Path
from typing import Any

from hydra_zen import ZenStore, make_custom_builds_fn, store

__all__ = [
    "builds",
    "MultiGroupZenStore",
    "resolve_decoding_strategy",
    "resolve_output_processor",
]

builds = make_custom_builds_fn(populate_full_signature=True)


class MultiGroupZenStore:
    """A ZenStore that applies the same function to multiple groups."""

    def __init__(self, groups: list[str]) -> None:
        """Initialize a MultiGroupZenStore."""
        self.stores: list[ZenStore] = [store(group=group) for group in groups]

    def __call__(self, target: Any, name: str) -> None:  # noqa: ANN401 - Any accepted by the zen store
        """Store the target in all stores."""
        for store_fn in self.stores:
            store_fn(target, name=name)


def function_path_to_name(function_path: str) -> str:
    """Convert a function path to a name."""
    return function_path.split(".")[-1]


def resolve_decoding_strategy(decoding_strategy: dict[str, str | int | float]) -> str:
    """Resolve the decoding strategy."""
    name: str = decoding_strategy.get("_target_", "")  # type:ignore[reportAssignmentType]
    if "path" in decoding_strategy:
        name = decoding_strategy.get("path")  # type:ignore[reportAssignmentType]

    other_params: list[str] = [
        key for key in decoding_strategy if key not in ["_target_", "path"]
    ]

    dec_str: Path = Path(function_path_to_name(name))
    for param in other_params:
        dec_str = dec_str / f"{param}_{decoding_strategy[param]}"

    return str(dec_str)


def resolve_output_processor(output_processor: dict[str, str | int | float]) -> str:
    """Resolve the output processor."""
    name: str = output_processor.get("_target_", "")  # type:ignore[reportAssignmentType]

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
