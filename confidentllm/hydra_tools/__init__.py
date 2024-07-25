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
"""Set up logging configuration for project."""

from typing import Any

from hydra_zen import ZenStore, make_custom_builds_fn, store

__all__ = ["builds", "MultiGroupZenStore"]

builds = make_custom_builds_fn(populate_full_signature=True)


class MultiGroupZenStore:
    """A ZenStore that applies the same function to multiple groups."""

    def __init__(self, groups: list[str]) -> None:
        """Initialize a MultiGroupZenStore."""
        self.stores: list[ZenStore] = [store(group=group) for group in groups]

    def __call__(self, target: Any, name: str) -> None:  # noqa: ANN401 - Any accepted by the zen store
        """Store the target in all stores."""
        for store_fn in self.stores:
            store_fn(target, name=name)  # type: ignore - Name is a string
