# coding=utf-8
# --------------------------------------------------------------------------------
# Project: ConfidentLLM
# Author: Carel van Niekerk
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
#     http: //www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Database module for ConfidentLLM."""

from pathlib import Path

from confidentllm.database.create_db import create_database
from confidentllm.database.schema import (
    Dataset,
    Label,
    Observation,
    Prediction,
    Tokenization,
)
from confidentllm.database.session import get_session

DEFAULT_DATABASE_PATH: Path = (
    Path(__file__).parent.parent.parent / "database" / "confidentllm.db"
)

__all__ = [
    "DEFAULT_DATABASE_PATH",
    "Dataset",
    "Label",
    "Observation",
    "Prediction",
    "Tokenization",
    "create_database",
    "get_session",
]
