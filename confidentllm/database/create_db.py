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
"""Create a SQLite database for the ConfidentLLM project."""

from pathlib import Path

from sqlalchemy import create_engine

from confidentllm.database.schema import Base

__all__ = ["create_database"]


def create_database(db_path: Path) -> None:
    """Create a SQLite database at the specified path."""
    # Ensure the directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create the SQLite engine
    engine = create_engine(f"sqlite:///{db_path}")

    # Create all tables in the database
    Base.metadata.create_all(engine)

    print(f"Database created at {db_path}")
