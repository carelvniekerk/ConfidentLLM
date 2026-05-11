# coding=utf-8
# --------------------------------------------------------------------------------
# Project: ConfidentLLM
# Author: Carel van Niekerk, Renato Vukovic
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
"""Database session management for ConfidentLLM."""

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

__all__ = ["get_session"]


@contextmanager
def get_session(db_path: Path) -> Generator[Session]:
    """Create a new SQLAlchemy session for the database at the specified path."""
    engine: Engine = create_engine(f"sqlite:///{db_path}")
    session_maker: sessionmaker = sessionmaker(bind=engine)
    session: Session = session_maker()

    try:
        yield session
    finally:
        session.close()
        engine.dispose()
