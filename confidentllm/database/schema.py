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
"""Database schema for ConfidentLLM."""

from datetime import UTC, datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base
from sqlalchemy.orm import Mapped, mapped_column, relationship

__all__ = ["Base"]

Base: DeclarativeMeta = declarative_base()


class Dataset(Base):
    """Datasets Table to store metadata about datasets."""

    __tablename__: str = "datasets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now(UTC))

    observations: Mapped[list["Observation"]] = relationship(
        argument="Observation",
        back_populates="dataset",
        uselist=True,
    )


class Observation(Base):
    """Observations Table to store individual sentences and their metadata."""

    __tablename__: str = "observations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("datasets.id"),
        nullable=False,
    )
    sentence: Mapped[str] = mapped_column(Text, nullable=False)

    dataset: Mapped["Dataset"] = relationship(
        "Dataset",
        back_populates="observations",
    )
    tokenizations: Mapped[list["Tokenization"]] = relationship(
        "Tokenization",
        back_populates="observation",
    )
    labels: Mapped[list["Label"]] = relationship(
        "Label",
        back_populates="observation",
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction",
        back_populates="observation",
    )


class Tokenization(Base):
    """Tokenizations Table to store tokenized representations of sentences."""

    __tablename__: str = "tokenizations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    observation_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("observations.id"),
        nullable=False,
    )
    tokenizer_name: Mapped[str] = mapped_column(String, nullable=False)
    tokens: Mapped[str] = mapped_column(Text, nullable=False)

    observation: Mapped["Observation"] = relationship(
        "Observation",
        back_populates="tokenizations",
    )


class Label(Base):
    """Labels Table to store labels associated with observations."""

    __tablename__: str = "labels"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    observation_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("observations.id"),
        nullable=False,
    )
    label_type: Mapped[str] = mapped_column(String, nullable=False)
    label_value: Mapped[str] = mapped_column(String, nullable=False)

    observation: Mapped["Observation"] = relationship(
        "Observation",
        back_populates="labels",
    )


class Prediction(Base):
    """Predictions Table to store model predictions for observations."""

    __tablename__: str = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    observation_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("observations.id"),
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    prediction_value: Mapped[str] = mapped_column(String, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=True)

    observation: Mapped["Observation"] = relationship(
        "Observation",
        back_populates="predictions",
    )
