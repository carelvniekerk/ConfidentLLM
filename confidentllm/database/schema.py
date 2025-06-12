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

from sqlalchemy import JSON, ForeignKey
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    MappedAsDataclass,
    mapped_column,
    relationship,
)
from typing_extensions import Annotated

__all__ = [
    "Base",
    "Dataset",
    "Label",
    "Observation",
    "Prediction",
    "Tokenization",
]


class Base(MappedAsDataclass, DeclarativeBase):
    """Base class for all database models."""


int_primary_key = Annotated[int, mapped_column(primary_key=True)]
now_datetime = Annotated[datetime, mapped_column(default=datetime.now(UTC))]
dataset_foreign_key = Annotated[int, mapped_column(ForeignKey("datasets.id"))]
observation_foreign_key = Annotated[int, mapped_column(ForeignKey("observations.id"))]
str_list = Annotated[list[str], mapped_column(JSON)]
int_list = Annotated[list[int], mapped_column(JSON)]
float_list = Annotated[list[float], mapped_column(JSON)]


class Dataset(Base):
    """Datasets Table to store metadata about datasets."""

    __tablename__: str = "datasets"

    id: Mapped[int_primary_key] = mapped_column(init=False)
    name: Mapped[str]  # type: ignore[misc] # The value is inferred from the type annotation
    description: Mapped[str | None] = mapped_column(default=None)
    citation: Mapped[str | None] = mapped_column(default=None)
    homepage: Mapped[str | None] = mapped_column(default=None)
    license: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[now_datetime] = mapped_column(init=False)

    observations: Mapped[list["Observation"]] = relationship(
        back_populates="dataset",
        default_factory=list,
    )


class Observation(Base):
    """Observations Table to store individual sentences and their metadata."""

    __tablename__: str = "observations"

    id: Mapped[int_primary_key] = mapped_column(init=False)
    dataset_id: Mapped[dataset_foreign_key] = mapped_column(init=False)
    sentence: Mapped[str]  # type: ignore[misc] # The value is inferred from the type annotation

    dataset: Mapped["Dataset"] = relationship(
        back_populates="observations",
        default=None,
    )
    tokenizations: Mapped[list["Tokenization"]] = relationship(
        back_populates="observation",
        default_factory=list,
    )
    labels: Mapped[list["Label"]] = relationship(
        back_populates="observation",
        default_factory=list,
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="observation",
        default_factory=list,
    )


class Tokenization(Base):
    """Tokenizations Table to store tokenized representations of sentences."""

    __tablename__: str = "tokenizations"

    id: Mapped[int_primary_key] = mapped_column(init=False)
    observation_id: Mapped[observation_foreign_key] = mapped_column(init=False)
    tokenizer_name: Mapped[str]  # type: ignore[misc] # The value is inferred from the type annotation
    tokens: Mapped[str_list]  # type: ignore[misc] # The value is inferred from the type annotation
    token_ids: Mapped[int_list]  # type: ignore[misc] # The value is inferred from the type annotation

    observation: Mapped["Observation"] = relationship(
        back_populates="tokenizations",
        default=None,
    )


class Label(Base):
    """Labels Table to store labels associated with observations."""

    __tablename__: str = "labels"

    id: Mapped[int_primary_key] = mapped_column(init=False)
    observation_id: Mapped[observation_foreign_key] = mapped_column(init=False)
    label_type: Mapped[str]  # type: ignore[misc] # The value is inferred from the type annotation
    label_value: Mapped[str]  # type: ignore[misc] # The value is inferred from the type annotation

    observation: Mapped["Observation"] = relationship(
        back_populates="labels",
        default=None,
    )


class Prediction(Base):
    """Predictions Table to store model predictions for observations."""

    __tablename__: str = "predictions"

    id: Mapped[int_primary_key] = mapped_column(init=False)
    observation_id: Mapped[observation_foreign_key] = mapped_column(init=False)
    model_name: Mapped[str]  # type: ignore[misc] # The value is inferred from the type annotation
    prediction_type: Mapped[str]  # type: ignore[misc] # The value is inferred from the type annotation
    prediction_value: Mapped[str | None] = mapped_column(default=None)
    prediction_values: Mapped[float_list | None] = mapped_column(default=None)
    confidence_score: Mapped[float | None] = mapped_column(default=None)

    observation: Mapped["Observation"] = relationship(
        back_populates="predictions",
        default=None,
    )
