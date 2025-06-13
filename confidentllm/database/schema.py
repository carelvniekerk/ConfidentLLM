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
    "Model",
    "Observation",
    "Prediction",
    "Token",
    "Tokenizer",
]


class Base(MappedAsDataclass, DeclarativeBase):
    """Base class for all database models."""


int_primary_key = Annotated[int, mapped_column(primary_key=True)]
now_datetime = Annotated[datetime, mapped_column(default=datetime.now(UTC))]
dataset_foreign_key = Annotated[int, mapped_column(ForeignKey("datasets.id"))]
observation_foreign_key = Annotated[int, mapped_column(ForeignKey("observations.id"))]
token_foreign_key = Annotated[int, mapped_column(ForeignKey("tokens.id"))]
tokenizer_foreign_key = Annotated[int, mapped_column(ForeignKey("tokenizers.id"))]
model_foreign_key = Annotated[int, mapped_column(ForeignKey("models.id"))]
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
    tokens: Mapped[list["Token"]] = relationship(
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


class Token(Base):
    """Tokens Table to store individual tokens for each observation."""

    __tablename__: str = "tokens"

    id: Mapped[int_primary_key] = mapped_column(init=False)
    observation_id: Mapped[observation_foreign_key] = mapped_column(init=False)
    tokenizer_id: Mapped[tokenizer_foreign_key] = mapped_column(init=False)
    position: Mapped[int]  # type: ignore[misc] # The value is inferred from the type annotation
    text: Mapped[str]  # type: ignore[misc] # The value is inferred from the type annotation
    token_id: Mapped[int]  # type: ignore[misc] # The value is inferred from the type annotation
    is_final: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[now_datetime] = mapped_column(init=False)
    updated_at: Mapped[now_datetime] = mapped_column(init=False)

    observation: Mapped["Observation"] = relationship(
        back_populates="tokens",
        default=None,
    )
    tokenizer: Mapped["Tokenizer"] = relationship(
        back_populates="tokens",
        default=None,
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="token",
        default_factory=list,
    )
    labels: Mapped[list["Label"]] = relationship(
        back_populates="token",
        default_factory=list,
    )


class Tokenizer(Base):
    """Tokenizers Table to store metadata about tokenizers."""

    __tablename__: str = "tokenizers"

    id: Mapped[int_primary_key] = mapped_column(init=False)
    name: Mapped[str]  # type: ignore[misc] # The value is inferred from the type annotation
    created_at: Mapped[now_datetime] = mapped_column(init=False)
    updated_at: Mapped[now_datetime] = mapped_column(init=False)

    models: Mapped[list["Model"]] = relationship(
        back_populates="tokenizer",
        default_factory=list,
    )
    tokens: Mapped[list["Token"]] = relationship(
        back_populates="tokenizer",
        default_factory=list,
    )


class Model(Base):
    """Models Table to store metadata about models."""

    __tablename__: str = "models"

    id: Mapped[int_primary_key] = mapped_column(init=False)
    name: Mapped[str]  # type: ignore[misc] # The value is inferred from the type annotation
    tokenizer_id: Mapped[tokenizer_foreign_key] = mapped_column(init=False)
    training_details: Mapped[str | None] = mapped_column(default=None)
    wandb_run_url: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[now_datetime] = mapped_column(init=False)
    updated_at: Mapped[now_datetime] = mapped_column(init=False)

    tokenizer: Mapped["Tokenizer"] = relationship(
        back_populates="models",
        default=None,
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="model",
        default_factory=list,
    )


class Prediction(Base):
    """Predictions Table to store model predictions for tokens."""

    __tablename__: str = "predictions"

    id: Mapped[int_primary_key] = mapped_column(init=False)
    token_id: Mapped[token_foreign_key] = mapped_column(init=False)
    model_id: Mapped[model_foreign_key] = mapped_column(init=False)
    type: Mapped[str]  # type: ignore[misc] # The value is inferred from the type annotation
    value: Mapped[list[str | float]] = mapped_column(JSON)  # type: ignore[misc] # The value is inferred from the type annotation
    wandb_run_url: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[now_datetime] = mapped_column(init=False)
    updated_at: Mapped[now_datetime] = mapped_column(init=False)

    token: Mapped["Token"] = relationship(
        back_populates="predictions",
        default=None,
    )
    model: Mapped["Model"] = relationship(
        back_populates="predictions",
        default=None,
    )


class Label(Base):
    """Labels Table to store labels associated with tokens."""

    __tablename__: str = "labels"

    id: Mapped[int_primary_key] = mapped_column(init=False)
    token_id: Mapped[token_foreign_key] = mapped_column(init=False)
    type: Mapped[str]  # type: ignore[misc] # The value is inferred from the type annotation
    value: Mapped[str | float]  # type: ignore[misc] # The value is inferred from the type annotation
    created_at: Mapped[now_datetime] = mapped_column(init=False)
    updated_at: Mapped[now_datetime] = mapped_column(init=False)

    token: Mapped["Token"] = relationship(
        back_populates="labels",
        default=None,
    )
