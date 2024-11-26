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
"""Types for the output processing module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypedDict, Unpack

from transformers import PreTrainedModel, PreTrainedTokenizer

from confidentllm.generation.types import GenerationOutput

__all__ = [
    "ProcessedOutput",
    "OutputProcessor",
    "OutputProcessorKwargs",
]


class OutputProcessorKwargs(TypedDict, total=False):
    """Keyword arguments for the output processor."""

    return_best_answer_idx: bool


@dataclass
class ProcessedOutput:
    """Dataclass for the processed output of the generation method."""


class OutputProcessor(ABC):
    """Class for processing the generated answers."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer

    def set_model(self, model: PreTrainedModel) -> None:
        """Set the model for the generation method."""
        self.model = model

    def set_tokenizer(self, tokenizer: PreTrainedTokenizer) -> None:
        """Set the tokenizer for the generation method."""
        self.tokenizer = tokenizer

    def set_choices(self, choices: list[str]) -> None:
        """Set the choices for the generation method."""
        self.choices = choices

    @abstractmethod
    def __call__(
        self,
        generation_output: GenerationOutput,
        **kwargs: Unpack[OutputProcessorKwargs],
    ) -> ProcessedOutput:
        """Process the generated answer."""
        ...
