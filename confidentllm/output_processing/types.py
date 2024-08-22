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
from typing import TYPE_CHECKING

from torch.nn import Module
from transformers.tokenization_utils import PreTrainedTokenizer

from confidentllm.generation.types import GenerationOutput

if TYPE_CHECKING:
    from confidentllm.generation.types import GenerateFunction

__all__ = [
    "ProcessedOutput",
    "OutputProcessor",
]


@dataclass
class ProcessedOutput:
    """Dataclass for the processed output of the generation method."""


class OutputProcessor(ABC):
    """Class for processing the generated answers."""

    def __init__(self) -> None:
        """Initialize the processor."""
        self.model: Module | None = None
        self.tokenizer: PreTrainedTokenizer | None = None
        self.generator: GenerateFunction | None = None

    def set_model(self, model: Module) -> None:
        """Set the model for the generation method."""
        if isinstance(self.generator, type(None)):
            msg = "The generator is not set."
            raise TypeError(msg)
        self.generator.set_model(model)

    def set_tokenizer(self, tokenizer: PreTrainedTokenizer) -> None:
        """Set the tokenizer for the generation method."""
        self.tokenizer = tokenizer

    @abstractmethod
    def __call__(
        self,
        generation_output: GenerationOutput,
        **kwargs: dict | None,
    ) -> ProcessedOutput:
        """Process the generated answer."""
        ...
