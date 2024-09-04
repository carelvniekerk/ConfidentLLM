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
"""Set up output processing configuration for project."""

from hydra_zen import store

from confidentllm.hydra_tools import builds
from confidentllm.output_processing.answer_processor import Answer, AnswerProcessor
from confidentllm.output_processing.numeric_answer_processor import (
    NumericAnswerProcessor,
)
from confidentllm.output_processing.types import OutputProcessor
from confidentllm.output_processing.verbalised_confidence import (
    VerbalisedConfidenceAnswerProcessor,
    VerbalisedConfidenceNumericAnswerProcessor,
)

__all__ = [
    "OutputProcessor",
    "Answer",
]

AnswerProcessorConfig = builds(AnswerProcessor)

NumericAnswerProcessorConfig = builds(NumericAnswerProcessor)

VerbalisedConfidenceProcessorConfig = builds(VerbalisedConfidenceAnswerProcessor)

NumericVerbalisedConfidenceProcessorConfig = builds(
    VerbalisedConfidenceNumericAnswerProcessor,
)

output_processor_store = store(group="output_processor")
output_processor_store(AnswerProcessorConfig, name="answer_with_token_confidence")
output_processor_store(
    NumericAnswerProcessorConfig,
    name="numeric_answer_with_token_confidence",
)
output_processor_store(
    VerbalisedConfidenceProcessorConfig,
    name="answer_with_verbalised_confidence",
)
output_processor_store(
    NumericVerbalisedConfidenceProcessorConfig,
    name="numeric_answer_with_verbalised_confidence",
)
