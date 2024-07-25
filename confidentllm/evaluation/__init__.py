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
"""Evaluation module for ConfidentLLM."""

from hydra_zen import store

from confidentllm.evaluation.accuracy import accuracy_config
from confidentllm.evaluation.calibration import calibration_config
from confidentllm.evaluation.combined import CombinedEvaluatorConfig

__all__ = []

accuracy_and_calibration_config = CombinedEvaluatorConfig(
    evaluators=[accuracy_config, calibration_config],  # type: ignore  # noqa: PGH003 - Configs return Evaluator objects during execution.
)

evaluator_store = store(group="evaluator")
evaluator_store(accuracy_config, name="accuracy")
evaluator_store(calibration_config, name="calibration")
evaluator_store(accuracy_and_calibration_config, name="accuracy_and_calibration")
