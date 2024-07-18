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
"""Module to evaluate the accuracy of the model on a dataset."""

from dataclasses import dataclass, field
from typing import Any

from hydra_zen import make_custom_builds_fn, store

from confidentllm.evaluation.accuracy import accuracy_config
from confidentllm.evaluation.calibration import calibration_config
from confidentllm.evaluation.types import BaseResults, Evaluator

__all__ = []
builds = make_custom_builds_fn(populate_full_signature=True)


@dataclass
class CombinedResults(BaseResults):
    """Combined results from multiple evaluators."""

    combined_results: dict[str, BaseResults] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert combined results to a dictionary."""
        combined_dict = super().to_dict()
        for results in self.combined_results.values():
            for key, value in results.to_dict().items():
                combined_dict[key] = value
        combined_dict["evaluator_name"] = self.evaluator_name
        return combined_dict

    def __str__(self) -> str:
        """Convert combined results to a string."""
        combined_str = super().__str__()
        for results in self.combined_results.values():
            combined_str += f"\n{results}"
        return combined_str


class CombinedEvaluator(Evaluator):
    """Pipeline for combining multiple evaluators."""

    def __init__(self, evaluators: list[Evaluator], padding_value: int = -1) -> None:
        """Initialize the combined evaluator with a list of evaluators."""
        super().__init__(padding_value)
        shared_buffer = {}
        self.buffer = shared_buffer
        for evaluator in evaluators:
            evaluator.buffer = shared_buffer
            evaluator.padding_value = padding_value
        self.evaluators = evaluators

    def evaluate(self) -> CombinedResults:
        """Evaluate the model using all evaluators in the combined evaluator.

        Returns
        -------
            A dictionary containing the results from each evaluator.

        """
        results = {}
        for evaluator in self.evaluators:
            _res = evaluator.evaluate()
            results[_res.evaluator_name] = _res

        return CombinedResults(evaluator_name="combined", combined_results=results)


CombinedEvaluatorConfig = builds(CombinedEvaluator)

accuracy_and_calibration_config = CombinedEvaluatorConfig(
    evaluators=[accuracy_config, calibration_config],  # type: ignore  # noqa: PGH003 - Configs return Evaluator objects during execution.
    padding_value=-1,
)


evaluator_store = store(group="evaluator")
evaluator_store(accuracy_and_calibration_config, name="accuracy_and_calibration")
