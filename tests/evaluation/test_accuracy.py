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
"""Test the accuracy evaluator."""

import pytest

from confidentllm.evaluation import EvaluationBatch
from confidentllm.evaluation.accuracy import AccuracyEvaluator, AccuracyResults


@pytest.fixture
def test_data() -> list[EvaluationBatch]:
    """Create a test dataset."""
    return [
        EvaluationBatch(
            labels=[0, 1, 2],
            predictions=[0, 1, 2],
        ),
        EvaluationBatch(
            labels=[-1, 3, 4],
            predictions=[0, 1, 1],
        ),
    ]


@pytest.fixture
def accuracy_evaluator() -> AccuracyEvaluator:
    """Create an accuracy evaluator."""
    return AccuracyEvaluator(padding_value=-1)


def test_adding_batches(
    test_data: list[EvaluationBatch],
    accuracy_evaluator: AccuracyEvaluator,
) -> None:
    """Test adding batches to the accuracy evaluator."""
    for batch in test_data:
        accuracy_evaluator.add_batch(batch)

    if accuracy_evaluator.buffer.labels != [0, 1, 2, -1, 3, 4]:
        msg = (
            f"The labels were not stored correctly, expected {[0, 1, 2, -1, 3, 4]},"
            f" got {accuracy_evaluator.buffer.labels}"
        )
        raise AssertionError(msg)
    if accuracy_evaluator.buffer.predictions != [0, 1, 2, 0, 1, 1]:
        msg = (
            f"The predictions were not stored correctly, expected {[0, 1, 2, 0, 1, 1]},"
            f" got {accuracy_evaluator.buffer.labels}"
        )
        raise AssertionError(msg)


def test_accuracy(
    test_data: list[EvaluationBatch],
    accuracy_evaluator: AccuracyEvaluator,
) -> None:
    """Test the accuracy calculation."""
    for batch in test_data:
        accuracy_evaluator.add_batch(batch)

    results: AccuracyResults = accuracy_evaluator.evaluate()

    if results.accuracy != 3 * 100 / 5:
        msg = f"Expected accuracy to be {3 * 100 / 5}, got {results.accuracy}"
        raise AssertionError(msg)
