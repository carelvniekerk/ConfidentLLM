# coding=utf-8
# --------------------------------------------------------------------------------
# Project: ConfidentLLM
# Author: Carel van Niekerk
# Year: 2024
# Group: Dialogue Systems and Machine Learning Group
# Institution: Heinrich Heine University Düsseldorf
# --------------------------------------------------------------------------------
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
"""Runner for question answering using the ConfidentLLM package."""

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

from confidentllm.generation.types import (
    CausalLMGenerationMethod,
    OutputProcessor,
)

__all__ = ["run_question_answering"]


class QuestionAnsweringRunner:
    """Class to run the question answering process."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        generation_method: CausalLMGenerationMethod,
        answer_processor: OutputProcessor,
    ) -> None:
        """Initialize the runner."""
        self.model = model
        self.tokenizer = tokenizer
        self.generation_method = generation_method
        self.answer_processor = answer_processor

        self.generation_method.set_model(model)
        self.answer_processor.set_model(model)
        self.generation_method.set_tokenizer(tokenizer)
        self.answer_processor.set_tokenizer(tokenizer)

    def answer_question(
        self,
        question: str,
        max_length: int = 256,
    ) -> tuple[str | int | None, torch.Tensor]:
        """Answer the given question.

        Args:
        ----
            question (str): The question to answer.
            max_length (int, optional): The maximum length of the generated answer.

        Returns:
        -------
            tuple[str, torch.Tensor]: The generated answer and its confidence.

        """
        generated_ids, generation_probs = self.generation_method(
            question,
            max_length,
        )
        answer, confidence = self.answer_processor(generated_ids, generation_probs)
        return answer, confidence

    def run(self, data: Dataset) -> None:
        """Run the question answering process."""
        for example in data:
            question: str = example.get("question", "")  # type: ignore  # noqa: PGH003
            answer, confidence = self.answer_question(question)
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"Confidence: {confidence}")
            print()


def run_question_answering(
    data: Dataset,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    generation_method: CausalLMGenerationMethod,
    output_processor: OutputProcessor,
) -> None:
    """Run the question answering process."""
    runner = QuestionAnsweringRunner(
        model,
        tokenizer,
        generation_method,
        output_processor,
    )
    runner.run(data)
