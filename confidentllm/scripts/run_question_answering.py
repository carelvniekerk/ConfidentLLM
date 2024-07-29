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
"""Runner for question answering using the ConfidentLLM package."""

import logging

import wandb
from datasets import Dataset
from hydra_zen import store, zen
from tqdm import tqdm

from confidentllm import data  # noqa: F401
from confidentllm.evaluation import Evaluator
from confidentllm.generation import CausalLMGenerationMethod
from confidentllm.models import ModelLoader
from confidentllm.output_processing import (
    Answer,
    OutputProcessor,
)
from confidentllm.scripts.setup_tools import (
    init_wandb,
    set_seed,
    setup_hydra_config_and_logging,
)

__all__ = ["main"]
logger = logging.getLogger("__main__")


class QuestionAnsweringRunner:
    """Class to run the question answering process."""

    def __init__(
        self,
        model: ModelLoader,
        generation_method: CausalLMGenerationMethod,
        answer_processor: OutputProcessor,
        evaluator: Evaluator,
    ) -> None:
        """Initialize the runner."""
        self.model, self.tokenizer = model.load()
        self.generation_method = generation_method
        self.answer_processor = answer_processor
        self.evaluator = evaluator

        self.generation_method.set_model(self.model)
        self.answer_processor.set_model(self.model)
        self.generation_method.set_tokenizer(self.tokenizer)
        self.answer_processor.set_tokenizer(self.tokenizer)

    def _answer_question(
        self,
        question: str,
        max_length: int = 256,
    ) -> tuple[Answer, str]:
        """Answer the given question.

        Args:
        ----
            question (str): The question to answer.
            max_length (int, optional): The maximum length of the generated answer.

        Returns:
        -------
            tuple[str, torch.Tensor]: The generated answer and its confidence.

        """
        generation_output = self.generation_method(
            question,
            max_length,
        )
        reasoning: str = self.tokenizer.decode(
            generation_output.generated_ids[0],
            skip_special_tokens=True,
        )
        answer: Answer = self.answer_processor(generation_output)  # type: ignore  # noqa: PGH003 - Answer processor here will always return answer.
        return answer, reasoning

    def run(self, data: Dataset) -> None:  # noqa: F811
        """Run the question answering process."""
        results_table = wandb.Table(
            columns=["Question", "Reasoning", "Answer", "True Answer", "Confidence"],
        )
        for example in tqdm(data, desc="Answering questions"):
            question: str = example.get("question", "")  # type: ignore  # noqa: PGH003
            answer, reasoning = self._answer_question(question)

            print(answer)

            # Add the batch to the evaluator
            self.evaluator.add_batch(
                {
                    "labels": [int(example.get("answer", "-1").replace(",", ""))],  # type: ignore  # noqa: PGH003
                    "predictions": [answer.answer],
                    "confidences": [answer.confidence.mean().item()],
                },
            )

            # Log the answers and predictions
            results_table.add_data(
                question,
                reasoning,
                answer.answer,
                int(example.get("answer", "-1").replace(",", "")),  # type: ignore  # noqa: PGH003
                answer.confidence.mean().item(),
            )

        results = self.evaluator.evaluate()
        logging_message: str = str(results)
        logger.info(logging_message)
        wandb_log = {"results_table": results_table}
        wandb_log.update(results.to_dict())
        wandb.log(wandb_log)


@store(
    name="question_answering",
    hydra_defaults=[
        "_self_",
        {"model": "default"},
        {"generation_method": "causal_lm_generation_method"},
        {"generation_method/generator/decoding_strategy": "greedy"},
        {"output_processor": "answer_processor"},
        {"output_processor/generator/decoding_strategy": "greedy"},
        {"data": "multiarith"},
        {"evaluator": "accuracy_and_calibration"},
    ],
)
def run_question_answering(  # noqa: PLR0913
    data: Dataset,  # noqa: F811
    model: ModelLoader,
    generation_method: CausalLMGenerationMethod,
    output_processor: OutputProcessor,
    evaluator: Evaluator,
    seed: int = 20244202,
) -> None:
    """Run the question answering process."""
    set_seed(seed)
    init_wandb("question_answering")

    runner = QuestionAnsweringRunner(
        model,
        generation_method,
        output_processor,
        evaluator,
    )
    runner.run(data)


def main() -> None:
    """Run the question answering process."""
    run_function = zen(run_question_answering)

    setup_hydra_config_and_logging(job_name="question_answering", add_hpc_launcher=True)

    # Generate the CLI for run_extraction
    run_function.hydra_main(
        config_name="question_answering",
        version_base="1.3",
    )


if __name__ == "__main__":
    main()
