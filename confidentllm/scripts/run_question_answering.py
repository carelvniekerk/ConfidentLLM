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

import torch
from datasets import Dataset
from hydra.conf import HydraConf, JobConf
from hydra.core.hydra_config import HydraConfig
from hydra_zen import store, zen
from tqdm import tqdm

import wandb
from confidentllm import data  # noqa: F401
from confidentllm.evaluation.types import Evaluator
from confidentllm.generation.types import (
    CausalLMGenerationMethod,
    OutputProcessor,
)
from confidentllm.logging import (
    create_logging_config,
    initialize_wandb,
    setup_exception_logging,
)
from confidentllm.models import ModelLoader
from hydra_plugins.hpc_submission_launcher.launcher import (
    HPCSubmissionLauncher,  # noqa: F401
)

__all__ = ["run_question_answering"]
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

    def answer_question(
        self,
        question: str,
        max_length: int = 256,
    ) -> tuple[str | int | None, torch.Tensor, str]:
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
        reasoning: str = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )
        answer, confidence = self.answer_processor(generated_ids, generation_probs)
        return answer, confidence, reasoning

    def run(self, data: Dataset) -> None:  # noqa: F811
        """Run the question answering process."""
        results_table = wandb.Table(
            columns=["Question", "Reasoning", "Answer", "True Answer", "Confidence"],
        )
        for example in tqdm(data, desc="Answering questions"):
            question: str = example.get("question", "")  # type: ignore  # noqa: PGH003
            answer, confidence, reasoning = self.answer_question(question)

            # Add the batch to the evaluator
            self.evaluator.add_batch(
                {
                    "labels": [int(example.get("answer", "-1").replace(",", ""))],  # type: ignore  # noqa: PGH003
                    "predictions": [answer if answer else -1],
                    "confidences": [confidence.mean().item()],
                },
            )

            # Log the answers and predictions
            results_table.add_data(
                question,
                reasoning,
                answer if answer else -1,
                int(example.get("answer", "-1").replace(",", "")),  # type: ignore  # noqa: PGH003
                confidence.mean().item(),
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
        {"model": "gemma_11_2b_it"},
        {"generation_method": "greedy_causal_lm_generation_method"},
        {"output_processor": "answer_processor"},
        {"data": "gsm8k"},
        {"evaluator": "accuracy_and_calibration"},
        {"override hydra/launcher": "hpc_submission"},
    ],
)
def run_question_answering(
    data: Dataset,  # noqa: F811
    model: ModelLoader,
    generation_method: CausalLMGenerationMethod,
    output_processor: OutputProcessor,
    evaluator: Evaluator,
) -> None:
    """Run the question answering process."""
    initialize_wandb(config=HydraConfig.get())  # type: ignore  # noqa: PGH003

    runner = QuestionAnsweringRunner(
        model,
        generation_method,
        output_processor,
        evaluator,
    )
    runner.run(data)


if __name__ == "__main__":
    setup_exception_logging(logger)

    store(
        HydraConf(
            job=JobConf(chdir=True, name="question_answering"),
            job_logging=create_logging_config(),
        ),
        name="config",
        group="hydra",
    )
    store.add_to_hydra_store()

    # Generate the CLI for run_extraction
    zen(run_question_answering).hydra_main(
        config_name="question_answering",
    )
