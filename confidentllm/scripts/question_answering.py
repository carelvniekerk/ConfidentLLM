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
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runner for question answering using the ConfidentLLM package."""

import gc
import logging
from dataclasses import dataclass
from itertools import cycle
from pprint import pformat

import torch
from datasets import Dataset
from hydra_zen import store, zen
from tqdm import tqdm

import wandb
from confidentllm.data import datasets  # noqa: F401
from confidentllm.evaluation import EvaluationBatch, Evaluator
from confidentllm.generation import CausalLMGenerationMethod
from confidentllm.generation.types import GenerationOutput
from confidentllm.models import ModelLoader
from confidentllm.output_processing import (
    Answer,
    OutputProcessor,
)
from confidentllm.scripts.setup_tools import (
    init_wandb,
    log_system_info,
    set_seed,
    setup_hydra_config_and_logging,
)

__all__ = ["main"]
logger = logging.getLogger("__main__")


@dataclass
class QARunConfig:
    """Configuration class for the mathematical reasoning process."""

    keep_all_generation_paths: bool = False
    seed: int = 20244202
    debug: bool = False


class QARunner:
    """Class to run the mathematical reasoning process."""

    def __init__(
        self,
        model: ModelLoader,
        generation_method: CausalLMGenerationMethod,
        answer_processor: OutputProcessor,
        evaluator: Evaluator,
        *,
        keep_all_generation_paths: bool = False,
    ) -> None:
        """Initialize the runner."""
        self.model, self.tokenizer = model.load()
        self.generation_method = generation_method
        self.answer_processor = answer_processor
        self.evaluator = evaluator
        self.keep_all_generation_paths = keep_all_generation_paths

        self.generation_method.set_model(self.model)
        self.answer_processor.set_model(self.model)
        self.generation_method.set_tokenizer(self.tokenizer)
        self.answer_processor.set_tokenizer(self.tokenizer)

    def _answer_question(
        self,
        question: str | None = None,
        responses: list[str] | None = None,
    ) -> Answer | list[Answer]:
        """Answer the given question.

        Args:
        ----
            question (str | None): The question to answer.
            responses (list[str] | None): Optional list of responses to consider.

        Returns:
        -------
            Answer | list[Answer]: The answer to the question.

        """
        generation_output: GenerationOutput = self.generation_method(
            prompt=question,
            responses=responses,
        )
        num_beams: int = (
            generation_output.generated_ids.size(0)
            if not isinstance(generation_output.generated_ids, list)
            else len(generation_output.generated_ids)
        )

        if self.keep_all_generation_paths and num_beams == 1:
            msg = "Only one beam was generated, output will only contain one path."
            raise RuntimeWarning(msg)

        if self.keep_all_generation_paths:
            single_paths: list[GenerationOutput] = [
                GenerationOutput(
                    generated_ids=generation_output.generated_ids[idx].unsqueeze(0),
                    generation_scores=generation_output.generation_scores[
                        idx,
                    ].unsqueeze(
                        dim=0,
                    ),
                )
                for idx in range(num_beams)
            ]

            answers: list[Answer] = [
                self.answer_processor(single_path)  # type: ignore[misc]
                for single_path in single_paths
            ]
            gc.collect()
            torch.cuda.empty_cache()
            return answers

        answer: Answer = self.answer_processor(generation_output)  # type: ignore[assignment]
        gc.collect()
        torch.cuda.empty_cache()
        return answer

    def run(self, data: Dataset) -> None:
        """Run the mathematical reasoning process."""
        results_table = wandb.Table(
            columns=["Question", "Reasoning", "Answer", "True Answer", "Confidence"],
        )

        # Table for storing all generation paths if required
        generation_path_data: wandb.Table | None = (
            wandb.Table(
                columns=[
                    "Question",
                    *[
                        f"Decoded Path {i}"
                        for i in range(self.generation_method.num_beams)
                    ],
                    *[
                        f"Confidence {i}"
                        for i in range(self.generation_method.num_beams)
                    ],
                ],
            )
            if self.keep_all_generation_paths
            else None
        )

        for example in tqdm(data, desc="Answering questions"):
            question: str = example.get("question")  # type: ignore[attr-access]
            responses: list[str] | None = example.get("preferred_response")  # type: ignore[attr-access]
            responses = responses[:-1] if responses else None
            answers: Answer | list[Answer] = self._answer_question(
                question=question,
                responses=responses,
            )

            if self.keep_all_generation_paths:
                confidences: list[float] = [
                    answer.confidence.item()
                    for answer in answers  # type: ignore[union-attr]
                ]
                best_answer_idx: int = confidences.index(max(confidences))

            answer: Answer = (
                answers[best_answer_idx] if self.keep_all_generation_paths else answers  # type: ignore[index, assignment]
            )

            # Add the batch to the evaluator
            self.evaluator.add_batch(
                batch=EvaluationBatch(
                    labels=[example.get("answer", "-1").upper()],  # type: ignore[attr-access]
                    predictions=[answer.answer.upper()],
                    confidences=[answer.confidence.mean().item()],
                ),
            )

            # Log the answers and predictions
            if question is None:
                question = ""
                for speaker, utterance in zip(
                    cycle(["User", "Assistant"]),
                    responses,
                    strict=False,
                ):
                    question += f"{speaker}: {utterance}\n"

            results_table.add_data(
                question,
                answer.reasoning,
                answer.answer,
                example.get("answer", "-1"),  # type: ignore[attr-access]
                answer.confidence.mean().item(),
            )

            # Log the generation paths if required
            if not self.keep_all_generation_paths:
                continue

            generation_path_data.add_data(  # type: ignore[union-attr]
                question,
                *[answer.reasoning for answer in answers],  # type: ignore[union-attr]
                *confidences,
            )

        wandb_log: dict[str, wandb.Table] = {}
        wandb_log["results_table"] = results_table
        if self.keep_all_generation_paths:
            wandb_log["generation_path_data"] = generation_path_data  # type: ignore[assignment]
        results = self.evaluator.evaluate()
        logging_message: str = str(results)
        logger.info(logging_message)
        wandb_log.update(results.to_dict())
        wandb.log(wandb_log)


@store(
    name="question_answering",
    hydra_defaults=[
        "_self_",
        {"model": "causal_lm"},
        {"model/lora": "no_lora"},
        {"generation_method": "greedy_decoding"},
        {"generation_method/confidence_metric": "probability_disparity"},
        {"output_processor": "numeric_answer_with_token_confidence"},
        {"data": "multiarith"},
        {"evaluator": "accuracy_and_calibration"},
        {"run_config": "default"},
    ],
)
def run_qa(  # noqa: PLR0913
    data: Dataset,
    model: ModelLoader,
    generation_method: CausalLMGenerationMethod,
    output_processor: OutputProcessor,
    evaluator: Evaluator,
    run_config: QARunConfig,
) -> None:
    """Run the question answering process."""
    if not run_config.debug:
        init_wandb(task_name="QuestionAnswering")
    log_system_info()
    set_seed(run_config.seed)

    logger.info(f"Data: {pformat(data.info)}")  # noqa: G004

    runner = QARunner(
        model=model,
        generation_method=generation_method,
        answer_processor=output_processor,
        evaluator=evaluator,
        keep_all_generation_paths=run_config.keep_all_generation_paths,
    )

    if hasattr(data, "choices"):
        runner.answer_processor.set_choices(data.choices)  # type: ignore[attr-defined] # All QA datasets should have the choices attribute

    # Select subset of data for debugging
    if run_config.debug:
        data = data.select(range(10))

    runner.run(data=data)


def main() -> None:
    """Run the question answering process."""
    store(
        QARunConfig,
        name="default",
        group="run_config",
    )
    run_function = zen(run_qa)

    config_keys: list[str] = [
        "data.name",
        "data.split",
        "model.pretrained_model_name_or_path",
        "resolve_generation_method:${generation_method}",
        "resolve_output_processor:${output_processor}",
        "run_config.seed",
    ]

    setup_hydra_config_and_logging(
        job_name="question_answering",
        config_keys=config_keys,
        add_hpc_launcher=True,
    )

    # Generate the CLI for run_extraction
    run_function.hydra_main(
        config_name="question_answering",
        version_base="1.3",
    )


if __name__ == "__main__":
    main()
