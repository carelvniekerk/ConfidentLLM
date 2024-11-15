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
"""Runner for question answering using the ConfidentLLM package."""

import logging
from dataclasses import dataclass
from pprint import pformat

from datasets import Dataset
from hydra_zen import store, zen
from tqdm import tqdm

import wandb
from confidentllm import data  # noqa: F401
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
class MathematicalReasoningRunConfig:
    """Configuration class for the mathematical reasoning process."""

    keep_all_generation_paths: bool = False
    seed: int = 20244202


store(
    MathematicalReasoningRunConfig,
    name="default",
    group="run_config",
)


class MathematicalReasoningRunner:
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
        question: str,
    ) -> Answer | list[Answer]:
        """Answer the given question.

        Args:
        ----
            question (str): The question to answer.
            keep_all_generation_paths (bool): Whether to keep all generation paths.

        Returns:
        -------
            Answer | list[Answer]: The answer to the question.

        """
        generation_output: GenerationOutput = self.generation_method(question)
        num_beams: int = generation_output.generated_ids.size(0)

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
            return answers

        answer: Answer = self.answer_processor(generation_output)  # type: ignore[assignment]
        return answer

    def run(self, data: Dataset) -> None:  # noqa: F811
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
            question: str = example.get("question", "")  # type: ignore[attr-access]
            answers: Answer | list[Answer] = self._answer_question(question)

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
                    labels=[float(example.get("answer", "-1").replace(",", ""))],  # type: ignore[attr-access]
                    predictions=[float(answer.answer)],
                    confidences=[answer.confidence.mean().item()],
                ),
            )

            # Log the answers and predictions
            results_table.add_data(
                question,
                answer.reasoning,
                float(answer.answer),
                float(example.get("answer", "-1").replace(",", "")),  # type: ignore[attr-access]
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

        results = self.evaluator.evaluate()
        logging_message: str = str(results)
        logger.info(logging_message)
        wandb_log: dict[str, wandb.Table] = {"results_table": results_table}
        if self.keep_all_generation_paths:
            wandb_log["generation_path_data"] = generation_path_data  # type: ignore[assignment]
        wandb_log.update(results.to_dict())
        wandb.log(wandb_log)


@store(
    name="mathematical_reasoning",
    hydra_defaults=[
        "_self_",
        {"model": "causal_lm"},
        {"model/lora": "no_lora"},
        {"generation_method": "greedy_decoding"},
        {"generation_method/confidence_extraction_method": "probability_disparity"},
        {"output_processor": "numeric_answer_with_token_confidence"},
        {"data": "multiarith"},
        {"evaluator": "accuracy_and_calibration"},
        {"run_config": "default"},
    ],
)
def run_mathematical_reasoning(  # noqa: PLR0913
    data: Dataset,  # noqa: F811
    model: ModelLoader,
    generation_method: CausalLMGenerationMethod,
    output_processor: OutputProcessor,
    evaluator: Evaluator,
    run_config: MathematicalReasoningRunConfig,
) -> None:
    """Run the question answering process."""
    init_wandb()
    log_system_info()
    set_seed(run_config.seed)

    logger.info(f"Data: {pformat(data.info)}")  # noqa: G004

    runner = MathematicalReasoningRunner(
        model=model,
        generation_method=generation_method,
        answer_processor=output_processor,
        evaluator=evaluator,
        keep_all_generation_paths=run_config.keep_all_generation_paths,
    )
    runner.run(data)


def main() -> None:
    """Run the question answering process."""
    run_function = zen(run_mathematical_reasoning)

    config_keys: list[str] = [
        "data.name",
        "data.split",
        "model.pretrained_model_name_or_path",
        "resolve_generation_method:${generation_method}",
        "resolve_output_processor:${output_processor}",
        "run_config.seed",
    ]

    setup_hydra_config_and_logging(
        job_name="mathematical_reasoning",
        config_keys=config_keys,
        add_hpc_launcher=True,
    )

    # Generate the CLI for run_extraction
    run_function.hydra_main(
        config_name="mathematical_reasoning",
        version_base="1.3",
    )


if __name__ == "__main__":
    main()
