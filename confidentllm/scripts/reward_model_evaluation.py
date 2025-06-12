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
"""Runner for reward model evaluation using the ConfidentLLM package."""

import logging
from dataclasses import dataclass
from pprint import pformat
from typing import TYPE_CHECKING

import torch
import wandb
from datasets import Dataset
from hydra_zen import store, zen
from torch.nn.functional import logsigmoid as log_sigmoid
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils.generic import PaddingStrategy, TensorType
from trl.trainer.utils import selective_log_softmax

from confidentllm import data  # noqa: F401
from confidentllm.conversations.create_chat import create_conversation
from confidentllm.evaluation import EvaluationBatch, Evaluator
from confidentllm.models import ModelLoader
from confidentllm.models.model_name_and_type import ModelType
from confidentllm.scripts.setup_tools import (
    init_wandb,
    log_system_info,
    set_seed,
    setup_hydra_config_and_logging,
)

if TYPE_CHECKING:
    from confidentllm.conversations.types import ChatConversation

__all__ = ["main"]
logger = logging.getLogger("__main__")


@dataclass
class RewardModelEvalConfig:
    """Configuration class for the reward model evaluation process."""

    seed: int = 20244202
    batch_size: int = 32
    max_length: int = 256
    debug: bool = False


class RewardModelEvalRunner:
    """Class to run the reward model evaluation process."""

    def __init__(
        self,
        model: ModelLoader,
        evaluator: Evaluator,
        max_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        """Initialize the runner."""
        self.model, self.tokenizer = model.load()
        self.evaluator = evaluator
        self.max_length = max_length
        self.batch_size = batch_size

    def _rate_responses(
        self,
        responses: list[str],
        questions: list[str] | None = None,
    ) -> torch.Tensor:
        """Rate the responses for the given question.

        Args:
        ----
            questions (list[str] | None): The optional list of questions corresponding
                to the responses.
            responses (list[str]): The list of responses corresponding to the prompts.

        Returns:
        -------
            torch.Tensor: The scores for the rated responses.

        """
        conversations: ChatConversation = create_conversation(
            questions=questions,
            responses=responses,
        )

        inputs: BatchEncoding = self.tokenizer.apply_chat_template(
            list(conversations),
            add_generation_prompt=False,
            return_tensors=TensorType.PYTORCH,
            return_dict=True,
            truncation=True,
            padding=PaddingStrategy.MAX_LENGTH,  # type: ignore[arg-type]
            max_length=self.max_length,
        )  # type: ignore[assignment] # When using return dict a type BatchEncoding is returned
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            scores: torch.Tensor = self.model(**inputs).logits.detach()

        if self.model.model_type == ModelType.SEQUENCE_CLS:
            return log_sigmoid(scores.reshape(-1))
        elif self.model.model_type == ModelType.CAUSAL_LM:  # noqa: RET505
            scores = selective_log_softmax(logits=scores, index=inputs.input_ids)
            scores = (scores * inputs.attention_mask).sum(dim=-1)
            scores /= inputs.attention_mask.sum(dim=-1)
            return scores.reshape(-1)
            # return scores[:, -1].reshape(-1)
        else:
            raise ValueError("Unsupported model type")  # noqa: EM101, TRY003

    @staticmethod
    def _collate_fn(
        batch: list[dict[str, str | list[str]]],
    ) -> dict[str, list[str | list[str]]]:
        """Collate function to handle variable-length sequences by padding."""
        output: dict[str, list[str | list[str]]] = {}
        if "question" in batch[0]:
            output["question"] = [item["question"] for item in batch]
        output["preferred_response"] = [item["preferred_response"] for item in batch]
        output["rejected_response"] = [item["rejected_response"] for item in batch]

        return output

    def run(self, data: Dataset) -> None:  # noqa: F811
        """Run the reward model evaluation."""
        results_table = wandb.Table(
            columns=[
                "Question",
                "Preferred Response",
                "Rejected Response",
                "Preferred Confidence Score",
                "Rejected Confidence Score",
            ],
        )

        dataloader: DataLoader = DataLoader(
            dataset=data,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

        preferred_scores_list: list[torch.Tensor] = []
        rejected_scores_list: list[torch.Tensor] = []

        for batch in tqdm(dataloader, desc="Evaluating Responses"):
            questions: list[str] = batch.get("question")
            preferred_responses: list[str] = batch.get("preferred_response", [])
            rejected_responses: list[str] = batch.get("rejected_response", [])

            preferred_scores: torch.Tensor = self._rate_responses(
                questions=questions,
                responses=preferred_responses,
            )
            rejected_scores: torch.Tensor = self._rate_responses(
                questions=questions,
                responses=rejected_responses,
            )

            preferred_scores_list.append(preferred_scores.exp())
            rejected_scores_list.append(rejected_scores.exp())

            # Get the predictions (1 if preferred is better, 0 if rejected is better)
            predictions: list[int] = (
                (preferred_scores > rejected_scores).int().reshape(-1).tolist()
            )
            # Get labels (preferred is better so label is 1)
            labels: list[int] = [1] * len(predictions)
            # Get the confidences (difference between preferred and rejected scores)
            confidences: list[float] = (
                (preferred_scores - rejected_scores).abs().exp().reshape(-1).tolist()
            )

            questions = questions if questions else [""] * len(preferred_responses)
            for prompt, preferred, rejected, preferred_score, rejected_score in zip(
                questions,
                preferred_responses,
                rejected_responses,
                preferred_scores,
                rejected_scores,
                strict=True,
            ):
                results_table.add_data(
                    prompt,
                    preferred,
                    rejected,
                    preferred_score.item(),
                    rejected_score.item(),
                )

            # Add the batch to the evaluator
            self.evaluator.add_batch(
                batch=EvaluationBatch(
                    labels=[str(label) for label in labels],  # type: ignore[arg-type]
                    predictions=[str(prediction) for prediction in predictions],  # type: ignore[arg-type]
                    confidences=confidences,
                ),
            )

            # Log the answers and predictions
            for prompt, preferred, rejected, preferred_score, rejected_score in zip(
                questions,
                preferred_responses,
                rejected_responses,
                preferred_scores,
                rejected_scores,
                strict=True,
            ):
                results_table.add_data(
                    prompt,
                    preferred,
                    rejected,
                    preferred_score.item(),
                    rejected_score.item(),
                )

        results = self.evaluator.evaluate()
        logging_message: str = str(results)
        logger.info(logging_message)
        wandb_log: dict[str, wandb.Table] = {"results_table": results_table}
        wandb_log.update(results.to_dict())
        wandb.log(wandb_log)

        preferred_scores = torch.cat(preferred_scores_list, dim=0)
        rejected_scores = torch.cat(rejected_scores_list, dim=0)
        logging_message = (
            f"Average Score for preferred responses: {preferred_scores.mean().item()}"
        )
        logger.info(logging_message)
        wandb.log({"average_score_preferred": preferred_scores.mean().item()})
        logging_message = (
            f"Average Score for rejected responses: {rejected_scores.mean().item()}"
        )
        logger.info(logging_message)
        wandb.log({"average_score_rejected": rejected_scores.mean().item()})


@store(
    name="reward_model_evaluation",
    hydra_defaults=[
        "_self_",
        {"model": "sequence_cls"},
        {"model/lora": "no_lora"},
        {"data": "reward_bench"},
        {"evaluator": "accuracy_and_calibration"},
        {"run_config": "default"},
    ],
)
def run_reward_model_evaluation(
    data: Dataset,  # noqa: F811
    model: ModelLoader,
    evaluator: Evaluator,
    run_config: RewardModelEvalConfig,
) -> None:
    """Run the reward model evaluation process."""
    if not run_config.debug:
        init_wandb(task_name="RewardModelEvaluation")
    log_system_info()
    set_seed(run_config.seed)

    logger.info(f"Data: {pformat(data.info)}")  # noqa: G004

    runner = RewardModelEvalRunner(
        model=model,
        evaluator=evaluator,
        batch_size=run_config.batch_size,
        max_length=run_config.max_length,
    )

    runner.run(data=data)


def main() -> None:
    """Run the reward model evaluation."""
    store(
        RewardModelEvalConfig,
        name="default",
        group="run_config",
    )
    run_function = zen(run_reward_model_evaluation)

    config_keys: list[str] = [
        "data.name",
        "model.pretrained_model_name_or_path",
        "run_config.seed",
    ]

    setup_hydra_config_and_logging(
        job_name="reward_model_evaluation",
        config_keys=config_keys,
        add_hpc_launcher=True,
    )

    # Generate the CLI for run_reward_model_evaluation
    run_function.hydra_main(
        config_name="reward_model_evaluation",
        version_base="1.3",
    )


if __name__ == "__main__":
    main()
