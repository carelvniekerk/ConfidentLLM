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
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING

import torch
import wandb
from datasets import Dataset
from hydra_zen import store, zen
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils.generic import PaddingStrategy, TensorType

from confidentllm import data  # noqa: F401
from confidentllm.conversations.create_chat import create_conversation
from confidentllm.database import (
    DEFAULT_DATABASE_PATH,
    Label,
    Observation,
    Prediction,
    Tokenization,
    create_database,
    get_session,
)
from confidentllm.database import Dataset as DBDataset
from confidentllm.evaluation import EvaluationBatch, Evaluator
from confidentllm.models import ModelLoader
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
class QuantileRegressionEvalConfig:
    """Configuration class for the quantile regression evaluation process."""

    seed: int = 20244202
    batch_size: int = 32
    max_length: int = 256
    debug: bool = False
    database_path: Path = DEFAULT_DATABASE_PATH


class QuantileRegressionEvalRunner:
    """Class to run the quantile regression evaluation process."""

    def __init__(
        self,
        model: ModelLoader,
        evaluator: Evaluator,
        run_config: QuantileRegressionEvalConfig,
    ) -> None:
        """Initialize the runner."""
        self.model, self.tokenizer = model.load()
        self.evaluator = evaluator
        self.max_length = run_config.max_length
        self.batch_size = run_config.batch_size
        self.quantiles: list[float] = self.model.config.quantiles
        self.db_path: Path = run_config.database_path

    def _predict_quantiles(
        self,
        responses: list[str],
        questions: list[str] | None = None,
    ) -> list[dict[str, dict[str, float | list[float]]]]:
        """Predict the quantiles for the given responses.

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
            quantiles: torch.Tensor = self.model(**inputs).logits.detach()

        return self._postprocess_quantiles(inputs.input_ids, quantiles)

    def _postprocess_quantiles(
        self,
        token_ids: torch.Tensor,
        quantiles: torch.Tensor,
    ) -> list[dict[str, dict[str, float | list[float]]]]:
        """Postprocess the quantiles to get the final scores."""
        results: list[dict[str, dict[str, float | list[float]]]] = []
        for obs_idx, obs_token_ids in enumerate(token_ids):
            obs_results: dict[str, dict[str, float | list[float]]] = {}
            for pos, token_id in enumerate(obs_token_ids):
                if token_id == self.tokenizer.eos_token_id:
                    break
                if token_id == self.tokenizer.pad_token_id:
                    continue
                quantile_scores: list[float] = quantiles[obs_idx, pos, :].tolist()
                token: str = self.tokenizer.convert_ids_to_tokens(token_id.item())  # type: ignore[overload]
                obs_results[token] = {
                    "quantiles": quantile_scores,
                    "token_id": token_id.item(),
                }
            results.append(obs_results)
        return results

    @staticmethod
    def _collate_fn(
        batch: list[dict[str, str | list[str]]],
    ) -> dict[str, list[str | list[str]]]:
        """Collate function to handle variable-length sequences by padding."""
        output: dict[str, list[str | list[str]]] = {}
        if "question" in batch[0]:
            output["question"] = [item["question"] for item in batch]
        output["response"] = [item["preferred_response"] for item in batch]
        output["response_confidence"] = [
            item["preferred_response_confidence"] for item in batch
        ]

        return output

    def run(self, data: Dataset) -> None:  # noqa: F811
        """Run the reward model evaluation."""
        dataloader: DataLoader = DataLoader(
            dataset=data,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

        dataset_entry: DBDataset = DBDataset(
            name=data.info.dataset_name,  # type: ignore[arg-type] # Dataset name is set
            description=data.info.description,
            citation=data.info.citation,
            homepage=data.info.homepage,
            license=data.info.license,
        )  # type: ignore[call-arg]

        with get_session(self.db_path) as session:
            session.add(dataset_entry)
            session.commit()

        for batch in tqdm(dataloader, desc="Evaluating Responses"):
            questions: list[str] = batch.get("question")
            responses: list[str] = batch.get("response", [])
            labels: list[float] = batch.get("response_confidence", [])

            quantiles: list[dict[str, dict[str, float | list[float]]]] = (
                self._predict_quantiles(
                    responses=responses,
                    questions=questions,
                )
            )

            for idx, question in enumerate(questions):
                sentence: str = question
                sentence += responses[idx] if responses else ""

                observation: Observation = Observation(
                    sentence=sentence,
                    dataset=dataset_entry,
                )

                token_ids: list[int] = [
                    token_info["token_id"]  # type: ignore[misc]
                    for token_info in quantiles[idx].values()
                ]  # type: ignore[union-attr]
                tokens: list[str] = list(quantiles[idx].keys())

                tokenization = Tokenization(
                    tokenizer_name=self.tokenizer.name_or_path,
                    tokens=tokens,
                    token_ids=token_ids,
                    observation=observation,
                )

                label = Label(
                    label_type="confidence",
                    label_value=str(labels[idx]) if labels else "",
                    observation=observation,
                )

                quantile_predictions: list[list[float]] = [
                    token_info["quantiles"]  # type: ignore[misc]
                    for token_info in quantiles[idx].values()
                ]

                prediction = Prediction(
                    model_name=self.model.name_or_path,
                    prediction_type="quantile_scores",
                    prediction_values=quantile_predictions,  # type: ignore[arg-type]
                    observation=observation,
                    tokenization=tokenization,
                )

                with get_session(self.db_path) as session:
                    session.add(observation)
                    session.add(tokenization)
                    session.add(label)
                    session.add(prediction)
                    session.commit()


@store(
    name="quantile_regression_evaluation",
    hydra_defaults=[
        "_self_",
        {"model": "quantile_regression"},
        {"model/lora": "no_lora"},
        {"data": "question_answering"},
        {"evaluator": "accuracy_and_calibration"},
        {"run_config": "default"},
    ],
)
def run_quantile_regression_evaluation(
    data: Dataset,  # noqa: F811
    model: ModelLoader,
    evaluator: Evaluator,
    run_config: QuantileRegressionEvalConfig,
) -> None:
    """Run the quantile regression evaluation process."""
    if not run_config.debug:
        init_wandb(task_name="QuantileRegressionEvaluation")
        create_database(db_path=run_config.database_path)
    log_system_info()
    set_seed(run_config.seed)

    logger.info(f"Data: {pformat(data.info)}")  # noqa: G004

    runner = QuantileRegressionEvalRunner(
        model=model,
        evaluator=evaluator,
        run_config=run_config,
    )

    runner.run(data=data)


def main() -> None:
    """Run the quantile regression evaluation."""
    store(
        QuantileRegressionEvalConfig,
        name="default",
        group="run_config",
    )
    run_function = zen(run_quantile_regression_evaluation)

    config_keys: list[str] = [
        "data.name",
        "model.pretrained_model_name_or_path",
        "run_config.seed",
    ]

    setup_hydra_config_and_logging(
        job_name="quantile_regression_evaluation",
        config_keys=config_keys,
        add_hpc_launcher=True,
    )

    # Generate the CLI for run_reward_model_evaluation
    run_function.hydra_main(
        config_name="quantile_regression_evaluation",
        version_base="1.3",
    )


if __name__ == "__main__":
    main()
