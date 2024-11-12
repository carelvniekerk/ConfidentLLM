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
#     http: //www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train a reward model."""

import logging
from functools import partial

from datasets import Dataset
from hydra_zen import store, zen

from confidentllm import data  # noqa: F401
from confidentllm.models import ModelLoader
from confidentllm.scripts.setup_tools import (
    init_wandb,
    set_seed,
    setup_hydra_config_and_logging,
)
from confidentllm.train import RewardModelTrainer, prepare_reward_model_data

__all__ = ["main"]
logger = logging.getLogger("__main__")


@store(
    name="reward_model_training",
    hydra_defaults=[
        "_self_",
        {"model": "causal_lm"},
        {"model/lora": "no_lora"},
        {"train_data": "cot_preference"},
        {"eval_data": "cot_preference"},
        {"trainer": "reward_model_trainer"},
    ],
)
def run_training(
    train_data: Dataset,
    eval_data: Dataset,
    model: ModelLoader,
    trainer: RewardModelTrainer,
) -> None:
    """Run the question answering process."""
    set_seed(trainer.seed)
    init_wandb()

    reward_model, tokenizer = model.load()
    trainer.set_model(reward_model)
    trainer.set_tokenizer(tokenizer)

    data_preperation_function = partial(prepare_reward_model_data, tokenizer=tokenizer)
    train_data = train_data.map(
        function=data_preperation_function,
        batched=True,
        batch_size=512,
    )
    eval_data = eval_data.map(
        function=data_preperation_function,
        batched=True,
        batch_size=512,
    )

    trainer.set_train_dataset(train_data)
    trainer.set_eval_dataset(eval_data)

    trainer.train()


def main() -> None:
    """Run the question answering process."""
    run_function = zen(run_training)

    setup_hydra_config_and_logging(
        job_name="reward_model_training",
        add_hpc_launcher=True,
    )

    # Generate the CLI for run_extraction
    run_function.hydra_main(
        config_name="reward_model_training",
        version_base="1.3",
    )


if __name__ == "__main__":
    main()
