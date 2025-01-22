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
"""Finetune the model supervisedly."""

from dataclasses import dataclass
from pprint import pformat

from datasets import Dataset
from hydra_zen import store, zen

from confidentllm.data import sft_preprocessing
from confidentllm.models import ModelLoader
from confidentllm.scripts.setup_tools import (
    get_logger,
    init_wandb,
    log_system_info,
    set_seed,
    setup_hydra_config_and_logging,
)
from confidentllm.train import SupervisedFinetuningTrainer

__all__ = ["main"]
logger = get_logger()


@dataclass
class SFTTrainingRunConfig:
    """Configuration class for the SFT training process."""

    debug: bool = False


@store(
    name="supervised_finetuning",
    hydra_defaults=[
        "_self_",
        {"model": "train_causal_lm"},
        {"model/lora": "causal_lm"},
        {"train_data": "cot_preference"},
        {"eval_data": "cot_preference"},
        {"trainer": "supervised_finetuning"},
        {"run_config": "default"},
    ],
)
def run_training(
    train_data: Dataset,
    eval_data: Dataset,
    model: ModelLoader,
    trainer: SupervisedFinetuningTrainer,
    run_config: SFTTrainingRunConfig,
) -> None:
    """Run the question answering process."""
    if not run_config.debug:
        init_wandb()
    log_system_info()
    set_seed(trainer.seed)

    model_instance, tokenizer = model.load()
    trainer.set_model(model_instance)
    trainer.set_tokenizer(tokenizer)

    train_data = train_data.map(
        function=sft_preprocessing,
        batched=True,
        batch_size=512,
        remove_columns=train_data.column_names,
        load_from_cache_file=train_data.cached_version,  # type: ignore[attr-defined]
    )
    eval_data = eval_data.map(
        function=sft_preprocessing,
        batched=True,
        batch_size=512,
        remove_columns=eval_data.column_names,
        load_from_cache_file=eval_data.cached_version,  # type: ignore[attr-defined]
    )

    logger.info(f"Training data: {pformat(train_data.info)}")  # noqa: G004
    logger.info(f"Evaluation data: {pformat(eval_data.info)}")  # noqa: G004

    trainer.set_train_dataset(train_data)
    trainer.set_eval_dataset(eval_data)

    trainer.train()


def main() -> None:
    """Run the question answering process."""
    store(
        SFTTrainingRunConfig,
        name="default",
        group="run_config",
    )
    run_function = zen(run_training)

    config_keys = [
        "train_data.name",
        "model.pretrained_model_name_or_path",
        "trainer.seed",
    ]

    setup_hydra_config_and_logging(
        job_name="supervised_finetuning",
        add_hpc_launcher=True,
        config_keys=config_keys,
    )

    # Generate the CLI for run_extraction
    run_function.hydra_main(
        config_name="supervised_finetuning",
        version_base="1.3",
    )


if __name__ == "__main__":
    main()
