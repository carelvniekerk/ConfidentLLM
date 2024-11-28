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

from functools import partial
from pprint import pformat

from datasets import Dataset
from hydra_zen import store, zen

from confidentllm import data  # noqa: F401
from confidentllm.models import ModelLoader
from confidentllm.models.model_name_and_type import ModelMode
from confidentllm.scripts.setup_tools import (
    get_logger,
    init_wandb,
    log_system_info,
    set_seed,
    setup_hydra_config_and_logging,
)
from confidentllm.train import PPORLTrainer, prepare_rl_data

__all__ = ["main"]
logger = get_logger()


@store(
    name="reward_ppo_training",
    hydra_defaults=[
        "_self_",
        {"model": "train_causal_lm"},
        {"model/lora": "sequence_cls"},
        {"reward_model": "sequence_cls"},
        {"reward_model/lora": "no_lora"},
        {"train_data": "multiarith"},
        {"eval_data": "multiarith"},
        {"trainer": "ppo"},
    ],
)
def run_training(
    train_data: Dataset,
    eval_data: Dataset,
    model: ModelLoader,
    reward_model: ModelLoader,
    trainer: PPORLTrainer,
) -> None:
    """Run the question answering process."""
    init_wandb()
    log_system_info()
    set_seed(trainer.seed)

    reward_model_instance, reward_tokenizer = reward_model.load()
    policy_model, policy_tokenizer = model.load()

    if reward_tokenizer.__class__ != policy_tokenizer.__class__:
        raise ValueError("The reward and policy models must use the same tokenizer.")  # noqa: EM101, TRY003

    # Disable Lora setup and set model mode to EVAL to load a reference model version of
    # the model. Further disable gradient computation for the reference model.
    model.lora.active = False
    model.model_mode = ModelMode.EVAL
    policy_reference_model, _ = model.load()
    for param in policy_reference_model.parameters():
        param.requires_grad = False

    trainer.set_model(policy_model)
    trainer.set_tokenizer(policy_tokenizer)
    trainer.set_reference_model(
        model=policy_reference_model,
    )
    trainer.set_reward_model(reward_model_instance)

    data_preperation_function = partial(
        prepare_rl_data,
        tokenizer=policy_tokenizer,
        max_length=trainer.max_input_length,
    )
    train_data = train_data.map(
        function=data_preperation_function,
        batched=True,
        batch_size=512,
        load_from_cache_file=eval_data.cached_version,  # type: ignore[attr-defined]
        remove_columns=train_data.column_names,
    )
    eval_data = eval_data.map(
        function=data_preperation_function,
        batched=True,
        batch_size=512,
        load_from_cache_file=eval_data.cached_version,  # type: ignore[attr-defined]
        remove_columns=eval_data.column_names,
    )

    logger.info(f"Training data: {pformat(train_data.info)}")  # noqa: G004
    logger.info(f"Evaluation data: {pformat(eval_data.info)}")  # noqa: G004

    trainer.set_train_dataset(train_data)
    trainer.set_eval_dataset(eval_data)

    trainer.train()


def main() -> None:
    """Run the question answering process."""
    run_function = zen(run_training)

    config_keys = [
        "train_data.name",
        "model.pretrained_model_name_or_path",
        "trainer.seed",
    ]

    setup_hydra_config_and_logging(
        job_name="reward_ppo_training",
        add_hpc_launcher=True,
        config_keys=config_keys,
    )

    # Generate the CLI for run_extraction
    run_function.hydra_main(
        config_name="reward_ppo_training",
        version_base="1.3",
    )


if __name__ == "__main__":
    main()
