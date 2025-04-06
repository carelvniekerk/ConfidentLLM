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
#     http: //www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run the reinforcement learning process."""

from pprint import pformat

from datasets import Dataset
from hydra_zen import store, zen
from transformers.modeling_utils import PreTrainedModel

from confidentllm.models import ModelLoader
from confidentllm.models.model_name_and_type import ModelMode
from confidentllm.scripts.setup_tools import (
    get_logger,
    init_wandb,
    log_system_info,
    set_seed,
    setup_hydra_config_and_logging,
)
from confidentllm.train.types import BaseModelTrainer

__all__ = ["main"]
logger = get_logger()


@store(
    name="reinforcement_learning",
    hydra_defaults=[
        "_self_",
        {"model": "train_causal_lm"},
        {"model/lora": "causal_lm"},
        {"reward_model": "train_sequence_cls"},
        {"reward_model/lora": "sequence_cls"},
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
    trainer: BaseModelTrainer,
) -> None:
    """Run the question answering process."""
    init_wandb(task_name="ReinforcementLearning")
    log_system_info()
    set_seed(trainer.seed)

    value_model, reward_tokenizer = reward_model.load()
    policy_model, policy_tokenizer = model.load()

    if reward_tokenizer.__class__ != policy_tokenizer.__class__:
        raise ValueError("The reward and policy models must use the same tokenizer.")  # noqa: EM101, TRY003

    # Disable Lora setup and set model mode to EVAL to load a reference model version of
    # the model. Further disable gradient computation for the reference model.
    if model.lora.active:
        # peft_config: LoraConfig | None = model.lora._get_lora_config_object()
        model.model_mode = ModelMode.EVAL
        policy_reference_model: PreTrainedModel | None = None
    else:
        # peft_config = None
        model.lora.active = False
        model.model_mode = ModelMode.EVAL
        policy_reference_model, _ = model.load()
        for param in policy_reference_model.parameters():
            param.requires_grad = False

    reward_model.lora.active = False
    reward_model.model_mode = ModelMode.EVAL
    reward_model_instance, _ = reward_model.load()
    for param in reward_model_instance.parameters():
        param.requires_grad = False

    trainer.set_model(policy_model)
    trainer.set_tokenizer(policy_tokenizer)
    trainer.set_reference_model(policy_reference_model)  # type: ignore[attr-defined] # Defined for RL Trainers
    trainer.set_reward_model(reward_model_instance)  # type: ignore[attr-defined]
    trainer.set_value_model(value_model)  # type: ignore[attr-defined]
    trainer.stop_token_id = policy_tokenizer.eos_token_id  # type: ignore[attr-defined]

    train_data = train_data.map(
        function=trainer.preprocessing_function,
        batched=True,
        batch_size=512,
        load_from_cache_file=eval_data.cached_version,  # type: ignore[attr-defined]
        remove_columns=train_data.column_names,
    )
    eval_data = eval_data.map(
        function=trainer.preprocessing_function,
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
        "resolve_target_name:${trainer}",
        "train_data.name",
        "model.pretrained_model_name_or_path",
        "trainer.seed",
    ]

    setup_hydra_config_and_logging(
        job_name="reinforcement_learning",
        add_hpc_launcher=True,
        config_keys=config_keys,
    )

    # Generate the CLI for run_extraction
    run_function.hydra_main(
        config_name="reinforcement_learning",
        version_base="1.3",
    )


if __name__ == "__main__":
    main()
