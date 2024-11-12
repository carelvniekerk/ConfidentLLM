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
"""Trainer for the reward model."""

from pathlib import Path

from hydra_zen import store
from trl import RewardConfig, RewardTrainer

from confidentllm.generation.types import ModelNotSetError, TokenizerNotSetError
from confidentllm.hydra_tools import builds
from confidentllm.train.types import BaseModelTrainer, IntervalStrategy

__all__ = ["RewardModelTrainer"]


class RewardModelTrainer(BaseModelTrainer):
    """Trainer for the reward model."""

    def _get_trainer_config(self) -> RewardConfig:
        config = RewardConfig(
            output_dir=str(Path.cwd()),
            eval_strategy=self.eval_strategy.value,
            eval_steps=self.eval_steps,
            logging_strategy=self.logging_strategy.value,
            logging_steps=self.logging_steps,
            log_level=self.log_level.value,
            report_to=self.report_to,
            save_strategy=self.save_strategy.value,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=self.load_best_model_at_end,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            max_length=self.max_length,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.num_train_epochs,
            seed=self.seed,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            adam_epsilon=self.adam_epsilon,
            max_grad_norm=self.max_grad_norm,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio,
            metric_for_best_model=self.metric_for_best_model,
            label_smoothing_factor=self.label_smoothing_factor,
            bf16=self.bf16,
            fp16=self.fp16,
        )
        return config

    def _set_trainer(self) -> None:
        if self.model is None:
            raise ModelNotSetError(self.model)
        if self.tokenizer is None:
            raise TokenizerNotSetError(self.tokenizer)

        if self.train_dataset is None:
            msg = "The training dataset is not set."
            raise ValueError(msg)
        if self.eval_dataset is None:
            self.eval_strategy = IntervalStrategy.NO
            self.eval_steps = 0

        self.trainer: RewardTrainer = RewardTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=self._get_trainer_config(),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )


RewardModelTrainerConfig = builds(RewardModelTrainer)

default_config = RewardModelTrainerConfig()
store(default_config, group="trainer", name="reward_model")
