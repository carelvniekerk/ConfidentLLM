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
"""Trainer for PPO RL Finetuning."""

import logging
from functools import partial
from pathlib import Path
from typing import Callable

from hydra_zen import store
from transformers.modeling_utils import PreTrainedModel
from trl import PPOConfig, PPOTrainer

from confidentllm.data import rl_preprocessing
from confidentllm.generation.types import ModelNotSetError, TokenizerNotSetError
from confidentllm.hydra_tools import builds
from confidentllm.train.types import BaseModelTrainer, IntervalStrategy, LoggingLevel

__all__ = ["PPORLTrainer"]

MIN_NORMALIZATION_BATCH_SIZE: int = 8


class PPORLTrainer(BaseModelTrainer):
    """Trainer for PPO RL Finetuning."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        eval_strategy: IntervalStrategy = IntervalStrategy.EPOCH,
        eval_steps: int = 1,
        log_level: LoggingLevel = LoggingLevel.INFO,
        save_strategy: IntervalStrategy = IntervalStrategy.EPOCH,
        save_steps: int = 1,
        save_total_limit: int | None = None,
        load_best_model_at_end: bool = False,
        per_device_train_batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        num_train_epochs: float = 3.0,
        normalize_rewards: bool = True,
        kl_coefficient: float = 0.05,
        ppo_clipping_range: float = 0.2,
        discount_factor: float = 1.0,
        gae_lambda: float = 0.95,
        stop_token_id: int | None = None,
        temperature: float = 0.7,
        response_length: int = 256,
        max_input_length: int = 256,
        seed: int = 42,
        learning_rate: float = 3e-6,
        weight_decay: float = 0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        max_grad_norm: float = 1.0,
        warmup_ratio: float = 0.0,
        bf16: bool = False,
        fp16: bool = False,
    ) -> None:
        """Configure the model trainer.

        Args:
        ----
            eval_strategy (IntervalStrategy, optional): The evaluation strategy.
                Default is IntervalStrategy.EPOCH.
            eval_steps (int, optional): The evaluation steps. Default is 1.
            log_level (LoggingLevel, optional): The logging level.
                Default is LoggingLevel.INFO.
            save_strategy (IntervalStrategy, optional): The saving strategy.
                Default is IntervalStrategy.EPOCH.
            save_steps (int, optional): The saving steps. Default is 1.
            save_total_limit (int, optional): The total limit for saving.
                Default is None.
            load_best_model_at_end (bool, optional): Load the best model at the end.
                Default is False.
            per_device_train_batch_size (int, optional): The batch size for training.
                Default is 8.
            gradient_accumulation_steps (int, optional): The gradient accumulation steps.
                Default is 4.
            num_train_epochs (float, optional): The number of training epochs.
                Default is 3.0.
            normalize_rewards (bool, optional): Whether to normalize rewards.
                Default is True.
            kl_coefficient (float, optional): The coefficient for KL divergence.
                Default is 0.05.
            ppo_clipping_range (float, optional): The clipping range for PPO.
                Default is 0.2.
            discount_factor (float, optional): The discount factor for future rewards.
                Default is 1.0.
            gae_lambda (float, optional): The lambda parameter for Generalized Advantage
                Estimation. Default is 0.95.
            stop_token_id (int, optional): The token ID to stop generation.
                Default is None.
            temperature (float, optional): The temperature for sampling.
                Default is 0.7.
            response_length (int, optional): The maximum length of the response.
                Default is 256.
            max_input_length (int, optional): The maximum length of the input.
                Default is 256.
            seed (int, optional): The seed for reproducibility. Default is 42.
            learning_rate (float, optional): The learning rate for optimization.
                Default is 5e-5.
            weight_decay (float, optional): The weight decay for optimization.
                Default is 0.
            adam_beta1 (float, optional): The beta1 for Adam. Default is 0.9.
            adam_beta2 (float, optional): The beta2 for Adam. Default is 0.999.
            adam_epsilon (float, optional): The epsilon for Adam. Default is 1e-8.
            max_grad_norm (float, optional): The maximum gradient norm. Default is 1.0.
            warmup_ratio (float, optional): The warmup ratio for the learning rate
                scheduler. Default is 0.0.
            metric_for_best_model (str, optional): The metric for the best model.
                Default is None.
            label_smoothing_factor (float, optional): The label smoothing factor.
                Default is 0.0.
            bf16 (bool, optional): Use bfloat16 precision. Default is False.
            fp16 (bool, optional): Use fp16 precision. Default is False.
            max_length (int, optional): The maximum length of the generated text.
                Default is 256.
            use_preference_margin (bool, optional): Use the preference margin.
                Default is True.
            center_rewards_coefficient (float, optional): The coefficient for centring
                the rewards. Default is None.

        """  # noqa: E501
        super().__init__(
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            log_level=log_level,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=0,
            gradient_accumulation_steps=0,
            num_train_epochs=num_train_epochs,
            seed=seed,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            max_grad_norm=max_grad_norm,
            warmup_ratio=warmup_ratio,
            metric_for_best_model=None,
            label_smoothing_factor=0,
            bf16=bf16,
            fp16=fp16,
        )

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.normalize_rewards = normalize_rewards
        self.kl_coefficient = kl_coefficient
        self.ppo_clipping_range = ppo_clipping_range
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.stop_token_id = stop_token_id
        self.temperature = temperature
        self.response_length = response_length
        self.max_input_length = max_input_length

        if (
            self.normalize_rewards
            and self.per_device_train_batch_size < MIN_NORMALIZATION_BATCH_SIZE
        ):
            self.normalize_rewards = False
            msg: str = (
                "Normalizing rewards requires a batch size of at least "
                f"{MIN_NORMALIZATION_BATCH_SIZE}. Setting normalize_rewards to False."
            )
            logging.warning(msg)

    @property
    def _trainer_config(self) -> PPOConfig:
        config = PPOConfig(
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
            num_ppo_epochs=int(self.num_train_epochs),
            num_train_epochs=self.num_train_epochs,
            whiten_rewards=self.normalize_rewards,
            kl_coef=self.kl_coefficient,
            cliprange=self.ppo_clipping_range,
            gamma=self.discount_factor,
            lam=self.gae_lambda,
            stop_token_id=self.stop_token_id,
            temperature=self.temperature,
            response_length=self.response_length,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            seed=self.seed,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            adam_epsilon=self.adam_epsilon,
            max_grad_norm=self.max_grad_norm,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio,
            bf16=self.bf16,
            fp16=self.fp16,
        )
        return config

    @property
    def preprocessing_function(self) -> Callable[[dict], dict]:
        """Return the preprocessing function for the dataset."""
        if self.tokenizer is None:
            raise TokenizerNotSetError(self.tokenizer)

        return partial(
            rl_preprocessing,
            tokenizer=self.tokenizer,
            max_length=self.max_input_length,
        )

    def set_reference_model(
        self,
        model: PreTrainedModel | None,
    ) -> None:
        """Set the model for reference."""
        self.reference_model: PreTrainedModel | None = model

    def set_reward_model(self, model: PreTrainedModel) -> None:
        """Set the reward model."""
        self.reward_model = model

    def set_value_model(self, model: PreTrainedModel) -> None:
        """Set the value model."""
        self.value_model = model

    def _set_trainer(self) -> None:
        if self.model is None:
            raise ModelNotSetError(self.model)
        if self.reward_model is None:
            raise ModelNotSetError(self.reward_model)
        if self.value_model is None:
            raise ModelNotSetError(self.value_model)
        if self.tokenizer is None:
            raise TokenizerNotSetError(self.tokenizer)

        if self.train_dataset is None:
            msg = "The training dataset is not set."
            raise ValueError(msg)
        if self.eval_dataset is None:
            self.eval_strategy = IntervalStrategy.NO
            self.eval_steps = 0

        self.trainer: PPOTrainer = PPOTrainer(
            model=self.model,
            ref_model=self.reference_model,
            processing_class=self.tokenizer,
            args=self._trainer_config,
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )


PPOTrainerConfig = builds(PPORLTrainer)

default_config = PPOTrainerConfig()
store(default_config, group="trainer", name="ppo")
