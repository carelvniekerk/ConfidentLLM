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
"""Trainer for supervised finetuning."""

from functools import partial
from pathlib import Path
from typing import Callable

import torch
from hydra_zen import store
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer import Trainer
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments

from confidentllm.data import sft_preprocessing
from confidentllm.generation.types import (
    ModelNotSetError,
    TokenizerNotSetError,
)
from confidentllm.hydra_tools import builds
from confidentllm.train.loss_functions import (
    LOSS_FUNCTIONS,
    ComputeLossFunction,
    LossFunction,
)
from confidentllm.train.types import (
    BaseModelTrainer,
    IntervalStrategy,
    LoggingLevel,
)

__all__ = ["SupervisedFinetuningTrainer"]


def _compute_eval_metrics(
    predictions: EvalPrediction,
    *,
    compute_result: bool = False,  # noqa: ARG001
    loss_func: ComputeLossFunction,
) -> dict[str, float]:
    """Compute evaluation metrics.

    Args:
        loss_func (ComputeLossFunction): The loss function to use for computing metrics.
        predictions (EvalPrediction): The predictions from the model.
        compute_result (bool, optional): Whether to compute the result.
                Default is False.

    Returns:
        dict[str, float]: A dictionary containing the computed metrics.

    """
    outputs = SequenceClassifierOutput(
        logits=torch.tensor(predictions.predictions[0], dtype=torch.float32),  # type: ignore[arg-type]
    )

    loss: torch.Tensor = loss_func(
        outputs=outputs,  # type: ignore[arg-type]
        labels=torch.tensor(predictions.label_ids, dtype=torch.float32),  # type: ignore[arg-type]
    )

    return {"loss": loss.item()}


class SupervisedFinetuningTrainer(BaseModelTrainer):
    """Trainer for supervised finetuning."""

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
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        num_train_epochs: float = 3.0,
        seed: int = 42,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.001,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        max_grad_norm: float = 1.0,
        warmup_ratio: float = 0.03,
        metric_for_best_model: str | None = None,
        loss_function: LossFunction = LossFunction.DEFAULT,
        label_smoothing_factor: float = 0.0,
        bf16: bool = False,
        fp16: bool = False,
        max_length: int = 256,
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
                Default is 4.
            per_device_eval_batch_size (int, optional): The batch size for evaluation.
                Default is 8.
            gradient_accumulation_steps (int, optional): The gradient accumulation
                steps. Default is 1.
            num_train_epochs (float, optional): The number of training epochs.
                Default is 3.0.
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
            loss_function (LossFunction, optional): The loss function.
                Default is LossFunction.DEFAULT.
            label_smoothing_factor (float, optional): The label smoothing factor.
                Default is 0.0.
            bf16 (bool, optional): Use bfloat16 precision. Default is False.
            fp16 (bool, optional): Use fp16 precision. Default is False.
            max_length (int, optional): The maximum length of the generated text.
                Default is 256.

        """
        super().__init__(
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            log_level=log_level,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            seed=seed,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            max_grad_norm=max_grad_norm,
            warmup_ratio=warmup_ratio,
            metric_for_best_model=metric_for_best_model,
            loss_function=loss_function,
            label_smoothing_factor=label_smoothing_factor,
            bf16=bf16,
            fp16=fp16,
        )

        self.max_length = max_length

    @property
    def _trainer_config(self) -> TrainingArguments:
        config = TrainingArguments(
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

    @property
    def preprocessing_function(self) -> Callable[[dict], dict]:
        """Preprocessing function for the dataset."""
        if self.tokenizer is None:
            raise TokenizerNotSetError(self.tokenizer)

        return partial(
            sft_preprocessing,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

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

        self.tokenizer.padding_side = "right"

        self.trainer: Trainer = Trainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=self._trainer_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )

        self.trainer.compute_loss_func = LOSS_FUNCTIONS.get(self.loss_function, None)

        # Set the quantiles for quantile regression if applicable
        if self.model.config.model_type == "quantile_regression":
            self.trainer.compute_loss_func.quantiles = torch.tensor(  # type: ignore[attr-defined] # For quantile loss function the quantiles are set
                self.model.config.quantiles,
                dtype=torch.float32,
            )

        # Add the loss as an evaluation metric if a loss function is set
        if self.trainer.compute_loss_func is not None:
            self.trainer.compute_metrics = partial(
                _compute_eval_metrics,
                loss_func=self.trainer.compute_loss_func,
            )
            self.trainer.label_names = ["labels"]


SupervisedFinetuningTrainerConfig = builds(SupervisedFinetuningTrainer)

default_config = SupervisedFinetuningTrainerConfig()
store(default_config, group="trainer", name="supervised_finetuning")
