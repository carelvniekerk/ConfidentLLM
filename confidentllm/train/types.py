# coding=utf-8
# --------------------------------------------------------------------------------
# Project: ConfidentLLM
# Author: Carel van Niekerk, Renato Vukovic
# Year: 202
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
"""Types used in the training process."""

from abc import ABC, abstractmethod
from enum import StrEnum, auto
from typing import Callable

import torch
from datasets import Dataset
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer import Trainer
from transformers.trainer_utils import SchedulerType
from transformers.training_args import TrainingArguments

from confidentllm.train.loss_functions import LossFunction

__all__ = [
    "BaseModelTrainer",
    "IntervalStrategy",
    "LoggingLevel",
]


class IntervalStrategy(StrEnum):
    """Enum for the interval strategies."""

    NO = auto()
    STEPS = auto()
    EPOCH = auto()


class LoggingLevel(StrEnum):
    """Enum for the logging level."""

    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class BaseModelTrainer(ABC):
    """Protocol for the generation method."""

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
        learning_rate: float = 5e-5,
        weight_decay: float = 0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        max_grad_norm: float = 1.0,
        warmup_ratio: float = 0.0,
        metric_for_best_model: str | None = None,
        loss_function: LossFunction = LossFunction.DEFAULT,
        label_smoothing_factor: float = 0.0,
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
            loss_function (LossFunction, optional): The loss function for the model.
                Default is LossFunction.DEFAULT.
            label_smoothing_factor (float, optional): The label smoothing factor.
                Default is 0.0.
            bf16 (bool, optional): Use bfloat16 precision. Default is False.
            fp16 (bool, optional): Use fp16 precision. Default is False.

        """
        self.tokenizer: PreTrainedTokenizer | None = None
        self.model: PreTrainedModel | None = None

        # Set the logging, evaluation and saving strategies
        self.eval_strategy = eval_strategy
        self.eval_steps = eval_steps
        self.logging_strategy = eval_strategy
        self.logging_steps = eval_steps
        self.log_level = log_level
        self.report_to = ["wandb"]
        self.save_strategy = save_strategy
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.load_best_model_at_end = load_best_model_at_end

        # Set the training arguments
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.seed = seed

        # Set the optimization arguments
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm

        # Set the learning rate scheduler arguments
        self.lr_scheduler_type = SchedulerType.LINEAR
        self.warmup_ratio = warmup_ratio

        # Loss and evaluation metrics
        self.metric_for_best_model = metric_for_best_model
        self.loss_function = loss_function
        self.label_smoothing_factor = label_smoothing_factor

        # Set the mixed precision arguments
        self.bf16 = bf16
        self.fp16 = fp16 if not bf16 else False

        # Set placeholders for the data and trainer
        self.train_dataset: Dataset | None = None
        self.eval_dataset: Dataset | None = None
        self.trainer: Trainer | None = None

    def set_tokenizer(self, tokenizer: PreTrainedTokenizer) -> None:
        """Set the tokenizer for the generation method."""
        self.tokenizer = tokenizer

    def set_model(self, model: PreTrainedModel) -> None:
        """Set the model for the generation method."""
        self.model = model
        if self.model.dtype == torch.bfloat16:
            self.bf16 = True
        elif self.model.dtype == torch.float16:
            self.fp16 = True

    def set_train_dataset(self, train_dataset: Dataset) -> None:
        """Set the training dataset."""
        self.train_dataset = train_dataset

    def set_eval_dataset(self, eval_dataset: Dataset) -> None:
        """Set the evaluation dataset."""
        self.eval_dataset = eval_dataset

    @property
    @abstractmethod
    def _trainer_config(self) -> TrainingArguments:
        """Get the training arguments for the trainer."""
        ...

    @property
    @abstractmethod
    def preprocessing_function(self) -> Callable[[dict], dict]:
        """Get the preprocessing function for the trainer."""
        ...

    @abstractmethod
    def _set_trainer(self) -> None:
        """Set the trainer for the model."""
        ...

    def train(self) -> None:
        """Train the model."""
        self._set_trainer()
        self.trainer.train()  # type: ignore[union-attr]
        self.trainer.save_model()  # type: ignore[union-attr]
