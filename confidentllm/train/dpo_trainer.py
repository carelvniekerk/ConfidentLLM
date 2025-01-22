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
"""Trainer for DPO finetuning."""

from enum import StrEnum, auto
from pathlib import Path

from hydra_zen import store
from trl import DPOConfig, FDivergenceType
from trl import DPOTrainer as DPOBaseTrainer

from confidentllm.generation.types import ModelNotSetError, TokenizerNotSetError
from confidentllm.hydra_tools import builds
from confidentllm.train.types import BaseModelTrainer, IntervalStrategy, LoggingLevel

__all__ = ["DPOTrainer"]


class LossFunction(StrEnum):
    """Loss type for DPO Training."""

    SIGMOID = auto()  # sigmoid loss from the original
    # [DPO](https://huggingface.co/papers/2305.18290) paper.
    HINGE = auto()  # hinge loss on the normalized likelihood from the
    # [SLiC](https://huggingface.co/papers/2305.10425) paper.
    IPO = auto()  # IPO loss from the
    # [IPO](https://huggingface.co/papers/2310.12036) paper.
    EXO_PAIR = auto()  # pairwise EXO loss from the
    # [EXO](https://huggingface.co/papers/2402.00856) paper.
    NCA_PAIR = auto()  # pairwise NCA loss from the
    # [NCA](https://huggingface.co/papers/2402.05369) paper.
    ROBUST = auto()  # unbiased estimate of the DPO loss that is robust to preference
    # noise from the [Robust DPO](https://huggingface.co/papers/2403.00409) paper.
    BCO_PAIR = auto()  # pairwise BCO loss from the
    # [BCO](https://huggingface.co/papers/2404.04656) paper.
    SPPO_HARD = auto()  # SPPO loss with hard label from the
    # [SPPO](https://huggingface.co/papers/2405.00675) paper.
    AOT = auto()  # AOT loss for paired datasets from the
    # [AOT](https://huggingface.co/papers/2406.05882) paper.
    AOT_PAIR = auto()  # AOT loss for unpaired datasets from the
    # [AOT](https://huggingface.co/papers/2406.05882) paper.
    APO_ZERO = auto()  # APO-zero loss from the
    # [APO](https://huggingface.co/papers/2408.06266) paper.
    APO_DOWN = auto()  # APO-down loss from the
    # [APO](https://huggingface.co/papers/2408.06266) paper.


class DPOTrainer(BaseModelTrainer):
    """Trainer for the reward model."""

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
        learning_rate: float = 1e-6,
        weight_decay: float = 0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        max_grad_norm: float = 1.0,
        warmup_ratio: float = 0.0,
        metric_for_best_model: str | None = None,
        label_smoothing_factor: float = 0.0,
        bf16: bool = False,
        fp16: bool = False,
        max_length: int = 256,
        max_prompt_length: int = 128,
        loss_beta: float = 0.2,
        loss_function: LossFunction = LossFunction.SIGMOID,
        use_weighting: bool = False,
        divergence_alpha_coefficient: float = 1.0,
        update_ref_model: bool = False,
        ref_model_mixup_alpha: float = 0.9,
        ref_model_update_steps: int = 64,
        rpo_alpha: float | None = None,
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
            label_smoothing_factor (float, optional): Robust DPO label smoothing
                parameter from the [cDPO](https://ericmitchell.ai/cdpo.pdf) report and
                [Robust DPO](https://huggingface.co/papers/2403.00409) paper that
                should be between `0.0` and `0.5`. Default is 0.0.
            bf16 (bool, optional): Use bfloat16 precision. Default is False.
            fp16 (bool, optional): Use fp16 precision. Default is False.
            max_length (int, optional): The maximum length of the generated text.
                Default is 256.
            max_prompt_length (int, optional): The maximum length of the prompt.
                Default is 128.
            loss_beta (float, optional): Parameter controlling the deviation from the
                reference model. Higher β means less deviation from the reference model.
                For the IPO loss (`loss_type="ipo"`), β is the regularization parameter
                denoted by τ in the [paper](https://huggingface.co/papers/2310.12036).
            loss_function (LossFunction, optional): The loss type for DPO training.
            use_weighting (bool, optional): Whether or not to weight the loss as done in
                the [WPO](https://huggingface.co/papers/2406.11827) paper.
                Default is False.
            divergence_alpha_coefficient (float, optional): α coefficient in the
                α-divergence u^-α regularization function for DPO loss.
                Default is 1.0.
            update_ref_model (bool, optional): When set to `True`, the reference model
                is synchronized with the active model every `ref_model_sync_steps`
                steps, using the `ref_model_mixup_alpha` parameter. This synchronization
                originites from the [TR-DPO](https://huggingface.co/papers/2404.09656)
                paper. Default is False.
            ref_model_mixup_alpha (float, optional): α parameter from the
                [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which controls
                the mix between the current policy and the previous reference policy
                during updates. The reference policy is updated according to the
                equation: `π_ref = α * π_θ + (1 - α) * π_ref_prev` To use this parameter
                , you must set `sync_ref_model=True`. Default is 0.9.
            ref_model_update_steps (int, optional): τ parameter from the
                [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which
                determines how frequently the current policy is synchronized with the
                reference policy. To use this parameter, you must set
                `sync_ref_model=True`. Default is 64.
            rpo_alpha (float, optional): α parameter from the
                [RPO](https://huggingface.co/papers/2404.19733) paper (v3), which
                controls the weighting of the NLL term in the loss. If `None`, no
                weighting is applied and the loss is the same as the DPO loss. The
                paper recommends `rpo_alpha=1.0`. Default is None.

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
            label_smoothing_factor=label_smoothing_factor,
            bf16=bf16,
            fp16=fp16,
        )

        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.loss_beta = loss_beta
        self.loss_function = loss_function  # type: ignore[assignment] # DPO has its own selection of loss funtions
        self.use_weighting = use_weighting
        self.divergence_type = FDivergenceType.REVERSE_KL
        self.divergence_alpha_coefficient = divergence_alpha_coefficient
        self.update_ref_model = update_ref_model
        self.ref_model_mixup_alpha = ref_model_mixup_alpha
        self.ref_model_update_steps = ref_model_update_steps
        self.rpo_alpha = rpo_alpha

    @property
    def _trainer_config(self) -> DPOConfig:
        config = DPOConfig(
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
            max_length=self.max_length,
            max_prompt_length=self.max_prompt_length,
            max_completion_length=self.max_length - self.max_prompt_length,
            beta=self.loss_beta,
            loss_type=self.loss_function.value,  # type: ignore[arg-type]
            use_weighting=self.use_weighting,
            f_divergence_type=self.divergence_type,
            f_alpha_divergence_coef=self.divergence_alpha_coefficient,
            sync_ref_model=self.update_ref_model,
            ref_model_mixup_alpha=self.ref_model_mixup_alpha,
            ref_model_sync_steps=self.ref_model_update_steps,
            rpo_alpha=self.rpo_alpha,
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

        self.trainer: DPOBaseTrainer = DPOBaseTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=self._trainer_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )


DPOTrainerConfig = builds(DPOTrainer)

default_config = DPOTrainerConfig()
store(default_config, group="trainer", name="dpo")
