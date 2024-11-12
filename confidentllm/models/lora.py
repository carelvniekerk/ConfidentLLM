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
"""LoRA Configuration for Models."""

from enum import StrEnum, auto
from typing import Literal

from hydra_zen import store
from peft import (
    LoraConfig,  # type: ignore[import] # LoraConfig is publically available in peft
)
from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
from peft.utils.peft_types import TaskType
from transformers import PreTrainedModel

from confidentllm.hydra_tools import builds

__all__ = ["LoRAConfig", "NoLoRAConfig"]


class LoRAWeightInitStrategy(StrEnum):
    """Weight initialization strategies for LoRA."""

    DEFAULT = auto()
    PISSA = auto()
    GAUSSIAN = auto()


class LoRAConfig:
    """Configuration for the LoRA model."""

    def __init__(  # noqa: D107, PLR0913
        self,
        *,
        active: bool = False,
        task_type: TaskType = TaskType.CAUSAL_LM,
        inference_mode: bool = False,
        r: int = 8,
        lora_alpha: int = -1,
        lora_dropout: float = 0.1,
        use_rslora: bool = True,
        weight_init_strategy: LoRAWeightInitStrategy = LoRAWeightInitStrategy.DEFAULT,
    ) -> None:
        self.active = active
        self.task_type = task_type
        self.inference_mode = inference_mode
        self.r = r
        self.lora_alpha = lora_alpha if lora_alpha > 0 else 2 * r
        self.lora_dropout = lora_dropout
        self.use_rslora = use_rslora
        self.weight_init_strategy: str | Literal[True] = (
            True
            if weight_init_strategy == LoRAWeightInitStrategy.DEFAULT
            else weight_init_strategy.value
        )

    def _get_lora_config_object(self) -> LoraConfig:
        """Get the LoraConfig object from the LoRAConfig object."""
        return LoraConfig(
            task_type=self.task_type,
            inference_mode=self.inference_mode,
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            use_rslora=self.use_rslora,
            init_lora_weights=self.weight_init_strategy,  # type: ignore[arg-type]
        )

    def get_lora_model(self, model: PreTrainedModel) -> PeftModel | PreTrainedModel:
        """Get the LoRA model."""
        if not self.active:
            return model
        return get_peft_model(model, self._get_lora_config_object())  # type: ignore[return-type]


HydraLoRAConfig = builds(LoRAConfig)
NoLoRAConfig = HydraLoRAConfig(active=False)
CausalLMLoRAConfig = HydraLoRAConfig(active=True, task_type=TaskType.CAUSAL_LM)
SeqClsLoRAConfig = HydraLoRAConfig(active=True, task_type=TaskType.SEQ_CLS)

lora_config_store = store(group="model/lora")
lora_config_store(NoLoRAConfig, name="no_lora")
lora_config_store(CausalLMLoRAConfig, name="causal_lm")
lora_config_store(SeqClsLoRAConfig, name="seq_cls")
