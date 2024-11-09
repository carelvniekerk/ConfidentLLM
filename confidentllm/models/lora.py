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

from dataclasses import dataclass

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


@dataclass
class LoRAConfig:
    """Configuration for the LoRA model."""

    active: bool = False
    task_type: TaskType = TaskType.CAUSAL_LM
    inference_mode: bool = False
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1

    def _get_lora_config_object(self) -> LoraConfig:
        """Get the LoraConfig object from the LoRAConfig object."""
        return LoraConfig(
            task_type=self.task_type,
            inference_mode=self.inference_mode,
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
        )

    def get_lora_model(self, model: PreTrainedModel) -> PeftModel | PreTrainedModel:
        """Get the LoRA model."""
        if not self.active:
            return model
        return get_peft_model(model, self._get_lora_config_object())  # type: ignore[return-type]


NoLoRAConfig = LoRAConfig()
CausalLMLoRAConfig = builds(LoRAConfig, active=True)
SeqClsLoRAConfig = builds(LoRAConfig, active=True, task_type=TaskType.SEQ_CLS)

LoRAConfigStore = store(group="models/lora")
LoRAConfigStore(NoLoRAConfig, name="no_lora")
LoRAConfigStore(CausalLMLoRAConfig, name="causal_lm")
LoRAConfigStore(SeqClsLoRAConfig, name="seq_cls")
