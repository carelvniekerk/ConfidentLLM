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
"""Initialise loss functions module."""

from confidentllm.train.loss_functions.huber_quantile_loss import HuberQuantileLoss
from confidentllm.train.loss_functions.types import ComputeLossFunction, LossFunction
from confidentllm.train.loss_functions.uncertainty_aware_clm import (
    UncertaintyAwareCLMLoss,
)

__all__ = ["LOSS_FUNCTIONS", "ComputeLossFunction", "LossFunction"]


LOSS_FUNCTIONS: dict[LossFunction, ComputeLossFunction | None] = {
    LossFunction.DEFAULT: None,
    LossFunction.UA_CLM: UncertaintyAwareCLMLoss(),  # type: ignore[dict-item] # Uncertainty-aware CLM loss is a ComputeLossFunction
    LossFunction.HUBER_QUANTILE: HuberQuantileLoss(),  # type: ignore[dict-item] # Huber quantile loss is a ComputeLossFunction
}
