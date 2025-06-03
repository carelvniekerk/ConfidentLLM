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
"""Custom model classes."""

from transformers import AutoConfig, AutoModel

from confidentllm.models.custom_models.quantile_regression_config import (
    QuantileRegressionConfig,
)
from confidentllm.models.custom_models.quantile_regression_model import (
    PreTrainedModelForQuantileRegression,
)

__all__: list[str] = []

AutoConfig.register(model_type="quantile_regression", config=QuantileRegressionConfig)
AutoModel.register(
    config_class=QuantileRegressionConfig,
    model_class=PreTrainedModelForQuantileRegression,
)
