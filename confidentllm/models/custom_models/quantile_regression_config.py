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
"""LLM model for quantile regression configuration."""

from transformers.configuration_utils import PretrainedConfig

from confidentllm.models.custom_models.types import ActivationFunction

__all__: list[str] = ["QuantileRegressionConfig"]


class QuantileRegressionConfig(PretrainedConfig):
    """Configuration for a quantile regression model."""

    model_type = "quantile_regression"

    def __init__(
        self,
        base_model_name_or_path: str = "",
        hidden_size: int = 128,
        quantiles: list[float] | None = None,
        output_activation_function: ActivationFunction = ActivationFunction.SIGMOID,
        *,
        freeze_base_model: bool = True,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Initialize the configuration for a quantile regression model.

        Args:
        ----
            base_model_name_or_path: The base model name or path.
            hidden_size: The size of the hidden layers in the model.
                    Defaults to 128.
            quantiles: A list of quantiles to predict.
                    Defaults to [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95].
            output_activation_function: The activation function for the output layer.
                    Defaults to ActivationFunction.SIGMOID.
            freeze_base_model: Whether to freeze the base model parameters.
                    Defaults to True.
            **kwargs: Additional keyword arguments for the configuration.

        Raises:
        ------
            TypeError: If `quantiles` is not a list or contains non-float elements.
            ValueError: If any quantile is not in the range (0, 1).

        """  # noqa: E501
        super().__init__(**kwargs)

        self.base_model_name_or_path: str = base_model_name_or_path
        self.hidden_size: int = hidden_size
        self.output_activation_function: str = output_activation_function.name
        self.freeze_base_model: bool = freeze_base_model

        if quantiles is None:
            quantiles = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        if not isinstance(quantiles, list):
            raise TypeError(f"Expected a list of quantiles, got {type(quantiles)}")  # noqa: EM102, TRY003
        if not all(isinstance(q, float) for q in quantiles):
            raise TypeError("All quantiles must be of type float")  # noqa: EM101, TRY003
        if not all(0 < q < 1 for q in quantiles):
            raise ValueError("All quantiles must be in the range (0, 1)")  # noqa: EM101, TRY003
        self.quantiles: list[float] = quantiles
