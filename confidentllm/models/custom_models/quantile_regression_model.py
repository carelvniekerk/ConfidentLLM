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
"""LLM model for quantile regression."""

import torch
from torch.nn import Linear
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModel

from confidentllm.models.custom_models.quantile_regression_config import (
    QuantileRegressionConfig,
)

__all__: list[str] = ["PreTrainedModelForQuantileRegression"]


class PreTrainedModelForQuantileRegression(PreTrainedModel):
    """Base class for quantile regression models."""

    config_class = QuantileRegressionConfig

    def __init__(self, *inputs, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Initialize the quantile regression model."""
        super().__init__(*inputs, **kwargs)
        self.base_pretrained_model: PreTrainedModel = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.config.base_model_name_or_path,
        )  # type: ignore[misc] # Overwriting the base model with a specific one

        # TODO: Add support for base model fine tuning via LoRA.
        if self.config.freeze_base_model:
            for param in self.base_pretrained_model.parameters():
                param.requires_grad = False

        self.regressor = Linear(
            in_features=self.base_pretrained_model.config.hidden_size,
            out_features=len(self.config.quantiles),
        )
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,  # noqa: ANN003
    ) -> SequenceClassifierOutput:
        """Forward pass for the quantile regression model.

        Args:
        ----
            input_ids: Input token IDs.
            attention_mask: Attention mask for the input tokens.
            **kwargs: Additional keyword arguments for the base model.

        Returns:
        -------
            BaseModelOutput: The quantile regression outputs.

        """
        outputs: BaseModelOutput = self.base_pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        quantile_outputs: torch.FloatTensor = self.regressor(outputs.hidden_states[-1])  # type: ignore[index] # Hidden states is not None when output_hidden_states=True

        return SequenceClassifierOutput(
            logits=quantile_outputs,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
