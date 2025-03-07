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
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module to load pretrained models and tokenizers from Hugging Face's model hub."""

import logging
import os
from functools import partial
from pathlib import Path
from typing import Protocol

import torch
from dotenv import load_dotenv
from hydra_zen import store
from hydra_zen.third_party.pydantic import pydantic_parser
from peft.auto import AutoPeftModelForCausalLM, AutoPeftModelForSequenceClassification
from peft.mixed_model import PeftMixedModel
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from confidentllm.hydra_tools import builds
from confidentllm.models.api_models.openai import ChatGPTModel
from confidentllm.models.api_models.vertexai import GeminiModel
from confidentllm.models.configuration import (
    get_chat_template,
    get_pretrained_model_name_or_path,
)
from confidentllm.models.lora import LoRAConfig, NoLoRAConfig
from confidentllm.models.model_name_and_type import (
    ModelDataTypes,
    ModelDevice,
    ModelMode,
    ModelName,
    ModelType,
)

__all__ = ["ModelLoader"]


# Default device to load the model on (Always use MPS or CUDA if available)
DEFAULT_DEVICE: ModelDevice = (
    ModelDevice.MPS if torch.backends.mps.is_available() else ModelDevice.CPU
)
DEFAULT_DEVICE = ModelDevice.CUDA if torch.cuda.is_available() else DEFAULT_DEVICE

API_MODELS: list[ModelType] = [ModelType.OPENAI, ModelType.VERTEXAI]


class ModelLoaderFunction(Protocol):
    def __call__(
        self,
        pretrained_model_name_or_path: str | Path,
    ) -> PreTrainedModel: ...


class ModelLoader:
    """Abstract class to load a pretrained model and tokenizer."""

    def __init__(  # noqa: PLR0913
        self,
        pretrained_model_name_or_path: ModelName | Path,
        *,
        model_type: ModelType,
        device: ModelDevice = DEFAULT_DEVICE,
        data_type: ModelDataTypes = ModelDataTypes.BFLOAT16,
        model_mode: ModelMode = ModelMode.EVAL,
        lora: LoRAConfig = NoLoRAConfig,  # type: ignore[assignment]
    ) -> None:
        """Initialize the model loader."""
        if (
            isinstance(pretrained_model_name_or_path, Path)
            and not pretrained_model_name_or_path.exists()
        ):
            if pretrained_model_name_or_path.name not in ModelName.__members__:
                msg = (
                    f"The specified path {pretrained_model_name_or_path} does not "
                    "exist and is not a valid model name."
                )
                raise FileNotFoundError(msg)
            pretrained_model_name_or_path = ModelName[
                pretrained_model_name_or_path.name
            ]

        self.pretrained_model_name_or_path: str | Path = (
            get_pretrained_model_name_or_path(
                pretrained_model_name_or_path,
            )
        )
        self.device: torch.device = torch.device(device.value)
        self.chat_template: str | None = get_chat_template(
            pretrained_model_name_or_path,  # type: ignore[arg-type]
        )
        self.data_type: torch.dtype = self._get_dtype(data_type)
        self.model_type: ModelType = model_type
        self.model_mode: ModelMode = model_mode
        self.lora: LoRAConfig = lora

        self.use_peft_model_class: bool = False
        self._get_model_class()

    @staticmethod
    def _get_dtype(data_type: ModelDataTypes) -> torch.dtype:
        """Get the torch data type."""
        if data_type == ModelDataTypes.BFLOAT16:
            dtype: torch.dtype = torch.bfloat16
        elif data_type == ModelDataTypes.FLOAT16:
            dtype = torch.float16
        elif data_type == ModelDataTypes.FLOAT32:
            dtype = torch.float32
        elif data_type == ModelDataTypes.FLOAT64:
            dtype = torch.float64
        else:
            raise ValueError(f"Invalid data type: {data_type}")  # noqa: EM102, TRY003

        return dtype

    def _get_model_class(self) -> None:
        """Get the model class."""
        if (
            isinstance(self.pretrained_model_name_or_path, Path)
            and (self.pretrained_model_name_or_path / "adapter_config.json").exists()
        ):
            self.use_peft_model_class = True

        if self.model_type == ModelType.CAUSAL_LM:
            self.model_class: AutoModel = (
                AutoModelForCausalLM
                if not self.use_peft_model_class
                else AutoPeftModelForCausalLM
            )  # type: ignore[assignment] # All auto models are of type AutoModel
        elif self.model_type == ModelType.SEQUENCE_CLS:
            self.model_class = (
                AutoModelForSequenceClassification
                if not self.use_peft_model_class
                else AutoPeftModelForSequenceClassification
            )  # type: ignore[assignment]
        elif self.model_type == ModelType.OPENAI:
            self.model_class = ChatGPTModel  # type: ignore[assignment]
        elif self.model_type == ModelType.VERTEXAI:
            self.model_class = GeminiModel  # type: ignore[assignment]
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")  # noqa: EM102, TRY003

        if self.model_type not in API_MODELS:
            self.model_loader: ModelLoaderFunction = partial(
                self.model_class.from_pretrained,
                device_map=self.device,
                torch_dtype=self.data_type,
            )  # type: ignore[assignment] # Paths can also be passed to from_pretrained

        if self.model_type == ModelType.SEQUENCE_CLS:
            self.model_loader = partial(self.model_loader, num_labels=1)  # type: ignore[call-arg]

    def _log_model_info(self, model: PeftMixedModel | PreTrainedModel) -> None:
        """Log model information and a list of all trainable parameters."""
        logger = logging.getLogger()
        logger.info(f"Model Summary:\n{model}")  # noqa: G004

        # List all trainable parameters (parameters with requires_grad=True)
        trainable_params = [
            name for name, param in model.named_parameters() if param.requires_grad
        ]

        if self.model_mode != ModelMode.TRAIN:
            return
        # Log the trainable parameters list
        logger.info("Trainable Parameters:")
        for param_name in trainable_params:
            logger.info(f" - {param_name}")  # noqa: G004

    def load(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load a pretrained model from Hugging Face's model hub.

        Args:
        ----
            pretrained_model_name_or_path (str): The name of or the path to the model.
            device (str): The device to load the model on.

        Returns:
        -------
            Module: The model loaded on the specified device.

        """
        if self.model_type in API_MODELS:
            load_dotenv()
            api_key_key: str = f"{self.model_type.name}_API_KEY"
            model: PeftMixedModel | PreTrainedModel = self.model_class(
                model_name=self.pretrained_model_name_or_path,
                api_key=os.getenv(api_key_key),
            )  # type: ignore[call-arg]
        else:
            model = self.model_loader(
                pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            )

        if self.model_type in API_MODELS:
            self.lora.inference_mode = True
        elif self.model_mode == ModelMode.TRAIN:
            model.train()
            self.lora.inference_mode = False
        elif self.model_mode == ModelMode.EVAL:
            model.eval()
            self.lora.inference_mode = True
        else:
            raise ValueError(f"Invalid model mode: {self.model_mode}")  # noqa: EM102, TRY003

        model = (
            self.lora.get_lora_model(model)
            if self.model_type not in API_MODELS
            else model
        )

        # During eval perform a merge and unload operation to incorporate the LoRA
        # adapters and avoid extra memory and computation overhead
        if (
            self.lora.active or self.use_peft_model_class
        ) and self.model_mode == ModelMode.EVAL:
            model = model.merge_and_unload()  # type: ignore[assignment]

        if self.model_type in API_MODELS:
            tokenizer: PreTrainedTokenizer = None  # type: ignore[assignment]
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_model_name_or_path,  # type: ignore[assignment]
                clean_up_tokenization_spaces=True,
                padding_side="left",
            )

        if self.chat_template:
            tokenizer.chat_template = self.chat_template

        if tokenizer is not None and tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token

        if tokenizer is not None and model.config.pad_token_id is None:  # type: ignore[attr-defined]
            model.config.pad_token_id = tokenizer.pad_token_id  # type: ignore[attr-defined]
            model.config.pad_token = tokenizer.pad_token  # type: ignore[attr-defined]

        if self.model_type not in API_MODELS:
            self._log_model_info(model)
            model = model.to(self.device)  # type: ignore[arg-type]

        return model, tokenizer  # type: ignore[arg-type]


# Add default model loader to the store
ModelLoaderConfig = builds(ModelLoader, zen_wrappers=[pydantic_parser])
CausalLMModelConfig = ModelLoaderConfig(
    pretrained_model_name_or_path=ModelName.GPT2_137M,
    model_type=ModelType.CAUSAL_LM,
    lora=NoLoRAConfig,  # type: ignore[arg-type]
)

TrainCausalLMModelConfig = ModelLoaderConfig(
    pretrained_model_name_or_path=ModelName.GPT2_137M,
    model_type=ModelType.CAUSAL_LM,
    model_mode=ModelMode.TRAIN,
    lora=NoLoRAConfig,  # type: ignore[arg-type]
)

TrainSequenceCLSModelConfig = ModelLoaderConfig(
    pretrained_model_name_or_path=ModelName.GPT2_137M,
    model_type=ModelType.SEQUENCE_CLS,
    model_mode=ModelMode.TRAIN,
    lora=NoLoRAConfig,  # type: ignore[arg-type]
)

SequenceCLSModelConfig = ModelLoaderConfig(
    pretrained_model_name_or_path=ModelName.GPT2_137M,
    model_type=ModelType.SEQUENCE_CLS,
    model_mode=ModelMode.EVAL,
    lora=NoLoRAConfig,  # type: ignore[arg-type]
)

OpenAIModelConfig = ModelLoaderConfig(
    pretrained_model_name_or_path=ModelName.GPT4O_MINI,
    model_type=ModelType.OPENAI,
)

VertexAIModelConfig = ModelLoaderConfig(
    pretrained_model_name_or_path=ModelName.GEMINI15_FLASH,
    model_type=ModelType.VERTEXAI,
)

store(CausalLMModelConfig, name="causal_lm", group="model")
store(TrainCausalLMModelConfig, name="train_causal_lm", group="model")
store(TrainSequenceCLSModelConfig, name="train_sequence_cls", group="reward_model")
store(SequenceCLSModelConfig, name="sequence_cls", group="reward_model")
store(TrainSequenceCLSModelConfig, name="train_sequence_cls", group="model")
store(SequenceCLSModelConfig, name="sequence_cls", group="model")
store(VertexAIModelConfig, name="vertex_ai", group="model")
store(OpenAIModelConfig, name="openai", group="model")
