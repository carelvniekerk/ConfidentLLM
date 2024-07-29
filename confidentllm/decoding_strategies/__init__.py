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
# limitations under the License."
"""Module to define decoding strategies for the model."""

from hydra_zen import just

from confidentllm.decoding_strategies.greedy import (
    greedy_decoding_strategy,
    greedy_decoding_with_disparity,
)
from confidentllm.decoding_strategies.sampling import (
    sampling_decoding_strategy as SamplingDecodingStrategyConfig,  # noqa: N812
)
from confidentllm.decoding_strategies.types import DecodingStrategy
from confidentllm.hydra_tools import MultiGroupZenStore

__all__ = [
    "DecodingStrategy",
    "greedy_decoding_strategy",
    "greedy_decoding_with_disparity",
    "SamplingDecodingStrategyConfig",
]

decoding_strategy_store = MultiGroupZenStore(
    groups=[
        "generation_method/generator/decoding_strategy",
        "output_processor/generator/decoding_strategy",
    ],
)
decoding_strategy_store(
    just(greedy_decoding_strategy),
    name="greedy",
)
decoding_strategy_store(
    just(greedy_decoding_with_disparity),
    name="greedy_with_disparity",
)
decoding_strategy_store(
    SamplingDecodingStrategyConfig,
    name="sampling",
)
