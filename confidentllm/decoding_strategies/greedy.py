# coding=utf-8
# --------------------------------------------------------------------------------
# Project: ConfidentLLM
# Author: Carel van Niekerk
# Year: 2024
# Group: Dialogue Systems and Machine Learning Group
# Institution: Heinrich Heine University Düsseldorf
# --------------------------------------------------------------------------------
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
"""Greedy decoding strategy."""

from torch import Tensor, argmax, softmax, topk


def greedy_decoding_strategy(scores: Tensor) -> tuple[Tensor, Tensor]:
    """Greedy decoding strategy."""
    probs = softmax(scores, dim=-1)
    next_token = argmax(probs, dim=-1)
    next_token_prob = probs[range(probs.size(0)), next_token]

    return next_token, next_token_prob


def greedy_decoding_with_disparity(scores: Tensor) -> tuple[Tensor, Tensor]:
    """Greedy decoding strategy with disparity."""
    probs = softmax(scores, dim=-1)
    next_token = argmax(probs, dim=-1)
    next_token_prob = probs[range(probs.size(0)), next_token]

    second_highest_prob = topk(probs, k=2, dim=-1)[0][:, 1]

    disparity = next_token_prob - second_highest_prob

    return next_token, disparity
