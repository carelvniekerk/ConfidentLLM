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
"""Sampling decoding strategy."""

import torch
from hydra_zen import make_custom_builds_fn, store

from confidentllm.decoding_strategies.types import (
    DecodingStrategyOutput,
)

__all__ = ["SamplingDecodingStrategy"]
builds = make_custom_builds_fn(populate_full_signature=True)


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.

    Args:
    ----
        logits: logits distribution shape (batch size, vocabulary size)
        top_k: if top_k > 0 then keep only top k tokens with highest probability (top-k filtering).
        top_p: if top_p < 1.0 then keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        filter_value: a float value to assign to tokens with probability < top_p
        min_tokens_to_keep: minimum number of tokens to keep in the output

    Returns:
    -------
        logits: the filtered logits

    """  # noqa: E501
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        # (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because
            # we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1,
            sorted_indices,
            sorted_indices_to_remove,
        )
        logits[indices_to_remove] = filter_value

    return logits


class SamplingDecodingStrategy:
    """Sampling decoding strategy."""

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ) -> None:
        """Initialize the sampling decoding strategy."""
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def sampling_decoding_strategy(
        self,
        scores: torch.Tensor,
    ) -> DecodingStrategyOutput:
        """Decode the next token."""
        scores = scores / self.temperature
        logits = top_k_top_p_filtering(
            scores,
            top_k=self.top_k,
            top_p=self.top_p,
            filter_value=self.filter_value,
            min_tokens_to_keep=self.min_tokens_to_keep,
        )

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        next_token_prob = probs[range(probs.size(0)), next_token]

        return DecodingStrategyOutput(
            next_token=next_token,
            next_token_score=next_token_prob,
        )


SamplingDecodingStrategyConfig = builds(SamplingDecodingStrategy)

sampling_decoding_strategy = SamplingDecodingStrategyConfig(
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    filter_value=-float("Inf"),
    min_tokens_to_keep=1,
)

decoding_strategy_store = [
    store(group="generation_method/generator/decoding_strategy"),
    store(group="output_processor/generator/decoding_strategy"),
]
[
    store_fn(sampling_decoding_strategy, name="sampling")
    for store_fn in decoding_strategy_store
]
