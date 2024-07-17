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
"""Generation Function."""

import torch
from hydra_zen import just, make_custom_builds_fn, store

from confidentllm.decoding_strategies.greedy import greedy_decoding_strategy
from confidentllm.decoding_strategies.types import DecodingStrategy
from confidentllm.generation.types import GenerateFunction, ModelNotSetError

__all__ = []

builds = make_custom_builds_fn(populate_full_signature=True)


def calc_banned_ngram_tokens(
    prev_input_ids: torch.Tensor,
    num_hypos: int,
    no_repeat_ngram_size: int,
    cur_len: int,
) -> list[list[int]]:
    """Copy from fairseq for no_repeat_ngram in beam_search.

    Args:
    ----
        prev_input_ids (torch.Tensor): The previous input IDs.
        num_hypos (int): The number of hypotheses.
        no_repeat_ngram_size (int): The size of the n-grams to avoid.
        cur_len (int): The current length of the generated sequences.

    Returns:
    -------
        list[list[int]]: The list of banned tokens.

    """
    # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
    if cur_len + 1 < no_repeat_ngram_size:
        return [[] for _ in range(num_hypos)]

    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(
            *[gen_tokens[i:] for i in range(no_repeat_ngram_size)],
            strict=False,
        ):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = [
                *generated_ngram.get(prev_ngram_tuple, []),
                ngram[-1],
            ]

    def _get_generated_ngrams(hypo_idx: int) -> list[int]:
        # Before decoding the next token, prevent decoding of ngrams that have appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    return [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]


def calc_banned_bad_words_ids(
    prev_input_ids: torch.Tensor,
    bad_words_ids: list[list[int]],
) -> list[int]:
    """Copy from fairseq for bad_words_ids in beam_search."""
    banned_tokens = []

    def _tokens_match(prev_tokens: list[int], tokens: list[int]) -> bool:
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_input_ids):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        return prev_tokens[-len(tokens) :] == tokens

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            if len(banned_token_seq) == 0:
                msg = (
                    f"Banned words token sequences {bad_words_ids}"
                    " cannot have an empty list"
                )
                raise ValueError(msg)

            if (
                _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1])
                is False
            ):
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens


def enforce_repetition_penalty_(
    lprobs: torch.Tensor,
    batch_size: int,
    num_beams: int,
    prev_output_tokens: torch.Tensor,
    repetition_penalty: float,
) -> None:
    """Repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)."""
    for i in range(batch_size * num_beams):
        for previous_token in set(prev_output_tokens[i].tolist()):
            # if score < 0 then repetition penalty has to multiplied to reduce
            # the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty


def postprocess_next_token_scores(  # noqa: PLR0913
    scores: torch.Tensor,
    input_ids: torch.Tensor,
    no_repeat_ngram_size: int,
    bad_words_ids: list[list[int]],
    cur_len: int,
    min_length: int,
    eos_token_id: int | None,
    repetition_penalty: float,
    batch_size: int,
    num_beams: int,
) -> torch.Tensor:
    """Postprocess the next token scores using of current scores and penalties."""
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0:
        enforce_repetition_penalty_(
            scores,
            batch_size,
            num_beams,
            input_ids,
            repetition_penalty,
        )

    # set eos token prob to zero if min_length is not reached
    if eos_token_id is not None and cur_len < min_length:
        scores[:, eos_token_id] = -float("inf")

    if no_repeat_ngram_size > 0:
        # calculate a list of banned tokens to prevent repetitively generating
        # the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = calc_banned_ngram_tokens(
            input_ids,
            num_batch_hypotheses,
            no_repeat_ngram_size,
            cur_len,
        )
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

    if bad_words_ids is not None:
        # calculate a list of banned tokens according to bad words
        banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

        for i, _banned_tokens in enumerate(banned_tokens):
            scores[i, _banned_tokens] = -float("inf")

    return scores


class GreedyGenerateFunction(GenerateFunction):
    """Greedy (single beam) generate function."""

    def __init__(self, decoding_strategy: DecodingStrategy) -> None:
        """Initialize the generate function.

        Args:
        ----
            decoding_strategy (DecodingStrategy): The decoding strategy to use.

        """
        super().__init__(decoding_strategy)

    def __call__(  # noqa: PLR0913
        self,
        input_ids: torch.Tensor,  # type: ignore  # noqa: PGH003
        max_length: int,
        min_length: int = 5,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        bad_words_ids: list[list[int]] | None = None,
        pad_token_id: int = 0,
        eos_token_id: int | None = None,
        batch_size: int = 1,
        attention_mask: torch.Tensor | None = None,  # type: ignore  # noqa: PGH003
        model_specific_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate sequences for each example without beam search (num_beams == 1).

        All returned sequence are generated independantly.
        """
        if isinstance(self.model, type(None)):
            raise ModelNotSetError(self.model)

        input_ids: torch.Tensor = input_ids.to(self.model.device)  # type: ignore  # noqa: PGH003
        attention_mask: torch.Tensor = (
            attention_mask.to(self.model.device) if attention_mask is not None else None  # type: ignore  # noqa: PGH003
        )
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)
        cur_len = 0

        if model_specific_kwargs is None:
            model_specific_kwargs = {}

        if bad_words_ids is None:
            bad_words_ids = []

        generation_probs: list[torch.Tensor] = [
            torch.ones(
                input_ids.size(),
                device=input_ids.device,
                dtype=torch.float32,
            ).fill_(-1.0),
        ]

        while cur_len < max_length:
            inputs = {"input_ids": input_ids}
            if attention_mask is not None:
                inputs["attention_mask"] = attention_mask

            inputs = self.model.prepare_inputs_for_generation(
                **inputs,
                **model_specific_kwargs,
            )  # type: ignore  # noqa: PGH003

            with torch.no_grad():
                outputs = self.model(**inputs)
            next_token_logits = outputs[0][:, -1, :]

            scores = postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=cur_len,
                min_length=min_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=1,
            )

            next_token, next_token_probs = self.decoding_strategy(
                scores=scores,
            )

            generation_probs.append(next_token_probs.unsqueeze(-1))

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (
                    1 - unfinished_sents
                )
            else:
                tokens_to_add = next_token

            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths
                # is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(
                    eos_in_sents.long(),
                ).bool()
                sent_lengths.masked_fill_(
                    is_sents_unfinished_and_token_to_add_is_eos,
                    cur_len,
                )
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the
            # maximul length
            if unfinished_sents.max() == 0:
                break

            attention_mask = (
                torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1)),
                    ],
                    dim=-1,
                )
                if attention_mask is not None
                else None
            )  # type: ignore  # noqa: PGH003

        return input_ids, torch.cat(generation_probs, dim=-1)


GeneratorConfig = builds(GreedyGenerateFunction)

greedy_generate_function = GeneratorConfig(
    decoding_strategy=just(greedy_decoding_strategy),  # type: ignore  # noqa: PGH003
)

generate_function_store = [
    store(group="generation_method/generator"),
    store(group="output_processor/generator"),
]
[
    store(greedy_generate_function, name="greedy_generate_function")
    for store in generate_function_store
]
