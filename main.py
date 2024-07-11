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
"""Main execution file for the project."""

from hydra_zen import store, zen

from confidentllm import data, models  # noqa: F401
from confidentllm.scripts.question_answering_runner import run_question_answering

store(
    run_question_answering,
    name="question_answering",
    hydra_defaults=[
        "_self_",
        {"model": "gemma_11_2b_it"},
        {"tokenizer": "gemma_11_2b_it"},
        {"generation_method": "greedy_causal_lm_generation_method"},
        {"output_processor": "answer_processor"},
        {"data": "gsm8k"},
        # {"override hydra/launcher": "hpc_submission"},
    ],
)

if __name__ == "__main__":
    store.add_to_hydra_store()

    # Generate the CLI for run_extraction
    zen(run_question_answering).hydra_main(
        config_name="question_answering",
    )
