from hydra_zen import builds, store

from confidentllm.decoding_strategies import greedy_decoding_strategy
from confidentllm.generation.answer_processor import AnswerProcessor
from confidentllm.generation.causal_lm_generation_methods import (
    create_generation_method,
)
from confidentllm.generation.generation_function import create_generate_function
from confidentllm.models import CausalLMConfig, TokenizerConfig

GeneratorConfig = builds(
    create_generate_function,
    model=CausalLMConfig,
    decoding_strategy=greedy_decoding_strategy,
)

store(
    create_generation_method,
    group="generation_method",
    name="causal_lm_generation_method",
    tokenizer=TokenizerConfig(),
    generator=GeneratorConfig(),
)

output_processor_store = store(group="output_processor")

output_processor_store(
    AnswerProcessor,
    name="answer_processor",
    tokenizer=TokenizerConfig(),
    generator=GeneratorConfig(),
)
