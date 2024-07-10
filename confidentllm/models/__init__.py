from hydra_zen import builds
from transformers import AutoModelForCausalLM, AutoTokenizer

CausalLMConfig = builds(
    AutoModelForCausalLM.from_pretrained,
    pretrained_model_name_or_path="gpt2",
)

TokenizerConfig = builds(
    AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path="gpt2",
)
