from hydra_zen import builds
from torch.nn import Module
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(pretrained_model_name_or_path: str, device: str) -> Module:
    """Load a pretrained model from Hugging Face's model hub.

    Args:
    ----
        pretrained_model_name_or_path (str): The name of or the path to the model.
        device (str): The device to load the model on.

    Returns:
    -------
        Module: The model loaded on the specified device.

    """
    model: Module = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)

    return model.to(device)


CausalLMConfig = builds(
    load_model,
    pretrained_model_name_or_path="google/gemma-1.1-2b-it",
    device="cuda",
)

TokenizerConfig = builds(
    AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path="google/gemma-1.1-2b-it",
)
