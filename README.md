# ConfidentLLM

A framework for evaluating and improving large‐language model confidence using Reinforcement Learning from Self‐Feedback (RLSF).

## Features

## Paper Code

This codebase implements the experiments from our EMNLP 2025 submission:
**"Reinforcement Learning from Self-feedback for Fine-tuning Large Language Models (RLSF)"**.

### Installation

We recommend installing using [`uv`](https://docs.astral.sh/uv/). UV can be easily installed using `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`. After installing UV, you can install the ConfidentLLM codebase with:

```bash
uv sync
```

### Models Used

- **Gemma 2 2B** – used for both reward model training and RL fine-tuning
- **Phi-2** – used for multiple choice QA
- **Qwen 2.5 (via DeepSeek R1)** – used for mathematical reasoning tasks

| Model Description                  | Config Name         |
|-----------------------------------|----------------------|
| Phi-2 (used in QA)                | `PHI2_3B`            |
| Gemma 2B IT (used in math QA)     | `GEMMA2_2B_IT`       |
| Gemma 9B IT (used in QA + reward) | `GEMMA2_9B_IT`       |
| Qwen 2.5 7B (via DeepSeek R1)     | `DEEPSEEK_R1_DISTILL_QWEN_7B` |
| URM (baseline reward model)       | `URM_LLAMA31_8B`     |

### Datasets

- **GSM8K** and **MultiArith** (math reasoning)
- **CommonsenseQA** and **ARC-Easy** (multiple choice QA)
- **RewardBench** for evaluating reward models

#### Dataset Configuration Mapping

| Dataset Description                          | Config Name       | Notes                                                                 |
|----------------------------------------------|-------------------|-----------------------------------------------------------------------|
| GSM8K (math reasoning)                       | `gsm8k`           | Used for reward model training and math QA evaluation                |
| MultiArith (math reasoning)                  | `multiarith`      | Evaluated for accuracy and calibration                               |
| CommonsenseQA (multiple choice)              | `commonsense_qa`  | Used for QA evaluation with Phi-2                                    |
| ARC Easy (multiple choice)                   | `arc`             | Used for QA evaluation with Gemma 2                                  |
| RewardBench (reward model eval)              | `reward_bench`    | Used to evaluate ranking performance of reward models                |
| CoT Preference (W&B table)                   | `cot_preference`  | Loaded using W&B run details; contains ranked completions for RLSF   |
| Union (multi-task dataset)                   | `union+arc+commonsense_qa+...` | Combine datasets using `+` in `data.name` (e.g. `union+arc+gsm8k`)   |

## Usage

All main workflows are exposed as console scripts:

| Command                   | Description                                  |
|---------------------------|----------------------------------------------|
| `download-and-sync`       | Download model weights & datasets            |
| `question-answering`      | Evaluate QA performance                      |
| `reward-model-evaluation` | Compute reward‐model accuracy                |
| `reinforcement-learning`  | RL fine‐tuning (RLSF)                        |
| `train`                   | Train SFT or DPO models                      |

They accept Hydra‐style args (`key=value`), e.g.:

```bash
download-and-sync \
  model.pretrained_model_name_or_path=GEMMA_2B \
  data.name=gsm8k \
  data.split=validation
```

### Reproducing EMNLP 2025 Experiments

- **Evaluation on different tasks**:
  - **Greedy Decoding**:

    Example command for evaluating the Phi-2 model on the GSM8K dataset:

    ```bash
    uv run question-answering model.pretrained_model_name_or_path=PHI2_3B data=gsm8k
    ```

  - **CoT Prompting**:

    Example command for evaluating the Phi-2 model on the GSM8K dataset with CoT prompting:

    ```bash
    uv run question-answering model.pretrained_model_name_or_path=PHI2_3B data=gsm8k output_processor.prompt="Please solve the math problem step by step."
    ```
  
  - **CoT Decoding**:

    Example command for evaluating the Phi-2 model on the GSM8K dataset with CoT decoding:

    ```bash
    uv run question-answering model.pretrained_model_name_or_path=PHI2_3B data=gsm8k generation_method=cot_decoding generation_method.num_beams=10 run_config.keep_all_generation_paths=True
    ```

    Setting `run_config.keep_all_generation_paths=True` allows you to keep all generated paths for further analysis and RLSF training.
  
- **Reward Model Training**:

  Example command for training a reward model using the Phi-2 model on the preference pairs created in a run called `example_run`:

  ```bash
  uv run train model=train_sequence_cls model.lora=sequence_cls model.pretrained_model_name_or_path=PHI2_3B data=cot_preference train_data.run_name=example_run trainer=reward_model
  ```

- **Reward Model Evaluation**:

  Example command for evaluating the reward model on the RewardBench dataset:

  ```bash
  uv run reward-model-evaluation model.pretrained_model_name_or_path="PHI2_3B/or/path/to/trained/reward/model"
  ```

- **RLSF Training**:
  
  Example command for fine tuning the Phi-2 model using RLSF with the reward model trained in the previous step:

  ```bash
  uv run reinforcement-learning model.pretrained_model_name_or_path=PHI2_3B reward_model.pretrained_model_name_or_path="/path/to/trained/reward/model/or/PHI2_3B/for/testing" train_data=gsm8k test_data=gsm8k
  ```

- **DPO Training**:

  Example command for training a model using DPO with the Phi-2 model:

  ```bash
  uv run train model.pretrained_model_name_or_path=PHI2_3B data=cot_preference train_data.run_name=example_run trainer=dpo
  ```

Further hyperparameters are easily accessible, for example changing the number of epochs for training or changing the parameters of the PPO algorithm can easily be done via `trainer.param_name=value` arguments.

## License

Apache License 2.0 – see [LICENSE](LICENSE) for details.
