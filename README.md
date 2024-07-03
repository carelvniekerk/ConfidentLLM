# ConfidentLLM

## Introduction
**Objective**: Evaluate the confidence of various LLM models (Gemma 2B, Llama 7B, Phi 2, Phi 3) using different confidence methods and generation techniques.
**Scope**: Implement and compare verbalized confidence and logit confidence using various generation methods. Extract logits and fine-tune models using transformers and TRL. Configure code with Pydantic and Hydra Zen.

## Project Setup

### Environment Setup
- Install Python and required libraries (Transformers, TRL, Pydantic, Hydra Zen).
- Set up Poetry for dependency management and packaging.
- Install GitLab for version control.

## Configuration Management

### Hydra Zen Configuration
- Set up hierarchical configuration management.
- Example config file for model selection, training parameters, and evaluation settings.

### Pydantic for Type Safety
- Define data models using Pydantic.
- Ensure type safety and data validation.

## Model Implementation

### Model Loading
- Load pre-trained models (Gemma 2B, Llama 7B, Phi 2, Phi 3) using Transformers.
- Implement functionality to switch between models based on configuration.

### Confidence Methods
- Implement verbalized confidence method.
- Implement logit confidence method.
- Extract logits for analysis.

### Generation Methods
- Implement greedy decoding in `generation/greedy.py`.
- Implement sampling method in `generation/sampling.py`.
- Implement Chain-of-Thought (CoT) decoding in `generation/cot.py`.

## Fine-Tuning

### Training Pipeline
- Implement fine-tuning process using TRL in `train/finetune.py`.
- Configure training parameters (learning rate, batch size, epochs) using Hydra Zen.

### Experiment Tracking
- Track experiments using GitLab CI/CD.
- Save model checkpoints and logs.

## Evaluation

### Metrics
- Define evaluation metrics for confidence methods (accuracy, calibration) in `evaluation/metrics.py`.
- Implement evaluation scripts in `evaluation/evaluate.py`.

### Initial Results
- Conduct initial experiments to gather results.
- Analyze results to identify areas for improvement.

## Documentation

### Code Documentation
- Document codebase using docstrings and comments.
- Generate documentation using tools like Sphinx.

### Research Documentation
- Document research methodology, experiments, and results.
- Prepare initial results for presentation to your advisor.

## Future Work

### Codebase Expansion
- Plan for future features such as co-annotation.
- Identify potential areas for further research and experimentation.