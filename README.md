# ConfidentLLM

## Introduction

**Objective**: Evaluate the confidence of various LLM models (Gemma 2B, Llama 7B, Phi 2, Phi 3) using different confidence methods and generation techniques.

**Scope**: Implement and compare verbalized confidence and logit confidence using various generation methods. Extract logits and fine-tune models using transformers and TRL.

## Installation

Install Python 3.12 and Poetry, then run:

```bash
poetry install
```

## Usage

Run the following command to download models and data:

```bash
poetry run download-and-sync model.pretrained_model_name_or_path=MODELNAME data=DATASETNAME data.split=DATASETSPLIT
```

Run the following command to evaluate models in QA tasks:

```bash
poetry run question-answering model.pretrained_model_name_or_path=MODELNAME data=DATASETNAME data.split=DATASETSPLIT generation_method/generator/decoding_strategy=GENERATIONMETHOD generation_method.generator.decoding_strategy.param=PARAM
```

## Models

This package currently supports the following models:

- Google Gemma Models:
  - Gemma 2B (GEMMA_2B and GEMMA_2B_IT)
  - Gemma2 9B (GEMMA2_9B and GEMMA2_9B_IT)
- Microsoft Phi Models:
  - Phi 2 (PHI2)
  - Phi 3 (PHI3_MINI_INSTRUCT)
- Meta-Llama Models:
  - Llama2 7B (LLAMA2_7B, LLAMA2_7B_CHAT)
  - Llama3 7B (LLAMA3_8B, LLAMA3_8B_INSTRUCT)
- Mistral Models:
  - Mistral 7B (MISTRAL_7B, MISTRAL_7B_INSTRUCT)

## Data

This package currently supports the following datasets:

- Grade School Maths (gsm8k)
- Multi Step Arithmetic Word Problems (multiarith)
