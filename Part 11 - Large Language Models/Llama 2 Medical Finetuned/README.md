# Llama 2 Medical Finetuned: Supervised Fine-Tuning for Medical Text Generation

This project demonstrates supervised fine-tuning of a Llama 2 language model for medical text generation using Hugging Face Transformers, PEFT (LoRA), and TRL libraries. The model is trained on a medical terms dataset and can generate medical explanations in response to user prompts.

---

## Overview

- **Task**: Supervised fine-tuning and inference for medical text generation
- **Model**: Llama 2 (quantized, LoRA PEFT)
- **Dataset**: `aboonaji/wiki_medical_terms_llam2_format` (Hugging Face Hub)
- **Frameworks**: PyTorch, Transformers, PEFT, TRL, Datasets
- **Input**: Medical prompt (e.g., "Please tell me about Tinea Pedis")
- **Output**: Generated medical explanation

---

## Steps

1. Load and quantize the Llama 2 model (4-bit, nf4)
2. Load the tokenizer and set padding
3. Optionally load checkpointed model weights
4. Set up training arguments (output directory, batch size, steps)
5. Create a supervised fine-tuning trainer (SFTTrainer) with LoRA configuration
6. Train the model on the medical terms dataset
7. Generate text responses to user prompts using a text-generation pipeline

---

## Usage

```bash
python "Llama 2 Medical Finetuned.py"
```

---

## Requirements

- torch
- transformers
- peft
- trl
- datasets

Install with:

```bash
pip install torch transformers peft trl datasets
```

---

## Output

- Prints generated medical text in response to a user prompt
- Saves model checkpoints in the `./results` directory

---

## References

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://huggingface.co/docs/peft/index)
- [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl/index)
- [Llama 2 Model Card](https://huggingface.co/meta-llama/Llama-2-7b)
- [Medical Terms Dataset](https://huggingface.co/datasets/aboonaji/wiki_medical_terms_llam2_format)
