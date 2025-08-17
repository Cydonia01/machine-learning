# DistilBERT IMDB Sentiment Classification (Fine-Tuned)

This project demonstrates fine-tuning the DistilBERT transformer model for binary sentiment classification on the IMDB movie review dataset using Hugging Face Transformers and Datasets libraries.

---

## Overview

- **Task**: Sentiment analysis (positive/negative movie reviews)
- **Dataset**: IMDB (via Hugging Face Datasets)
- **Model**: DistilBERT (pre-trained, then fine-tuned)
- **Frameworks**: PyTorch, Hugging Face Transformers, Datasets, Evaluate
- **Input**: Movie review text
- **Output**: Sentiment label (positive/negative)

---

## Steps

1. Load and preprocess the IMDB dataset
2. Tokenize text using DistilBERT tokenizer
3. Fine-tune DistilBERT for sequence classification
4. Evaluate model performance (accuracy)
5. Save the fine-tuned model and tokenizer
6. Run inference on new text using a pipeline

---

## Usage

```bash
python distilbert-imdb-finetuned.py
```

---

## Requirements

- torch
- transformers
- datasets
- evaluate
- numpy

Install with:

```bash
pip install torch transformers datasets evaluate numpy
```

---

## Output

- Prints device used (CPU or GPU)
- Prints evaluation results (accuracy)
- Prints sentiment prediction for a sample review
- Saves the fine-tuned model and tokenizer in the `./model` directory

---

## References

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/index)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [IMDB Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
