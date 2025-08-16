# Natural Language Processing (NLP) Projects

This directory contains two NLP projects: a classic Bag-of-Words sentiment analysis pipeline and a BERT-based sentiment classifier using ktrain.

---

## 1. Bag-of-Words Sentiment Analysis (Naive Bayes)

A traditional NLP pipeline for classifying restaurant reviews as positive or negative using text preprocessing, Bag-of-Words, and a Naive Bayes classifier.

### Overview

- **Dataset**: `Restaurant_Reviews.tsv` (tab-separated, with review text and label)
- **Preprocessing**: Regex cleaning, lowercasing, stopword removal (except 'not'), stemming
- **Feature Extraction**: Bag-of-Words (CountVectorizer, max 1500 features)
- **Model**: Gaussian Naive Bayes
- **Evaluation**: Confusion matrix, accuracy score
- **Single Review Prediction**: Classifies new reviews as positive or negative

### Usage

```bash
python nlp.py
```

### Requirements

- pandas
- numpy
- scikit-learn
- nltk

### Installation

Install the required packages with:

```bash
pip install pandas numpy scikit-learn nltk
```

### Output

- Prints confusion matrix and accuracy
- Prints sentiment prediction for a sample review

---

## 2. Sentiment Analysis with BERT (ktrain)

A modern sentiment classifier using a pre-trained BERT model, fine-tuned on the IMDB movie review dataset with ktrain.

### Overview

- **Dataset**: IMDB reviews in this [link](https://www.kaggle.com/datasets/pranayprasad/aclimdb). Download and extract the dataset.
- **Preprocessing**: Tokenization and input formatting for BERT
- **Model**: Pre-trained BERT, fine-tuned for binary sentiment classification
- **Framework**: ktrain (TensorFlow + transformers)
- **Training**: One epoch, batch size 6 (configurable)
- **Validation**: Prints accuracy and classification report
- **Export**: Saves a predictor for later use

### Usage

```bash
python nlpwbert.py
```

### Requirements

- ktrain
- tensorflow
- transformers

### Installation

Install the required packages with:

```bash
pip install ktrain tensorflow transformers
```

### Output

- Prints validation results
- Saves the trained predictor as `bert_imdb_predictor`

---

## References

- [NLTK Documentation](https://www.nltk.org/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [ktrain Documentation](https://amaiya.github.io/ktrain/)
- [BERT Paper (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
