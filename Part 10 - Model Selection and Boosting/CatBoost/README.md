# CatBoost Classifier for Model Selection and Boosting

This project demonstrates the use of CatBoost, a gradient boosting library, for binary classification tasks. The workflow includes data preprocessing, model training, evaluation, and k-fold cross-validation.

---

## Overview

- **Task**: Binary classification with boosting
- **Dataset**: `Data.csv`
- **Frameworks**: scikit-learn, catboost, numpy, pandas
- **Input**: Feature matrix (X), binary target (y)
- **Output**: Predicted class labels

---

## Steps

1. Load and preprocess the dataset
2. Encode labels if necessary
3. Split into training and test sets
4. Train a CatBoostClassifier
5. Evaluate with confusion matrix and accuracy
6. Perform k-fold cross-validation

---

## Usage

```bash
python catb.py
```

---

## Requirements

- catboost
- numpy
- pandas
- scikit-learn

Install with:

```bash
pip install catboost numpy pandas scikit-learn
```

---

## Output

- Prints confusion matrix and accuracy
- Prints mean and standard deviation of k-fold cross-validation accuracy

---

## References

- [CatBoost Documentation](https://catboost.ai/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
