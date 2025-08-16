# XGBoost Classifier for Model Selection and Boosting

This project demonstrates the use of XGBoost, a popular gradient boosting library, for binary classification tasks. The workflow includes data preprocessing, model training, evaluation, and k-fold cross-validation.

---

## Overview

- **Task**: Binary classification with boosting
- **Dataset**: `Data.csv`
- **Frameworks**: scikit-learn, xgboost, numpy, pandas
- **Input**: Feature matrix (X), binary target (y)
- **Output**: Predicted class labels

---

## Steps

1. Load and preprocess the dataset
2. Encode labels if necessary
3. Split into training and test sets
4. Train an XGBClassifier
5. Evaluate with confusion matrix and accuracy
6. Perform k-fold cross-validation

---

## Usage

```bash
python xgb.py
```

---

## Requirements

- xgboost
- numpy
- pandas
- scikit-learn

Install with:

```bash
pip install xgboost numpy pandas scikit-learn
```

---

## Output

- Prints confusion matrix and accuracy
- Prints mean and standard deviation of k-fold cross-validation accuracy

---

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
