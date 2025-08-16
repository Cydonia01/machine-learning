# Artificial Neural Network (ANN) for Classification

This project implements an Artificial Neural Network (ANN) for binary classification using TensorFlow and Keras. The example provided predicts customer churn based on the "Churn_Modelling.csv" dataset.

---

## Overview

- **Task**: Binary classification (predicting customer churn)
- **Dataset**: `Churn_Modelling.csv`
- **Framework**: TensorFlow (Keras API)
- **Input**: Customer features (after encoding and scaling)
- **Output**: Probability of churn (0 or 1)

---

## Steps

1. Data preprocessing: label encoding, one-hot encoding, feature scaling
2. ANN construction: 2 hidden layers (ReLU), 1 output layer (sigmoid)
3. Model training: Adam optimizer, binary cross-entropy loss
4. Evaluation: Confusion matrix, accuracy score
5. Single prediction: Predicts churn probability for a new customer

---

## Usage

```bash
python ann.py
```

---

## Requirements

- tensorflow
- pandas
- numpy
- scikit-learn

Install with:

```bash
pip install tensorflow pandas numpy scikit-learn
```

---

## Output

- Prints confusion matrix and accuracy on the test set
- Prints probability of churn for a sample customer

---

## References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API](https://keras.io/)
- [Churn Modelling Dataset](https://www.kaggle.com/datasets/shubhendra7/churn-modelling)
