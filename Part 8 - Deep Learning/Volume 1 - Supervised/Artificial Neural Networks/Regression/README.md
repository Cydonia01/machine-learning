# Artificial Neural Network (ANN) for Regression

This project implements an Artificial Neural Network (ANN) for regression using TensorFlow and Keras. The example provided predicts power plant energy output based on the "Folds5x2_pp.xlsx" dataset.

---

## Overview

- **Task**: Regression (predicting continuous values)
- **Dataset**: `Folds5x2_pp.xlsx`
- **Framework**: TensorFlow (Keras API)
- **Input**: Power plant features
- **Output**: Predicted energy output

---

## Steps

1. Data preprocessing: train-test split
2. ANN construction: 2 hidden layers (ReLU), 1 output layer (linear)
3. Model training: Adam optimizer, mean squared error loss
4. Evaluation: Mean Squared Error (MSE), R² score

---

## Usage

```bash
python ann_regression.py
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

- Prints Mean Squared Error and R² score on the test set

---

## References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API](https://keras.io/)