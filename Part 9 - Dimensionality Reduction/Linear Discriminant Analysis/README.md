# Linear Discriminant Analysis (LDA) for Dimensionality Reduction and Classification

This project demonstrates the use of Linear Discriminant Analysis (LDA) for dimensionality reduction, followed by classification with logistic regression. The example uses the Wine dataset for multi-class classification and visualizes the results in 2D.

---

## Overview

- **Task**: Dimensionality reduction and classification
- **Dataset**: `Wine.csv`. Can be found in parent directory
- **Frameworks**: scikit-learn, matplotlib, numpy, pandas
- **Input**: Wine features (continuous variables)
- **Output**: Predicted wine class

---

## Steps

1. Load and preprocess the dataset (feature scaling)
2. Split into training and test sets
3. Apply LDA (2 components)
4. Train a logistic regression classifier
5. Evaluate with confusion matrix and accuracy
6. Visualize decision boundaries for both training and test sets

---

## Usage

```bash
python lda.py
```

---

## Requirements

- numpy
- pandas
- matplotlib
- scikit-learn

Install with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## Output

- Prints predicted and actual labels for the test set
- Prints confusion matrix and accuracy
- Shows decision boundary plots for training and test sets

---

## References

- [scikit-learn LDA Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
- [Wine Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/wine)
