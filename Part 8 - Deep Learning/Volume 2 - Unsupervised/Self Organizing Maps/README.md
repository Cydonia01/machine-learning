# Self-Organizing Maps (SOM) and Hybrid SOM+ANN for Fraud Detection

This directory contains two projects for unsupervised and hybrid fraud detection on credit card applications: a classic Self-Organizing Map (SOM) and a hybrid approach combining SOM with an Artificial Neural Network (ANN).

---

## 1. Self-Organizing Map (SOM) for Fraud Detection

A SOM is used to identify potential frauds in credit card applications by mapping high-dimensional data to a 2D grid and visualizing outliers.

### Overview

- **Dataset**: `Credit_Card_Applications.csv`
- **Preprocessing**: Min-max scaling
- **Model**: MiniSom (10x10 grid)
- **Visualization**: Outliers (potential frauds) are visualized on the SOM grid
- **Output**: List of suspected frauds (original feature values)

### Usage

```bash
python som.py
```

### Requirements

- numpy
- pandas
- matplotlib
- minisom
- scikit-learn

Install with:

```bash
pip install numpy pandas matplotlib minisom scikit-learn
```

### Output

- Visualizes the SOM grid with outliers
- Prints the suspected frauds

---

## 2. Hybrid SOM + ANN for Fraud Prediction

A two-step approach: first, a SOM identifies potential frauds; then, an ANN is trained to predict the probability of fraud for each customer.

### Overview

- **Step 1**: Use SOM to detect outliers (potential frauds)
- **Step 2**: Use ANN to predict fraud probability for all customers
- **Preprocessing**: Min-max scaling for SOM, standard scaling for ANN
- **Model**: Keras Sequential ANN (2 layers)
- **Output**: Probabilities of fraud for each customer, sorted by risk

### Usage

```bash
python hybrid.py
```

### Requirements

- numpy
- pandas
- matplotlib
- minisom
- scikit-learn
- keras

Install with:

```bash
pip install numpy pandas matplotlib minisom scikit-learn keras
```

### Output

- Visualizes the SOM grid
- Prints and sorts customers by predicted fraud probability

---

## References

- [MiniSom Documentation](https://github.com/JustGlowing/minisom)
- [Keras API](https://keras.io/)
- [Self-Organizing Maps (Wikipedia)](https://en.wikipedia.org/wiki/Self-organizing_map)
