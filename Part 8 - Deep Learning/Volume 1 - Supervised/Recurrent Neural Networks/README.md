# Recurrent Neural Network (RNN) for Stock Price Prediction

This project implements a Recurrent Neural Network (RNN) with LSTM layers to predict Google stock prices using TensorFlow/Keras. The model is trained on historical stock price data and predicts future prices based on previous time steps.

---

## Overview

- **Task**: Time series regression (stock price prediction)
- **Dataset**: `Google_Stock_Price_Train.csv` and `Google_Stock_Price_Test.csv`
- **Framework**: TensorFlow (Keras API)
- **Input**: 60 previous stock prices (windowed time steps)
- **Output**: Predicted stock price for the next day

---

## Steps

1. Load and scale the training data (MinMaxScaler)
2. Create sequences of 60 time steps for training
3. Build an RNN with 4 LSTM layers and dropout regularization
4. Train the model with mean squared error loss and Adam optimizer
5. Prepare test data and make predictions
6. Visualize real vs. predicted stock prices

---

## Usage

```bash
python rnn.py
```

---

## Requirements

- tensorflow
- numpy
- pandas
- matplotlib
- scikit-learn

Install with:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

---

## Output

- Plots real and predicted Google stock prices for the test period

---

## References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API](https://keras.io/)
- [LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
