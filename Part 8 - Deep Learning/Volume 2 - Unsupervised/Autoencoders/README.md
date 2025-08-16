# Stacked Autoencoder (SAE) for Movie Recommendation

This project implements a Stacked Autoencoder (SAE) using PyTorch for collaborative filtering and movie recommendation. The model is trained on the MovieLens dataset to reconstruct user ratings and predict unseen ratings.

---

## Overview

- **Task**: Collaborative filtering for movie recommendation
- **Dataset**: MovieLens 1M (`ml-1m/`) and 100K (`ml-100k/`) datasets
- **Framework**: PyTorch
- **Input**: User-movie rating matrix
- **Output**: Predicted ratings for movies not yet rated by the user

---

## Steps

1. Load and preprocess the MovieLens datasets
2. Convert ratings to user-movie matrices for training and testing
3. Build a Stacked Autoencoder with 4 fully connected layers
4. Train the model to reconstruct user ratings
5. Evaluate the model on the test set using RMSE

---

## Usage

```bash
python ae.py
```

---

## Requirements

- torch
- numpy
- pandas

Install with:

```bash
pip install torch numpy pandas
```

---

## Data Files

- `ml-1m/movies.dat`, `ml-1m/users.dat`, `ml-1m/ratings.dat`
- `ml-100k/u1.base`, `ml-100k/u1.test`

Download MovieLens datasets from:

- [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
- [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)

---

## Output

- Prints training loss per epoch
- Prints test loss (RMSE) after training

---

## References

- [PyTorch Documentation](https://pytorch.org/)
- [MovieLens Datasets](https://grouplens.org/datasets/movielens/)
- [Autoencoders for Collaborative Filtering (Blog)](https://towardsdatascience.com/autoencoders-for-collaborative-filtering-6c4cf4d9cd4f)
