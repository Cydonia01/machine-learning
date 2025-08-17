# Restricted Boltzmann Machine (RBM) for Movie Recommendation

This project implements a Restricted Boltzmann Machine (RBM) using PyTorch for collaborative filtering and movie recommendation. The model is trained on the MovieLens dataset to learn user preferences and predict movie ratings.

---

## Overview

- **Task**: Collaborative filtering for movie recommendation
- **Dataset**: MovieLens 1M (`ml-1m/`) and 100K (`ml-100k/`) datasets
- **Framework**: PyTorch
- **Input**: User-movie binary rating matrix (like/dislike)
- **Output**: Predicted binary ratings for movies not yet rated by the user

---

## Steps

1. Load and preprocess the MovieLens datasets
2. Convert ratings to user-movie matrices for training and testing
3. Binarize ratings: 1 (like), 0 (not like), -1 (unrated)
4. Build an RBM with 100 hidden nodes
5. Train the model using Contrastive Divergence
6. Evaluate the model on the test set using mean absolute error

---

## Usage

```bash
python rbm.py
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
- Prints test loss (mean absolute error) after training

---

## References

- [PyTorch Documentation](https://pytorch.org/)
- [RBMs for Collaborative Filtering (Blog)](https://www.datacamp.com/tutorial/recommender-systems-python)
