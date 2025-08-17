# Q-Learning for Route Planning in a Graph Environment

This project demonstrates a simple **Q-Learning** algorithm for finding the optimal route between locations in a graph-based environment using **NumPy**.

---

## Overview

- **Environment**: Custom graph with 12 locations (A-L)
- **Algorithm**: [Q-Learning](https://en.wikipedia.org/wiki/Q-learning)
- **Framework**: NumPy (no deep learning required)
- **Action Space**: Discrete (move between connected locations)
- **Input**: Start, intermediate, and end locations

---

## How It Works

- The environment is represented as a reward matrix (R) for possible transitions between locations.
- Q-Learning is used to learn the best actions (routes) from any location to a target.
- The `route` function computes the optimal path from a start to an end location.
- The `best_route` function finds the best path from a start location to an end location via an intermediate location.

---

## Usage

Run the script to see an example of finding the best route from E to G via K:

```bash
python q_learning.py
```

---

## Features

- Simple, interpretable Q-Learning implementation
- Customizable environment (edit the reward matrix for new graphs)
- Returns the optimal path as a list of locations
- No external dependencies except NumPy

---

## Output

- Prints the best route as a list of locations, e.g.:
  ```
  ['E', 'I', 'J', 'K', 'J', 'F', 'B', 'C', 'G']
  ```

---

## Customization Tips

- Modify the `R` matrix to represent different graphs or reward structures
- Adjust `gamma` (discount factor) and `alpha` (learning rate) for different learning behaviors
- Change the number of training iterations for faster or more thorough learning

---
