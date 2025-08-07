# Thompson Sampling
This project demonstrates the implementation of the **Thompson Sampling** algorithm to optimize ad selection in a simulated environment using the `Ads_CTR_Optimisation.csv` dataset.

---

## Objective

Select the best ad to display to users in order to **maximize click-through rate (CTR)** using a probabilistic approach.

---

## Files

- `Ads_CTR_Optimisation.csv`: Simulated dataset of user interactions with 10 ads. Can be found in the parent directory.
- `thompson.py`: Python script implementing the Thompson Sampling algorithm.
- **Output**: A histogram showing the number of times each ad was selected over 10,000 rounds.

---

## Workflow

1. **Dataset Overview**
    - Each row simulates one user impression.
    - Each column (10 in total) represents whether the user clicked (`1`) or did not click (`0`) a specific ad.

2. **Thompson Sampling Algorithm**
    - For each round:
      - Draw a sample from the **Beta distribution** for each ad.
      - Select the ad with the **highest sampled value**.
      - Observe the reward (click or no click).
      - Update the success/failure count for that ad.
    - Repeat for **10,000 rounds**.

3. **Visualization**
    - A histogram shows how often each ad was selected.
    - Ideally, the most effective ad gets selected the most.

---

## How to Run

1. Make sure the dataset `Ads_CTR_Optimisation.csv` is in the same directory or update the path in the script.
2. Run the script:
    ```bash
    python thompson.py
    ```

---

## Output

A histogram will be displayed:

- **X-axis**: Ad indices (0 to 9)
- **Y-axis**: Number of times each ad was selected

The distribution shows how Thompson Sampling converges toward the optimal ad over time.

---

## Dependencies

- `numpy`
- `matplotlib`
- `pandas`

Install via:

```bash
pip install numpy matplotlib pandas
```