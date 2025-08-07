# Random Forest Regression

This project demonstrates how to train and visualize a **Random Forest Regressor** using a regression dataset.

---

## Objective

Predict the salary based on:
- **Position**
- **Level**

---

## Files

- `Position_Salaries.csv`: Dataset used for training/testing. Can be found in the parent directory.
- `rfregression.py`: Main Python script with full model pipeline.
- Visualizations of regression results and decision boundaries.

---

## Workflow

1. **Data Preprocessing**
    - Import data
2. **Model Training**
    - `RandomForestRegressor` from `sklearn.ensemble`
    - Set parameters as needed (e.g., `n_estimators`, `random_state`)
3. **Evaluation**
    - Compare predictions (`y_test` vs `y_pred`)
4. **Visualization**
    - Plot regression predictions and decision boundaries using `matplotlib`

---

## How to Run

1. Make sure the dataset `Position_Salaries.csv` is in the same directory or update the path in the code.
2. Run the script:
    ```bash
    python rfregression.py
    ```

## Results

- Plots are generated to visualize regression predictions and decision boundaries.

---

## Dependencies

- `numpy`
- `matplotlib`
- `pandas`
- `sklearn`

Install via:

```bash
pip install numpy matplotlib pandas scikit-learn
```