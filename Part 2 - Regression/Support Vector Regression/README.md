# Support Vector Regression

This project demonstrates how to train and visualize a **Support Vector Regressor (SVR)** using a regression dataset.

---

## Objective

Predict the salary based on:
- **Position**
- **Level**

---

## Files

- `Position_Salaries.csv`: Dataset used for training/testing. Can be found in the parent directory.
- `svr.py`: Main Python script with full model pipeline.
- Visualizations of regression results and decision boundaries.

---

## Workflow

1. **Data Preprocessing**
    - Import data
    - Feature scaling (important for SVR)
2. **Model Training**
    - `SVR` from `sklearn.svm`
    - Set kernel and parameters as needed (e.g., `kernel`, `C`, `epsilon`)
3. **Evaluation**
    - Compare predictions (`y_test` vs `y_pred`)
4. **Visualization**
    - Plot regression predictions and decision boundaries using `matplotlib`

---

## How to Run

1. Make sure the dataset `Position_Salaries.csv` is in the same directory or update the path in the code.
2. Run the script:
    ```bash
    python svr.py
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