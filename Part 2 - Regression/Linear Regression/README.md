# Simple Linear Regression

This project demonstrates how to train and visualize a **Linear Regression** model using a regression dataset.

---

## Objective

Predict the salary based on:
- **Years**
- **Experience**

---

## Files

- `Salary_Data.csv`: Dataset used for training/testing. Can be found in the parent directory.
- `simpleLinearRegression.py`: Main Python script with full model pipeline.
- Visualizations of regression results.

---

## Workflow

1. **Data Preprocessing**
    - Import data
    - Train-test split
2. **Model Training**
    - `LinearRegression` from `sklearn.linear_model`
    - Fit the model to the training data
3. **Evaluation**
    - Compare predictions (`y_test` vs `y_pred`)
4. **Visualization**
    - Plot regression line and predictions using `matplotlib`

---

## How to Run

1. Make sure the dataset `Salary_Data.csv` is in the same directory or update the path in the code.
2. Run the script:
    ```bash
    python simpleLinearRegression.py
    ```

## Results

- Plots are generated to visualize the regression line and predictions.

## Dependencies

- `matplotlib`
- `pandas`
- `sklearn`

Install via:

```bash
pip install matplotlib pandas scikit-learn
```