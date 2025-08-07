# Multiple Linear Regression

This project demonstrates how to train and visualize a **Multiple Linear Regression** model using a regression dataset.

---

## Objective

Predict the salary based on multiple features:
- **R&D Spend**
- **Administration**
- **Marketing Spend**
- **State**

---

## Files

- `50_Startups.csv`: Dataset used for training/testing. Can be found in the parent directory.
- `multipleLinearRegression.py`: Main Python script with the full model pipeline for multiple linear regression.
- Visualizations of regression results.

---

## Workflow

1. **Data Preprocessing**
    - Import data
    - Encode categorical variables
    - Train-test split
2. **Model Training**
    - `LinearRegression` from `sklearn.linear_model`
    - Fit the model to the training data with multiple features
3. **Evaluation**
    - Compare predictions (`y_test` vs `y_pred`)
4. **Visualization**
    - Print actual vs predicted values

---

## How to Run

1. Make sure the dataset `50_Startups.csv` is in the same directory or update the path in the code.
2. Run the script:
    ```bash
    python multipleLinearRegression.py
    ```

## Results

- Prints the actual vs predicted salary values.

---

## Dependencies

- `numpy`
- `pandas`
- `sklearn`

Install via:

```bash
pip install numpy pandas scikit-learn
```