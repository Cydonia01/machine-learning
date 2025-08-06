# 📈 Polynomial Regression

This project demonstrates how to train and visualize a **Polynomial Regression** model using a regression dataset.

---

## 📌 Objective

Predict the salary based on:
- **Position**
- **Level**

---

## 📁 Files

- `Position_Salaries.csv`: Dataset used for training/testing. Can be found in the parent directory.
- `polynomialRegression.py`: Main Python script with full model pipeline.
- Visualizations of polynomial regression results.

---

## 🔍 Workflow

1. **Data Preprocessing**
    - Import data
2. **Feature Transformation**
    - Transform features to polynomial terms using `PolynomialFeatures` from `sklearn.preprocessing`
3. **Model Training**
    - `LinearRegression` from `sklearn.linear_model`
    - Fit the model to the transformed training data
4. **Evaluation**
    - Compare predictions (`y_test` vs `y_pred`)
5. **Visualization**
    - Plot polynomial regression curve and predictions using `matplotlib`

---

## ▶️ How to Run

1. Make sure the dataset `Position_Salaries.csv` is in the same directory or update the path in the code.
2. Run the script:
    ```bash
    python polynomialRegression.py
    ```

### 📊 Results

- Plots are generated to visualize the polynomial regression curve and predictions.

---

## 📦 Dependencies

- `numpy`
- `matplotlib`
- `pandas`
- `sklearn`

Install via:

```bash
pip install numpy matplotlib pandas scikit-learn
```