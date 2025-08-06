# 🧹 Data Preprocessing

This project demonstrates how to preprocess data for machine learning tasks using a regression dataset.

---

## 📌 Objective

Prepare the dataset for modeling by:
- Handling missing values
- Encoding categorical variables
- Feature scaling

---

## 📁 Files

- `Data.csv`: Raw dataset for preprocessing.
- `preprocessing.py`: Main Python script for all preprocessing steps.

---

## 🔍 Workflow

1. **Import Data**
    - Load dataset using `pandas`
2. **Handle Missing Values**
    - Impute or remove missing data as needed
3. **Encode Categorical Variables**
    - Use techniques like One-Hot Encoding or Label Encoding
4. **Split Data**
    - Separate training and testing sets
5. **Feature Scaling**
    - Apply Standardization or Normalization

---

## ▶️ How to Run

1. Ensure `Data.csv` is available in the directory or update the path in the script.
2. Run the preprocessing script:
    ```bash
    python preprocessing.py
    ```

---

## 📦 Dependencies

- `numpy`
- `pandas`
- `sklearn`

Install via:

```bash
pip install numpy pandas scikit-learn
```