# 🤖 Naive Bayes Classification

This project demonstrates how to train and visualize a **Naive Bayes Classifier** using the `Social_Network_Ads.csv` dataset.

---

## 📌 Objective

Classify whether a user purchased a product based on:
- **Age**
- **Estimated Salary**

---

## 📁 Files

- `Social_Network_Ads.csv`: Dataset used for training/testing. Can be found in parent directory.
- `naivebayes.py`: Main Python script with full model pipeline.
- Visualizations of decision boundaries for both training and test sets.

---

## 🔍 Workflow

1. **Data Preprocessing**
    - Import data
    - Train-test split
    - Feature scaling
2. **Model Training**
    - `GaussianNB` from `sklearn.naive_bayes`
    - Default parameters (e.g., `priors=None`)
3. **Evaluation**
    - Confusion matrix and accuracy score
    - Result comparison (`y_test` vs `y_pred`)
4. **Visualization**
    - Decision boundary plots for both training and test sets using `matplotlib`

---

## ▶️ How to Run

1. Make sure the dataset `Social_Network_Ads.csv` is in the same directory or update the path in the code.
2. Run the script:
    ```bash
    python naivebayes.py
    ```

### 📊 Results

- Accuracy is printed in the console.
- Two plots are generated:
    - Training set decision boundary
    - Test set decision boundary

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