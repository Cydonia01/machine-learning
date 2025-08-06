# 🛒 Eclat Association Rule Learning

This project demonstrates how to perform **Association Rule Learning** using the **Eclat** algorithm on the `Market_Basket_Optimisation.csv` dataset.

---

## 📌 Objective

Discover frequent itemsets in transaction data:
- **Identify products often purchased together**
- **Find high-support item combinations**

---

## 📁 Files

- `Market_Basket_Optimisation.csv`: Transaction dataset for association rule mining. Can be found in parent directory.
- `eclat.py`: Main Python script with full Eclat pipeline.

---

## 🔍 Workflow

1. **Data Preprocessing**
    - Import transaction data
    - Format data for Eclat algorithm
2. **Frequent Itemset Mining**
    - Use Eclat algorithm to find frequent itemsets
    - Extract itemsets with high support
3. **Evaluation**
    - Analyze support metrics
    - Filter and interpret frequent itemsets
4. **Visualization**
    - Print itemsets descending by support

---

## ▶️ How to Run

1. Make sure the dataset `Market_Basket_Optimisation.csv` is in the same directory or update the path in the code.
2. Run the script:
    ```bash
    python eclat.py
    ```

### 📊 Results

- Frequent itemsets are displayed with their support values.
- Printings help interpret the most significant product combinations.

---

## 📦 Dependencies

- `numpy`
- `apyori`

Install via:

```bash
pip install numpy apyori
```