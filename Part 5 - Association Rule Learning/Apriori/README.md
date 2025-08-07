# Apriori Association Rule Learning

This project demonstrates how to perform **Association Rule Learning** using the **Apriori** algorithm on the `Market_Basket_Optimisation.csv` dataset.

---

## Objective

Discover interesting associations and frequent itemsets in transaction data:
- **Identify products often purchased together**
- **Generate actionable association rules**

---

## Files

- `Market_Basket_Optimisation.csv`: Transaction dataset for association rule mining. Can be found in parent directory.
- `apriori.py`: Main Python script with full Apriori pipeline.

---

## Workflow

1. **Data Preprocessing**
    - Import transaction data
    - Format data for Apriori algorithm
2. **Association Rule Learning**
    - Use Apriori algorithm from `apyori` library
    - Extract frequent itemsets and generate association rules
3. **Evaluation**
    - Analyze support, confidence, and lift metrics
    - Filter and interpret rules
4. **Visualization**
    - Print rules descending by lift

---

## How to Run

1. Make sure the dataset `Market_Basket_Optimisation.csv` is in the same directory or update the path in the code.
2. Run the script:
    ```bash
    python apriori.py
    ```

## Results

- Frequent itemsets and association rules are displayed with their metrics.
- Printings help interpret the most significant associations.

---

## Dependencies

- `numpy`
- `apyori`

Install via:

```bash
pip install numpy apyori
```