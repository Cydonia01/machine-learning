# 🤖 K Means Clustering

This project demonstrates how to perform and visualize **K Means Clustering** using the `Mall_Customers.csv` dataset.

---

## 📌 Objective

Group users into clusters based on:
- **Annual Income**
- **Spending Score**

---

## 📁 Files

- `Mall_Customers.csv`: Dataset used for clustering. Can be found in parent directory.
- `kmeans.py`: Main Python script with full clustering pipeline.
- Visualizations of cluster assignments and elbow method plot.

---

## 🔍 Workflow

1. **Data Preprocessing**
    - Import data
    - Select relevant features
2. **Clustering**
    - Use `KMeans` from `sklearn.cluster`
    - Determine optimal number of clusters using the elbow method
3. **Evaluation**
    - Visualize clusters
    - Analyze cluster characteristics
4. **Visualization**
    - Elbow method plot for cluster selection
    - Scatter plot of clusters using `matplotlib`

---

## ▶️ How to Run

1. Make sure the dataset `Mall_Customers.csv` is in the same directory or update the path in the code.
2. Run the script:
    ```bash
    python kmeans.py
    ```

### 📊 Results

- Elbow method plot is displayed to help select the number of clusters.
- Scatter plot shows clustered data points.

---

## 📦 Dependencies

- `matplotlib`
- `pandas`
- `sklearn`

Install via:

```bash
pip install matplotlib pandas scikit-learn
```