# Machine Learning & Deep Learning Learning Journey

## Current Directory Structure

```
├── Model Selection/
│   ├── Classification/
│   └── Regression/
├── Part 1 - Data Preprocessing/
├── Part 2 - Regression/
│   ├── Decision Tree Regression/
│   ├── Linear Regression/
│   ├── Multiple Linear Regression/
│   ├── Polynomial Regression/
│   ├── Random Forest Regression/
│   └── Support Vector Regression/
├── Part 3 - Classification/
│   ├── Decision Tree Classification/
│   ├── K-Nearest Neighbors/
│   ├── Kernel SVM/
│   ├── Logistic Regression/
│   ├── Naive Bayes Classification/
│   ├── Random Forest Classification/
│   └── Support Vector Machine/
├── Part 4 - Clustering/
│   ├── Hierarchial Clustering/
│   └── K Means Clustering/
├── Part 5 - Association Rule Learning/
│   ├── Apriori/
│   └── Eclat/
├── Part 6 - Reinforcement Learning/
│   ├── Bandit Algorithms/
│   │   ├── Thompson Sampling/
│   │   └── Upper Confidence Bound/
│   ├── Policy Based/
│   │   ├── Asynchronous Advantage Actor-Critic/
│   │   ├── Proximal Policy Optimization/
│   │   └── Soft Actor-Critic/
│   └── Value Based/
│       ├── Deep Convolutional Q-Learning/
│       ├── Deep Q-Learning/
│       └── Q-Learning/
├── Part 7 - Natural Language Processing/
├── Part 8 - Deep Learning/
│   ├── Volume 1 - Supervised/
│   │   ├── Artificial Neural Networks/
│   │   │   ├── Classification/
│   │   │   └── Regression/
│   │   ├── Convolutional Neural Networks/
│   │   └── Recurrent Neural Networks/
│   └── Volume 2 - Unsupervised/
│       ├── Autoencoders/
│       ├── Boltzmann Machines/
│       └── Self Organizing Maps/
├── Part 9 - Dimensionality Reduction/
│   ├── Kernel PCA/
│   ├── Linear Discriminant Analysis/
│   └── Principal Component Analysis/
├── Part 10 - Model Selection and Boosting/
│   ├── CatBoost/
│   ├── Model Selection/
│   └── XGBoost/
├── Part 11 - Large Language Models/
```

Each folder contains:

- Python implementations of algorithms
- Datasets (or links to them)
- Folder-level `README.md` files for context

---

## Topics Covered

### Machine Learning

- **Regression**: Linear, Polynomial, SVR, Decision Tree, Random Forest
- **Classification**: Logistic, KNN, SVM, Naive Bayes, Decision Trees, Random Forest
- **Clustering**: K-Means, Hierarchical
- **Association Rules**: Apriori, Eclat
- **Model Selection**: Cross-validation, Grid Search, Boosting (XGBoost, CatBoost)

### Deep Learning

- **ANNs** for classification and regression
- **CNNs** for image classification (cats vs. dogs)
- **RNNs** for sequence modeling
- **Autoencoders, SOMs, Boltzmann Machines** for unsupervised learning

### Reinforcement Learning

- **Bandit algorithms**: Thompson Sampling, UCB
- **Value-based**: Q-Learning, Deep Q-Learning
- **Policy-based**: PPO, A3C, Soft Actor-Critic
- **Deep RL**: Deep Convolutional Q-Learning

### NLP & LLMs

- **Text preprocessing**, tokenization, sentiment analysis
- **Large Language Models** (BERT, GPT-2, etc.)

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Cydonia01/machine-learning.git
   cd machine-learning
   ```

Navigate to any part and install the specific dependency for the algorithm and run the .py or .ipynb files. Dataset paths may need adjusting based on local setup.

## Data & Model Files

Small datasets and models are included in the repo. For large files (videos, .pth, .csv > 50MB), you may find a download link or instructions inside a README.md file in each folder.

## Tools & Libraries

- Python 3.10+
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- TensorFlow
- Keras
- PyTorch
- OpenCV
- NLTK
- Hugging Face Transformers (for LLMs)

## Notes

- This repository is not a tutorial series, but a self-curated, project-based notebook to reinforce theory through code.
