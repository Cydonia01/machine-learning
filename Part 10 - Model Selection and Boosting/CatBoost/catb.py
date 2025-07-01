import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from catboost import CatBoostClassifier
# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Convert labels to 0 and 1 if they are not already
unique_labels = np.unique(y)
if set(unique_labels) != {0, 1}:
    label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    y = np.vectorize(label_mapping.get)(y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Trainin CatBoost on the Training set
classifier = CatBoostClassifier()
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)

print("Confusion Matrix:\n", cm)
print("Accuracy:", f"{accuracy_score(y_test, y_pred):.4f}")
print("Mean Accuracy:", accuracies.mean().round(4))
print("Standard Deviation of Accuracy:", accuracies.std().round(4))
