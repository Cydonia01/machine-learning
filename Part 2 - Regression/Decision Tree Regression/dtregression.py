import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values  # Feature matrix (independent variable)
y = dataset.iloc[:, 2].values  # Target vector (dependent variable)

# Training the Decision Tree Regression model on the whole dataset
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predicting a new result with Decision Tree Regression
y_pred = regressor.predict([[6.5]])
print(y_pred)
# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X.ravel()), max(X.ravel()), 0.1)  # Create
# a grid of values for better visualization
X_grid = X_grid.reshape((len(X_grid), 1))  # Reshape for prediction
plt.scatter(X, y, color='red')  # Scatter plot of the original data
plt.plot(X_grid, regressor.predict(X_grid), color='blue')  # Plot the predictions
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()  # Display the plot