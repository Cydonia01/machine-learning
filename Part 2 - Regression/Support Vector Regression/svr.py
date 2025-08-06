import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values  # Feature matrix (independent variable)
y = dataset.iloc[:, 2].values  # Target variable (dependent variable)
y = y.reshape(len(y), 1)  # Reshape y to be a 2D array

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)  # Scale features
y = sc_y.fit_transform(y)  # Scale target variable

# Fitting SVR to the dataset
regressor = SVR(kernel='rbf')
regressor.fit(X, y.ravel())  # Fit the SVR model

# Predicting a new result with SVR
print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1, 1)))  # Predicting for a new value

# Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')  # Original data points
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color='blue')  # SVR predictions
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results with higher resolution and smoother curve
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.01)  # Create a grid for smoother curve
X_grid = X_grid.reshape((len(X_grid), 1))  # Reshape for prediction
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')  # Original data points
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1)), color='blue')  # SVR predictions
plt.title('Truth or Bluff (SVR - High Resolution)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
