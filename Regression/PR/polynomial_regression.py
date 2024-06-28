# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing data
df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
# print(X)

# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(X, y)

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_matrix = PolynomialFeatures(degree=5)
X_poly = poly_matrix.fit_transform(X)
# print(X_poly)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y)

# Visualising the Linear Regression results
plt.figure(1)
plt.scatter(X, y, color='red')
plt.plot(X, lin_regressor.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')

# Visualising the Polynomial Regression results
plt.figure(2)
plt.scatter(X, y, color='red')
plt.plot(X, poly_regressor.predict(X_poly), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
plt.figure(3)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, poly_regressor.predict(poly_matrix.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
print(lin_regressor.predict([[6.5]]))

# Predicting a new result with Polynomial Regression
print(poly_regressor.predict(poly_matrix.fit_transform([[6.5]])))
