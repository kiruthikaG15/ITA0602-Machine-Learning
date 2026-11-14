
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Create dataset directly
data = {
    'X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Y': [3, 8, 15, 24, 35, 48, 63, 80, 99, 120]
}

df = pd.DataFrame(data)

# Input and output
X = df[['X']]
y = df['Y']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
poly_pred = poly_model.predict(X_test_poly)

# Compare results
print("Linear Regression")
print("MSE:", mean_squared_error(y_test, linear_pred))
print("R2 Score:", r2_score(y_test, linear_pred))

print("\nPolynomial Regression")
print("MSE:", mean_squared_error(y_test, poly_pred))
print("R2 Score:", r2_score(y_test, poly_pred))

# Plot graph
X_plot = pd.DataFrame({'X': range(1, 11)})

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_plot, linear_model.predict(X_plot),
         color='red', label='Linear Regression')
plt.plot(X_plot, poly_model.predict(poly.transform(X_plot)),
         color='green', label='Polynomial Regression')

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()

