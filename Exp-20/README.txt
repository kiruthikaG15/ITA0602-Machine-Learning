
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create dataset
data = {
    'Area': [1000, 1200, 1500, 1800, 2000,
             2200, 2500, 2800, 3000, 3200,
             1100, 1400, 1700, 2100, 2600],
    'Bedrooms': [2, 2, 3, 3, 3, 4, 4, 4,
                 5, 5, 2, 3, 3, 4, 4],
    'Bathrooms': [1, 1, 2, 2, 2, 3, 3, 3,
                  3, 4, 1, 2, 2, 3, 3],
    'Price': [2000000, 2400000, 3000000, 3600000,
              4000000, 4500000, 5000000, 5600000,
              6000000, 6500000, 2200000, 2800000,
              3400000, 4300000, 5200000]
}

df = pd.DataFrame(data)

# Separate input and output
X = df[['Area', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Output
print("House Price Prediction")
print("Actual Prices:", list(y_test))
print("Predicted Prices:", y_pred)

print("Mean Squared Error:",
      mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict new house price
new_house = pd.DataFrame(
    [[2000, 3, 2]],
    columns=X.columns
)

print("Predicted House Price:",
      model.predict(new_house)[0])

# Plot graph
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("House Price Prediction")
plt.show()
