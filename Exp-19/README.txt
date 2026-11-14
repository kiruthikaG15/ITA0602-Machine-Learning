
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create dataset
data = {
    'Year': [2015, 2016, 2017, 2018, 2019,
             2020, 2021, 2022, 2014, 2013,
             2018, 2019, 2020, 2021, 2022],
    'Kilometers': [50000, 45000, 40000, 35000, 30000,
                   25000, 20000, 15000, 60000, 70000,
                   38000, 32000, 27000, 22000, 12000],
    'Age': [9, 8, 7, 6, 5, 4, 3, 2, 10, 11, 6, 5, 4, 3, 2],
    'Price': [300000, 350000, 400000, 450000, 500000,
              550000, 600000, 650000, 250000, 200000,
              430000, 480000, 530000, 580000, 700000]
}

df = pd.DataFrame(data)

# Separate input and output
X = df[['Year', 'Kilometers', 'Age']]
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
print("Car Price Prediction")
print("Actual Prices:", list(y_test))
print("Predicted Prices:", y_pred)

print("Mean Squared Error:",
      mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict new car price
new_car = pd.DataFrame(
    [[2022, 10000, 2]],
    columns=X.columns
)

print("Predicted Car Price:",
      model.predict(new_car)[0])
