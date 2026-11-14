
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create dataset
data = {
    'Income': [25000, 30000, 45000, 50000, 60000,
               70000, 80000, 90000, 35000, 55000,
               100000, 40000, 75000, 28000, 85000],
    'Age': [22, 25, 30, 35, 40, 45, 50, 55,
            28, 38, 48, 32, 42, 24, 52],
    'Loan': [10000, 15000, 12000, 20000, 18000,
             25000, 20000, 30000, 10000, 22000,
             25000, 15000, 20000, 12000, 28000],
    'Payment_History': [1, 0, 1, 1, 1, 0, 1, 1,
                        0, 1, 1, 0, 1, 0, 1],
    'Credit_Score': ['Low', 'Low', 'Average', 'Average',
                     'Good', 'Average', 'Good', 'Good',
                     'Low', 'Average', 'Good', 'Low',
                     'Good', 'Low', 'Good']
}

df = pd.DataFrame(data)

# Separate features and target
X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']

# Encode target
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Output
print("Credit Score Classification")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(
    y_test, y_pred,
    labels=[0, 1, 2],
    target_names=encoder.classes_,
    zero_division=0
))

# Predict new customer
new_customer = pd.DataFrame(
    [[65000, 35, 15000, 1]],
    columns=X.columns
)

prediction = model.predict(new_customer)
print("Predicted Credit Score:",
      encoder.inverse_transform(prediction)[0])
