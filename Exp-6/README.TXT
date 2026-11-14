import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data = {
    'RAM': [2, 4, 6, 8, 3, 12, 16, 8],
    'Battery': [2000, 3000, 4000, 5000, 2500, 5000, 6000, 4500],
    'Camera': [5, 8, 12, 16, 5, 24, 48, 20],
    'Storage': [16, 32, 64, 128, 16, 256, 512, 128],
    'Price': [0, 0, 1, 2, 0, 3, 3, 2]
}
df = pd.DataFrame(data)
X = df.drop('Price', axis=1)
y = df['Price']
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
print("Predicted Price Range:",
      model.predict([[8, 5000, 48, 128]])[0])
