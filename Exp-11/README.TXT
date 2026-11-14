from sklearn.ensemble import RandomForestRegressor

X = [[1, 100], [2, 150], [3, 200], [4, 250],
     [5, 300], [6, 350], [7, 400], [8, 450],
     [9, 500], [10, 550]]

y = [120, 150, 180, 220, 260, 300, 340, 380, 420, 460]

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

prediction = model.predict([[11, 600]])

print("Predicted Future Sales:", prediction[0])
