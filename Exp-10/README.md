from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

X = [[25, 30000, 650], [35, 60000, 750],
     [45, 80000, 800], [22, 20000, 550],
     [40, 70000, 780], [28, 35000, 600],
     [50, 90000, 820], [30, 45000, 680]]

y = ['No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']

le = LabelEncoder()
y = le.fit_transform(y)

model = GaussianNB()
model.fit(X, y)

new_loan = [[32, 50000, 720]]
pred = model.predict(new_loan)

print("Loan Prediction:", le.inverse_transform(pred)[0])
