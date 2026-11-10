from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

X = [[22, 25000], [25, 30000], [35, 60000],
     [45, 80000], [28, 35000], [50, 90000],
     [32, 55000], [40, 75000], [23, 28000], [48, 85000]]

y = [0, 0, 1, 1, 0, 1, 1, 1, 0, 1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
