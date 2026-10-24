
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target
print("Dataset Shape:", X.shape)
print("Flower Classes:", iris.target_names)

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

sc = StandardScaler()
Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)
Y = np.eye(3)[ytr]

np.random.seed(42)
W1, W2 = np.random.randn(4, 8)*0.1, np.random.randn(8, 3)*0.1
b1, b2 = np.zeros((1, 8)), np.zeros((1, 3))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e/e.sum(axis=1, keepdims=True)

# Backpropagation
for epoch in range(5000):
    A1 = sigmoid(Xtr @ W1 + b1)
    A2 = softmax(A1 @ W2 + b2)
    loss = -np.mean(np.sum(Y*np.log(A2+1e-9), axis=1))

    d2 = (A2-Y)/len(Xtr)
    d1 = (d2 @ W2.T)*A1*(1-A1)

    W2 -= 0.1*(A1.T @ d2)
    b2 -= 0.1*d2.sum(axis=0, keepdims=True)
    W1 -= 0.1*(Xtr.T @ d1)
    b1 -= 0.1*d1.sum(axis=0, keepdims=True)

    if epoch % 1000 == 0:
        print("Epoch:", epoch, "Loss:", round(loss, 4))

# Testing
def predict(X):
    return softmax(sigmoid(X @ W1+b1) @ W2+b2).argmax(axis=1)

print("\nModel Accuracy:",
      round(accuracy_score(yte, predict(Xte))*100, 2), "%")

sample = np.array([[5.1, 3.5, 1.4, 0.2]])
print("\nNew Sample:", sample[0])
print("Predicted Flower:", iris.target_names[predict(sc.transform(sample))[0]])

