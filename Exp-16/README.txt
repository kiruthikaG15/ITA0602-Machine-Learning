
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# Create dataset
X, y = make_blobs(
    n_samples=200,
    centers=3,
    random_state=42
)

# Implement EM using Gaussian Mixture Model
model = GaussianMixture(
    n_components=3,
    random_state=42
)

# Train model
model.fit(X)

# Predict clusters
labels = model.predict(X)

# Display results
print("Expectation Maximization Algorithm")
print("Number of clusters:", model.n_components)
print("Predicted clusters:", np.unique(labels))
print("Model converged:", model.converged_)
print("Iterations:", model.n_iter_)

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("EM Algorithm - Gaussian Mixture Model")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
