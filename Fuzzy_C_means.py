#In this example, I'll use the Iris dataset, a popular dataset in machine learning. The Iris dataset consists of measurements of 
#sepal length, sepal width, petal length, and petal width for 150 iris flowers, with 50 samples from each of three different species (setosa, versicolor, and virginica).
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt

class FuzzyCMeans:
    def __init__(self, n_clusters, max_iter=100, m=2, error=1e-5, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error
        self.random_state = random_state
        self.centers = None
        self.membership_matrix = None

    def initialize_membership_matrix(self, n_samples):
        np.random.seed(self.random_state)
        membership_matrix = np.random.rand(n_samples, self.n_clusters)
        membership_matrix /= np.sum(membership_matrix, axis=1)[:, np.newaxis]
        return membership_matrix

    def update_centers(self, X):
        um_power_m = np.power(self.membership_matrix, self.m)
        centers = np.dot(um_power_m.T, X) / um_power_m.sum(axis=0)[:, np.newaxis]
        return centers

    def update_membership_matrix(self, X):
        distances = pairwise_distances_argmin_min(X, self.centers)[1]
        distances = np.maximum(distances, np.finfo(np.float64).eps)
        inv_distances = 1.0 / distances ** (2 / (self.m - 1))
        inv_distances_sum = np.sum(inv_distances, axis=1)
        membership_matrix = inv_distances / inv_distances_sum[:, np.newaxis]
        return membership_matrix

    def fit(self, X):
        n_samples, n_features = X.shape

        # Initialize membership matrix randomly
        self.membership_matrix = self.initialize_membership_matrix(n_samples)

        for _ in range(self.max_iter):
            # Update cluster centers
            self.centers = self.update_centers(X)

            # Update membership matrix
            new_membership_matrix = self.update_membership_matrix(X)

            # Check for convergence
            if np.linalg.norm(new_membership_matrix - self.membership_matrix) < self.error:
                break

            self.membership_matrix = new_membership_matrix

    def predict(self, X):
        return np.argmax(self.membership_matrix, axis=1)

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Fuzzy C-means clustering on the training data
fcm = FuzzyCMeans(n_clusters=3, max_iter=100, random_state=42)
fcm.fit(X_train)

# Predict cluster labels for training and testing data
train_labels = fcm.predict(X_train)
test_labels = fcm.predict(X_test)

# Visualize the results on the training data
plt.scatter(X_train[:, 0], X_train[:, 1], c=train_labels, cmap='viridis', edgecolors='k')
plt.scatter(fcm.centers[:, 0], fcm.centers[:, 1], marker='x', s=200, linewidths=3, color='red', label='Centers')
plt.title("Fuzzy C-means Clustering on Training Data")
plt.legend()
plt.show()

# Evaluate the accuracy on the testing data
predicted_species = np.argmax(fcm.membership_matrix, axis=1)
accuracy = accuracy_score(y_test, predicted_species)
print(f"Accuracy: {accuracy * 100:.2f}%")
