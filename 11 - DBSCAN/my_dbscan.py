import numpy as np


class DBSCAN:
    def __init__(self, epsilon=0.2, min_samples=5):
        self.is_fitted = False
        self.epsilon = epsilon
        self.min_samples = min_samples

    def fit(self, X):
        X = np.asarray(X)

        self.labels = [-1] * len(X)

        next_label = 0

        for i in range(len(X)):
            x = X[i]
            sq_diffs = (X - x) ** 2
            distances = np.sqrt(np.sum(sq_diffs, axis=1))

            radius_mask = distances <= self.epsilon
            neighbors_indices = np.where(radius_mask)[0]

            if len(neighbors_indices) >= self.min_samples - 1:
                if self.labels[i] == -1:
                    self.labels[i] = next_label
                    next_label += 1
                for neighbor_index in neighbors_indices:
                    self.labels[neighbor_index] = self.labels[i]

        self.is_fitted = True

    def fit_predict(self, X):
        self.fit(X)
        print(self.labels)
        return self.labels
