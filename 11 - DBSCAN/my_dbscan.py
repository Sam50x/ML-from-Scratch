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
            if self.labels[i] != -1:
                continue
            self.__label_neighborhood(X, i, next_label)
            next_label += 1

        self.is_fitted = True

    def fit_predict(self, X):
        self.fit(X)

        return self.labels

    def __label_neighborhood(self, X, index, label):
        x = X[index]
        sq_diffs = (X - x) ** 2
        distances = np.sqrt(np.sum(sq_diffs, axis=1))

        radius_mask = distances <= self.epsilon
        neighbors_indices = np.where(radius_mask)[0]

        if len(neighbors_indices) >= self.min_samples - 1:
            if self.labels[index] == -1:
                self.labels[index] = label
            for neighbor_index in neighbors_indices:
                if self.labels[neighbor_index] != -1:
                    continue
                self.labels[neighbor_index] = self.labels[index]
                self.__label_neighborhood(X, neighbor_index, label)
