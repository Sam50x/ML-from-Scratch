import numpy as np
from collections import deque

class DBSCAN:
    def __init__(self, epsilon=0.2, min_samples=5):
        self.epsilon = epsilon
        self.min_samples = min_samples

    def fit(self, X):
        X = np.asarray(X)

        self.labels = [-1] * len(X)
        self.__calculate_distances(X)

        self.is_core_instance = [False] * len(X)
        self.__identify_core_instances()

        self.explored = [False] * len(X)
        self.n_clusters = 0

        for i in range(len(self.is_core_instance)):
            if self.is_core_instance[i] and not self.explored[i]:
                label = self.n_clusters
                self.n_clusters += 1
                self.__label_neighborhood(i, label)

    def fit_predict(self, X):
        self.fit(X)

        return self.labels

    def __calculate_distances(self, X):
        self.distances = []

        for x in X:
            self.distances.append(np.sum((X - x) ** 2, axis=1))

    def __identify_core_instances(self):
        self.neighborhood = []
        for i in range(len(self.distances)):
            neighbors_indices = np.where(self.distances[i] <= self.epsilon ** 2)[0]
            self.neighborhood.append(neighbors_indices)
            if len(neighbors_indices) >= self.min_samples:
                self.is_core_instance[i] = True

    def __label_neighborhood(self, index, label):
        queue = deque([index])

        while queue:
            current_index = queue.popleft()

            if self.explored[current_index]:
                continue

            self.labels[current_index] = label
            self.explored[current_index] = True

            if self.is_core_instance[current_index]:
                for neighbor_index in self.neighborhood[current_index]:
                    if not self.explored[neighbor_index]:
                        queue.append(neighbor_index)
