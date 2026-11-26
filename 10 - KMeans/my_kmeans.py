import numpy as np


class KMeans:
    def __init__(self, n_clusters, max_iterations=100, random_state=42):
        self.is_fitted = False
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)

        self.centroids = self.__initiate_random_centroids(X)

        for _ in range(self.max_iterations):
            self.labels = np.asarray(self.__label_instances(X))
            updated_centroids = np.asarray(self.__update_centroids(X))

            if np.array_equal(self.centroids, updated_centroids):
                break

            self.centroids = updated_centroids

        self.is_fitted = True

        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError('Model is not fitted')

        X = np.asarray(X)

        labels = []

        for x in X:
            sq_diffs = (self.centroids - x) ** 2
            distances = np.sqrt(np.sum(sq_diffs, axis=1))

            label = np.argmin(distances)
            labels.append(label)

        return labels

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def __initiate_random_centroids(self, X):
        random_centroids = np.random.randn(self.n_clusters, X.shape[1])
        return random_centroids

    def __label_instances(self, X):
        X = np.asarray(X)

        labels = []

        for x in X:
            sq_diffs = (self.centroids - x) ** 2
            distances = np.sqrt(np.sum(sq_diffs, axis=1))

            label = np.argmin(distances)
            labels.append(label)

        return labels

    def __update_centroids(self, X):
        X = np.asarray(X)

        updated_clusters = []

        for i in range(self.n_clusters):
            cluster_indices = self.labels == i
            cluster_instances = X[cluster_indices]

            if len(cluster_instances) == 0:
                updated_clusters.append(self.centroids[i])
            else:
                updated_cluster = cluster_instances.mean(axis=0).tolist()
                updated_clusters.append(updated_cluster)

        return updated_clusters
