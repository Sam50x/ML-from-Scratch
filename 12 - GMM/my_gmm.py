import numpy as np

class GMM:
    def __init__(self, n_clusters, max_iterations=100, random_state=42):
        self.is_fitted = False
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X)

        np.random.seed(self.random_state)

        self.__initialize_random_parameters(X)

        for _ in range(self.max_iterations):
            self.__expectation_step(X)
            self.__maximization_step(X)

        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError('Model is not fitted')

        X = np.asarray(X)

        self.__expectation_step(X)

        return np.argmax(self.gammas, axis=1)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError('Model is not fitted')

        X = np.asarray(X)

        self.__expectation_step(X)

        return self.gammas

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def __initialize_random_parameters(self, X):
        self.means = np.random.randn(self.n_clusters, X.shape[1])

        self.covariances = np.array([np.eye(X.shape[1]) for _ in range(self.n_clusters)])

        self.weights = np.array([1 / self.n_clusters] * self.n_clusters)

    def __expectation_step(self, X):
        probas = np.array([])

        for i in range(self.n_clusters):
            proba = self.weights[i] * self.__gaussian_pdf(X, i)
            probas = np.append(probas, proba)

        probas = probas.reshape(self.n_clusters, X.shape[0]).T

        self.gammas = probas / np.sum(probas, axis=1, keepdims=True)

        self.N_k = np.sum(self.gammas, axis=0)

    def __maximization_step(self, X):
        self.weights = self.N_k / X.shape[0]

        self.means = (self.gammas.T @ X) / self.N_k[:, None]

        for i in range(self.n_clusters):
            diff = X - self.means[i]
            self.covariances[i] = (diff.T * self.gammas[:, i]) @ diff
            self.covariances[i] /= self.N_k[i]
            self.covariances[i] += 1e-6 * np.eye(X.shape[1])

    def __gaussian_pdf(self, X, n_cluster):
        d = X.shape[1]

        det = np.linalg.det(self.covariances[n_cluster])
        inv = np.linalg.inv(self.covariances[n_cluster])

        diff = X - self.means[n_cluster]

        denominator = np.sqrt((2 * np.pi) ** d * det)

        exponent = np.exp(-0.5 * np.sum(diff @ inv * diff, axis=1))

        return exponent / denominator
