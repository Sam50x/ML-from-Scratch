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



    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def __initialize_random_parameters(self, X):
        pass

    def __expectation_step(self, X):
        pass

    def __maximization_step(self, X):
        pass

    def __gaussian_pdf(self, X):
        pass
