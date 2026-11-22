import numpy as np
np.random.seed(42)

class KMeans:
    def __init__(self, n_clusters):
        self.is_fitted = False
        self.n_clusters = n_clusters

    def fit(self, X):


        self.is_fitted = True

        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError('Model is not fitted')



    def fit_predict(self, X):
        return self.fit(X).predict(X)
