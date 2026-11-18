import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.is_fitted = False
        self.n_components = n_components

    def fit(self, X):
        X_centered = X - X.mean(axis=0)

        _, S, Vt = np.linalg.svd(X_centered)

        self.V = Vt.T
        if self.n_components is None:
            self.W = self.V
        else:
            self.W = self.V[:, :self.n_components]

        explained_variance = S**2 / (X.shape[0] - 1)
        explained_variance_ratio = explained_variance / explained_variance.sum()
        self.cumulative_explained_variance = np.cumsum(explained_variance_ratio)

        self.is_fitted = True

        return self

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError('Model is not fitted')

        X_projected = X @ self.W
        return X_projected

    def fit_transform(self, X):
        return self.fit(X).transform(X)