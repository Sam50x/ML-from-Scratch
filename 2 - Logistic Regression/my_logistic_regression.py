import numpy as np

class Logistic_Regression():
    def __init__(self):
        self.is_fitted = False
        
        self._bias = None
        self._weights = None

    def fit(self, X, y, epochs=1000, learning_rate=0.1):
        p = X.shape[1]

        w = np.random.randn(p)
        b = np.random.randn()

        for _ in range(epochs):
            linear = X @ w.T + b

            proba = self._predict_proba(linear)

            grad_b, grad_w = self._calculate_gradients(X, y, proba)

            b = b - learning_rate * grad_b
            w = w - learning_rate * grad_w

        self._bias = b
        self._weights = w
        self.is_fitted = True

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError('Model is not fitted')

        linear = X @ self._weights.T + self._bias
        proba = 1 / (1 + np.exp(-linear))

        return proba

    def predict(self, X, threshold=0.5):
        if not self.is_fitted:
            raise ValueError('Model is not fitted')

        proba = self.predict_proba(X)
        y_pred = np.where(proba >= threshold, 1, 0)

        return y_pred

    def _predict_proba(self, linear):
        proba = 1 / (1 + np.exp(-linear))

        return proba

    def _calculate_gradients(self, X, y, proba):
        m = y.shape[0]

        grad_b = 1 / m * np.sum(proba - y)
        grad_w = 1 / m * (X.T @ (proba - y))

        return grad_b, grad_w