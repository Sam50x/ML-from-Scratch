import numpy as np
from scipy.optimize import minimize

np.random.seed(42)

class SVM:
    def __init__(self, mode='dual_problem'):
        self.is_fitted = False

        if mode != 'dual_problem' and mode != 'gradient_descent':
            raise ValueError('SVM mode can either be \'dual_problem\' or \'gradient_descent\'')

        self.mode = mode
        self.weights_ = None
        self.bias_ = None

    def fit(self, X, y, C=1.0, epochs=1000, learning_rate=0.01, regularization_rate=0.01):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        y_sign = np.where(y == 1, 1, -1)

        if self.mode == 'dual_problem':
            self.__fit_using_dual_problem(X, y_sign, C)
        elif self.mode == 'gradient_descent':
            self.__fit_using_gradient_descent(X, y_sign, epochs, learning_rate, regularization_rate)

        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError('Model is not fitted')

        f_x = X @ self.weights_.T + self.bias_

        y_pred = np.where(f_x > 0, 1, 0)
        return y_pred

    def __fit_using_dual_problem(self, X, y_sign, C):
        n_instances = X.shape[0]

        Q = X @ X.T

        def objective(a):
            return 0.5 * np.sum(np.outer(a*y_sign, a*y_sign) * Q) - np.sum(a)

        constraints = {
            'type': 'eq',
            'fun': lambda a: np.dot(a, y_sign)
        }

        result = minimize(
            objective,
            np.zeros(n_instances),
            constraints=constraints,
            bounds=[(0, C)],
            method='SLSQP'
        )

        alpha = result.x
        sv = (alpha > 1e-5) & (alpha < C - 1e-5)

        w = X.T @ (alpha * y_sign)
        b = np.mean(y_sign[sv] - X[sv] @ w.T)

        self.bias_ = b
        self.weights_ = w

    def __fit_using_gradient_descent(self, X, y_sign, epochs, learning_rate, regularization_rate):
        n_features = X.shape[1]

        b = np.random.randn(1)
        w = np.random.randn(n_features)

        for _ in range(epochs):

            grad_b, grad_w = self.__calculate_gradients(X, y_sign, w, b, regularization_rate)

            b -= learning_rate * grad_b
            w -= learning_rate * grad_w

        self.bias_ = b
        self.weights_ = w

    def __calculate_gradients(self, X, y_sign, w, b, regularization_rate):
        n_instances = X.shape[0]

        grad_b = 0
        grad_w = 2 * regularization_rate * w

        for i in range(n_instances):
            sign = y_sign[i]
            x = X[i]

            y_f_x = sign * (x @ w.T + b) >= 1

            if not y_f_x:
                grad_b += -sign
                grad_w += -sign * x

        return grad_b / n_instances, grad_w / n_instances