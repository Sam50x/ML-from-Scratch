import numpy as np

np.random.seed(42)

class Linear_Regression:
    def __init__(self, mode='normal_equation'):
        if mode != 'normal_equation' and mode != 'gradient_descent':
            raise ValueError('Linear Regression mode can either be \'normal_equation\' or \'gradient_descent\'')

        self.is_fitted = False

        self.mode = mode
        self._weights = None
        self._intercept = None

    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        if self.mode == 'normal_equation':
            self.__fit_using_normal_equation(X, y)
        elif self.mode == 'gradient_descent':
            self.__fit_using_gradient_descent(X, y, epochs, learning_rate)

        self.is_fitted = True


    def predict(self, X):
        if not self.is_fitted:
            raise ValueError('Model is not fitted')

        y_pred = X @ self._weights.T + self._intercept

        return y_pred

    def __fit_using_normal_equation(self, X, y):
        X_b = np.c_[np.ones(X.shape[0]), X]
        
        best_theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        
        self._intercept = best_theta[0]
        self._weights = best_theta[1:]

    def __fit_using_gradient_descent(self, X, y, epochs, learning_rate):
        n = X.shape[1]

        b = np.random.random()
        w = np.random.random(n)

        for _ in range(epochs):
            y_pred = X @ w.T + b

            grad_b, grad_w = self.__calculate_gradients(X, y, y_pred)
            b = b - learning_rate * grad_b
            w = w - learning_rate * grad_w

        self._intercept = b
        self._weights = w

    def __calculate_gradients(self, X, y_true, y_pred):
        m = y_true.shape[0]

        grad_b = (2/m * np.sum(y_pred - y_true))
        grad_w = (2/m * (X.T @ (y_pred - y_true)))

        return grad_b, grad_w
