import numpy as np

np.random.seed(42)

class Linear_Regression:
    def __init__(self, mode='normal_equation'):
        if mode != 'normal_equation' and mode != 'gradient_descent':
            raise ValueError('Linear Regression mode can either be \'normal_equation\' or \'gradient_descent\'')

        self.mode = mode
        self._weights = None
        self._intercept = None

    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        if self.mode == 'normal_equation':
            self.__fit_using_normal_equation(X, y)
        elif self.mode == 'gradient_descent':
            self.__fit_using_gradient_descent(X, y, epochs, learning_rate)

    def predict(self, X):
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

        for epoch in range(epochs):
            y_pred = X @ w.T + b

            if epoch % 100 == 0:
                mse = self.__calculate_mse(y, y_pred)
                print(f'MSE at epoch {epoch} = {mse}')

            grad_b, grad_w = self.__calculate_gradients(X, y, y_pred)
            b = b - learning_rate * grad_b
            w = w - learning_rate * grad_w

        mse = self.__calculate_mse(y, y_pred)
        print(f'MSE at epoch {epochs} = {mse}')

        self._intercept = b
        self._weights = w

    def __calculate_mse(self, y_true, y_pred):
        m = y_true.shape[0]

        mse = 1/m * np.sum((y_pred - y_true) ** 2)

        return mse

    def __calculate_gradients(self, X, y_true, y_pred):
        m = y_true.shape[0]

        grad_b = (2/m * np.sum(y_pred - y_true))
        grad_w = (2/m * (X.T @ (y_pred - y_true)))

        return grad_b, grad_w