import numpy as np

class SVM:
    def __init__(self):
        self.is_fitted = False
        self.weights_ = None
        self.bias_ = None

    def fit(self, X, y, epochs=1000, learning_rate=0.01, regularization_rate=0.01):
        

        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError('Model is not fitted')

        