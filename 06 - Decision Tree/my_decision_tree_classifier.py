import numpy as np

np.random.seed(42)

class Decision_Tree_Classifier:
    def __init__(self, mode='gini'):
        self.is_fitted = False

        if mode != 'gini' and mode != 'entropy':
            raise ValueError('Decision Tree mode can either be \'gini\' or \'entropy\'')

        self.mode = mode

    def fit(self, X, y):
        

        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError('Model is not fitted')

        


class Node:
    def __init__(self):
        pass