import numpy as np

import sys
sys.path.append('../06 - Decision Tree')

from my_decision_tree_regressor import Decision_Tree_Regressor

class Gradient_Boosting_Regressor:
    def __init__(self, max_depth=4, n_estimators=50, learning_rate=0.1):
        self.is_fitted = False
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, X, y):
        self.initial_leaf = y.mean()

        self.estimators = np.array([])
        current_residuals = y - self.initial_leaf

        for i in range(self.n_estimators):
            tree = Decision_Tree_Regressor(max_depth=self.max_depth)
            tree.fit(X, current_residuals)
            tree_pred = tree.predict(X)
            current_residuals -= self.learning_rate * tree_pred

            self.estimators = np.append(self.estimators, tree)

        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError('Model is not fitted')

        y_pred = np.array([self.initial_leaf] * X.shape[0])

        for estimator in self.estimators:
            estimator_pred = estimator.predict(X)
            y_pred += self.learning_rate * estimator_pred

        return y_pred
