import numpy as np

import sys
sys.path.append('../06 - Decision Tree')

import importlib
import my_decision_tree_regressor

importlib.reload(my_decision_tree_regressor)
from my_decision_tree_regressor import Decision_Tree_Regressor

class Gradient_Boosting_Classifier:
    def __init__(self):
        self.is_fitted = False

    def fit(self, X, y, max_depth=4, n_estimators=50, learning_rate=0.1):
        class_counts = np.unique_counts(y)
        self.initial_leaf = np.log((class_counts.counts[1] + 1e+15) / (class_counts.counts[0] + 1e+15))

        self.learning_rate = learning_rate

        self.estimators = np.array([])

        current_raw = np.array([self.initial_leaf] * len(y))
        current_proba = self.__calculate_proba(current_raw)
        current_residuals = y - current_proba

        for i in range(n_estimators):
            tree = Decision_Tree_Regressor()
            tree.fit(X, current_residuals, max_depth)
            tree_pred = tree.predict(X)

            current_raw += learning_rate * tree_pred
            current_proba = self.__calculate_proba(current_raw)
            current_residuals = y - current_proba

            self.estimators = np.append(self.estimators, tree)

        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError('Model is not fitted')

        y_raw = np.array([self.initial_leaf] * X.shape[0])

        for estimator in self.estimators:
            estimator_pred = estimator.predict(X)
            y_raw += self.learning_rate * estimator_pred

        y_proba = self.__calculate_proba(y_raw)

        y_pred = np.where(y_proba > 0.5, 1, 0)

        return y_pred

    def __calculate_proba(self, y):
        proba = np.exp(y) / (1 + np.exp(y))
        return proba