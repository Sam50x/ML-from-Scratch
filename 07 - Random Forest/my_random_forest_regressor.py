import numpy as np

import sys
sys.path.append('../06 - Decision Tree')

import importlib
import my_decision_tree_regressor

importlib.reload(my_decision_tree_regressor)
from my_decision_tree_regressor import Decision_Tree_Regressor

class Random_Forest_Regressor:
    def __init__(self):
        self.is_fitted = False

    def fit(self, X, y, max_depth=None, n_estimators=100, frac_instances=0.8, frac_features=0.8):
        self.estimators = np.array([])

        for n in range(n_estimators):
            sampled_instances = X.sample(frac=frac_instances, replace=True, random_state=n).index

            tree = Decision_Tree_Regressor()
            tree.fit(X.loc[sampled_instances], y[sampled_instances], max_depth, frac_features=frac_features, random_state=n)

            self.estimators = np.append(self.estimators, tree)

        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError('Model is not fitted')

        predictions = []

        for estimator in self.estimators:
            estimator_predictions = estimator.predict(X)

            predictions.append(estimator_predictions)

        predictions = np.vstack(predictions)

        y_pred = np.array([], dtype='int')

        for i in range(len(X)):
            votes = predictions[:, i]

            final_pred = votes.sum() / len(votes)

            y_pred = np.append(y_pred, final_pred)

        return y_pred