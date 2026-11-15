import numpy as np

import sys
sys.path.append('../06 - Decision Tree')

from my_decision_tree_regressor import Decision_Tree_Regressor

class Random_Forest_Regressor:
    def __init__(self, max_depth=None, n_estimators=100, frac_instances=0.8, frac_features=0.8):
        self.is_fitted = False
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.frac_instances = frac_instances
        self.frac_features = frac_features

    def fit(self, X, y):
        self.estimators = np.array([])

        for n in range(self.n_estimators):
            sampled_instances = X.sample(frac=self.frac_instances, replace=True, random_state=n).index

            tree = Decision_Tree_Regressor(max_depth=self.max_depth, frac_features=self.frac_features, random_state=n)
            tree.fit(X.loc[sampled_instances], y[sampled_instances])

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

        y_pred = np.array([])

        for i in range(len(X)):
            votes = predictions[:, i]

            final_pred = votes.sum() / len(votes)

            y_pred = np.append(y_pred, final_pred)

        return y_pred