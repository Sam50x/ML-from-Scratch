import numpy as np

import sys
sys.path.append('../06 - Decision Tree')

import importlib
import my_decision_tree_regressor

importlib.reload(my_decision_tree_regressor)
from my_decision_tree_regressor import Decision_Tree_Regressor

class Gradient_Boosting_Regressor:
    def __init__(self):
        self.is_fitted = False

    def fit(self, X, y, max_depth=None, n_estimators=100):
        
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError('Model is not fitted')

        