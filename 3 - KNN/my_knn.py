import numpy as np
import pandas as pd

class KNN():
    def __init__(self, n_neighbors):
        self.is_fitted = False
        self.k = n_neighbors
        self.X_ = None
        self.y_ = None

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y

        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError('Model is not fitted')

        y_pred = np.array([])

        for i in X:
            sq_diff = (self.X_ - i) ** 2
            distances = np.sqrt(np.sum(sq_diff, axis=1))

            distances_classes = pd.DataFrame(distances, self.y_, columns=['distance']).sort_values(by='distance', ascending=True)

            classes = np.array([])
            for j in range(self.k):
                current_class = distances_classes.iloc[j].name
                classes = np.append(classes, current_class)

            value_counts = np.unique_counts(classes)
            max_count = np.argmax(value_counts.counts)
            most_frequent_class_in_k = value_counts.values[max_count]

            y_pred = np.append(y_pred, most_frequent_class_in_k)

        return y_pred