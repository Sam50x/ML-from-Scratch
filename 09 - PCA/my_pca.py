class PCA:
    def __init__(self):
        self.is_fitted = False

    def fit(self):
        

        self.is_fitted = True

        return self

    def transform(self):
        if not self.is_fitted:
            raise ValueError('Model is not fitted')

        

    def fit_transform(self):
        return self.fit().transform()