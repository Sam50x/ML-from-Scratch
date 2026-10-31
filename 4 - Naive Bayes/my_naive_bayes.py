class Naive_Bayes:
    def __init__(self):
        self.is_fitted = False

    def fit(self, X, y, alpha=1):
        df = X.copy()
        df['label'] = y

        label_counts = df['label'].value_counts()
        word_counts = df.groupby('label').sum() + alpha
        total_words_in_label = word_counts.sum(axis=1)

        proba_labels = label_counts / label_counts.sum()
        proba_words = word_counts.div(total_words_in_label, axis=0)

        self.proba_labels = proba_labels
        self.proba_words = proba_words

        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError('Model is not fitted')

        y_pred = []

        for x in X.T:
            row = X.iloc[x]
            row_words = row[row > 0]

            class_proba = self.proba_labels.copy()
            for word, count in row_words.items():
                class_proba *= ((self.proba_words[word]) ** count)

            y_pred.append(class_proba.idxmax())

        return y_pred