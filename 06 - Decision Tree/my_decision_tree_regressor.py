import numpy as np

class Decision_Tree_Regressor:
    def __init__(self):
        self.is_fitted = False

        self.root = None

    def fit(self, X, y, max_depth=None, frac_features=None, random_state=42):
        overall_mse = self.__calculate_mse(y)

        self.root = Node(y, overall_mse)

        self.__create_split_nodes(X, y, parent=self.root, max_depth=max_depth, current_depth=0, frac_features=frac_features, random_state=random_state)

        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError('Model is not fitted')

        y_pred = np.array([], dtype='int')

        for _, x in X.iterrows():
            current_node = self.root

            while current_node.true_node:
                if x[current_node.split_feature_name] <= current_node.split_feature_value:
                    current_node = current_node.true_node
                else:
                    current_node = current_node.false_node

            y_pred = np.append(y_pred, current_node.prediction)

        return y_pred

    def __calculate_mse(self, y_node):
        y_pred_node = y_node.mean()

        mse_node = np.sum((y_pred_node - y_node) ** 2) / len(y_node)
        return mse_node

    def __find_split(self, X_node, y_node):
        min_cost = float('inf')
        min_cost_feature_name = None
        min_cost_feature_value = None

        for c in X_node.columns:
            values = sorted(set(X_node[c]))

            for i in range(len(values) - 1):
                val = (values[i] + values[i + 1]) / 2

                current_cost = self.__calculate_cart_cost(X_node, y_node, c, val)

                if current_cost < min_cost:
                    min_cost = current_cost
                    min_cost_feature_name = c
                    min_cost_feature_value = val

        return min_cost_feature_name, min_cost_feature_value

    def __calculate_cart_cost(self, X_node, y_node, feature_name, feature_value):
        mask_left = (X_node[feature_name] <= feature_value)
        mask_right = (X_node[feature_name] > feature_value)

        y_left = y_node[mask_left]
        y_right = y_node[mask_right]

        mse_left = self.__calculate_mse(y_left)
        mse_right = self.__calculate_mse(y_right)

        count_left = len(y_left)
        count_right = len(y_right)
        count_total = len(y_node)

        cart_cost = ((count_left / count_total) * mse_left) + ((count_right / count_total) * mse_right)

        return cart_cost

    def __create_split_nodes(self, X_node, y_node, parent, max_depth, current_depth, frac_features, random_state):
        if (max_depth and max_depth < current_depth) or len(X_node) == 1 or parent.mse == 0:
            return

        if not frac_features is None:
            sampled_cols = X_node.sample(frac=frac_features, random_state=random_state * current_depth, axis=1).columns

            X_node = X_node.loc[:, sampled_cols]

        split_feature_name, split_feature_value = self.__find_split(X_node, y_node)

        if split_feature_name is None:
            return

        mask_true = (X_node[split_feature_name] <= split_feature_value)
        mask_false = (X_node[split_feature_name] > split_feature_value)

        mse_true = self.__calculate_mse(y_node[mask_true])
        mse_false = self.__calculate_mse(y_node[mask_false])

        if mse_true < parent.mse or mse_false < parent.mse:
            parent.split_feature_name = split_feature_name
            parent.split_feature_value = split_feature_value

            parent.true_node = Node(y_node[mask_true], mse_true)
            parent.false_node = Node(y_node[mask_false], mse_false)

            self.__create_split_nodes(X_node[mask_true], y_node[mask_true], parent.true_node, max_depth, current_depth + 1, frac_features, random_state)
            self.__create_split_nodes(X_node[mask_false], y_node[mask_false], parent.false_node, max_depth, current_depth + 1, frac_features, random_state + 1)

        return


class Node:
    def __init__(self, y_node, mse):
        self.prediction = y_node.mean()

        self.mse = mse

        self.split_feature_name = None
        self.split_feature_value = None

        self.true_node = None
        self.false_node = None

    def set_split(self, feature_name, feature_value):
        self.feature_name = feature_name
        self.feature_value = feature_value

    def set_true_node(self, true_node):
        self.true_node = true_node

    def set_false_node(self, false_node):
        self.false_node = false_node