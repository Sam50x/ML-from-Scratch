# ML from Scratch

Core machine learning algorithms implemented from scratch with NumPy, then compared against their scikit-learn counterparts on real data.

The goal of this repo is to understand how each algorithm works under the hood: every model lives in its own folder with a clean implementation and a notebook that tests it on synthetic data and on a real Kaggle dataset.

## Models

| # | Model | Status | Data |
|---|-------|--------|------|
| 1 | [Linear Regression](./1%20-%20Linear%20Regression) | Done | Synthetic data, [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/) |
| 2 | [Logistic Regression](./2%20-%20Logistic%20Regression) | Done | Synthetic data, [Titanic](https://www.kaggle.com/competitions/titanic) |
| 3 | [K-Nearest Neighbors (KNN)](./3%20-%20KNN) | Done | Synthetic data, [Iris](https://www.kaggle.com/competitions/iris-classification-challenge) |
| 4 | Naive Bayes | Planned | |
| 5 | Support Vector Machine (SVM) | Planned | |
| 6 | Decision Tree | Planned | |
| 7 | Random Forest | Planned | |
| 8 | Boosting Trees | Planned | |
| 9 | Principal Component Analysis (PCA) | Planned | |
| 10 | K-Means | Planned | |
| 11 | DBSCAN | Planned | |
| 12 | Gaussian Mixture Model (GMM) | Planned | |

## 1 - Linear Regression

**Implementation:** [`my_linear_regression.py`](./1%20-%20Linear%20Regression/my_linear_regression.py)
**Walkthrough:** [`notebook.ipynb`](./1%20-%20Linear%20Regression/notebook.ipynb)

### How it works

The `Linear_Regression` class supports two ways of fitting the model, chosen with the `mode` argument:

- **`normal_equation`** (default): solves for the weights in closed form, `θ = (XᵀX)⁻¹ Xᵀy`, after adding a column of ones for the intercept.
- **`gradient_descent`**: starts from random weights and minimizes the MSE loss iteratively, using the gradients `∂L/∂w = (2/m) Xᵀ(ŷ − y)` and `∂L/∂b = (2/m) Σ(ŷ − y)`. Defaults: `epochs=1000`, `learning_rate=0.01`.

After fitting, the learned parameters are available as `weights_` and `intercept_`. Calling `predict` before `fit` raises a `ValueError`.

### Usage

```python
from my_linear_regression import Linear_Regression

# Closed-form solution
model = Linear_Regression(mode='normal_equation')
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

# Gradient descent
model = Linear_Regression(mode='gradient_descent')
model.fit(X_train, y_train, epochs=1000, learning_rate=0.01)
y_pred = model.predict(X_val)
```

### Experiments

**1. Synthetic data** (10 samples, 5 features, standard-normal inputs and weights, an intercept of 5, and standard-normal noise). Both solvers reach essentially the same MSE, which is close to the variance of the noise added to the data:

| Method | MSE |
|--------|-----|
| Normal Equation | 0.3750 |
| Gradient Descent | 0.3754 |

**2. House Prices (Kaggle).** The notebook walks through:

- **EDA:** summary statistics, missing values, columns that are mostly zeros, and a correlation heatmap.
- **Cleaning:** outliers in the highest-variance numeric columns (including `SalePrice`) are clipped to the 10th and 90th percentiles. Columns with more than 50% missing values, more than 50% zeros, or high correlation with other features are dropped.
- **Preprocessing:** a scikit-learn `ColumnTransformer` that imputes (median / most frequent), standardizes numeric features, and one-hot encodes categorical ones.
- **Training and evaluation:** a 70/30 train/validation split, comparing my two solvers against scikit-learn's `LinearRegression`. For the Kaggle submissions, each model is retrained on the full training set.

| Model | Validation RMSE | Kaggle score |
|-------|-----------------|--------------|
| Mine (Normal Equation) | 18,884.06 | 0.16408 |
| Mine (Gradient Descent) | 18,412.13 | 0.16216 |
| scikit-learn `LinearRegression` | 18,884.06 | 0.16408 |

Validation RMSE is measured in dollars on the clipped `SalePrice` values. The Kaggle score is the competition's metric (RMSE on the log of the price), so lower is better.

My normal equation implementation matches scikit-learn's results (the validation RMSEs differ by less than 0.000000001), which confirms that it is correct. Gradient descent scores slightly better on both the validation set and Kaggle, likely because it never fits the training data as tightly as the closed-form solution does.

## 2 - Logistic Regression

**Implementation:** [`my_logistic_regression.py`](./2%20-%20Logistic%20Regression/my_logistic_regression.py)
**Walkthrough:** [`notebook.ipynb`](./2%20-%20Logistic%20Regression/notebook.ipynb)

### How it works

The `Logistic_Regression` class is a binary classifier trained with gradient descent on the log loss (binary cross-entropy):

- **Model:** a linear combination of the features is passed through the sigmoid function, `p = 1 / (1 + e^(−(Xw + b)))`, which gives the probability of class 1.
- **Training:** weights and bias start from random normal values and are updated for a fixed number of epochs using the gradients `∂L/∂w = (1/m) Xᵀ(p − y)` and `∂L/∂b = (1/m) Σ(p − y)`. Defaults: `epochs=1000`, `learning_rate=0.1`.
- **Fitted parameters:** available as `weights_` and `bias_` after training.
- **Prediction:** `predict_proba` returns the probabilities, and `predict` turns them into 0/1 labels using a `threshold` (default `0.5`). Both raise a `ValueError` if the model has not been fitted.

### Usage

```python
from my_logistic_regression import Logistic_Regression

model = Logistic_Regression()
model.fit(X_train, y_train, epochs=1000, learning_rate=0.1)

proba = model.predict_proba(X_val)
y_pred = model.predict(X_val, threshold=0.5)
```

### Experiments

**1. Synthetic data** (100 samples, 5 features, labels generated from random weights plus noise through a sigmoid). The model reaches an accuracy of **0.96**, measured on the same data it was trained on.

**2. Titanic (Kaggle).** Predicting passenger survival. The notebook walks through:

- **EDA:** missing values, summary statistics, and plots of the survival distribution, age, and fare.
- **Feature engineering:** `FamilySize` (siblings/spouses + parents/children + 1) and an `IsAlone` flag.
- **Preprocessing:** a scikit-learn `ColumnTransformer` that imputes and standardizes the numeric features, one-hot encodes `Embarked`, ordinally encodes `Sex`, and passes `Pclass` and `IsAlone` through. `Name`, `Ticket`, and `Cabin` are dropped.
- **Training and evaluation:** a stratified 70/30 train/validation split, comparing my model against scikit-learn's `LogisticRegression`. For the Kaggle submissions, both models are retrained on the full training set.

| Model | Validation accuracy | Kaggle score |
|-------|---------------------|--------------|
| Mine | 0.8022 | 0.76315 |
| scikit-learn `LogisticRegression` | 0.8022 | 0.76315 |

My model matches scikit-learn on both the validation set and the Kaggle leaderboard.

## 3 - K-Nearest Neighbors (KNN)

**Implementation:** [`my_knn.py`](./3%20-%20KNN/my_knn.py)
**Walkthrough:** [`notebook.ipynb`](./3%20-%20KNN/notebook.ipynb)

### How it works

KNN is a lazy learner: `fit` just stores the training data, and all the work happens in `predict`. For each sample to classify, the model:

1. Computes the Euclidean distance from the sample to every training point.
2. Takes the `k` closest points.
3. Returns the most common class among them (a majority vote).

If two classes tie in the vote, the one that comes first in sorted order wins. `predict` raises a `ValueError` if the model has not been fitted. Because KNN is distance-based, features should be scaled first.

### Usage

```python
from my_knn import KNN

model = KNN(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
```

### Experiments

**1. Synthetic data** (100 samples, 5 features, an 80/20 train/test split, `k=3`). The labels are random, so there is no pattern to learn, and the accuracy of **0.30** is about what chance (1 in 3) would give. This is a sanity check that the model runs, not a performance test.

**2. Iris (Kaggle).** Classifying iris flowers into three species from sepal and petal measurements, using the [Iris Classification Challenge](https://www.kaggle.com/competitions/iris-classification-challenge) (105 training samples, 45 test samples). The notebook walks through:

- **EDA:** dataset info, summary statistics, and scatter plots of sepal and petal dimensions by species.
- **Preprocessing:** a stratified 80/20 train/validation split, then standardization with a `StandardScaler` fitted on the training set.
- **Training and evaluation:** my model with `k` = 3, 5, and 10, compared against scikit-learn's `KNeighborsClassifier`. For the Kaggle submissions, the `k=10` models are retrained on the full training set.

| Model | Validation accuracy | Kaggle score |
|-------|---------------------|--------------|
| Mine (k = 3) | 0.9524 | |
| Mine (k = 5) | 1.0000 | |
| Mine (k = 10) | 1.0000 | 0.94444 |
| scikit-learn `KNeighborsClassifier` (k = 10) | 1.0000 | 0.94444 |

My model matches scikit-learn on both the validation set and the leaderboard. The validation set has only 21 samples, so each mistake costs about 4.8 percentage points, and `k=5` and `k=10` cannot be told apart.

## Repo structure

```
ML-from-Scratch/
├── 1 - Linear Regression/
│   ├── my_linear_regression.py   # the model
│   └── notebook.ipynb            # experiments
├── 2 - Logistic Regression/
│   ├── my_logistic_regression.py # the model
│   └── notebook.ipynb            # experiments
├── 3 - KNN/
│   ├── my_knn.py                 # the model
│   └── notebook.ipynb            # experiments
└── data/
    ├── house_prices/             # Kaggle data and submission files
    ├── titanic/                  # Kaggle data and submission files
    └── iris/                     # Kaggle data and submission files
```

## Requirements

`numpy` (2.1 or newer), `pandas`, `scikit-learn` (preprocessing and metrics only), `seaborn`, `matplotlib`