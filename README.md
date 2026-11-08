# ML from Scratch

Core machine learning algorithms implemented from scratch with NumPy, then compared against their scikit-learn counterparts on real data.

The goal of this repo is to understand how each algorithm works under the hood: every model lives in its own folder with a clean implementation and a notebook that tests it on synthetic data and on a real Kaggle dataset.

## Models

| # | Model | Status | Data |
|---|-------|--------|------|
| 01 | [Linear Regression](./01%20-%20Linear%20Regression) | Done | Synthetic data, [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/) |
| 02 | [Logistic Regression](./02%20-%20Logistic%20Regression) | Done | Synthetic data, [Titanic](https://www.kaggle.com/competitions/titanic) |
| 03 | [K-Nearest Neighbors (KNN)](./03%20-%20KNN) | Done | Synthetic data, [Iris](https://www.kaggle.com/competitions/iris-classification-challenge) |
| 04 | [Naive Bayes](./04%20-%20Naive%20Bayes) | Done | Synthetic data, [Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) |
| 05 | [Support Vector Machine (SVM)](./05%20-%20SVM) | Done | Synthetic data, [Titanic](https://www.kaggle.com/competitions/titanic) |
| 06 | Decision Tree | Planned | |
| 07 | Random Forest | Planned | |
| 08 | Boosting Trees | Planned | |
| 09 | Principal Component Analysis (PCA) | Planned | |
| 10 | K-Means | Planned | |
| 11 | DBSCAN | Planned | |
| 12 | Gaussian Mixture Model (GMM) | Planned | |
| 13 | Stacking | Planned | |
| 14 | Projects | Planned | |

## 01 - Linear Regression

**Implementation:** [`my_linear_regression.py`](./01%20-%20Linear%20Regression/my_linear_regression.py)
**Walkthrough:** [`notebook.ipynb`](./01%20-%20Linear%20Regression/notebook.ipynb)

### How it works

The `Linear_Regression` class fits a linear model by minimizing the mean squared error, `L = (1/m) Σ (ŷ − y)²`, with no regularization. It supports two ways of doing so, chosen with the `mode` argument:

- **`normal_equation`** (default): solves for the weights in closed form, `θ = (XᵀX)⁻¹ Xᵀy`, after adding a column of ones for the intercept. This requires `XᵀX` to be invertible.
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
- **Training and evaluation:** a 70/30 train/validation split, comparing my two solvers against scikit-learn's `LinearRegression`. For the Kaggle submissions, the preprocessing pipeline is refitted on the full training set and each model is retrained on it.

| Model | Validation RMSE | Kaggle score (RMSE) |
|-------|-----------------|--------------|
| Mine (Normal Equation) | 18,884.06 | 0.16664 |
| Mine (Gradient Descent) | 18,412.13 | 0.16219 |
| scikit-learn `LinearRegression` | 18,884.06 | 0.16664 |

Validation RMSE is measured in dollars on the clipped `SalePrice` values. The Kaggle score is the competition's metric (RMSE on the log of the price), so lower is better.

My normal equation implementation matches scikit-learn's results (the validation RMSEs differ by less than 0.000000001), which confirms that it is correct. Gradient descent scores slightly better on both the validation set and Kaggle, likely because it never fits the training data as tightly as the closed-form solution does.

## 02 - Logistic Regression

**Implementation:** [`my_logistic_regression.py`](./02%20-%20Logistic%20Regression/my_logistic_regression.py)
**Walkthrough:** [`notebook.ipynb`](./02%20-%20Logistic%20Regression/notebook.ipynb)

### How it works

The `Logistic_Regression` class is a binary classifier trained with gradient descent on the log loss (binary cross-entropy), `L = −(1/m) Σ [y·log(p) + (1 − y)·log(1 − p)]`, with no regularization:

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
- **Training and evaluation:** a stratified 70/30 train/validation split, comparing my model against scikit-learn's `LogisticRegression`. For the Kaggle submissions, the preprocessing pipeline is refitted on the full training set and both models are retrained on it.

| Model | Validation accuracy | Kaggle score (accuracy) |
|-------|---------------------|--------------|
| Mine | 0.8022 | 0.76315 |
| scikit-learn `LogisticRegression` | 0.8022 | 0.76555 |

My model matches scikit-learn's validation accuracy exactly. On Kaggle, scikit-learn scores 0.76555 against my 0.76315, which is a difference of one passenger out of 418. Note that scikit-learn's `LogisticRegression` applies L2 regularization by default and mine does not, so the two models are not mathematically identical, which may be why one prediction differs.

## 03 - K-Nearest Neighbors (KNN)

**Implementation:** [`my_knn.py`](./03%20-%20KNN/my_knn.py)
**Walkthrough:** [`notebook.ipynb`](./03%20-%20KNN/notebook.ipynb)

### How it works

KNN is a lazy learner: `fit` just stores the training data, and all the work happens in `predict`. For each sample to classify, the model:

1. Computes the Euclidean distance, `√Σ(aᵢ − bᵢ)²`, from the sample to every training point.
2. Takes the `k` closest points.
3. Returns the most common class among them (a majority vote).

If two classes tie in the vote, the one that comes first in sorted order wins. `predict` raises a `ValueError` if the model has not been fitted. Because KNN is distance-based, features should be scaled first.

KNN has no loss function and nothing to optimize. The only hyperparameter is `n_neighbors`, and every prediction compares the sample against all the stored training points, so prediction gets slower as the training set grows.

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
- **Training and evaluation:** my model with `k` = 3, 5, and 10, compared against scikit-learn's `KNeighborsClassifier`. For the Kaggle submissions, the scaler is refitted on the full training set and the `k=10` models are retrained on it.

| Model | Validation accuracy | Kaggle score (accuracy) |
|-------|---------------------|--------------|
| Mine (k = 3) | 0.9524 | |
| Mine (k = 5) | 1.0000 | |
| Mine (k = 10) | 1.0000 | 0.94444 |
| scikit-learn `KNeighborsClassifier` (k = 10) | 1.0000 | 0.94444 |

My model matches scikit-learn on both the validation set and the leaderboard. The validation set has only 21 samples, so each mistake costs about 4.8 percentage points, and `k=5` and `k=10` cannot be told apart.

## 04 - Naive Bayes

**Implementation:** [`my_naive_bayes.py`](./04%20-%20Naive%20Bayes/my_naive_bayes.py)
**Walkthrough:** [`notebook.ipynb`](./04%20-%20Naive%20Bayes/notebook.ipynb)

### How it works

`Naive_Bayes` is a multinomial Naive Bayes classifier for word-count features, such as the output of a bag-of-words vectorizer. It applies Bayes' rule with the "naive" assumption that words are independent given the class:

`P(class | document) ∝ P(class) × Π P(word | class)^count`

- **Training:** the class priors `P(class)` come from how often each label appears. `P(word | class)` is the number of times the word appears in documents of that class, with Laplace smoothing (`alpha`, default `1`) added to every word count so that a word never seen in a class does not zero out the whole product.
- **Prediction:** for each document, the class prior is multiplied by `P(word | class)` raised to the word's count, for each word that appears in the document. The class with the highest score wins.

There is no loss function to minimize: the probabilities are estimated directly by counting words, and `alpha` is the only hyperparameter.

`predict` raises a `ValueError` if the model has not been fitted. Both `fit` and `predict` expect a **pandas DataFrame** of word counts (one row per document, one column per word), and the columns must be the same at training and prediction time.

### Usage

```python
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from my_naive_bayes import Naive_Bayes

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_train = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())

model = Naive_Bayes()
model.fit(X_train, y_train, alpha=1)

X_val = pd.DataFrame(vectorizer.transform(val_texts).toarray(), columns=vectorizer.get_feature_names_out())
y_pred = model.predict(X_val)
```

### Experiments

**1. Synthetic data** (a tiny spam-vs-normal-message dataset: 4 training texts and 4 validation texts, turned into word counts with `CountVectorizer`). The model gets an F1 score of **1.0**, which confirms that the pipeline runs end to end.

**2. Disaster Tweets (Kaggle).** Predicting whether a tweet is about a real disaster, using [NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) (7,613 labelled training tweets). The notebook walks through:

- **Cleaning:** the `keyword` and `location` columns are dropped. Tweet text is lowercased, and everything except letters and spaces is removed.
- **Preprocessing:** a stratified 80/20 train/validation split (6,090 / 1,523 tweets), then a `CountVectorizer` fitted on the training text.
- **Training and evaluation:** my model compared against scikit-learn's `MultinomialNB`. For the Kaggle submissions, the vectorizer is refitted on all the labelled tweets and both models are retrained on them.

| Model | Validation F1 | Kaggle score (F1) |
|-------|---------------|-------------------|
| Mine | 0.7738 | 0.79957 |
| scikit-learn `MultinomialNB` | 0.7738 | 0.79957 |

My model matches scikit-learn exactly on both the validation set and the leaderboard.

## 05 - Support Vector Machine (SVM)

**Implementation:** [`my_svm.py`](./05%20-%20SVM/my_svm.py)
**Walkthrough:** [`notebook.ipynb`](./05%20-%20SVM/notebook.ipynb)

### How it works

The `SVM` class is a linear, soft-margin binary classifier trained with the **hinge loss**. For a point with label `y` (−1 or +1) and score `f(x) = w·x + b`, the hinge loss is `max(0, 1 − y·f(x))`. It is zero when the point is on the correct side and outside the margin, and it grows linearly as the point moves inside the margin or onto the wrong side.

Both modes solve the same soft-margin problem, `minimize ½‖w‖² + C Σ max(0, 1 − yᵢ(w·xᵢ + b))`. Labels are converted from 0/1 to −1/+1 internally, and the `mode` argument chooses how the problem is solved:

- **`dual_problem`** (default): solves the dual form of the problem above. It minimizes `½ ΣΣ αᵢαⱼ yᵢyⱼ (xᵢ·xⱼ) − Σ αᵢ` subject to `Σ αᵢyᵢ = 0` and `0 ≤ αᵢ ≤ C`, using SciPy's `minimize` with the constraint and bounds. The weights come from `w = Σ αᵢ yᵢ xᵢ`, and the bias is the average of `yₖ − w·xₖ` over the free support vectors (those with `0 < α < C`). The parameter `C` controls how strongly hinge-loss violations are penalized.
- **`gradient_descent`**: minimizes the regularized hinge loss directly, `λ‖w‖² + (1/n) Σ max(0, 1 − yᵢ(w·xᵢ + b))`, with subgradient descent. Only points that violate the margin (`y·f(x) < 1`) contribute to the gradient. Defaults: `epochs=1000`, `learning_rate=0.01`, `regularization_rate=0.01`. The regularization rate `λ` and `C` describe the same trade-off, with `C = 1 / (2·λ·n)` for `n` training samples.

`predict` returns 1 when `w·x + b > 0` and 0 otherwise, and raises a `ValueError` if the model has not been fitted. After fitting, the parameters are available as `weights_` and `bias_`.

### Usage

```python
from my_svm import SVM

# Dual problem
model = SVM(mode='dual_problem')
model.fit(X_train, y_train, C=1.0)
y_pred = model.predict(X_val)

# Gradient descent
model = SVM(mode='gradient_descent')
model.fit(X_train, y_train, epochs=1000, learning_rate=0.01, regularization_rate=0.01)
y_pred = model.predict(X_val)
```

### Experiments

**1. Synthetic data** (100 samples, 5 features, labels generated from random weights plus noise through a sigmoid). Accuracy on the training data:

| Method | Accuracy |
|--------|----------|
| Dual problem | 0.92 |
| Gradient descent | 0.90 |

**2. Titanic (Kaggle).** The same task and preprocessing as in the logistic regression notebook (family features, imputation, scaling, and encoding), with a stratified 70/30 train/validation split (623 / 268 passengers). I compare my two modes against scikit-learn's `SVC(kernel='linear')`. For the Kaggle submissions, the preprocessing pipeline is refitted on the full training set and all models are retrained on it.

| Model | Validation accuracy | Kaggle score (accuracy) |
|-------|---------------------|--------------|
| Mine (Dual Problem) | 0.7761 | 0.77511 |
| Mine (Gradient Descent) | 0.7276 | 0.79186 |
| scikit-learn `SVC` (linear) | 0.7761 | 0.77511 |

The dual problem version matches scikit-learn exactly: the same validation accuracy, the same Kaggle score, and identical predictions on all 418 test passengers. Gradient descent only approximates the solution (a constant learning rate and a fixed number of epochs), so its predictions differ from scikit-learn's for about 9% of the test passengers. Its default settings also correspond to `C ≈ 0.08` on the 623 training samples, compared with `C = 1` for the dual mode and scikit-learn. Its Kaggle score is higher but its validation accuracy is lower, and both gaps are only a handful of passengers, so I would not read it as the better model.

The dual problem is solved with a general-purpose optimizer, so training time grows quickly with the number of samples.

## Repo structure

```
ML-from-Scratch/
├── 01 - Linear Regression/
│   ├── my_linear_regression.py   # the model
│   └── notebook.ipynb            # experiments
├── 02 - Logistic Regression/
│   ├── my_logistic_regression.py # the model
│   └── notebook.ipynb            # experiments
├── 03 - KNN/
│   ├── my_knn.py                 # the model
│   └── notebook.ipynb            # experiments
├── 04 - Naive Bayes/
│   ├── my_naive_bayes.py         # the model
│   └── notebook.ipynb            # experiments
├── 05 - SVM/
│   ├── my_svm.py                 # the model
│   └── notebook.ipynb            # experiments
└── data/
    ├── house_prices/             # Kaggle data and submission files
    ├── titanic/                  # Kaggle data and submission files
    ├── iris/                     # Kaggle data and submission files
    └── disaster_tweets/          # Kaggle data and submission files
```

## Requirements

`numpy` (2.1 or newer), `pandas`, `scipy`, `scikit-learn` (preprocessing and metrics only), `seaborn`, `matplotlib`