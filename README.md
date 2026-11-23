# ML from Scratch

Core machine learning algorithms implemented from scratch with NumPy and compared against their scikit-learn counterparts on Kaggle datasets.

## Models

| #  | Model                                      | Status  | Data |
|----|--------------------------------------------|---------|------|
| 01 | [Linear Regression](./01%20-%20Linear%20Regression) | Done    | Synthetic data, [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/) |
| 02 | [Logistic Regression](./02%20-%20Logistic%20Regression) | Done    | Synthetic data, [Titanic](https://www.kaggle.com/competitions/titanic) |
| 03 | [K-Nearest Neighbors (KNN)](./03%20-%20KNN) | Done    | Synthetic data, [Iris](https://www.kaggle.com/competitions/iris-classification-challenge) |
| 04 | [Naive Bayes](./04%20-%20Naive%20Bayes)     | Done    | Synthetic data, [Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) |
| 05 | [Support Vector Machine (SVM)](./05%20-%20SVM) | Done    | Synthetic data, [Titanic](https://www.kaggle.com/competitions/titanic) |
| 06 | [Decision Tree](./06%20-%20Decision%20Tree) | Done    | Synthetic data, [Titanic](https://www.kaggle.com/competitions/titanic) (classifier), [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/) (regressor) |
| 07 | [Random Forest](./07%20-%20Random%20Forest) | Done    | Synthetic data, [Titanic](https://www.kaggle.com/competitions/titanic) (classifier), [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/) (regressor) |
| 08 | [Gradient Boosting](./08%20-%20Gradient%20Boosting) | Done    | Synthetic data, [Titanic](https://www.kaggle.com/competitions/titanic) (classifier), [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/) (regressor) |
| 09 | [Principal Component Analysis (PCA)](./09%20-%20PCA) | Done    | Synthetic data, [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/) + [Titanic](https://www.kaggle.com/competitions/titanic) (as preprocessing) |
| 10 | [K-Means](./10%20-%20KMeans)               | Done    | Synthetic data, [Ibiy Thala Hidden Customer Groups](https://www.kaggle.com/competitions/ibiy-thala-hidden-customer-groups) |
| 11 | [DBSCAN](./11%20-%20DBSCAN)                | Done    | Synthetic data, [Ibiy Thala Hidden Customer Groups](https://www.kaggle.com/competitions/ibiy-thala-hidden-customer-groups) |
| 12 | Gaussian Mixture Model (GMM)               | Planned | |
| 13 | Stacking                                   | Planned | |
| 14 | Projects                                   | Planned | One end-to-end project for each task: classification, regression, and clustering |

All models share one interface: hyperparameters go in the constructor, and `fit(X, y)` / `predict(X)` (or `transform(X)` for PCA) take only the data. The unsupervised models (PCA, K-Means, and DBSCAN) call `fit(X)` without labels.

## Experiment setup

- **Titanic:** stratified 70/30 split (623 / 268 passengers), with `FamilySize` and `IsAlone` features, median imputation and scaling for numeric columns, one-hot `Embarked`, and ordinal `Sex`.
- **House Prices:** 70/30 split (1,022 / 438 houses). Outliers in the highest-variance columns (`SalePrice` included) are clipped to the 10th and 90th percentiles, and columns that are mostly missing, mostly zero, or highly correlated are dropped. Numeric columns are imputed and scaled, and categorical ones are one-hot encoded.
- **Preprocessed data:** Notebooks now load preprocessed CSVs from `data/<dataset>/preprocessed/` (full training set, test set, and targets) so the same cleaned features are reused across models.
- **Kaggle submissions:** the preprocessing is refitted on the full training set and the models are retrained on it.
- **scikit-learn baselines** use matching settings. Tree depth starts at 0 at the root, so my `max_depth=k` equals scikit-learn's `max_depth=k+1`, and the tables list scikit-learn's own value.
- **Metrics:** House Prices validation RMSE is in dollars on the clipped `SalePrice`, and the Kaggle score is RMSE on the log of the price. Lower is better.

## 01 - Linear Regression

[`my_linear_regression.py`](./01%20-%20Linear%20Regression/my_linear_regression.py) · [`notebook.ipynb`](./01%20-%20Linear%20Regression/notebook.ipynb)

Minimizes the mean squared error, with no regularization, using either the closed-form normal equation (`mode='normal_equation'`, the default) or gradient descent (`mode='gradient_descent'`, `epochs=1000`, `learning_rate=0.01`).

```python
from my_linear_regression import Linear_Regression

model = Linear_Regression(mode='gradient_descent', epochs=1000, learning_rate=0.01)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
```

Synthetic data (10 samples): MSE 0.3750 with the normal equation and 0.3754 with gradient descent. House Prices:

| Model | Validation RMSE | Kaggle score (RMSE) |
|-------|-----------------|---------------------|
| Mine (Normal Equation) | 18,884.06 | 0.16664 |
| Mine (Gradient Descent) | 18,412.13 | 0.16219 |
| scikit-learn `LinearRegression` | 18,884.06 | 0.16664 |

## 02 - Logistic Regression

[`my_logistic_regression.py`](./02%20-%20Logistic%20Regression/my_logistic_regression.py) · [`notebook.ipynb`](./02%20-%20Logistic%20Regression/notebook.ipynb)

Binary classifier trained with gradient descent on the log loss, with no regularization (`epochs=1000`, `learning_rate=0.1`). It also provides `predict_proba`.

```python
from my_logistic_regression import Logistic_Regression

model = Logistic_Regression(epochs=1000, learning_rate=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
```

Synthetic data (100 samples): 0.96 accuracy on the training data. Titanic, against scikit-learn with `penalty=None`:

| Model | Validation accuracy | Kaggle score (accuracy) |
|-------|---------------------|-------------------------|
| Mine | 0.8022 | 0.76315 |
| scikit-learn `LogisticRegression` | 0.8022 | 0.76315 |

The two models disagree on 16 of the 418 test passengers, and the differences cancel out in the score.

## 03 - K-Nearest Neighbors (KNN)

[`my_knn.py`](./03%20-%20KNN/my_knn.py) · [`notebook.ipynb`](./03%20-%20KNN/notebook.ipynb)

Predicts the majority class among the `n_neighbors` closest training points, using Euclidean distance. The features should be scaled.

```python
from my_knn import KNN

model = KNN(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
```

Synthetic data (random labels): 0.30 accuracy, which is about chance. Iris:

| Model | Validation accuracy | Kaggle score (accuracy) |
|-------|---------------------|-------------------------|
| Mine (k = 3) | 0.9524 | |
| Mine (k = 5) | 1.0000 | |
| Mine (k = 10) | 1.0000 | 0.94444 |
| scikit-learn `KNeighborsClassifier` (k = 10) | 1.0000 | 0.94444 |

The validation set has only 21 samples, so `k=5` and `k=10` cannot be told apart.

## 04 - Naive Bayes

[`my_naive_bayes.py`](./04%20-%20Naive%20Bayes/my_naive_bayes.py) · [`notebook.ipynb`](./04%20-%20Naive%20Bayes/notebook.ipynb)

Multinomial Naive Bayes with Laplace smoothing (`alpha=1`). It takes DataFrames of word counts, for example from a `CountVectorizer`.

```python
from my_naive_bayes import Naive_Bayes

model = Naive_Bayes(alpha=1)
model.fit(X_train_counts, y_train)
y_pred = model.predict(X_val_counts)
```

Synthetic data: F1 of 1.0. Disaster Tweets:

| Model | Validation F1 | Kaggle score (F1) |
|-------|---------------|-------------------|
| Mine | 0.7738 | 0.79957 |
| scikit-learn `MultinomialNB` | 0.7738 | 0.79957 |

## 05 - Support Vector Machine (SVM)

[`my_svm.py`](./05%20-%20SVM/my_svm.py) · [`notebook.ipynb`](./05%20-%20SVM/notebook.ipynb)

Linear soft-margin SVM trained with the hinge loss. It is solved either in the dual form with SciPy (`mode='dual_problem'`, `C=1.0`) or directly with gradient descent (`mode='gradient_descent'`, `epochs=1000`, `learning_rate=0.01`, `regularization_rate=0.01`).

```python
from my_svm import SVM

model = SVM(mode='dual_problem', C=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
```

Synthetic data: 0.92 accuracy with the dual problem and 0.90 with gradient descent (training data). Titanic:

| Model | Validation accuracy | Kaggle score (accuracy) |
|-------|---------------------|-------------------------|
| Mine (Dual Problem) | 0.7761 | 0.77511 |
| Mine (Gradient Descent) | 0.7276 | 0.79186 |
| scikit-learn `SVC` (linear) | 0.7761 | 0.77511 |

The dual version matches scikit-learn exactly, with identical predictions for all 418 test passengers. Gradient descent only approximates the solution and differs on about 9% of them, and its higher Kaggle score is a handful of passengers.

## 06 - Decision Tree

[`my_decision_tree_classifier.py`](./06%20-%20Decision%20Tree/my_decision_tree_classifier.py) · [`notebook_classification.ipynb`](./06%20-%20Decision%20Tree/notebook_classification.ipynb) · [`my_decision_tree_regressor.py`](./06%20-%20Decision%20Tree/my_decision_tree_regressor.py) · [`notebook_regression.ipynb`](./06%20-%20Decision%20Tree/notebook_regression.ipynb)

Binary CART trees. The classifier splits on Gini impurity or entropy (`mode='gini'` or `'entropy'`) and predicts the majority class, and the regressor splits on the mean squared error and predicts the mean. Both take `max_depth` in the constructor, and 0 or `None` means no limit.

```python
from my_decision_tree_classifier import Decision_Tree_Classifier
from my_decision_tree_regressor import Decision_Tree_Regressor

clf = Decision_Tree_Classifier(mode='gini', max_depth=2)
reg = Decision_Tree_Regressor(max_depth=3)
```

**Classifier.** Synthetic data: 1.00 training accuracy with both Gini and entropy. Titanic:

| Model | Validation accuracy | Kaggle score (accuracy) |
|-------|---------------------|-------------------------|
| Mine (Gini, `max_depth=2`) | 0.7985 | 0.77511 |
| Mine (Entropy, `max_depth=2`) | 0.8097 | 0.77511 |
| scikit-learn `DecisionTreeClassifier` (Gini, `max_depth=3`) | 0.7985 | 0.77511 |

The Gini tree matches scikit-learn exactly, with identical predictions for all 418 test passengers.

**Regressor.** House Prices:

| Model | Validation RMSE | Kaggle score (RMSE) |
|-------|-----------------|---------------------|
| Mine (`max_depth=3`) | 24,940.43 | 0.24192 |
| scikit-learn `DecisionTreeRegressor` (`max_depth=4`) | 24,988.23 | 0.24192 |

The test predictions are identical for all 1,459 houses. The validation RMSEs differ because one validation house sits exactly on a split threshold, which scikit-learn's single-precision arithmetic sends to the other side.

## 07 - Random Forest

[`my_random_forest_classifier.py`](./07%20-%20Random%20Forest/my_random_forest_classifier.py) · [`notebook_classification.ipynb`](./07%20-%20Random%20Forest/notebook_classification.ipynb) · [`my_random_forest_regressor.py`](./07%20-%20Random%20Forest/my_random_forest_regressor.py) · [`notebook_regression.ipynb`](./07%20-%20Random%20Forest/notebook_regression.ipynb)

Ensembles of the trees from `06 - Decision Tree` (keep the folders side by side). Each tree trains on a bootstrap sample of the rows (`frac_instances`, repeated rows kept) and searches a random subset of `frac_features` of the columns at every split. The classifier takes a majority vote and the regressor averages.

```python
from my_random_forest_classifier import Random_Forest_Classifier
from my_random_forest_regressor import Random_Forest_Regressor

clf = Random_Forest_Classifier(max_depth=2, n_estimators=100, frac_instances=0.8, frac_features=0.8)
reg = Random_Forest_Regressor(max_depth=3, n_estimators=100, frac_instances=0.4, frac_features=0.4)
```

**Classifier.** Synthetic data: 0.97 training accuracy. Titanic (100 trees, 80% of the rows and features):

| Model | Validation accuracy | Kaggle score (accuracy) |
|-------|---------------------|-------------------------|
| Mine (`max_depth=2`) | 0.8022 | 0.78229 |
| scikit-learn `RandomForestClassifier` (`max_depth=3`) | 0.8060 | 0.78468 |

**Regressor.** Synthetic data: 1.00 training MSE. House Prices (100 trees, 40% of the rows and features):

| Model | Validation RMSE | Kaggle score (RMSE) |
|-------|-----------------|---------------------|
| Mine (`max_depth=3`) | 19,634.59 | 0.22817 |
| scikit-learn `RandomForestRegressor` (`max_depth=4`) | 19,762.60 | 0.22887 |

The forests use the same design but different random draws, so they are not identical. On the Titanic test set they disagree on 3 of the 418 passengers.

## 08 - Gradient Boosting

[`my_gradient_boosting_classifier.py`](./08%20-%20Gradient%20Boosting/my_gradient_boosting_classifier.py) · [`notebook_classification.ipynb`](./08%20-%20Gradient%20Boosting/notebook_classification.ipynb) · [`my_gradient_boosting_regressor.py`](./08%20-%20Gradient%20Boosting/my_gradient_boosting_regressor.py) · [`notebook_regression.ipynb`](./08%20-%20Gradient%20Boosting/notebook_regression.ipynb)

Gradient boosting with `Decision_Tree_Regressor` trees from `06 - Decision Tree`, each fitted to the current residuals. The regressor minimizes the squared error, and the classifier minimizes the log loss on a raw log-odds score, fitting trees to `y − p`. The defaults are `max_depth=4`, `n_estimators=50`, and `learning_rate=0.1`.

```python
from my_gradient_boosting_classifier import Gradient_Boosting_Classifier
from my_gradient_boosting_regressor import Gradient_Boosting_Regressor

clf = Gradient_Boosting_Classifier(max_depth=4, n_estimators=50, learning_rate=0.1)
reg = Gradient_Boosting_Regressor(max_depth=4, n_estimators=50, learning_rate=0.1)
```

**Classifier.** Synthetic data: 0.99 training accuracy. Titanic (50 estimators, learning rate 0.1):

| Model | Validation accuracy | Kaggle score (accuracy) |
|-------|---------------------|-------------------------|
| Mine (`max_depth=4`) | 0.7985 | 0.78229 |
| scikit-learn `GradientBoostingClassifier` (`max_depth=5`) | 0.7985 | 0.76794 |

The two models disagree on 22 of the 418 test passengers.

**Regressor.** Synthetic data: about 0.00004 training MSE. House Prices (50 estimators, learning rate 0.1):

| Model | Validation RMSE | Kaggle score (RMSE) |
|-------|-----------------|---------------------|
| Mine (`max_depth=4`) | 16,303.76 | 0.19286 |
| scikit-learn `GradientBoostingRegressor` (`max_depth=5`) | 16,086.96 | 0.19286 |

Their test predictions differ by about $327 per house on average.

## 09 - Principal Component Analysis (PCA)

[`my_pca.py`](./09%20-%20PCA/my_pca.py) · [`notebook.ipynb`](./09%20-%20PCA/notebook.ipynb)

Dimensionality reduction via SVD of the centered data matrix. Returns the principal components (`transform`) and the cumulative explained variance ratio. `n_components=None` keeps all components.

```python
from my_pca import PCA

pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)
# or
pca.fit(X)
X_reduced = pca.transform(X)
```

**Synthetic data** (1000 samples, 5 features generated from 3 latent factors + noise): the first 3 components explain >99.9% of the variance.

**House Prices** (used as preprocessing before Linear Regression):

- 93 components retain ≈98% of the variance (out of 176 features).
- Linear Regression after PCA → **Kaggle RMSE 0.15865** (improved from 0.16664 without PCA, and faster).

**Titanic** (used as preprocessing before Logistic Regression):

- 8 components retain ≈98% of the variance (out of 11 features).
- Logistic Regression after PCA → **Kaggle accuracy 0.76315** (same score, faster training).

The implementation matches scikit-learn’s `PCA` on the cumulative explained variance curves.

## 10 - K-Means

[`my_kmeans.py`](./10%20-%20KMeans/my_kmeans.py) · [`notebook.ipynb`](./10%20-%20KMeans/notebook.ipynb)

Lloyd's algorithm: the centroids start as random points (standard-normal draws, seeded with 42), then every sample is assigned to its nearest centroid by Euclidean distance and each centroid moves to the mean of its samples, until the centroids stop changing or `max_iterations` (default 100) is reached. An empty cluster keeps its previous centroid. It provides `fit`, `predict`, and `fit_predict`.

```python
from my_kmeans import KMeans

model = KMeans(n_clusters=3, max_iterations=100)
labels = model.fit_predict(X)
```

**Synthetic data** (blobs): adjusted Rand index of 1.0.

**Ibiy Thala Hidden Customer Groups** (customer segmentation, scored by adjusted Rand index). Silhouette tuning over `n_clusters` in {2, 4, 6, 8, 10} selects the best `n`.

| Model | Kaggle score (adjusted Rand index) |
|-------|-------------------------------------|
| Mine | 0.40981 |
| scikit-learn `KMeans` | 0.40981 |

Both models produce the same score on the competition.

## 11 - DBSCAN

[`my_dbscan.py`](./11%20-%20DBSCAN/my_dbscan.py) · [`notebook.ipynb`](./11%20-%20DBSCAN/notebook.ipynb)

Density-based clustering. Points with at least `min_samples` neighbors inside radius `epsilon` are core points; clusters grow by recursively expanding neighborhoods. Points that never join a dense region stay noise (`label = -1`). It provides `fit` and `fit_predict`.

```python
from my_dbscan import DBSCAN

model = DBSCAN(epsilon=0.2, min_samples=5)
labels = model.fit_predict(X)
```

**Synthetic data** (`make_moons`, noise=0.05): recovers the two interlocking half-circles (adjusted Rand index near 1.0 with suitable `epsilon`).

**Ibiy Thala Hidden Customer Groups** (same customer segmentation data as K-Means):

| Model | Kaggle score (adjusted Rand index) |
|-------|-------------------------------------|
| Mine (`epsilon=0.5`, `min_samples=5`) | 0.00096 |
| scikit-learn `DBSCAN` (`eps=0.5`, `min_samples=5`) | 0.00096 |

The two implementations match. The low ARI indicates that the chosen density parameters are not well suited to this particular dataset (most points become noise or form few meaningful clusters); the exercise still validates that the from-scratch version behaves like scikit-learn.

## Model comparison

All my models, ranked by Kaggle score (best first). Within each table the models use the same preprocessing and split. KNN (Iris), Naive Bayes (Disaster Tweets), K-Means, and DBSCAN (customer segmentation) use their own datasets, so they are not included in the Titanic / House Prices tables.

**Titanic** (higher is better):

| Rank | Model | Validation accuracy | Kaggle score (accuracy) |
|------|-------|---------------------|-------------------------|
| 1 | SVM (gradient descent) | 0.7276 | 0.79186 |
| 2 (tie) | Random Forest (`max_depth=2`) | 0.8022 | 0.78229 |
| 2 (tie) | Gradient Boosting (`max_depth=4`) | 0.7985 | 0.78229 |
| 4 (tie) | SVM (dual problem) | 0.7761 | 0.77511 |
| 4 (tie) | Decision Tree (Gini, `max_depth=2`) | 0.7985 | 0.77511 |
| 4 (tie) | Decision Tree (entropy, `max_depth=2`) | 0.8097 | 0.77511 |
| 7 | Logistic Regression | 0.8022 | 0.76315 |

**House Prices** (lower is better):

| Rank | Model | Validation RMSE | Kaggle score (RMSE) |
|------|-------|-----------------|---------------------|
| 1 | Linear Regression + PCA (93 comps) | — | **0.15865** |
| 2 | Linear Regression (gradient descent) | 18,412.13 | 0.16219 |
| 3 | Linear Regression (normal equation) | 18,884.06 | 0.16664 |
| 4 | Gradient Boosting (`max_depth=4`) | 16,303.76 | 0.19286 |
| 5 | Random Forest (`max_depth=3`) | 19,634.59 | 0.22817 |
| 6 | Decision Tree (`max_depth=3`) | 24,940.43 | 0.24192 |

On Titanic only 12 passengers out of 418 separate first from last, so small differences are partly luck. The ensembles beat a single tree on both datasets, and linear regression (especially after PCA) wins on House Prices.

## Running the notebooks

Each notebook reads data from `../data` (now preferring the `preprocessed/` subfolder) and imports models by path, so run it from its own folder inside a full clone of the repo. On Colab, run this first (change the folder name to match):

```python
!git clone -q https://github.com/Sam50x/ML-from-Scratch.git
%cd "ML-from-Scratch/01 - Linear Regression"
```

Colab may need `!pip install -U numpy` and a runtime restart, because the code needs NumPy 2.1 or newer and Python 3.12 or newer.

| Model | Notebook | Colab |
|-------|----------|-------|
| Linear Regression | [`01 - Linear Regression/notebook.ipynb`](./01%20-%20Linear%20Regression/notebook.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sam50x/ML-from-Scratch/blob/main/01%20-%20Linear%20Regression/notebook.ipynb) |
| Logistic Regression | [`02 - Logistic Regression/notebook.ipynb`](./02%20-%20Logistic%20Regression/notebook.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sam50x/ML-from-Scratch/blob/main/02%20-%20Logistic%20Regression/notebook.ipynb) |
| KNN | [`03 - KNN/notebook.ipynb`](./03%20-%20KNN/notebook.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sam50x/ML-from-Scratch/blob/main/03%20-%20KNN/notebook.ipynb) |
| Naive Bayes | [`04 - Naive Bayes/notebook.ipynb`](./04%20-%20Naive%20Bayes/notebook.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sam50x/ML-from-Scratch/blob/main/04%20-%20Naive%20Bayes/notebook.ipynb) |
| SVM | [`05 - SVM/notebook.ipynb`](./05%20-%20SVM/notebook.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sam50x/ML-from-Scratch/blob/main/05%20-%20SVM/notebook.ipynb) |
| Decision Tree (classifier) | [`06 - Decision Tree/notebook_classification.ipynb`](./06%20-%20Decision%20Tree/notebook_classification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sam50x/ML-from-Scratch/blob/main/06%20-%20Decision%20Tree/notebook_classification.ipynb) |
| Decision Tree (regressor) | [`06 - Decision Tree/notebook_regression.ipynb`](./06%20-%20Decision%20Tree/notebook_regression.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sam50x/ML-from-Scratch/blob/main/06%20-%20Decision%20Tree/notebook_regression.ipynb) |
| Random Forest (classifier) | [`07 - Random Forest/notebook_classification.ipynb`](./07%20-%20Random%20Forest/notebook_classification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sam50x/ML-from-Scratch/blob/main/07%20-%20Random%20Forest/notebook_classification.ipynb) |
| Random Forest (regressor) | [`07 - Random Forest/notebook_regression.ipynb`](./07%20-%20Random%20Forest/notebook_regression.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sam50x/ML-from-Scratch/blob/main/07%20-%20Random%20Forest/notebook_regression.ipynb) |
| Gradient Boosting (classifier) | [`08 - Gradient Boosting/notebook_classification.ipynb`](./08%20-%20Gradient%20Boosting/notebook_classification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sam50x/ML-from-Scratch/blob/main/08%20-%20Gradient%20Boosting/notebook_classification.ipynb) |
| Gradient Boosting (regressor) | [`08 - Gradient Boosting/notebook_regression.ipynb`](./08%20-%20Gradient%20Boosting/notebook_regression.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sam50x/ML-from-Scratch/blob/main/08%20-%20Gradient%20Boosting/notebook_regression.ipynb) |
| PCA | [`09 - PCA/notebook.ipynb`](./09%20-%20PCA/notebook.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sam50x/ML-from-Scratch/blob/main/09%20-%20PCA/notebook.ipynb) |
| K-Means | [`10 - KMeans/notebook.ipynb`](./10%20-%20KMeans/notebook.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sam50x/ML-from-Scratch/blob/main/10%20-%20KMeans/notebook.ipynb) |
| DBSCAN | [`11 - DBSCAN/notebook.ipynb`](./11%20-%20DBSCAN/notebook.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sam50x/ML-from-Scratch/blob/main/11%20-%20DBSCAN/notebook.ipynb) |

## Repo structure

```
ML-from-Scratch/
├── README.md
├── requirements.txt
├── .gitignore
├── 01 - Linear Regression/
│   ├── my_linear_regression.py
│   └── notebook.ipynb
├── 02 - Logistic Regression/
│   ├── my_logistic_regression.py
│   └── notebook.ipynb
├── 03 - KNN/
│   ├── my_knn.py
│   └── notebook.ipynb
├── 04 - Naive Bayes/
│   ├── my_naive_bayes.py
│   └── notebook.ipynb
├── 05 - SVM/
│   ├── my_svm.py
│   └── notebook.ipynb
├── 06 - Decision Tree/
│   ├── my_decision_tree_classifier.py
│   ├── notebook_classification.ipynb
│   ├── my_decision_tree_regressor.py
│   └── notebook_regression.ipynb
├── 07 - Random Forest/
│   ├── my_random_forest_classifier.py
│   ├── notebook_classification.ipynb
│   ├── my_random_forest_regressor.py
│   └── notebook_regression.ipynb
├── 08 - Gradient Boosting/
│   ├── my_gradient_boosting_classifier.py
│   ├── notebook_classification.ipynb
│   ├── my_gradient_boosting_regressor.py
│   └── notebook_regression.ipynb
├── 09 - PCA/
│   ├── my_pca.py
│   └── notebook.ipynb
├── 10 - KMeans/
│   ├── my_kmeans.py
│   └── notebook.ipynb
├── 11 - DBSCAN/
│   ├── my_dbscan.py
│   └── notebook.ipynb
└── data/
    ├── house_prices/
    │   ├── preprocessed/          # full.csv, test.csv, y_full.csv
    │   └── submissions/
    ├── titanic/
    │   ├── preprocessed/
    │   └── submissions/
    ├── iris/
    ├── disaster_tweets/
    └── tabular_playground_jul_2022/
```

## Requirements

Python 3.12 or newer, and the packages in [`requirements.txt`](./requirements.txt): `numpy` (2.1 or newer), `pandas`, `scipy`, `scikit-learn` (preprocessing and metrics only), `seaborn`, `matplotlib`. Install them with `pip install -r requirements.txt`.
