# ML from Scratch

Core machine learning algorithms implemented from scratch with NumPy, then compared against their scikit-learn counterparts on real data.

The goal of this repo is to understand how each algorithm works under the hood: every model lives in its own folder with a clean implementation and a notebook that tests it on synthetic data and on a real Kaggle dataset.

## Models

| # | Model | Status | Data |
|---|-------|--------|------|
| 1 | [Linear Regression](./1%20-%20Linear%20Regression) | Done | Synthetic data, [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/) |
| 2 | Logistic Regression | Planned | |
| 3 | K-Nearest Neighbors (KNN) | Planned | |
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

**1. Synthetic data** (10 samples, 5 features, known weights plus noise). Both solvers recover a good fit:

| Method | MSE |
|--------|-----|
| Normal Equation | 0.0219 |
| Gradient Descent | 0.0500 |

**2. House Prices (Kaggle).** The notebook walks through:

- **EDA:** summary statistics, missing values, columns that are mostly zeros, and a correlation heatmap.
- **Cleaning:** outliers in the highest-variance numeric columns (including `SalePrice`) are clipped to the 10th and 90th percentiles. Columns with more than 50% missing values, more than 50% zeros, or high correlation with other features are dropped.
- **Preprocessing:** a scikit-learn `ColumnTransformer` that imputes (median / most frequent), standardizes numeric features, and one-hot encodes categorical ones.
- **Training and evaluation:** a 70/30 train/validation split, comparing my two solvers against scikit-learn's `LinearRegression`.

| Model | Validation RMSE | Kaggle score |
|-------|-----------------|--------------|
| Mine (Normal Equation) | 18,884.06 | 0.17177 |
| Mine (Gradient Descent) | 18,412.11 | 0.16515 |
| scikit-learn `LinearRegression` | 18,884.06 | 0.17177 |

Validation RMSE is measured in dollars on the clipped `SalePrice` values. The Kaggle score is the competition's metric (RMSE on the log of the price), so lower is better.

My normal equation implementation matches scikit-learn's results (the validation RMSE agrees to about 12 decimal places), which confirms that it is correct. Gradient descent scores slightly better here, likely because it never fits the training data as tightly as the closed-form solution does.

### Repo structure

```
ML-from-Scratch/
├── 1 - Linear Regression/
│   ├── my_linear_regression.py   # the model
│   └── notebook.ipynb            # experiments
└── data/
    └── house_prices/             # Kaggle data and submission files
```

### Requirements

`numpy`, `pandas`, `scikit-learn` (preprocessing and metrics only), `seaborn`, `matplotlib`