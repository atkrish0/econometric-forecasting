# VARIFMOD: Conceptual Implementation Document

## 1. Overview

Inflation forecasting is treated here as a system-level macroeconomic problem rather than a single-equation prediction task. The project compares four forecasting pipelines that all end in a Vector Autoregression (VAR), but differ in how predictors are chosen:

1. A theory-driven benchmark specification.
2. A Lasso-selected specification.
3. An XGBoost-selected specification.
4. A PCA-ranked specification.

The central point is to test whether data-driven feature selection improves downstream dynamic forecasting once variables interact endogenously in a multivariate time-series model.

### Why Lasso, XGBoost, and PCA Were Chosen

The model-selection methods were chosen to represent complementary philosophies before the common VAR endpoint:

- `Lasso` is a sparse linear selector, useful when macro variables are numerous and collinear. It offers coefficient-level interpretability and an explicit regularization path through

$$
\hat{\beta}^{\text{lasso}} = \arg\min_{\beta}
\left(
\frac{1}{2n}\|y-X\beta\|_2^2 + \lambda\|\beta\|_1
\right).
$$

- `XGBoost` is a nonlinear tree-ensemble selector, useful when signal may come from interactions and threshold effects that linear sparsity may miss. In additive form:

$$
\hat{f}(x) = \sum_{m=1}^{M} \gamma_m h_m(x).
$$

- `PCA` is a dimensionality-reduction method that identifies dominant directions of predictor variation. In this project it is used as a feature-ranking device rather than as the final forecasting model. A loading-weighted score is assigned to each original variable:

$$
\text{score}_j = \sum_{k=1}^{K} |\ell_{jk}|\, w_k,
$$

where \(\ell_{jk}\) is the loading of feature \(j\) on component \(k\), and \(w_k\) is the explained-variance ratio of component \(k\).

- Using both gives a methodologically clean comparison against the theory-driven benchmark:
  - theory-prior selection,
  - sparse linear ML selection,
  - nonlinear ML selection,
  - unsupervised factor-based ranking.

- This design isolates the real research question in this project: not only which model predicts inflation directly, but which selected variables produce the strongest downstream multivariate forecast once all predictors are embedded in the same VAR dynamics.

## 2. Data and Economic Framing

The analysis is built on the FRED-MD monthly macro panel, with inflation proxied from `CPIAUCSL` after transformation. The dataset gives broad coverage across real activity, labor, policy rates, housing, and financial indicators, which makes it suitable for high-dimensional feature selection experiments.

Conceptual tension:

- Economic theory suggests a compact, interpretable macro block.
- ML methods can screen large feature sets and may surface predictive signals not selected manually.

Notebook code block that sets up the panel:

```python
fred = pd.read_csv("fredmd.csv")
fred = fred.drop(0)
fred["sasdate"] = pd.to_datetime(fred["sasdate"])
fred = fred.rename(columns={"sasdate": "date"})
fred = fred.set_index("date")
```

## 3. Transformations and Stationarity Logic

Two annual transformations appear in the notebook.

For EDA:

$$
\Delta_{12}^{\%} x_t = 100\cdot\frac{x_t - x_{t-12}}{x_{t-12}}.
$$

For modeling (main path):

$$
\Delta_{12} x_t = x_t - x_{t-12}.
$$

The model-building workflow relies on \(\Delta_{12}x_t\) (`*_diff`) features because they are empirically more stable for regularized selection and VAR estimation in this notebook.

Notebook code blocks:

```python
# EDA transform
for c in list(fred.columns.values):
    fred[c + "_diff"] = (fred[c] - fred[c].shift(12))/fred[c].shift(12) * 100
```

```python
# Modeling transform
for c in list(fred.columns.values):
    fred[c + "_diff"] = fred[c] - fred[c].shift(12)
```

## 4. Exploratory Diagnostics as Model Prior

EDA is used to build macro prior intuition before formal variable selection.

Questions addressed:

- Does inflation co-move with unemployment (Phillips-curve intuition)?
- How does inflation behave relative to policy rates?
- Is equity index information useful for inflation dynamics?

Notebook code blocks:

```python
for f, c in dict(zip(["UNRATE", "S&P 500", "FEDFUNDS"], ["navy", "firebrick", "orange"])).items():
    scatter_plot(data=fred, col_1="CPIAUCSL_diff", col_2=f, color=c)
```

```python
scatter_subs(data=fred, col_1="CPIAUCSL_diff", col_2="UNRATE", color="navy")
scatter_subs(data=fred, col_1="CPIAUCSL_diff", col_2="S&P 500", color="firebrick")
scatter_subs(data=fred, col_1="CPIAUCSL_diff", col_2="FEDFUNDS", color="orange")
```

## 5. Benchmark VAR (Theory-Driven Specification)

The benchmark uses a compact macro block (inflation, income, labor, policy rate, USD, housing) and estimates a monthly VAR with annual lag depth.

Let

$$
y_t \in \mathbb{R}^{k}
$$

be the transformed macro vector. VAR(\(p\)):

$$
y_t = c + A_1 y_{t-1} + \cdots + A_p y_{t-p} + \varepsilon_t,
\quad \varepsilon_t \sim (0,\Sigma_\varepsilon).
$$

In notebook implementation, \(p=12\). Forecast accuracy is measured with:

$$
\text{MSE} = \frac{1}{n}\sum_{t=1}^{n}(\hat{\pi}_t - \pi_t)^2.
$$

Specific implementation detail:

- The notebook fits the VAR on a fixed training window (`1973-01` to `2009-01`) and generates a `120`-month forecast horizon.
- The same `var_create(...)` function is reused across all model branches, so only the input variable set changes.

Notebook code blocks:

```python
from statsmodels.tsa.api import VAR

results = VAR(data_train).fit(12)
lag_order = results.k_ar
forecasted = pd.DataFrame(results.forecast(data_train.values[-lag_order:], 120))
```

```python
var_mse = metrics.mean_squared_error(
    final_data.loc[start_date:end_date, "cpi_diff_fcast"],
    final_data.loc[start_date:end_date, "cpi_diff"]
)
```

Traditional variable set construction in the notebook:

```python
old_names_diff = ["CPIAUCSL_diff", "RPI_diff", "UNRATE_diff", "FEDFUNDS_diff", "TWEXAFEGSMTHx_diff", "HOUST_diff"]
new_names_diff = ["cpi_diff", "rpi_diff", "unemp_diff", "fedrate_diff", "usd_diff", "houst_diff"]
```

## 6. Lasso-Selected VAR Pipeline

Lasso is used to select sparse predictors before passing them into the same VAR forecasting framework.

Optimization:

$$
\hat{\beta}^{\text{lasso}} = \arg\min_{\beta}
\left(
\frac{1}{2n}\|y-X\beta\|_2^2 + \lambda\|\beta\|_1
\right).
$$

This section separates two tasks:

1. Supervised sparse selection for inflation.
2. Dynamic multivariate forecasting via VAR on selected features.

Specific implementation detail:

- `LassoCV` is fit with `TimeSeriesSplit(n_splits=5)` rather than random cross-validation, which preserves time ordering.
- The selected regularization strength `alpha` is then reused in a second `Lasso(...)` fit to extract non-zero coefficients explicitly.
- Those surviving predictors are not used as the final model directly; they are passed into the shared VAR forecast pipeline.

Notebook code blocks:

```python
cpi_target = fred.dropna().cpi_diff
fred_features = fred.dropna().drop(["cpi_diff"], axis=1)
X_train, y_train, X_test, y_test = timeseries_train_test_split(X=fred_features, y=cpi_target, testsize=0.25)

lasso = linear_model.LassoCV(cv=model_selection.TimeSeriesSplit(n_splits=5), random_state=0)
fred_lasso = lasso.fit(X_train, y_train)
optimal_alpha = fred_lasso.alpha_
```

```python
lasso2 = linear_model.Lasso(alpha=optimal_alpha)
lasso2.fit(X_train, y_train)
lasso_coefs = pd.DataFrame({"features": list(X_train), "coef": lasso2.coef_})
lasso_coefs = lasso_coefs[lasso_coefs.coef != 0.0]
```

Then selected features are mapped, renamed, and re-run through `var_create(...)` for forecast MSE comparison.

## 7. XGBoost-Selected VAR Pipeline

XGBoost provides nonlinear feature ranking, then VAR remains the terminal forecasting model.

Boosting form:

$$
\hat{f}(x) = \sum_{m=1}^{M} \gamma_m h_m(x).
$$

Conceptually, this asks whether nonlinear one-step predictive relevance produces better linear system forecasts over horizon.

Specific implementation detail:

- The feature matrix is first restricted to transformed (`*_diff`) predictors and standardized with `StandardScaler()`.
- `XGBRegressor()` is used as the ranking model, and `plot_importance(...)` is used to identify the most influential predictors.
- As with the Lasso branch, the selected variables are then fed into the same downstream VAR function for a controlled comparison.

Notebook code blocks:

```python
fred_features = fred_features[list(fred_features.filter(regex = "_diff"))]
X_train, y_train, X_test, y_test = timeseries_train_test_split(X=fred_features, y=cpi_target, testsize=0.25)

scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

xgb = XGBRegressor()
xgb.fit(X_train_scaled, y_train)
```

```python
fig_xgb, ax_xgb = plt.subplots(figsize=(10,6))
plot_importance(xgb, max_num_features=5, ax=ax_xgb)
```

After ranking, chosen predictors are renamed and passed to `var_create(...)` for apples-to-apples MSE comparison with the other two pipelines.

## 8. PCA-Ranked VAR Pipeline

PCA is used here as an unsupervised feature-ranking stage before re-estimating VAR on a compact subset of original variables.

The predictor matrix is standardized:

$$
Z_{ij} = \frac{X_{ij} - \mu_j}{\sigma_j}.
$$

PCA is then fit on the training predictors only:

$$
Z = U \Lambda V^\top.
$$

Two related diagnostics are used:

- a scree plot, which shows explained variance component by component,
- a cumulative explained variance plot, which shows how quickly total variance is captured as more components are added.

Specific implementation detail:

- The notebook first fits a full PCA on the training matrix to inspect the explained-variance profile.
- It then fits a retained PCA with `n_components=0.95`, keeping enough components to explain about 95% of predictor variance.
- Original variables are ranked by weighted absolute loading contribution across retained components.
- The top 5 PCA-ranked variables are then passed into the same downstream `VAR` function.

Notebook code blocks:

```python
scaler_pca = preprocessing.StandardScaler()
X_train_scaled = scaler_pca.fit_transform(X_train)

pca_full = PCA(svd_solver='full')
pca_full.fit(X_train_scaled)
```

```python
pca_fs = PCA(n_components=0.95, svd_solver='full')
pca_fs.fit(X_train_scaled)

loadings = pca_fs.components_.T
feature_scores = (abs(loadings) * pca_fs.explained_variance_ratio_).sum(axis=1)
```

```python
pca_selected_features = score_df['feature'].head(5).tolist()
pca_var_cols = pca_selected_features + ['cpi_diff']
mse4, df4 = var_create(columns=pca_var_cols, data=fred_pca)
```

## 9. Comparative Findings and Interpretation

The notebook’s comparative evaluation indicates the theory-driven benchmark VAR performs best on inflation forecast MSE among the compared approaches.

Interpretation:

- Good feature ranking in a supervised predictive model does not guarantee good multi-step dynamic forecasting in a VAR.
- Endogenous propagation in VAR can reward structurally coherent macro sets more than purely top-ranked predictive subsets.
- PCA can summarize broad predictor structure well, but that does not automatically imply the top loading-driven variables will produce the best downstream inflation forecasts.

Notebook comparison block:

```python
ax = sns.barplot(x=["Traditional", "Lasso", "XGBoost", "PCA"], y=[mse1, mse2, mse3, mse4])
ax.set_title("Comparing MSEs", fontsize=16, fontname="Verdana")
ax.set_ylabel("MSE", fontname="Verdana")
```

## 10. Programmatic Backbone and Practical Value

The notebook keeps one common evaluation spine:

- Same forecast target (`cpi_diff`), same split logic, same VAR forecasting function.
- Only the feature-selection front-end changes across experiments.

This controls the experiment properly: differences in forecast quality can be attributed to variable-selection strategy rather than downstream model changes.

Core reusable blocks:

```python
def var_create(columns, data):
    data_train = data.loc["1973-01":"2009-01", columns]
    var_train = VAR(data_train)
    results = var_train.fit(12)
    lag_order = results.k_ar
    forecasted = pd.DataFrame(results.forecast(data_train.values[-lag_order:], 120))
    ...
    return var_mse, final_data
```

```python
mse1, df1 = var_create(columns=[...traditional set...], data=fred)
mse2, df2 = var_create(columns=[...lasso-selected set...], data=fred)
mse3, df3 = var_create(columns=[...xgb-selected set...], data=fred)
mse4, df4 = var_create(columns=[...pca-selected set...], data=fred_pca)
```

That design makes the project technically defensible for research and interview discussion: same endpoint model, controlled feature-selection interventions, explicit quantitative comparison.

## 11. Interview Narrative

### Context

- The project addresses inflation forecasting with a high-dimensional macroeconomic dataset in which predictors are correlated, regime-sensitive, and often only usable after transformation.
- The practical problem is not just prediction accuracy in isolation, but whether a chosen feature set remains useful once placed inside a multivariate dynamic forecasting system.

### Goal

- Evaluate whether alternative feature-selection strategies improve inflation forecasting when the terminal model is a `VAR`.
- Keep the forecasting engine fixed and vary only the variable-selection logic.

### Execution

1. Built a theory-driven benchmark VAR.
- What happened: selected a compact macro block from economic priors, transformed variables to annual differences, and fit a fixed-lag monthly VAR.
- Why it matters: this establishes a structurally coherent baseline and prevents the comparison from being purely algorithmic.

2. Built a Lasso-selected VAR branch.
- What happened: used `LassoCV` with `TimeSeriesSplit` to select sparse predictors, then passed the surviving variables into the same `VAR` forecasting routine.
- Why it matters: this tests whether sparse linear screening improves downstream system forecasting rather than just one-step supervised fit.

3. Built an XGBoost-selected VAR branch.
- What happened: standardized transformed predictors, fit `XGBRegressor`, ranked features by importance, and reused the top variables inside the same `VAR` pipeline.
- Why it matters: this tests whether nonlinear predictive salience translates into stronger linear multi-step forecasts.

4. Built a PCA-ranked VAR branch.
- What happened: standardized training predictors, examined a scree plot and cumulative explained variance, retained enough components to explain most predictor variance, and ranked original variables using weighted absolute loadings.
- Why it matters: this introduces an unsupervised, factor-based ranking mechanism that captures broad predictor structure without directly fitting to the target.

5. Enforced a controlled experiment design.
- What happened: kept target definition, train/test period split, lag order, forecast horizon, and `MSE` metric consistent across all branches.
- Why it matters: differences in performance can be attributed to feature-selection strategy rather than to changes in the downstream model.

### Outcome

- The theory-driven benchmark remained the strongest forecasting specification in this setup.
- Lasso and XGBoost were useful as screening mechanisms, but neither guaranteed better multi-step inflation forecasts once endogenous VAR feedback was introduced.
- PCA added a useful structural comparison by showing that broad predictor variance can be summarized effectively, but that strong factor loadings do not necessarily imply the best downstream forecast behavior.

### Main Technical Insight

- Strong one-step feature ranking does not automatically transfer to stronger multi-step system forecasts.
- In macro forecasting, the interaction between variable choice, regime sensitivity, and endogenous dynamics can dominate pure predictive ranking quality.

### Practical Takeaway

- Machine learning and PCA are valuable for screening and structure discovery.
- Final model quality still has to be validated in the exact forecasting system used for decision-making, not just in the preliminary selection stage.
