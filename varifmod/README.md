# VARIFMOD: Conceptual Implementation Document

## 1. Overview

Inflation forecasting is treated here as a system-level macroeconomic problem rather than a single-equation prediction task. The project compares three forecasting pipelines that all end in a Vector Autoregression (VAR), but differ in how predictors are chosen:

1. A theory-driven benchmark specification.
2. A Lasso-selected specification.
3. An XGBoost-selected specification.

The central point is to test whether data-driven feature selection improves downstream dynamic forecasting once variables interact endogenously in a multivariate time-series model.

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
old_names_diff = ["CPIAUCSL_diff", "RPI_diff", "UNRATE_diff", "FEDFUNDS_diff", "TWEXMMTH_diff", "HOUST_diff"]
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
lasso2 = linear_model.Lasso(alpha=optimal_alpha, normalize=True)
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

## 8. Comparative Findings and Interpretation

The notebook’s comparative evaluation indicates the theory-driven benchmark VAR performs best on inflation forecast MSE among the three approaches.

Interpretation:

- Good feature ranking in a supervised predictive model does not guarantee good multi-step dynamic forecasting in a VAR.
- Endogenous propagation in VAR can reward structurally coherent macro sets more than purely top-ranked predictive subsets.

Notebook comparison block:

```python
ax = sns.barplot(x=["Traditional", "Lasso", "XGB"], y=[mse1, mse2, mse3])
ax.set_title("Comparing MSEs", fontsize=16, fontname="Verdana")
ax.set_ylabel("MSE", fontname="Verdana")
```

## 9. Programmatic Backbone and Practical Value

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
```

That design makes the project technically defensible for research and interview discussion: same endpoint model, controlled feature-selection interventions, explicit quantitative comparison.
