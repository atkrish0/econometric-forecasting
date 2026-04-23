# CVM: Conceptual Implementation Document

## 1. Overview

`cvm` is a commodity-volatility modeling project that translates a MATLAB-style research workflow into Python notebooks, with `soft_v2.ipynb` as the primary implementation and `soft_v1.ipynb` as an earlier scaffold.

Core objective:

- Build an end-to-end volatility pipeline from raw commodity and CPI data to structural decomposition, cross-commodity dependency analysis, and model-based volatility simulation.

Conceptual flow:

1. Construct real (inflation-adjusted) commodity prices.
2. Convert prices to returns.
3. Decompose price series into trend/seasonal/irregular components.
4. Cluster commodities using irregular-behavior similarity.
5. Estimate nonparametric volatility (EWMA).
6. Extract latent volatility factors (PCA / FA).
7. Fit ARIMA+GARCH on factor dynamics and simulate forward volatility.

## 2. Why This Stack Was Chosen

The modeling choices are complementary and intentional:

- `Additive decomposition` separates long-run, seasonal, and shock-like components before dependency analysis.
- `Correlation-distance + clustering` reveals commodity co-movement regimes in idiosyncratic behavior.
- `EWMA` provides a robust, low-assumption volatility proxy for many commodity series.
- `PCA / Factor Analysis` compress high-dimensional volatility panels into latent common drivers.
- `ARIMA(1,0,1) + GARCH(1,1)` models mean-reverting factor dynamics and conditional heteroskedasticity for forward simulation.

This creates a layered architecture:

1. Data conditioning.
2. Structural extraction.
3. Dependency geometry.
4. Volatility state estimation.
5. Stochastic forward projection.

## 3. Notebook Roles: `soft_v1` vs `soft_v2`

`soft_v1.ipynb`:

- Imports CMO data.
- Establishes date/index handling.
- Builds early commodity-class and soft/hard categorization logic.

`soft_v2.ipynb`:

- Implements the complete production pipeline end-to-end.
- Includes all transformations, outputs, clustering, factor work, and simulation diagnostics.

Conceptually, `soft_v1` is lineage and taxonomy scaffolding; `soft_v2` is the full modeling engine.

## 4. Data Ingestion and CPI Layer

Primary inputs:

- `CMO-Historical-Data-Monthly.xlsx` (commodity panel).
- `CPI-Data.xlsx` sheet `CUUR0000SA0R`.

Core notebook path:

```python
DATA = Path.cwd()
CMO_PATH = DATA / "CMO-Historical-Data-Monthly.xlsx"
CPI_PATH = DATA / "CPI-Data.xlsx"

prices_nominal = read_worldbank_cmo_monthly_prices(CMO_PATH)
cpi = read_bls_cpi_sheet(CPI_PATH, sheet="CUUR0000SA0R")
```

The notebook includes a CPI reconstruction step:

$$
\text{CPILevel}_t = \prod_{i \le t} \left(1 + \frac{-c_i}{100}\right),
$$

then normalization to index scale.

Code anchor:

```python
cpi_series = pd.to_numeric(cpi[col_name], errors="coerce")
cpi_level = (1 + (-cpi_series) / 100).cumprod()
cpi_level = cpi_level / cpi_level.iloc[0] * 100
cpi = pd.DataFrame({"CPI": cpi_level}, index=cpi.index)
```

## 5. Inflation Adjustment and Real Price Construction

The pipeline rebases CPI to January 2022:

$$
\text{CPI}^{\text{rebased}}_t = \frac{\text{CPI}_t}{\text{CPI}_{\text{Jan-2022}}}.
$$

Real prices are computed as:

$$
P^{\text{real}}_{t,j} = P^{\text{nominal}}_{t,j}\cdot \text{CPI}^{\text{rebased}}_t.
$$

Notebook code:

```python
anchor = pd.Timestamp("2022-01-31")
cpi_rebased = cpi.copy()
cpi_rebased["CPI_rebased"] = cpi_rebased["CPI"] / cpi_rebased.loc[anchor, "CPI"]
aligned = prices_nominal.join(cpi_rebased["CPI_rebased"], how="inner")
prices_real = aligned.drop(columns=["CPI_rebased"]).mul(aligned["CPI_rebased"], axis=0)
prices_real.to_csv("prices_real_jan2022usd.csv")
```

This stage defines the valuation-consistent baseline for all downstream volatility analysis.

## 6. Returns and Structural Decomposition

Arithmetic returns:

$$
r_{t,j} = \frac{P^{\text{real}}_{t,j} - P^{\text{real}}_{t-1,j}}{P^{\text{real}}_{t-1,j}}.
$$

Notebook code:

```python
returns = prices_real.pct_change().dropna(how='all')
returns = returns.replace([np.inf, -np.inf], np.nan)
returns.to_csv("returns.csv")
```

Additive decomposition per commodity:

$$
P_{t,j}^{\text{real}} = T_{t,j} + S_{t,j} + I_{t,j},
$$

where \(T\) is trend, \(S\) is seasonal, and \(I\) is irregular.

Code anchor:

```python
res = seasonal_decompose(s, model='additive', period=12, extrapolate_trend='freq')
```

Saved outputs:

- `trend_component.csv`
- `seasonal_component.csv`
- `irregular_component.csv`

Conceptual role:

- isolate idiosyncratic shocks (`irregular`) for distance-based clustering.

## 7. Dependency Structure: Clustering and MDS

Irregular-component correlation matrix:

$$
\rho_{ij} = \operatorname{corr}(I_{\cdot,i}, I_{\cdot,j}).
$$

Distance transform:

$$
d_{ij} = 1 - \rho_{ij}.
$$

Complete-linkage clustering is applied to the condensed distance matrix:

```python
corr = irregular_df.corr(method="pearson", min_periods=12)
dist_mat = 1.0 - corr
dist_condensed = squareform(dist_mat.values, checks=False)
Z = linkage(dist_condensed, method="complete", optimal_ordering=True)
k = 4
clusters = fcluster(Z, k, criterion="maxclust")
cluster_map.to_csv("clusters_irregular_complete.csv")
```

Classical MDS is then used for low-dimensional geometry:

$$
B = -\frac{1}{2}J D^2 J,\quad J=I-\frac{1}{n}\mathbf{1}\mathbf{1}^\top.
$$

Embedding:

$$
X = V_m \Lambda_m^{1/2}.
$$

Code anchor:

```python
J = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * J @ (D ** 2) @ J
eigvals, eigvecs = np.linalg.eigh(B)
X = eigvecs[:, :m] @ np.diag(np.sqrt(eigvals[:m]))
```

This gives an interpretable 3D map of commodity shock-behavior similarity.

## 8. Nonparametric Volatility Estimation (EWMA)

The project uses EWMA with \(\lambda=0.94\):

$$
\sigma_{t,j}^2 = (1-\lambda)r_{t,j}^2 + \lambda \sigma_{t-1,j}^2.
$$

Implemented in two forms:

1. IIR filter-based custom function:

```python
def ewma_volatility(returns_df, lam=lambda_):
    var = lfilter([1 - lam], [1, -lam], r**2)
```

2. Pandas `ewm` form for aggregation stage:

```python
span = 1 / (1 - lambda_)
ewma_vols = pd.DataFrame({col: returns[col].ewm(span=span).std() for col in returns.columns})
```

Why this matters:

- EWMA produces a stable, scalable inferred-volatility matrix for many commodities without heavy parametric assumptions.

## 9. Volatility Factors: PCA and Factor Analysis

### 9.1 PCA on Volatility Matrix

Volatility matrix \(V\) is z-score normalized:

$$
Z_{t,j} = \frac{V_{t,j}-\mu_j}{\sigma_j}.
$$

PCA is applied:

$$
Z = U\Lambda W^\top.
$$

Notebook code:

```python
mu = vol_df.mean(axis=0)
sigma = vol_df.std(axis=0, ddof=0)
vol_z = (vol_df - mu) / sigma
pca = PCA(svd_solver="full")
scores = pca.fit_transform(vol_z.values)
coeffs = pca.components_.T
eigVals = pca.explained_variance_
```

Outputs:

- Common volatility factors (scores).
- Commodity loadings (coefficients).
- Scree diagnostics.

### 9.2 Factor Analysis

Factor model:

$$
v_t = \Lambda f_t + \epsilon_t.
$$

Code anchor:

```python
fa = FactorAnalysis(n_components=3, random_state=0)
fa.fit(vol_ewma.dropna())
```

This complements PCA by emphasizing latent-factor structure with explicit idiosyncratic terms.

## 10. Model-Based Volatility Simulation

### 10.1 ARIMA + GARCH on Leading PCs

For each leading PC series:

1. Mean dynamics via ARIMA(1,0,1):

$$
x_t = c + \phi x_{t-1} + \theta \epsilon_{t-1} + \epsilon_t.
$$

2. Residual variance via GARCH(1,1):

$$
h_t = \omega + \alpha \epsilon_{t-1}^2 + \beta h_{t-1}.
$$

Notebook function:

```python
def fit_arima_garch_t(series, set_omega=None):
    arima_res = ARIMA(series, order=(1,0,1)).fit()
    resid = arima_res.resid
    garch = arch_model(resid, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
```

### 10.2 PC Simulation and Back-Transformation

Simulate PC paths for horizon \(H=60\):

```python
num_forecast_steps = 60
sim_pc1 = garch1.model.simulate(garch1.params, nobs=num_forecast_steps)
sim_pc2 = garch2.model.simulate(garch2.params, nobs=num_forecast_steps)
sim_components = np.column_stack((sim_pc1_vals, sim_pc2_vals))
```

Back-transform to commodity volatility:

$$
\hat{V}_{\text{future}} = \mu + \Sigma \left(S \cdot W_{1:2}^\top\right),
$$

implemented as:

```python
approx_vol = mu + sigma * (sim_components @ coeffs[:, :2].T)
```

This projects latent-factor simulations back into commodity-level volatility forecasts.

## 11. Validation and Diagnostics

Post-simulation validation compares historical vs simulated behavior for selected commodities:

1. Mean/std comparison.
2. Distribution overlay.
3. ACF of squared volatility.
4. GARCH persistence check:

$$
\text{Persistence} = \alpha + \beta.
$$

Code anchor:

```python
alpha1_pc1 = garch1.params['alpha[1]']
beta1_pc1  = garch1.params['beta[1]']
```

If \(\alpha+\beta\) is near 1, volatility shocks are highly persistent.

## 12. Source Alignment (MathWorks) and Current Divergences

Conceptual source:

- MathWorks example “Volatility Modeling for Soft Commodities”.

Where this implementation is aligned:

1. Overall pipeline stages.
2. EWMA + factor-based volatility workflow.
3. ARIMA-GARCH-based model simulation with 60-step horizon.

Where it currently diverges:

1. Universe scope:
- Current notebook processes broader commodities (including hard commodities), not only soft-commodity complete-history subset.
2. Cluster cut:
- Current notebook uses `k = 4`; source example uses 3-cluster framing.
3. Decomposition routine:
- Python uses `seasonal_decompose`; source example uses MATLAB `trenddecomp`.
4. CPI handling:
- Python includes a reconstruction step before rebasing.

These differences are explicit implementation choices and should be kept in mind when interpreting parity claims.

## 13. Interview Narrative (Concise)

Context:

- Needed a robust volatility framework for a large cross-commodity universe with inflation adjustment, latent common factors, and forward simulation.

Goal:

- Build a pipeline that explains historical commodity volatility structure and generates defensible forward volatility scenarios.

Execution:

1. Ingested CMO + CPI data and converted nominal prices to Jan-2022 real prices.
2. Converted to returns and decomposed price series into trend/seasonal/irregular components.
3. Built irregular-component clustering and MDS geometry for cross-commodity structure.
4. Estimated EWMA volatilities and extracted latent factors via PCA/FA.
5. Modeled leading factors with ARIMA(1,0,1)+GARCH(1,1) and simulated 60-step forward volatility.
6. Validated via moments, distributions, ACF, and persistence.

Outcome:

- Established a full-stack, research-grade volatility pipeline from raw panels to simulated commodity-level volatility paths.
- Identified common volatility regimes and persistent latent dynamics.
- Clarified where Python implementation is aligned with, and where it diverges from, the original MATLAB reference.
