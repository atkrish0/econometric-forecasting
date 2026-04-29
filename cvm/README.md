# CVM: Commodity Volatility Modeling

`cvm` is a Python notebook project for modeling commodity volatility across a broad monthly commodity panel. The project starts with raw nominal commodity prices, converts them into inflation-adjusted real prices, estimates historical volatility, extracts common volatility factors, and simulates forward commodity-level volatility paths.

Primary notebooks:

- `soft_v2.ipynb`: main end-to-end implementation.
- `soft_v1.ipynb`: earlier ingestion and taxonomy scaffold.

Primary data inputs:

- `CMO-Historical-Data-Monthly.xlsx`: World Bank monthly commodity price workbook.
- `CPI-Data.xlsx`: local CPI workbook used for inflation adjustment.

Primary generated outputs:

- `prices_real_jan2022usd.csv`
- `returns.csv`
- `trend_component.csv`
- `seasonal_component.csv`
- `irregular_component.csv`
- `clusters_irregular_complete.csv`
- `ewma_volatility_matrix.csv`
- `volatility_pca_components.csv`
- `volatility_summary.csv`

## 1. Project Scope

The current implementation models the broader commodity universe available in `CMO-Historical-Data-Monthly.xlsx`. The saved price, return, volatility, and factor outputs contain 71 commodity series plus the date column.

The universe includes:

- Energy commodities such as crude oil, coal, and natural gas.
- Agricultural and soft commodities such as cocoa, coffee, tea, oils, grains, livestock, sugar, tobacco, cotton, wood products, and rubber.
- Fertilizers such as phosphate rock, DAP, TSP, urea, and potassium chloride.
- Metals and minerals such as aluminum, iron ore, copper, lead, tin, nickel, zinc, gold, platinum, and silver.

This is not a supervised learning project with a single label or target variable. It is a time-series and cross-sectional volatility modeling project. The goal is to understand how commodity volatility behaves through time, how commodities group by shock behavior, and how common volatility factors can be simulated forward.

## 2. Modeling Objective

The project answers four linked questions:

1. What do commodity prices look like after removing broad inflation effects?
2. Which commodities have similar irregular, shock-like price behavior?
3. Can a high-dimensional volatility panel be represented by a smaller number of common factors?
4. Can those common volatility factors be modeled and simulated to produce commodity-level forward volatility scenarios?

The implementation uses a layered approach:

1. Data conditioning:
   Convert raw nominal commodity prices into aligned real-price series.

2. Structural decomposition:
   Separate each price series into trend, seasonal, and irregular components.

3. Dependency geometry:
   Cluster commodities using correlation distances of irregular components.

4. Volatility state estimation:
   Estimate smooth historical volatility states with EWMA.

5. Latent factor compression:
   Use PCA and Factor Analysis to summarize common volatility drivers.

6. Stochastic simulation:
   Fit ARIMA-GARCH models to leading volatility factors, simulate factor paths, and back-transform them into commodity-level volatility paths.

## 3. End-To-End Pipeline

The main workflow is:

1. Load monthly commodity prices from the World Bank CMO workbook.
2. Load and prepare the CPI adjustment series.
3. Convert nominal prices to January 2022 real USD.
4. Convert real price levels to arithmetic returns.
5. Decompose real prices into trend, seasonal, and irregular components.
6. Cluster commodities by irregular-component correlation distance.
7. Build a low-dimensional MDS map of commodity shock similarity.
8. Estimate EWMA volatility from returns.
9. Run PCA and Factor Analysis on the volatility panel.
10. Fit ARIMA(1,0,1) plus GARCH(1,1) models to leading PCA volatility factors.
11. Simulate factor paths over a 60-month horizon.
12. Back-transform simulated factors into approximate commodity-level volatility paths.
13. Validate the simulated behavior with moment, distribution, autocorrelation, and persistence diagnostics.

## 4. Data Ingestion

The World Bank commodity workbook contains monthly nominal price levels. The notebook reads the `Monthly Prices` sheet, locates the header row containing `Date`, drops non-data unit rows, parses World Bank date codes such as `1960M01`, and converts the panel to a month-end `DatetimeIndex`.

Core paths:

```python
DATA = Path.cwd()
CMO_PATH = DATA / "CMO-Historical-Data-Monthly.xlsx"
CPI_PATH = DATA / "CPI-Data.xlsx"

prices_nominal = read_worldbank_cmo_monthly_prices(CMO_PATH)
cpi = read_bls_cpi_sheet(CPI_PATH, sheet="CUUR0000SA0R")
```

The commodity parser returns a numeric panel:

- Rows are monthly dates.
- Columns are commodity names.
- Values are nominal price levels in the workbook's native units.

The CPI parser returns a monthly inflation-adjustment series. In this local workflow, the CPI workbook is handled defensively because its values may not arrive as a clean level index. The notebook reconstructs a consistent CPI-like level series before rebasing.

## 5. CPI Reconstruction And Real Prices

Commodity prices in the CMO workbook are nominal. Nominal prices combine two effects:

- Commodity-specific movement, such as supply shocks, demand shocks, inventory pressure, seasonality, or weather-sensitive production effects.
- Broad price-level movement, which can make older and newer price observations less directly comparable.

The project uses the CPI layer to express commodity prices in January 2022 real-dollar terms. This makes downstream returns and volatility estimates more focused on commodity behavior rather than general inflation drift.

The local CPI reconstruction step is:

```python
cpi_series = pd.to_numeric(cpi[col_name], errors="coerce")
cpi_level = (1 + (-cpi_series) / 100).cumprod()
cpi_level = cpi_level / cpi_level.iloc[0] * 100
cpi = pd.DataFrame({"CPI": cpi_level}, index=cpi.index)
```

This creates a continuous level-style series that can be rebased:

```python
anchor = pd.Timestamp("2022-01-31")
cpi_rebased = cpi.copy()
cpi_rebased["CPI_rebased"] = cpi_rebased["CPI"] / cpi_rebased.loc[anchor, "CPI"]
```

Real prices are computed as:

```math
P^{real}_{t,j} = P^{nominal}_{t,j} \times \frac{CPI_t}{CPI_{Jan2022}}
```

where:

- `t` is the month.
- `j` is the commodity.
- `P^{nominal}_{t,j}` is the raw nominal commodity price.
- `CPI_t / CPI_{Jan2022}` is the rebased inflation-adjustment multiplier.

Saved output:

- `prices_real_jan2022usd.csv`

This stage defines the valuation baseline for the rest of the project. Every return, decomposition, volatility estimate, factor model, and simulation inherits this real-price adjustment.

## 6. Returns

Volatility is modeled on returns rather than price levels. Price levels are not directly comparable across commodities because units and scales differ. Returns normalize each series by its own prior price level.

The notebook uses arithmetic returns:

```math
r_{t,j} = \frac{P^{real}_{t,j} - P^{real}_{t-1,j}}{P^{real}_{t-1,j}}
```

Notebook code:

```python
returns = prices_real.pct_change().dropna(how="all")
returns = returns.replace([np.inf, -np.inf], np.nan)
returns.to_csv("returns.csv")
```

Saved output:

- `returns.csv`

Returns are the input to EWMA volatility estimation. Squared returns act as the observed shock magnitude used to update the inferred variance state.

## 7. Additive Price Decomposition

Each real price series is decomposed into:

```math
P_t = T_t + S_t + I_t
```

where:

- `T_t` is the trend component.
- `S_t` is the seasonal component.
- `I_t` is the irregular component.

The implementation uses `statsmodels.tsa.seasonal.seasonal_decompose` with an additive model and a 12-month period:

```python
res = seasonal_decompose(
    s,
    model="additive",
    period=12,
    extrapolate_trend="freq",
)
```

Saved outputs:

- `trend_component.csv`
- `seasonal_component.csv`
- `irregular_component.csv`

The decomposition has a specific modeling purpose. Raw commodity prices can be similar because they share long-run drift, recurring seasonal patterns, or short-run shock behavior. Those are not the same concept. By extracting the irregular component, the clustering step focuses on residual shock behavior rather than grouping commodities only because their trends or seasonal patterns look similar.

## 8. Irregular-Component Clustering

The project clusters commodities using the correlation distance between irregular components:

```math
d_{ij} = 1 - corr(I_i, I_j)
```

where:

- `I_i` is the irregular component of commodity `i`.
- `I_j` is the irregular component of commodity `j`.
- Highly correlated irregular components have smaller distance.
- Weakly or negatively correlated irregular components have larger distance.

Notebook code:

```python
corr = irregular_df.corr(method="pearson", min_periods=12)
dist_mat = 1.0 - corr
np.fill_diagonal(dist_mat.values, 0.0)
dist_condensed = squareform(dist_mat.values, checks=False)
Z = linkage(dist_condensed, method="complete", optimal_ordering=True)
k = 4
clusters = fcluster(Z, k, criterion="maxclust")
```

Saved output:

- `clusters_irregular_complete.csv`

The current notebook uses `k = 4` clusters. This is a modeling choice based on the broader commodity universe and the dendrogram structure. Because the project includes energy, fertilizers, metals, and agricultural commodities, the clustering problem spans multiple commodity classes.

## 9. Classical MDS Geometry

The notebook also maps the irregular-component distance matrix into a low-dimensional geometry with classical multidimensional scaling.

Given a distance matrix `D`, the centered Gram matrix is:

```math
B = -\frac{1}{2}J D^2 J
```

with:

```math
J = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top
```

The embedding is:

```math
X = V_m \Lambda_m^{1/2}
```

where:

- `V_m` contains the leading eigenvectors.
- `Lambda_m` contains the corresponding positive eigenvalues.
- `X` is the lower-dimensional coordinate representation.

The point of MDS is interpretability. It turns the abstract distance matrix into a visual map where commodities with similar irregular shock behavior appear closer together.

## 10. EWMA Volatility

The project uses exponentially weighted moving average volatility with:

```python
lambda_ = 0.94
```

EWMA estimates an instantaneous variance state from squared returns:

```math
y_1 = r_1^2
```

```math
y_t = \lambda y_{t-1} + (1-\lambda)r_t^2
```

Volatility is the square root of the inferred variance:

```math
\sigma_t = \sqrt{y_t}
```

Interpretation:

- `lambda` controls memory.
- A higher `lambda` gives more weight to the previous variance state.
- A lower `lambda` gives more weight to the newest squared return.
- With `lambda = 0.94`, volatility is smooth but still responsive to persistent shocks.

IIR filter implementation:

```python
def ewma_volatility(returns_df, lam=lambda_):
    var = lfilter([1 - lam], [1, -lam], r**2)
    return np.sqrt(var)
```

Pandas implementation used in the model-based section:

```python
alpha = 1 - lambda_
vol_df = (returns**2).ewm(alpha=alpha).mean() ** 0.5
```

Saved output:

- `ewma_volatility_matrix.csv`

EWMA is useful here because the project has many commodity series. It gives a consistent nonparametric volatility estimate across the panel without requiring a separate fitted volatility model for every commodity.

## 11. Volatility PCA

The EWMA output is a time-by-commodity volatility matrix. Modeling every volatility column separately would be high-dimensional and difficult to interpret. PCA compresses this volatility matrix into common factors.

First, volatility is standardized commodity by commodity:

```math
Z_{t,j} = \frac{V_{t,j} - \mu_j}{\sigma_j}
```

where:

- `V_{t,j}` is EWMA volatility for commodity `j` at time `t`.
- `mu_j` is commodity `j`'s average volatility.
- `sigma_j` is commodity `j`'s volatility standard deviation.

Then PCA is applied:

```python
mu = vol_df.mean(axis=0)
sigma = vol_df.std(axis=0, ddof=0)
vol_z = (vol_df - mu) / sigma
pca = PCA(svd_solver="full")
scores = pca.fit_transform(vol_z.values)
coeffs = pca.components_.T
eigVals = pca.explained_variance_
```

Saved output:

- `volatility_pca_components.csv`

Important objects:

- `scores`: time series of common volatility factors. These are the values saved in `volatility_pca_components.csv`.
- `coeffs`: PCA eigenvectors/loadings. Each column is a principal-component direction across commodities, and each row corresponds to a commodity's loading on that component.
- `eigVals`: eigenvalues of the volatility covariance matrix, equal to the variance explained by each principal component.
- `mu` and `sigma`: needed to map simulated standardized volatility back to original volatility units.

The current saved CSV output contains PCA scores, not PCA eigenvectors. The eigenvectors exist in the notebook as `coeffs = pca.components_.T` and are used later in:

```python
approx_vol = mu + sigma * (sim_components @ coeffs[:, :2].T)
```

That back-transform works because `coeffs[:, :2]` contains the first two PCA eigenvectors/loadings. If the eigenvectors need to be inspected outside the notebook, they should be saved as a separate loadings table indexed by commodity name.

PCA is the bridge between historical volatility estimation and model-based simulation. Instead of fitting a model to each commodity's volatility, the notebook models the leading factor scores.

## 12. Factor Analysis

Factor Analysis is the second latent-factor view of the volatility panel. It is related to PCA, but the modeling interpretation is different. PCA finds orthogonal directions that maximize explained variance. Factor Analysis treats each observed commodity volatility series as the combination of a smaller set of latent common factors plus a commodity-specific residual term.

### 12.1 Factor Model

The factor model can be written as:

```math
v_t = \Lambda f_t + \epsilon_t
```

where:

- `v_t` is the vector of observed commodity volatilities at time `t`.
- `f_t` is the vector of latent volatility factors.
- `Lambda` is the factor loading matrix.
- `epsilon_t` is the idiosyncratic volatility component not explained by the common factors.

For commodity `j`, the same idea is:

```math
v_{t,j} = \lambda_{j,1}f_{t,1} + \lambda_{j,2}f_{t,2} + \lambda_{j,3}f_{t,3} + \epsilon_{t,j}
```

This says each commodity's volatility can be decomposed into exposure to shared latent factors plus a residual component unique to that commodity.

### 12.2 Implementation

The notebook estimates a three-factor model:

```python
fa = FactorAnalysis(n_components=3, random_state=0)
fa.fit(vol_ewma.dropna())
```

The loadings are extracted as:

```python
loadings = pd.DataFrame(
    fa.components_.T,
    index=vol_ewma.columns,
    columns=[f"Factor {i+1}" for i in range(num_latent_factors)]
)
```

Rows correspond to commodities. Columns correspond to latent factors. A large positive or negative loading means that the commodity's volatility is strongly associated with that latent factor. A loading near zero means that factor does not explain much of that commodity's volatility movement.

### 12.3 Interpretation

Factor loadings answer a different question from PCA scores:

- PCA scores describe how the common volatility factors move through time.
- PCA loadings/eigenvectors describe each commodity's direction in the variance-maximizing PCA basis.
- Factor Analysis loadings describe how strongly each commodity is exposed to latent common volatility drivers after allowing for idiosyncratic residual variation.

The useful interpretation is cross-sectional. Commodities with similar loading patterns are exposed to similar latent volatility forces. A commodity with one dominant loading is mainly tied to one common volatility factor. A commodity with meaningful loadings across several factors is exposed to multiple shared volatility channels. A commodity with weak loadings may be driven more by idiosyncratic behavior than by the common factors.

### 12.4 Common Versus Idiosyncratic Volatility

Factor Analysis is valuable because it makes the common-versus-specific split explicit:

```math
\operatorname{Var}(v_j) \approx \sum_{k=1}^{K}\lambda_{j,k}^2 + \psi_j
```

where:

- `sum(lambda_{j,k}^2)` is the part of commodity `j`'s volatility associated with common factors.
- `psi_j` is the commodity-specific residual variance, often called uniqueness.

Conceptually, this helps distinguish between commodities that are volatile because they load heavily on broad panel-wide volatility factors and commodities that are volatile because of more commodity-specific shocks.

### 12.5 Role In This Project

In this project, Factor Analysis is used as a diagnostic and interpretability layer, not as the primary simulation engine.

It helps answer:

- Which commodities share common latent volatility exposures?
- Which commodities appear more idiosyncratic?
- Whether the volatility panel has a coherent low-dimensional factor structure beyond the PCA decomposition.
- Whether the PCA-first simulation approach is reasonable from a latent-factor perspective.

PCA remains the simulation basis because its orthogonal scores, eigenvectors/loadings, and normalization parameters give a direct path for back-transforming simulated factor values into approximate commodity-level volatility:

```python
approx_vol = mu + sigma * (sim_components @ coeffs[:, :2].T)
```

Factor Analysis complements that workflow by explaining the latent exposure structure in a way that is closer to a common-factor risk model.

## 13. ARIMA-GARCH Factor Modeling

EWMA gives historical inferred volatility, but it does not by itself simulate future volatility paths. The project therefore models the leading PCA volatility factors.

For each leading factor, the notebook uses:

- ARIMA(1,0,1) for conditional mean dynamics.
- GARCH(1,1) for conditional variance dynamics.
- Student-t innovations to allow heavy-tailed shocks.

Mean equation:

```math
x_t = c + \phi x_{t-1} + \theta \epsilon_{t-1} + \epsilon_t
```

Variance equation:

```math
h_t = \omega + \alpha\epsilon_{t-1}^2 + \beta h_{t-1}
```

Notebook function:

```python
def fit_arima_garch_t(series, set_omega=None):
    arima_res = ARIMA(series, order=(1, 0, 1)).fit()
    resid = arima_res.resid
    garch = arch_model(resid, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
    res_garch = garch.fit(update_freq=0, disp="off")
```

Conceptually:

- ARIMA captures persistence and short-memory dynamics in the factor level.
- GARCH captures volatility clustering in the factor innovations.
- Student-t innovations reduce the normality assumption and better accommodate large shocks.

## 14. Simulation And Back-Transformation

The notebook simulates the first two fitted factor models over a 60-month horizon:

```python
num_forecast_steps = 60
sim_pc1 = garch1.model.simulate(garch1.params, nobs=num_forecast_steps)
sim_pc2 = garch2.model.simulate(garch2.params, nobs=num_forecast_steps)
sim_components = np.column_stack((sim_pc1_vals, sim_pc2_vals))
```

The simulated factor paths are mapped back to commodity volatility units with:

```math
\hat{V}_{future} = \mu + \sigma \odot (S W_{1:2}^{T})
```

where:

- `S` is the matrix of simulated factor values.
- `W_{1:2}` is the loading matrix for the first two principal components.
- `mu` and `sigma` undo the earlier z-score normalization.
- `odot` denotes elementwise multiplication.

Notebook code:

```python
approx_vol = mu + sigma * (sim_components @ coeffs[:, :2].T)
```

This is an approximation because only two principal components are simulated. Modeling more principal components would improve reconstruction fidelity but would require more fitted time-series models and more validation. The current choice prioritizes interpretability and tractability.

## 15. Diagnostics

The notebook checks whether simulated volatility behavior is plausible relative to historical volatility.

Diagnostics include:

- Historical versus simulated mean comparison.
- Historical versus simulated standard deviation comparison.
- Distribution overlays.
- Autocorrelation of squared volatility.
- GARCH persistence checks.

GARCH persistence is:

```math
\alpha + \beta
```

Interpretation:

- Values near 1 imply persistent volatility shocks.
- Lower values imply faster decay of volatility shocks.
- Persistence diagnostics help identify whether simulated paths have realistic volatility memory.

Saved output:

- `volatility_summary.csv`

## 16. Current Outputs

Generated files:

- `prices_real_jan2022usd.csv`: CPI-adjusted real commodity prices in January 2022 real-dollar terms.
- `returns.csv`: arithmetic returns from real prices.
- `trend_component.csv`: additive trend components from real prices.
- `seasonal_component.csv`: additive seasonal components from real prices.
- `irregular_component.csv`: additive irregular components from real prices.
- `clusters_irregular_complete.csv`: complete-linkage clusters based on irregular-component correlation distance.
- `ewma_volatility_matrix.csv`: inferred EWMA volatility matrix.
- `volatility_pca_components.csv`: first three PCA volatility factor score series, not the PCA eigenvectors/loadings.
- `volatility_summary.csv`: summary statistics and persistence-style diagnostics.

The current saved outputs reflect the broad 71-commodity panel available after local parsing and alignment.

## 17. Technical Caveats

1. CPI reconstruction is local to this workbook.
   The notebook reconstructs a CPI-like level series before rebasing. If a clean CPI level index is supplied in the future, this step should be reviewed and simplified.

2. Decomposition is additive.
   The implementation assumes price components combine additively. This is appropriate for the current workflow, but multiplicative decomposition may be worth testing if proportional seasonal effects dominate some commodities.

3. Clustering is based on irregular components.
   The clusters describe similarity in residual shock behavior after trend and seasonality are removed. They should not be interpreted as clusters of raw price levels or long-run economic sectors.

4. `k = 4` is a chosen cluster cut.
   The number of clusters is not estimated by a formal model-selection criterion in the current notebook. It is a dendrogram-driven modeling choice.

5. EWMA is an inferred volatility proxy.
   EWMA is nonparametric and scalable, but it is not a full generative volatility model. It estimates historical volatility states that are later used for factor modeling.

6. PCA simulation uses only the first two factors.
   The back-transformed volatility paths are approximate because omitted components are not simulated. This is a deliberate tradeoff between fidelity and tractability.

7. ARIMA and GARCH are fitted in two stages.
   The implementation fits ARIMA to factor levels and GARCH to ARIMA residuals. This is practical and interpretable, but it is not the same as estimating a single joint state-space model.

## 18. Conceptual Summary

The project turns a high-dimensional commodity panel into a structured volatility system:

```text
nominal prices
    -> CPI-adjusted real prices
    -> returns
    -> EWMA volatility states
    -> PCA volatility factors
    -> ARIMA-GARCH factor simulations
    -> approximate commodity-level volatility paths
```

The decomposition and clustering branch adds cross-sectional interpretation:

```text
real prices
    -> trend + seasonal + irregular components
    -> irregular-component correlation distances
    -> hierarchical clusters and MDS geometry
```

The core modeling idea is that commodity volatility is high-dimensional but not fully independent across commodities. A factor-first workflow captures shared volatility structure, makes simulation more tractable, and keeps the outputs interpretable at the commodity level.

## 19. Interview Narrative

### 19.1 Situation

Commodity prices are difficult to model because they combine long-run trends, seasonality, short-run shocks, shared macro drivers, and inflation effects. The local commodity panel is also broad, covering energy, agricultural commodities, fertilizers, metals, and precious metals. Modeling each series independently would create a large collection of noisy models and would make it hard to explain common volatility behavior across the panel.

The main issue was that raw nominal price levels were not a clean volatility input. They mix commodity-specific movements with broad price-level effects, and they are measured in different units and scales across commodities.

### 19.2 Task

The task was to build an end-to-end Python workflow that could:

- Convert nominal commodity prices into January 2022 real-dollar prices.
- Convert real prices into returns.
- Separate persistent structure from irregular shocks.
- Identify commodities with similar shock behavior.
- Estimate historical volatility states consistently across all commodities.
- Compress the volatility panel into common factors.
- Simulate forward volatility paths in factor space.
- Map those simulated factor paths back into commodity-level volatility scenarios.

The project was not designed as a one-target prediction model. It was designed as a volatility modeling system that explains historical structure and produces forward risk scenarios.

### 19.3 Action

I built the data layer first. The notebook parses the World Bank monthly commodity workbook into a clean month-end price panel and reads the local CPI workbook into a monthly adjustment series. Because the CPI input is handled as a non-canonical local workbook, I reconstructed a level-style CPI series, rebased it to January 2022, and multiplied nominal prices by the rebased adjustment multiplier to create real prices.

I then converted real prices into arithmetic returns. This made the series comparable across commodities with different units and price scales and created the squared-return input needed for volatility estimation.

Next, I decomposed each real price series into trend, seasonal, and irregular components using additive seasonal decomposition. I used the irregular components for clustering because the goal was to compare shock behavior, not raw levels or long-run trends. I transformed irregular-component correlations into distances, applied complete-linkage hierarchical clustering, and used classical MDS to visualize the commodity similarity geometry.

For volatility estimation, I implemented EWMA with `lambda = 0.94`. This gave a smooth inferred-volatility matrix across all commodities without fitting a separate parametric volatility model to each return series.

I then standardized the EWMA volatility matrix and applied PCA. PCA reduced the 71-commodity volatility panel into common volatility factors, with loadings that map factor movement back to commodities. I also estimated a three-factor Factor Analysis model as a diagnostic check on latent structure.

Finally, I fit ARIMA(1,0,1) plus GARCH(1,1) models with Student-t innovations to the leading PCA volatility factors. I simulated the first two factors over a 60-month horizon and back-transformed those simulated factors through the PCA loadings and volatility normalization parameters to recover approximate commodity-level volatility paths.

### 19.4 Result

The project produces a complete volatility workflow from raw data to forward simulation:

- Real commodity prices are saved in `prices_real_jan2022usd.csv`.
- Returns are saved in `returns.csv`.
- Decomposition outputs are saved in `trend_component.csv`, `seasonal_component.csv`, and `irregular_component.csv`.
- Irregular-component clusters are saved in `clusters_irregular_complete.csv`.
- EWMA volatility states are saved in `ewma_volatility_matrix.csv`.
- Volatility PCA factors are saved in `volatility_pca_components.csv`.
- Summary and persistence diagnostics are saved in `volatility_summary.csv`.

The main result is a technically coherent volatility system: inflation-adjusted prices feed returns, returns feed EWMA volatility, volatility feeds PCA factors, and PCA factors feed ARIMA-GARCH simulations. The workflow is more interpretable and scalable than fitting unrelated per-commodity models because it captures shared volatility structure before simulating forward paths.
