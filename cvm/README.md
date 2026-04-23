# CVM: Authoritative Implementation Document (Living)

## 1) Project Scope and Objective

`cvm` is a commodity volatility modeling project inspired by the MathWorks soft-commodities example and implemented in Python. It builds an end-to-end workflow from raw commodity price data and CPI data to:

- Inflation-adjusted commodity price panels (real Jan-2022 USD)
- Return series and additive decomposition (trend/seasonal/irregular)
- Cross-commodity structure discovery (hierarchical clustering + MDS)
- Nonparametric volatility estimation (EWMA)
- Volatility factor extraction (PCA + Factor Analysis)
- Model-based volatility simulation (ARIMA + GARCH on latent factors)
- Post-simulation diagnostics and persistence validation

This file is the single authoritative implementation record for `cvm`.

## 1.1) Source Authority Baseline

Primary conceptual/code authority:

- MathWorks example: `Volatility Modeling for Soft Commodities`
- URL: `https://www.mathworks.com/help/finance/volatility-modeling-for-soft-commodities.html`

The Python implementation follows this structure but introduces some implementation-specific deviations documented in Section 12.

## 2) Repository Contents (Current State)

```text
cvm/
  soft_v1.ipynb
  soft_v2.ipynb
  CMO-Historical-Data-Monthly.xlsx
  CPI-Data.xlsx
  prices_real_jan2022usd.csv
  returns.csv
  trend_component.csv
  seasonal_component.csv
  irregular_component.csv
  clusters_irregular_complete.csv
  ewma_volatility_matrix.csv
  volatility_summary.csv
  volatility_pca_components.csv
  artifacts/
    csv/
    figures/
```

Notes:

- `soft_v2.ipynb` is the main implementation notebook.
- `soft_v1.ipynb` is an earlier scaffold for ingestion/classification.
- `artifacts/csv` and `artifacts/figures` exist but are currently empty.

## 3) Runtime, Libraries, and Execution Surface

- Notebook kernel metadata: Python `3.11.5`, kernel `python3` (`display_name: base`).
- Primary stack:
  - Data: `pandas`, `numpy`, `pathlib`, `re`
  - Visualization: `matplotlib`, `seaborn`, 3D plotting toolkit
  - Econometrics/TS: `statsmodels` (`seasonal_decompose`, ARIMA), `arch` (GARCH)
  - ML/Factor methods: `sklearn` (`PCA`, `FactorAnalysis`)
  - Clustering/geometry: `scipy` hierarchy and distance tools
- Main entrypoint for full pipeline: `soft_v2.ipynb`.

## 4) Data Sources and Input Contracts

## 4.1 Commodity Prices

- Source file: `CMO-Historical-Data-Monthly.xlsx`
- Reader in notebook: `read_worldbank_cmo_monthly_prices(...)`
- Expected outcome:
  - Monthly date index at month-end
  - Numeric commodity price columns across soft/hard commodities

## 4.2 CPI Data

- Source file: `CPI-Data.xlsx`
- Reader in notebook: `read_bls_cpi_sheet(path, sheet="CUUR0000SA0R")`
- Reader behavior:
  - Detects header row dynamically
  - Requires `Year` and `Period` structure
  - Converts to monthly month-end index
  - Produces single `CPI` column

## 4.3 Derived Data Products (persisted)

- `prices_real_jan2022usd.csv`: 775 rows, 72 columns, `1960-01-31` to `2024-07-31`
- `returns.csv`: 774 rows, 72 columns, `1960-02-29` to `2024-07-31`
- `trend_component.csv`: 248 rows, 72 columns, `2003-12-31` to `2024-07-31`
- `seasonal_component.csv`: 248 rows, 72 columns, `2003-12-31` to `2024-07-31`
- `irregular_component.csv`: 248 rows, 72 columns, `2003-12-31` to `2024-07-31`
- `clusters_irregular_complete.csv`: 71 rows, 2 columns (commodity/cluster)
- `ewma_volatility_matrix.csv`: 774 rows, 72 columns
- `volatility_summary.csv`: 71 rows, 5 columns (cross-commodity volatility stats)
- `volatility_pca_components.csv`: 774 rows, 4 columns (date + PCs)

## 5) End-to-End Pipeline in `soft_v2.ipynb`

## Section A: Project Scaffolding

- Defines file paths from `Path.cwd()`.
- Asserts existence of core input files.
- Prints available CPI sheets for schema validation.

## Section B: Data Ingestion

- Loads and parses World Bank CMO monthly price panel.
- Loads CPI series from the BLS sheet `CUUR0000SA0R`.
- Performs sanity checks on shape/date ranges and numeric parse quality.

Key implementation detail:

- The notebook includes a CPI reconstruction step from interpreted percentage changes:
  - `cpi_level = (1 + (-cpi_series) / 100).cumprod()`
  - Then normalized to index base 100.
- This is a project-specific transformation choice and should be considered part of current model assumptions.

## Section C: Inflation Adjustment (Jan-2022 USD)

- Normalizes CPI (MATLAB-style convention) and rebases to anchor:
  - `anchor = 2022-01-31`
  - `CPI_rebased_t = CPI_t / CPI_anchor`
- Aligns commodity prices and CPI by date intersection.
- Converts nominal prices to real prices:
  - `real_price_t = nominal_price_t * CPI_rebased_t`
- Saves `prices_real_jan2022usd.csv`.

## Section D: Returns Conversion

- Computes arithmetic returns:
  - `returns = prices_real.pct_change()`
- Handles problematic values:
  - Replaces `+/-inf` with `NaN`
- Saves `returns.csv`.
- Tracks valid observation counts over time.

## Section E: Additive Decomposition

- Applies `seasonal_decompose(..., model="additive", period=12, extrapolate_trend="freq")` commodity by commodity.
- Decomposition outputs:
  - Trend
  - Seasonal
  - Irregular (residual)
- Aligns and trims edge NaNs generated by convolution windows.
- Saves:
  - `trend_component.csv`
  - `seasonal_component.csv`
  - `irregular_component.csv`

## Section F1: Hierarchical Clustering (Irregular Component)

- Computes pairwise Pearson correlation on irregular components.
- Converts correlation to distance:
  - `distance = 1 - corr`
- Runs complete-linkage hierarchical clustering with optimal ordering:
  - `linkage(..., method="complete", optimal_ordering=True)`
- Uses `k = 4` max-cluster cut for current grouping snapshot.
- Saves cluster membership:
  - `clusters_irregular_complete.csv`

## Section F2: Classical MDS (3D)

- Uses the irregular-component distance matrix.
- Classical MDS via double-centering on squared distances:
  - `B = -0.5 * J @ D^2 @ J`
- Eigen-decomposition and embedding into top positive dimensions (up to 3).
- Produces 3D scatter by cluster and optional convex hull overlays.

## Part I: Nonparametric Volatility Estimation

## Section G: EWMA Volatility

- Primary smoothing parameter:
  - `lambda = 0.94`
- Implements EWMA via IIR filtering in a custom function:
  - `ewma_volatility(returns_df, lam=0.94)`
- Also computes an alternate EWMA form via pandas `ewm` in later sections.
- Produces per-commodity volatility panel `vol_ewma`.

## Section H: Volatility Aggregation and Cross-Commodity Analysis

- Builds summary metrics per commodity:
  - Mean volatility
  - Volatility std
  - One-lag volatility persistence proxy
- Computes volatility correlation heatmap.
- Applies PCA (`n_components=3`) to volatility panel.
- Saves:
  - `volatility_summary.csv`
  - `ewma_volatility_matrix.csv`
  - `volatility_pca_components.csv`

## Section H2: Factor Analysis on Volatility

- Uses `FactorAnalysis(n_components=3)` on volatility matrix.
- Extracts loadings and strongest-factor indicator matrix.
- Visualizes heatmaps to mirror MATLAB-style diagnostics.

## Part II: Model-Based Volatility Simulation

## MB-1: PCA of Inferred Volatilities

- Rebuilds inferred volatility from returns:
  - `vol = sqrt(EWMA of squared returns)`
- Column-wise z-score normalization.
- PCA fit with full SVD.
- Captures:
  - Scores (latent factor time series)
  - Coefficients/loadings
  - Eigenvalues (scree diagnostics)

## MB-2: ARIMA + GARCH on Principal Components

- For each primary component (PC1, PC2):
  - Fit mean model: `ARIMA(1,0,1)`
  - Fit variance model on residuals: `GARCH(1,1)` with Student-t innovations
- Helper function: `fit_arima_garch_t(series, set_omega=None)`
- Outputs include printed ARIMA and GARCH fit summaries.

## MB-3: Forward Volatility Simulation

- Forecast horizon:
  - `num_forecast_steps = 60` (monthly steps)
- Simulates PC paths from fitted GARCH models via `.model.simulate(...)`.
- Combines simulated components and back-transforms to commodity volatility domain:
  - `approx_vol = mu + sigma * (sim_components @ coeffs[:, :2].T)`
- Creates forecast date index after last historical observation.
- Produces overlay plot of historical vs model-based volatility for selected commodity.

## MB-4: Post-Simulation Validation

- Compares historical and simulated moments (mean/std) for selected commodity.
- Overlays distributions (histograms).
- Compares ACF of squared volatility (historical vs simulated).
- Computes persistence indicator from GARCH parameters:
  - `alpha + beta`

## 6) Legacy Notebook (`soft_v1.ipynb`) Role

`soft_v1.ipynb` primarily contains early-stage setup:

- Importing CMO data
- Date/index setup
- Initial commodity class mapping
- Soft vs hard category tagging logic

Current relevance:

- Useful as lineage/context, but the complete pipeline is implemented in `soft_v2.ipynb`.

## 7) Core Methodological Assumptions

1. CPI reconstruction and rebasing choices are custom and embedded in notebook logic.
2. Arithmetic returns are used (not log returns) in the current baseline flow.
3. Additive decomposition assumes 12-month seasonality and linear additive structure.
4. Correlation-distance (`1 - corr`) on irregular components is the clustering similarity basis.
5. EWMA decay fixed at `lambda = 0.94`.
6. Volatility latent dynamics are represented by low-dimensional PCs, modeled with ARIMA(1,0,1)+GARCH(1,1).

## 12) Alignment Check vs MathWorks Source

Status summary:

- `cvm/README.md` is in sync with current Python outputs and notebook behavior.
- Current Python implementation is directionally aligned with the MathWorks flow, but not fully equivalent.

### 12.1 Areas in Sync (Conceptual)

1. Pipeline shape is aligned.
- Import prices/CPI, inflation-adjust prices, compute returns, decompose, cluster, estimate EWMA volatility, perform factor/PCA analysis, then model/simulate volatility with ARIMA+GARCH.

2. Core model choices are aligned.
- EWMA variance recursion with `lambda = 0.94`.
- PCA on normalized volatility.
- ARIMA(1,0,1) + GARCH(1,1) for leading components.
- Back-transform from PC space to volatility domain.

3. Simulation horizon is aligned.
- `numForecastSteps = 60` in both source and Python notebook.

### 12.2 Divergences from Source (Important)

1. Universe selection diverges (major).
- Source: explicitly filters to **soft commodities with complete histories**.
- Current Python: processes the broader commodity set from CMO (including hard commodities), evidenced by columns such as crude oil, natural gas, aluminum, gold, and silver in `prices_real_jan2022usd.csv`.

2. Cluster count differs.
- Source clustering cut: `numClusters = 3`.
- Python notebook currently uses `k = 4` for `fcluster(...)`.

3. Decomposition method differs.
- Source uses MATLAB `trenddecomp` (supports multiple seasonal components).
- Python uses `statsmodels.seasonal_decompose(..., period=12, additive)`.

4. CPI handling differs.
- Source uses CPI timetable directly (`CUUR0000SA0R`) and rebases to Jan-2022.
- Python includes an additional CPI reconstruction step from interpreted percentage-change series before rebasing, which is a custom adaptation.

5. Factor-analysis sample scope differs.
- Source factor analysis is on inferred volatility for soft-only universe.
- Python factor analysis is on the broader universe unless manually filtered.

### 12.3 Practical Implication

If strict source parity is required, the highest-impact fixes are:

1. Enforce soft-only + no-missing commodity filter before downstream steps.
2. Set clustering cut to 3 groups.
3. Decide whether to replicate `trenddecomp` behavior more closely or keep current decomposition with explicit caveat.
4. Standardize CPI ingest path to mirror source assumptions.

## 8) Current Strengths

- Full traceable pipeline from raw inputs to saved modeling outputs.
- Clear separation of nonparametric and model-based volatility modules.
- Persisted intermediate artifacts enable reproducibility and downstream reuse.
- Includes both structural analysis (clusters/MDS) and dynamic simulation.

## 9) Known Gaps and Technical Debt

1. Notebook-centric architecture
- Logic is in notebook cells; no packaged module/pipeline runner yet.

2. Repeated implementations
- EWMA/volatility transformations appear in multiple forms in different sections.

3. Assumption documentation in code
- CPI reconstruction sign-flip logic is implemented but not parameterized/config-managed.

4. Artifact management
- Outputs are written in project root while `artifacts/` directories are unused.

5. Validation breadth
- Post-simulation diagnostics are strong for selected commodities but not yet automated across all series.

6. Dependency pinning
- No environment lockfile or explicit `requirements` file in this folder.

## 10) Reproducible Run Order (Current)

From project root:

```bash
cd /Users/atheeshkrishnan/AK/DEV/econometric-forecasting/cvm
jupyter notebook
```

Then run `soft_v2.ipynb` in section order:

1. Project Scaffolding
2. Data Ingestion
3. Inflation Adjustment
4. Returns Conversion
5. Decomposition
6. Clustering + MDS
7. EWMA + Volatility Aggregation
8. PCA/FA volatility factors
9. ARIMA-GARCH simulation and validation

## 11) Living Document Update Protocol

For every project change, update this file with:

- What changed (data logic, model spec, output schemas, diagnostics)
- Why it changed (bug fix, empirical mismatch, modeling upgrade)
- Evidence (metrics/plots/tests)
- Backward compatibility impact (if any)
- New limitations and next priority tasks

This keeps `cvm/README.md` as the single definitive implementation reference.
