# IRD Pricer: Conceptual Implementation Document

## 1. Overview

`ird-pricer` is an interest-rate derivatives modeling workflow centered on the following research question:

- How do historical U.S. yield-curve dynamics map into forward-rate simulations, and how do those simulated dynamics affect cap/floor pricing under Monte Carlo?

The primary modeling stack in `NAIR.ipynb` combines:

1. Term-structure construction from Treasury constant-maturity yields and SOFR.
2. PCA-based volatility factor extraction from historical curve movements.
3. HJM-style forward-rate simulation under multiple random-number schemes.
4. Cap/floor payoff pricing with convergence diagnostics.
5. Variance-reduction experiments (standard, antithetic, control-variate variants).

The notebook is the source of truth for model flow; utility modules provide reusable building blocks and experiments around that flow.

## 2. Why This Modeling Stack Was Chosen

The method choices are complementary and deliberate:

- `PCA` was chosen to compress cross-maturity yield movements into dominant latent factors (level/slope/curvature), reducing dimensional noise before simulation.
- `HJM-style` simulation was chosen because forward rates are the natural state variables for interest-rate derivative payoffs.
- Multiple RNG schemes (`normal`, `uniform->inverse-normal`, `sobol`, `halton`, `lhs`) were chosen to study Monte Carlo stability and dispersion sensitivity to sampling design.
- Variance-reduction methods were included because vanilla Monte Carlo can converge slowly for path-dependent rate payoffs.

This design separates three concerns cleanly:

1. Historical structure extraction (PCA).
2. Stochastic propagation (simulation engine + RNG).
3. Pricing efficiency and robustness (payoff + variance reduction).

## 3. Data Engineering and Discounting Layer

The notebook ingests:

- SOFR history (`FRED_SOFR.csv`) for discounting.
- Treasury series (`DGS1`, `DGS2`, `DGS5`, `DGS10`, `DGS30`) for curve construction.

Core code path in `NAIR.ipynb`:

```python
sofr = pd.read_csv('data/FRED_SOFR.csv', parse_dates=['DATE'], index_col='DATE')
dgs1 = pd.read_csv('data/DGS1.csv', parse_dates=['DATE'], index_col='DATE')
dgs2 = pd.read_csv('data/DGS2.csv', parse_dates=['DATE'], index_col='DATE')
dgs5 = pd.read_csv('data/DGS5.csv', parse_dates=['DATE'], index_col='DATE')
dgs10 = pd.read_csv('data/DGS10.csv', parse_dates=['DATE'], index_col='DATE')
dgs30 = pd.read_csv('data/DGS30.csv', parse_dates=['DATE'], index_col='DATE')
```

Rates are cleaned, converted to numeric, and aligned on time. SOFR discount factors are built with daily step size:

$$
dt = \frac{1}{252},
\quad r_t = \frac{\text{SOFR}_t}{100},
\quad D_t = \exp\left(-\sum_{i=1}^{t} r_i \, dt\right).
$$

Notebook block:

```python
dt = 1 / 252
sofr['Rate'] = sofr['SOFR'] / 100
sofr['Discount_Factor'] = np.exp(-np.cumsum(sofr['Rate'] * dt))
```

Conceptually, this creates the discounting measure used later when mapping simulated forward payoffs into present values.

## 4. Curve Construction and Forward-Rate Geometry

A spot-yield panel is formed as:

$$
\mathcal{Y}_t = \{ y_t^{1Y}, y_t^{2Y}, y_t^{5Y}, y_t^{10Y}, y_t^{30Y} \}.
$$

The notebook computes interval forward rates using compounding-consistent transformation:

$$
f(t_1,t_2)=\left(\frac{(1+r_{t_2})^{t_2}}{(1+r_{t_1})^{t_1}}\right)^{\frac{1}{t_2-t_1}}-1.
$$

Implementation function:

```python
def calculate_forward_rate(yield_curve, t1, t2):
    r_t1 = yield_curve[f"{t1}Y"]
    r_t2 = yield_curve[f"{t2}Y"]
    forward_rate = ((1 + r_t2 / 100) ** t2 / (1 + r_t1 / 100) ** t1) ** (1 / (t2 - t1)) - 1
    return forward_rate * 100
```

Constructed forward buckets:

- `1Y-2Y`
- `2Y-5Y`
- `5Y-10Y`
- `10Y-30Y`

This stage converts static term snapshots into economically meaningful carry/expectation segments used in both interpretation and simulation setup.

## 5. PCA Volatility Calibration from Yield-Curve Dynamics

The yield panel is standardized, then decomposed with PCA:

$$
Z = \text{Standardize}(Y), \quad Z = U \Lambda V^\top.
$$

Key output:

- Eigenvectors (`pca.components_`) as maturity loadings.
- Eigenvalues / explained variance as factor strength.

Notebook path:

```python
yield_data = yc[['1Y', '2Y', '5Y', '10Y', '30Y']]
yield_data_std = scaler.fit_transform(yield_data)
pca = PCA(n_components=5)
pca.fit(yield_data_std)
explained_variance = pca.explained_variance_ratio_
```

The notebook’s volatility function combines factor loadings and variance weights:

```python
def calculate_volatility(t, T_index, eigenvectors, eigenvalues, n_factors=3, scaling_factor=10):
    ...
```

Conceptually:

$$
\sigma_j(t) \propto \sum_{k=1}^{K} \sqrt{\lambda_k}\, v_{k,j} \, g_k(t),
$$

where:

- \(j\) indexes maturity bucket,
- \(k\) indexes retained PCA factors,
- \(g_k(t)\) encodes notebook time scaling behavior.

This is the core link from historical curve movement structure to stochastic diffusion intensity.

## 6. HJM-Style Forward Simulation and RNG Design

Forward rates are simulated over maturities and paths using multiplicative diffusion:

$$
F_{t+\Delta t,j}^{(s)} =
F_{t,j}^{(s)} \exp\left(\mu_{t,j}\Delta t + \sigma_{t,j}\Delta W_{t,j}^{(s)}\right).
$$

In current notebook sections, drift is often set to zero in practice:

$$
\mu_{t,j}=0.
$$

Core block:

```python
simulated_forward_rates = np.zeros((n_steps, n_maturities, n_simulations))
simulated_forward_rates[0, :, :] = initial_forward_rates[:, np.newaxis]

for i in range(1, n_steps):
    dW = rng.generate_random_numbers(rng_method, (n_maturities, n_simulations), dt=dt)
    for j in range(n_maturities):
        volatility = calculate_volatility(i * dt, j, pca.components_, pca.explained_variance_)
        diffusion = volatility * dW[j, :]
        simulated_forward_rates[i, j, :] = simulated_forward_rates[i - 1, j, :] * np.exp(diffusion)
```

### RNG Schemes and Their Roles

The utility `random_number_generator.py` exposes:

- `normal`: baseline Gaussian MC increments.
- `uniform`: transformed with `norm.ppf`.
- `sobol`: low-discrepancy sequence (current implementation returns scaled sequence directly).
- `halton`: low-discrepancy + inverse-normal transform.
- `lhs`: Latin Hypercube + inverse-normal transform.

Dispatcher:

```python
def generate_random_numbers(method, shape, dt=1/252):
    ...
```

This gives a controlled testbed for sampling-driven variance and convergence behavior across identical dynamics.

## 7. Derivative Pricing Layer (Cap/Floor)

Given simulated forward rates at expiry index \(T^*\), the notebook prices cap/floor style payoffs:

Caplet-style payoff:

$$
\Pi^{\text{cap}} = \max(F_{T^*}-K,0).
$$

Floorlet-style payoff:

$$
\Pi^{\text{floor}} = \max(K-F_{T^*},0).
$$

Discounted estimator:

$$
V_0 = \mathbb{E}\left[D_{T^*}\Pi\right].
$$

Notebook implementations:

```python
def price_cap(simulated_forward_rates, strike, expiry_index, sofr_discount_factors):
    forward_rates = simulated_forward_rates[expiry_index, :, :]
    payoffs = np.maximum(forward_rates - strike, 0).mean(axis=0)
    discount_factor = sofr_discount_factors.iloc[expiry_index]
    return (payoffs * discount_factor).mean()
```

```python
def price_floor(simulated_forward_rates, strike, expiry_index, sofr_discount_factors):
    forward_rates = simulated_forward_rates[expiry_index, :, :]
    payoffs = np.maximum(strike - forward_rates, 0)
    discount_factor = sofr_discount_factors.iloc[expiry_index]
    return (payoffs.mean(axis=0) * discount_factor).mean()
```

The notebook evaluates convergence over simulation grids such as:

- `100`, `500`, `1000`, `2000` paths.

This stage is where curve calibration quality and sampling design become economically visible as price stability.

## 8. Variance Reduction Layer

`NAIR.ipynb` and `utility/variance_reduction.py` include multiple variance-reduction experiments.

### 8.1 Standard Monte Carlo

Baseline estimator:

$$
\hat{V}_{\text{MC}} = \frac{1}{N}\sum_{i=1}^{N} D_i \Pi_i.
$$

### 8.2 Antithetic Variates

Construct paired increments:

$$
\Delta W^{(a)} = -\Delta W.
$$

Average paired payoff:

$$
\hat{V}_{\text{anti}} = \frac{1}{N}\sum_{i=1}^{N}
\frac{D_i\Pi_i + D_i\Pi_i^{(a)}}{2}.
$$

### 8.3 Control Variates

Generic control-variate form:

$$
\hat{V}_{\text{cv}} = \hat{V} + c\left(Z-\mathbb{E}[Z]\right),
\quad
c^* = -\frac{\operatorname{Cov}(\hat{V},Z)}{\operatorname{Var}(Z)}.
$$

Notebook versions use forward-rate summary controls and covariance-based adjustment in later sections.

### Code Anchors

Notebook functions:

```python
def standard_mc(...): ...
def antithetic_variates(...): ...
def control_variates(...): ...
```

Utility dispatcher:

```python
def price_with_variance_reduction(method, simulated_forward_rates, strike, discount_factors, analytical_price=None, **kwargs):
    ...
```

Conceptual role:

- reduce estimator variance,
- improve convergence speed,
- separate model error from sampling error during pricing validation.

## 9. Utility Modules and Conceptual Fit

### `utility/random_number_generator.py`

Purpose:

- Abstracts sampling backend from simulation loop so RNG effect can be evaluated independently from model dynamics.

### `utility/variance_reduction.py`

Purpose:

- Encodes reusable pricing estimators under alternative variance controls.

### `utility/nelson_siegel.py`

Provides Nelson-Siegel curve object:

$$
y(T) = \beta_0 + \beta_1\frac{1-e^{-T/\tau}}{T/\tau}
      + \beta_2\left(\frac{1-e^{-T/\tau}}{T/\tau}-e^{-T/\tau}\right).
$$

And instantaneous forward form:

$$
f(T)=\beta_0 + \beta_1 e^{-T/\tau} + \beta_2 e^{-T/\tau}\frac{T}{\tau}.
$$

Current conceptual status:

- strong curve-modeling utility,
- not yet tightly integrated into the main `NAIR.ipynb` simulation chain.

### `utility/sensitivity_dashboard.py`

Purpose:

- Intended interactive sensitivity analysis for cap/floor pricing.

Current status:

- app structure exists, but callback/function argument mismatch prevents direct run without wiring `forward_curve` and `sofr_curve` inputs.

## 10. Comparative Findings in the Notebook

The notebook’s empirical narrative indicates:

1. First PCA factors capture most curve-variance structure.
2. Simulated forward-rate behavior is sensitive to RNG design.
3. Cap/floor price paths converge with increasing path count, but estimator dispersion depends on method.
4. Variance-reduction variants can materially alter convergence behavior relative to standard Monte Carlo.

Technical interpretation:

- Feature extraction (PCA) and sampling strategy (RNG + variance reduction) are both first-order drivers of pricing reliability in this setup.

## 11. Interview Narrative (Concise)

Context:

- Needed a practical, interpretable framework for pricing interest-rate derivatives from historical U.S. rate data while controlling simulation uncertainty.

Goal:

- Build an end-to-end pipeline from raw term-structure data to derivative prices, then evaluate how factor calibration and sampling design affect forecasted rates and prices.

Execution:

1. Constructed SOFR discount factors and Treasury term-structure panel.
2. Converted spot yields to forward segments and extracted latent curve factors with PCA.
3. Calibrated HJM-style volatility from PCA loadings and simulated forward-rate paths under multiple RNG schemes.
4. Priced cap/floor payoffs and benchmarked convergence across path counts.
5. Implemented standard, antithetic, and control-variate estimators to compare variance behavior.

Outcome:

- Established a reproducible research framework linking curve factors to pricing outputs.
- Showed that sampling method and variance reduction can materially affect price stability, even under the same structural model.
- Identified clear next integration points: consistent control-variate baseline, tighter Nelson-Siegel coupling, and dashboard wiring for interactive analysis.
