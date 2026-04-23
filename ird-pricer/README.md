# IRD Pricer: Implementation Document (Living)

## 1) Project Purpose

`ird-pricer` is an interest-rate derivatives prototyping project focused on:

- Building Treasury/SOFR term-structure inputs
- Constructing forward rates from spot Treasury yields
- Calibrating HJM volatility factors using PCA on historical yield-curve movements
- Simulating forward-rate paths via Monte Carlo with multiple random number generators
- Pricing cap/floor-style payoffs and evaluating variance-reduction techniques

This document captures the **current implementation state** and should be updated as the code/notebook evolves.

## 2) Current Project Layout

```text
ird-pricer/
  NAIR.ipynb
  data/
    DGS1.csv
    DGS2.csv
    DGS5.csv
    DGS10.csv
    DGS30.csv
    FRED_SOFR.csv
    NYFED_SOFR.xlsx
  utility/
    nelson_siegel.py
    random_number_generator.py
    variance_reduction.py
    sensitivity_dashboard.py
  .venv/
```

## 3) Runtime and Environment

- Notebook kernel metadata: Python `3.11.5` (`.venv (3.11.5)`).
- Core packages used in code: `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `scikit-learn`, `scipy`, `dash`.
- Main execution surface today: `NAIR.ipynb`.
- Supporting modules are imported from `utility/` for RNG and variance-reduction experiments.

## 4) Data Assets and Coverage

### Yield curve data (FRED DGS series)

- `DGS1.csv`: `1962-01-02` to `2024-10-31` (`16393` rows)
- `DGS2.csv`: `1976-06-01` to `2024-10-31` (`12633` rows)
- `DGS5.csv`: `1962-01-02` to `2024-10-31` (`16393` rows)
- `DGS10.csv`: `1962-01-02` to `2024-10-31` (`16393` rows)
- `DGS30.csv`: `1977-02-15` to `2024-10-31` (`12448` rows)

### Overnight rate data

- `FRED_SOFR.csv`: `2018-04-03` to `2024-11-01` (`1719` rows)
- `NYFED_SOFR.xlsx`: present but not currently the primary source in notebook flow.

### Data preprocessing pattern in notebook

- Parse `DATE` as index
- Convert rate columns to numeric with coercion
- Drop invalid/NaN rows
- Interpolate Treasury series on time index where needed
- Build SOFR discount factor using cumulative daily discounting with `dt = 1/252`

## 5) End-to-End Implementation Flow (NAIR.ipynb)

## Part I: Setup and Data Loading

- Imports numerical, plotting, PCA, and RNG tooling.
- Loads SOFR and Treasury maturities (`1Y`, `2Y`, `5Y`, `10Y`, `30Y`).
- Performs initial cleaning and creates a daily discount-factor series from SOFR.

## Part II: Exploratory Data Analysis

- Visualizes SOFR and moving averages.
- Visualizes term yields and spread diagnostics (`10Y-2Y`, `5Y-1Y`, etc.).
- Includes contextual annotations (for example 2020 policy regime period).

## Part III: Forward-Curve Construction

- Defines `calculate_forward_rate(yield_curve, t1, t2)` using compounding relation:
  - Convert spot yields to implied interval forward rates
  - Build historical forward curve columns:
    - `1Y-2Y`
    - `2Y-5Y`
    - `5Y-10Y`
    - `10Y-30Y`

## Part IV: PCA-Based Factor/Volatility Modeling

- Standardizes selected yield tenors.
- Fits PCA (`n_components=5`).
- Interprets dominant PCs as level/slope/curvature factors.
- Defines `calculate_volatility(...)` using PCA eigenvectors/eigenvalues and factor truncation (`n_factors=3` by default), then scales magnitude.

## Part V: HJM-Style Stochastic Simulation

- Simulates forward-rate trajectories over:
  - `n_steps = 252 * 10` (10 years of daily steps)
  - multiple maturities (`1Y`, `2Y`, `5Y`, `10Y`, `30Y`)
  - configurable simulation count
- Uses multiplicative evolution form:
  - next rate = prior rate * `exp(drift*dt + diffusion)`
- Diffusion volatility at each tenor/time uses PCA-driven function.
- Includes comparison of RNG methods:
  - `normal`, `uniform` (inverse-CDF transformed), `sobol`, `halton`, `lhs`

## Part VI: Pricing IR Derivatives (Caps/Floors)

- Defines cap/floor payoff calculators at expiry index:
  - Cap payoff: `max(F - K, 0)`
  - Floor payoff: `max(K - F, 0)`
- Uses SOFR-derived discount factor at expiry.
- Runs convergence experiments for simulation sizes:
  - `100`, `500`, `1000`, `2000`
- Stores and plots convergence trajectories and summary tables.

## Part VII: Variance Reduction Experiments

- Notebook defines/compares:
  - Standard Monte Carlo
  - Antithetic-style pricing variant
  - Control-variate-style adjustment
- Tracks convergence by method and simulation count.

Note: The notebook contains repeated/iterative sections for pricing functions and convergence analysis, consistent with research-style development.

## 6) Utility Modules: What Is Implemented

## `utility/random_number_generator.py`

- Provides RNG backends:
  - Pseudorandom normal
  - Uniform transformed via `norm.ppf`
  - Sobol
  - Halton
  - Latin Hypercube
- Dispatcher: `generate_random_numbers(method, shape, dt)`

Implementation note:
- `gen_sobol_rn` currently returns Sobol samples scaled by `sqrt(dt)` without inverse-normal transform, so distribution differs from Gaussian-driven methods.

## `utility/variance_reduction.py`

- `standard_monte_carlo(...)`
- `monte_carlo_with_antithetic(...)`
- `monte_carlo_with_control_variates(...)`
- `price_with_variance_reduction(method, ...)` dispatcher

Implementation note:
- Control-variate implementation currently uses `analytical_price - mean(analytical_price)`; if `analytical_price` is scalar, this correction is zero.

## `utility/nelson_siegel.py`

- `NelsonSiegelCurve` dataclass:
  - Parameters: `beta0`, `beta1`, `beta2`, `tau`
  - Methods: `factors`, `factor_matrix`, `zero`, `forward`
- Usable as a term-structure helper for rate curve fitting/representation.

Current state:
- Not visibly integrated into `NAIR.ipynb` main flow yet.

## `utility/sensitivity_dashboard.py`

- Dash app intended for interactive cap/floor sensitivity sliders.

Current state:
- Callback calls `calculate_prices(volatility, cap_rate, floor_rate, discount_rate)` but function signature expects additional `forward_curve` and `sofr_curve` inputs, so file is not directly runnable as-is.

## 7) Known Gaps and Technical Debt (Current Snapshot)

1. **Notebook-first architecture**
- Core logic is concentrated in `NAIR.ipynb`; limited modular separation for reuse/testing.

2. **Interactive input dependency**
- Notebook includes `input("Enter your choice...")`, which interrupts non-interactive runs.

3. **Duplicated pricing blocks**
- Multiple redefinitions of pricing functions and convergence loops increase maintenance overhead and risk divergence.

4. **Dashboard integration gap**
- `sensitivity_dashboard.py` has argument mismatch in callback path.

5. **Methodological consistency**
- Some variance-reduction/RNG implementations are experimental and not yet standardized against a single reference pricer.

## 8) How To Run (Current)

From project root:

```bash
cd /Users/atheeshkrishnan/AK/DEV/econometric-forecasting/ird-pricer
source .venv/bin/activate
jupyter notebook
```

Then open `NAIR.ipynb` and run sections in order.

## 9) Suggested Living-Doc Update Protocol

After each enhancement, update this file in-place with:

- What changed (feature/model/data)
- Why it changed (motivation or issue)
- Validation evidence (plots, metrics, backtest deltas, runtime impact)
- New known limitations
- Next prioritized tasks

This keeps `ird-pricer/README.md` as the single source of current implementation truth.
