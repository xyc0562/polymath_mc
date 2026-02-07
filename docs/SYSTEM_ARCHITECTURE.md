# System Architecture: Musk Tweet Count Trading System

> Comprehensive documentation of the forecasting pipeline, production trading system, and backtest infrastructure.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Forecasting Pipeline](#2-forecasting-pipeline)
   - [2.1 Config Layer](#21-config-layer)
   - [2.2 Data Layer](#22-data-layer)
   - [2.3 Intraday Models](#23-intraday-models)
   - [2.4 Interday Models](#24-interday-models)
   - [2.5 GAS Model](#25-gas-model)
   - [2.6 PIG Model](#26-pig-model)
   - [2.7 Monte Carlo Simulation](#27-monte-carlo-simulation)
   - [2.8 Projection Models](#28-projection-models)
   - [2.9 Ensemble Forecaster](#29-ensemble-forecaster)
   - [2.10 Forecaster Orchestrator](#210-forecaster-orchestrator)
3. [Production Trading System](#3-production-trading-system)
   - [3.1 Kelly Config](#31-kelly-config)
   - [3.2 Kelly Math](#32-kelly-math)
   - [3.3 Candidate Generation](#33-candidate-generation)
   - [3.4 Portfolio Management](#34-portfolio-management)
   - [3.5 Order Execution](#35-order-execution)
   - [3.6 Integration (KellyTradingBot)](#36-integration-kellytradingbot)
   - [3.7 Multi-Event Manager](#37-multi-event-manager)
   - [3.8 WebSocket Clients](#38-websocket-clients)
4. [Backtest Infrastructure](#4-backtest-infrastructure)
   - [4.1 BacktestRunner (Unified)](#41-backtestrunner-unified)
   - [4.2 run_backtest.py CLI](#42-run_backtestpy-cli)
   - [4.3 Data Provider](#43-data-provider)
   - [4.4 Trade Analysis Tool](#44-trade-analysis-tool)
5. [Key Mathematical Formulas](#5-key-mathematical-formulas)
6. [Configuration Reference](#6-configuration-reference)
7. [Data Flow Diagrams](#7-data-flow-diagrams)

---

## 1. System Overview

```
                              ┌──────────────────────────────────────────┐
                              │            XTracker API                  │
                              │  (xtracker.polymarket.com/api)           │
                              └────────────────┬─────────────────────────┘
                                               │
                                        fetch posts
                                               │
                              ┌────────────────▼─────────────────────────┐
                              │           EventStore                     │
                              │  (thread-safe, contract-day grouped)     │
                              └──┬──────────────┬────────────────────┬───┘
                                 │              │                    │
                          ┌──────▼──────┐ ┌─────▼──────┐  ┌─────────▼────────┐
                          │  Intraday   │ │  Interday  │  │  Monte Carlo     │
                          │  Nowcast    │ │  Regime    │  │  Simulation      │
                          │(Ridge/Bucket)│ │(EWMA/GAS) │  │  (10K samples)   │
                          └──────┬──────┘ └─────┬──────┘  └─────────┬────────┘
                                 │              │                    │
                                 └──────────────┼────────────────────┘
                                                │
                                     ┌──────────▼──────────┐
                                     │  Projection Model   │
                                     │  (Asymmetric/Normal │
                                     │   /SkewNorm/Gamma)  │
                                     └──────────┬──────────┘
                                                │
                                        bin probabilities
                                                │
                    ┌───────────────────────────▼───────────────────────────┐
                    │              Kelly Optimizer                          │
                    │  (reservation prices, edge model, candidate gen)     │
                    └───────────────────────────┬───────────────────────────┘
                                                │
                                         trade orders
                                                │
                    ┌───────────────────────────▼───────────────────────────┐
                    │           Polymarket CLOB (FAK Orders)                │
                    │  (order placement, WebSocket fills, API sync)         │
                    └──────────────────────────────────────────────────────┘
```

### Component Map

| Layer | Component | Source File |
|-------|-----------|-------------|
| **Data** | XTrackerClient | `forecaster/data.py` |
| **Data** | EventStore | `forecaster/data.py` |
| **Data** | ContractDayUtils | `forecaster/data.py` |
| **Intraday** | IntradayProgressCurve (F(τ)) | `forecaster/intraday.py` |
| **Intraday** | IntradayNowcast (Ridge) | `forecaster/intraday.py` |
| **Intraday** | BucketIntradayForecaster | `forecaster/intraday.py` |
| **Interday** | RegimeModel (EWMA) | `forecaster/interday.py` |
| **Interday** | GASRegimeModel (NB-GAS) | `forecaster/gas.py` |
| **Interday** | PIGGASRegimeModel | `forecaster/pig.py` |
| **Interday** | DispersionEstimator | `forecaster/interday.py` |
| **Interday** | WeekendEffect | `forecaster/interday.py` |
| **Simulation** | MonteCarloForecaster | `forecaster/monte_carlo.py` |
| **Simulation** | EnsembleMonteCarloForecaster | `forecaster/ensemble.py` |
| **Projection** | Asymmetric/Normal/SkewNorm/Gamma | `forecaster/projection.py` |
| **Orchestration** | TweetCountForecaster | `forecaster/forecaster.py` |
| **Trading** | KellyConfig | `kelly/config.py` |
| **Trading** | Kelly math functions | `kelly/kelly_math.py` |
| **Trading** | TradeCandidate generation | `kelly/candidates.py` |
| **Trading** | Portfolio, BinPosition | `kelly/portfolio.py` |
| **Trading** | OrderExecutor, KellyExecutor | `kelly/executor.py` |
| **Trading** | KellyTradingBot | `kelly/integration.py` |
| **Trading** | MultiEventManager | `forecaster/multi_event_manager.py` |
| **Trading** | CapitalPool | `kelly/capital_pool.py` |
| **Trading** | OrderbookManager | `kelly/orderbook.py` |
| **Trading** | WebSocket orderbook | `kelly/websocket_client.py` |
| **Trading** | UserStreamClient | `kelly/user_stream.py` |
| **Backtest** | UnifiedBacktestRunner | `backtest/runner.py` |
| **Backtest** | run_backtest CLI | `backtest/run_backtest.py` |
| **Backtest** | HistoricalDataProvider | `backtest/data_provider.py` |
| **Backtest** | Trade analysis (FIFO) | `backtest/analyze_trades.py` |

All source files are under `src/algo/musk_tweet_count/`.

---

## 2. Forecasting Pipeline

### 2.1 Config Layer

**File**: `forecaster/config.py` (357 lines)

All configuration is expressed as Python `@dataclass` classes with sensible defaults.

#### IntradayCurveConfig
Controls the intraday progress curve F(τ).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bin_size_minutes` | `5` | Discretization granularity for F(τ) |
| `half_life_days` | `21.0` | Exponential decay weight for historical curves |
| `exclude_zero_days` | `True` | Skip days with zero tweets |
| `min_tweets_per_day` | `5` | Minimum for inclusion in fitting |

#### BurstFeaturesConfig
Controls burst feature extraction for the Ridge nowcast model.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_15m` | `15` | Short-term activity window (min) |
| `window_60m` | `60` | Medium-term activity window (min) |
| `window_180m` | `180` | Long-term activity window (min) |
| `session_gap_threshold` | `10` | Gap threshold for "in session" flag (min) |
| `burst_window` | `10` | Max burst detection window (min) |

#### NowcastConfig
Controls the Ridge regression nowcast model.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ridge_alpha` | `1.0` | Ridge regularization parameter |
| `training_window_days` | `45` | Training window |
| `weight_half_life_days` | `14.0` | Exponential decay for sample weighting |
| `implied_rate_threshold` | `0.05` | Minimum F(τ) for rate computation |
| `min_training_days` | `10` | Minimum samples before using residual std |

#### BucketNowcastConfig
Controls the bucket-based intraday forecaster.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_buckets` | `8` | 8 buckets = 180-min windows (3 hours each) |
| `training_window_days` | `45` | Training window |
| `weight_half_life_days` | `14.0` | Exponential decay for sample weighting |
| `regime_min` | `0.85` | Clamp multiplier lower bound |
| `regime_max` | `1.15` | Clamp multiplier upper bound |
| `min_expected_for_regime` | `1.0` | Avoid division by near-zero |
| `min_dispersion_k` | `0.5` | Floor for Negative Binomial k |

#### RegimeConfig
Controls the EWMA interday regime model.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ewma_alpha` | `0.35` | Smoothing parameter (~3 day memory) |
| `initialization_window_days` | `21` | Initial training window |
| `mean_reversion_rate` | `0.05` | Per-day reversion rate |
| `max_log_intensity` | `5.70` | Cap at ln(300) tweets/day |
| `initialization_half_life_days` | `7.0` | Recency weighting for initialization |

#### DispersionConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `estimation_window_days` | `45` | Rolling window |
| `min_k` | `0.5` | Floor (higher = less dispersed) |
| `underdispersion_buffer` | `1.01` | Buffer for underdispersion check |
| `half_life_days` | `14.0` | Recency weighting |

#### WeekendConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weekend_days` | `(5, 6)` | Saturday/Sunday |
| `estimation_window_days` | `45` | Rolling window |
| `half_life_days` | `14.0` | Recency weighting |

#### MonteCarloConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_simulations` | `10000` | Number of MC samples |
| `random_seed` | `None` | Reproducibility |
| `regime_adj_f_min` | `0.05` | Min F(τ) to avoid division |
| `regime_adj_sigma_prior` | `0.35` | Prior std on log intensity |
| `regime_adj_sigma0` | `1.00` | Observation noise |
| `regime_adj_min` | `0.9` | Adjustment clamp lower |
| `regime_adj_max` | `1.1` | Adjustment clamp upper |
| `regime_adj_tau_gate` | `360` | Min minutes before applying (6 hours) |
| `regime_adj_f_gate` | `0.20` | Min F(τ) before applying (20%) |
| `regime_adj_decay` | `0.5` | Decay factor across future days |
| `dispersion_inflation_factor` | `2.5` | k' = k / factor; grid-search optimized |
| `today_std_inflation_factor` | `2.5` | today_std' = today_std * sqrt(factor) |
| `max_daily_forecast` | `200` | Per-day hard cap |
| `max_horizon_caps` | `(0,0,350,500,650,800,950,1100)` | Multi-day sum p99 caps |

#### GASConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `omega_init` | `0.1` | Initial intercept |
| `alpha_init` | `0.05` | Initial score impact |
| `beta_init` | `0.95` | Initial persistence |
| `beta_min` / `beta_max` | `0.5` / `0.999` | Persistence bounds |
| `alpha_min` / `alpha_max` | `0.001` / `0.5` | Score impact bounds |
| `max_log_intensity` | `5.70` | Safety cap |
| `cpd_threshold` | `0.4` | 40% deviation triggers boost |
| `cpd_alpha_multiplier` | `3.0` | Multiply α during change point |
| `cpd_alpha_cap` | `0.3` | Cap boosted α |

#### EnsembleConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_slow` | `0.2` | ~5 day memory |
| `alpha_fast` | `0.5` | ~2 day memory |
| `weight_slow` | `0.5` | Equal weighting |
| `weight_fast` | `0.5` | Equal weighting |

#### ForecasterConfig (Main)
Top-level config aggregating all sub-configs.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timezone` | `"America/New_York"` | Eastern time |
| `contract_boundary_hour` | `12` | Noon ET boundary |
| `intraday_mode` | `"ridge"` | `"ridge"` or `"bucket"` |
| `bins` | 21 bins from (0,74) to (550,10000) | Polymarket market bins |

### 2.2 Data Layer

**File**: `forecaster/data.py` (~800 lines)

#### TweetEvent
```python
@dataclass
class TweetEvent:
    timestamp: datetime   # Timezone-aware
    event_type: str       # "tweet" or "retweet"
    event_id: Optional[str] = None  # For deduplication
```

#### ContractDayUtils
Maps timestamps to "contract days" running noon-to-noon ET.

**Key methods**:
- `get_contract_date(timestamp) -> date`: Maps timestamp to contract-date. Formula: `adjusted = ts_local - timedelta(hours=boundary_hour); return adjusted.date()`
- `get_tau(timestamp, contract_date) -> int`: Minutes since contract-day start (noon). Range: τ ∈ [0, 1440)
- `get_contract_day_bounds(contract_date) -> (start_dt, end_dt)`: Returns noon-to-noon bounds
- `is_weekend(contract_date) -> bool`: `weekday() in (5, 6)`

#### XTrackerClient
Fetches official post data from XTracker API.

- **Base URL**: `https://xtracker.polymarket.com/api`
- **Data start**: 2025-11-01
- **Key endpoint**: `GET /api/users/{handle}/posts` with `startDate`/`endDate` params
- **Retweet detection**: `content.startswith("RT @")`

#### EventStore
Thread-safe storage with contract-day aggregation. Uses `threading.RLock`.

**Key methods**:
- `add_event(event)`: Deduplicates by `event_id`, invalidates count cache
- `get_contract_day_count(date) -> int`: Fast cached count
- `get_contract_day_counts(n_days) -> Dict[date, int]`: Last N completed days
- `refresh_from_api(n_days)`: Fetch and update with regression prevention (rejects if new total < old * 0.9)
- `cleanup_old_data(keep_days=60)`: Prevents memory growth

### 2.3 Intraday Models

**File**: `forecaster/intraday.py` (~1055 lines)

Two intraday forecaster modes: Ridge regression and Bucket-based.

#### IntradayProgressCurve: F(τ)

Expected fraction of day's tweets completed by minute τ. Separate curves for weekday vs weekend.

**Fitting**: For each historical day with ≥ `min_tweets_per_day` tweets:
1. Compute cumulative fraction curve: `count(tau < bin_end) / total`
2. Weight by recency: `w = exp(-days_ago / half_life * ln(2))`
3. Weighted average across days

**Lookup**: `get_expected_progress(tau, is_weekend)` returns F(τ) via bin index lookup.

#### Ridge Regression Nowcast (IntradayNowcast)

**11-feature vector**:
1. `cum_so_far` - Tweets observed so far
2. `tau` - Minutes since noon
3. `F_tau` - Progress curve value
4. `implied_rate` - `cum_so_far / F_tau` (if F_tau > 0.05, else `historical_mean`)
5. `count_15m` - Recent 15-minute count
6. `count_60m` - Recent 60-minute count
7. `count_180m` - Recent 180-minute count
8. `last_gap_min` - Minutes since last tweet
9. `in_session` - Boolean indicator (0 or 1)
10. `max_burst_60m` - Max burst in sliding window
11. `is_weekend` - Boolean indicator

**Training**: For each historical day, samples at τ = {360, 720, 1080} minutes (25%, 50%, 75% of day). Weights samples by exponential decay. Fits `Ridge(alpha=1.0)` with `sample_weight`.

**Prediction**: `prediction = max(ridge.predict(features), cum_so_far)`

**Uncertainty scaling**: `std = residual_std * sqrt(max(1 - F_tau, 0.01))`
- At start of day (F_tau ≈ 0): full uncertainty
- At end of day (F_tau ≈ 1): ~1% of base uncertainty

#### Bucket-Based Forecaster (BucketIntradayForecaster)

Divides the day into N buckets (default 8 = 180-minute windows).

**Fitting**: Counts tweets per bucket per historical day. Fits NegBin distribution per bucket:
- `mean = np.mean(bucket_counts)`
- `k = mean² / (var - mean)` (method of moments), floored at `min_dispersion_k`

**Prediction**:
1. Determine current bucket and partial fraction
2. Compute `expected_so_far = sum(completed bucket means) + partial_current * mean`
3. Regime multiplier: `regime = observed / expected_so_far`, clamped to [0.85, 1.15]
4. Monte Carlo (n=1000): sample remaining buckets from NegBin scaled by regime
5. Return `(mean, std)` of samples

**Expected progress**: `F(τ) = expected_so_far / total_daily_mean`

### 2.4 Interday Models

**File**: `forecaster/interday.py` (~870 lines)

#### RegimeModel (EWMA)

Tracks latent posting intensity in log-space.

**Initialization**: Weighted mean of recent `log(count + 1)` values with exponential decay (half-life 7 days). Long-term mean uses 3x longer half-life.

**Update**: `λ̂_d = α * log(C_d + 1) + (1 - α) * λ̂_{d-1}`, α = 0.35

**Long-term mean update**: `μ̄ = 0.01 * log(C + 1) + 0.99 * μ̄` (very slow)

**h-step forecast**: `λ_h = μ̄ + (1 - ρ)^h * (λ̂ - μ̄)` where ρ = 0.05

#### DispersionEstimator

Estimates NegBin k using weighted method of moments.

`k = μ² / (Var - μ)` where μ and Var use exponential-decay weights. Bias-corrected with effective sample size. Floored at 0.5. If `Var ≤ μ * 1.01` (underdispersed), uses k = 100.

#### WeekendEffect

Multiplicative adjustment: `effect = weekend_weighted_mean / weekday_weighted_mean`. Applied as multiplier to forecasted intensity on weekends.

#### InterdayForecaster (Combined)

`forecast_day(horizon, base_date) -> (mean, k)`:
1. `log_intensity = regime.forecast_intensity(horizon)`
2. `mean = exp(log_intensity) * weekend_effect`
3. Returns `(mean, dispersion.k)`

#### GASInterdayForecaster

Same interface, uses `GASRegimeModel` internally. Fits via MLE, updates with adaptive α.

#### PIGInterdayForecaster

Same interface, uses `PIGGASRegimeModel`. Returns `(mean, sigma)` instead of `(mean, k)` where `Var = μ + σ²μ²`.

### 2.5 GAS Model

**File**: `forecaster/gas.py` (273 lines)

NB-GAS (Negative Binomial Generalized Autoregressive Score) regime model.

**Model specification**:
```
f_t = log(μ_t)                              # log-intensity state
f_{t+1} = ω + β·f_t + α_t·s_t              # GAS recursion with adaptive α
s_t = (y_t - μ_t) / μ_t                     # scaled Pearson score
Y_t ~ NegBin(μ_t = exp(f_t), k)             # observation density
```

**Fitting**: MLE via L-BFGS-B minimizing negative log-likelihood. Bounds: ω ∈ (-2, 2), α ∈ (0.001, 0.5), β ∈ (0.5, 0.999). Log-likelihood: `LL = Σ nbinom.logpmf(y_t, n=k, p=k/(k+μ_t))`.

**Filter initialization**: `f[0] = log(mean(first 7 observations))`

**Change point detection**: Compares 2-day average to 7-day average. If `|avg_2d - avg_7d| / avg_7d > 0.4`, boosts α to `min(α * 3.0, 0.3)`.

**h-step forecast**: `f_h = f̄ * (1 - β^h) + β^h * f_t` where `f̄ = ω / (1 - β)` is the unconditional mean.

### 2.6 PIG Model

**File**: `forecaster/pig.py` (320 lines)

Poisson-Inverse Gaussian GAS model for heavier tails.

**Distribution**: `Y | Z ~ Poisson(μ·Z)`, `Z ~ InverseGaussian(1, 1/σ²)`
- Mean: μ
- Variance: μ + σ²μ² (vs NegBin: μ + μ²/k)

**Log-PMF**: Uses modified Bessel function K:
```
P(Y=y) = [μ^y √(φ/(2π)) / y!] * exp(φ) * 2 * (b/a)^((y-0.5)/2) * K_{y-0.5}(2√(ab))
```
where `φ = 1/σ²`, `a = μ + φ/2`, `b = φ/2`.

**Sampling** (composition method):
1. Sample Z from Inverse Gaussian via Michael et al. (1976) algorithm
2. Sample Y from Poisson(μ·Z)

### 2.7 Monte Carlo Simulation

**File**: `forecaster/monte_carlo.py` (~790 lines)

Generates the full distribution of the N-day sum by combining intraday and interday models.

#### ForecastResult
```python
@dataclass
class ForecastResult:
    mean, median, std: float          # Point estimates
    p5, p25, p75, p95: float          # Percentiles
    bin_probabilities: List[BinProbability]
    today_estimate: float             # Nowcast component
    future_days_estimate: float       # Adjusted interday component
    today_interday_estimate: float    # Baseline comparison
    future_days_pure: float           # Pure interday (no adjustment)
    regime_adjustment: float          # Kalman-adjusted multiplier
    n_simulations: int
    simulation_time_ms: float
    samples: Optional[np.ndarray]     # Raw MC samples
```

#### Sampling Today (Log-Normal)
```
σ² = log(1 + (std/mean)²)
μ = log(mean) - σ²/2
sample = max(exp(Normal(μ, σ)), cum_so_far)
```

#### Sampling Future Days (Negative Binomial with regime adjustment)
For each day h:
1. `(mean_h, k) = interday.forecast_day(h, base_date)`
2. `decayed_adj = 1 + (regime_adj - 1) * decay^(h-1)` (decay = 0.5)
3. `adjusted_mean = mean_h * decayed_adj`
4. `k_inflated = k / dispersion_inflation_factor` (2.5)
5. `p = k_inflated / (k_inflated + adjusted_mean)`
6. `sample = NegBin(k_inflated, p)`, capped at `max_daily_forecast` (200)

Total: `sum = today_sample + Σ(future_samples)`, capped at horizon cap.

#### Bayesian Regime Adjustment (Kalman filter in log-space)

**Gates**: Requires τ ≥ 360 min (6 hours) and F(τ) ≥ 0.20 (20% of day).

**Update**:
```
m_prior = log(λ_prior + 1)                          # Prior from interday regime
y_obs = log(cum_so_far / F_tau + 1)                  # Observation (implied full-day)
σ²_prior = 0.35²                                     # Prior variance
σ²_obs = 1.0² / F_tau                                # Observation variance (scales with progress)
w = σ²_prior / (σ²_prior + σ²_obs)                   # Kalman gain
m_post = (1-w) * m_prior + w * y_obs                  # Posterior
adjustment = exp(m_post) / exp(m_prior)               # Clamped to [0.9, 1.1]
```

For future days: `decayed_adj = 1 + (adj - 1) * 0.5^(h-1)`

#### Scoring Functions
- **Log score**: `log(P(actual_bin))` — higher is better, max 0
- **Brier score**: `Σ(p_i - y_i)²` — lower is better, min 0
- **Calibration stats**: Coverage at 50%, 80%, 90%, 95% confidence levels; MAE, RMSE

### 2.8 Projection Models

**File**: `forecaster/projection.py` (469 lines)

Converts forecast distributions into bin probabilities.

#### AsymmetricProjection (default)
Uses actual MC samples. Counts fraction of samples in each bin. Preserves right-skew of the LogNormal + NegBin sum.

#### NormalProjection
Analytic Normal CDF: `P(lower ≤ X ≤ upper) = Φ((upper+0.5 - μ) / σ) - Φ((lower-0.5 - μ) / σ)`. Supports truncated normal for floor values.

#### SkewNormalProjection
Extends Normal with skewness parameter (default 0.43). Maps to shape α via `α ≈ 1.5 * skewness`. Solves for location ξ and scale ω from mean/std.

#### GammaProjection
Natural for sums of positive random variables. Always right-skewed.
- Shape: `k = mean² / variance`
- Scale: `θ = variance / mean`

#### Factory
`create_projection_model(model_type)` — accepts `"asymmetric"`, `"normal"`, `"skew_normal"`, `"gamma"`.

### 2.9 Ensemble Forecaster

**File**: `forecaster/ensemble.py` (383 lines)

Combines multiple EWMA-based interday models with different adaptation speeds via **sample-level pooling**.

**Architecture**:
- Multiple ensemble members with different α values (slow=0.2, fast=0.5)
- Shared intraday nowcast
- Each member gets its own random seed (base + member_idx * 1000)

**Simulation flow**:
1. Get today's nowcast (shared)
2. Distribute `n_simulations` by weight: `n_per_member = int(n_simulations * w)`
3. Each member runs full simulation with its own regime adjustment
4. Pool all samples: `sums = np.concatenate(all_samples)`
5. Compute statistics on pooled samples

### 2.10 Forecaster Orchestrator

**File**: `forecaster/forecaster.py` (808 lines)

`TweetCountForecaster` — main interface integrating all components.

**Initialization**:
- Creates/uses EventStore (shared for multi-event)
- Selects intraday mode: Ridge (with progress curve + burst extractor) or Bucket
- Creates InterdayForecaster
- Creates MonteCarloForecaster after fit

**`fit(n_days=90, skip_fetch=False, as_of_date=None)`**:
1. Fetch historical data via XTracker API
2. Fit progress curve (Ridge mode only)
3. Fit nowcast on historical events
4. Fit interday (regime, dispersion, weekend)
5. Initialize MonteCarloForecaster

**`forecast_for_event_window(market_start_date, settlement_date, now)`**:
- Accounts for completed days: counts actual tweets from past days
- Forecasts only remaining days
- Supports pre-counting window (pure interday) and in-window (nowcast + interday)
- Shifts samples by `past_count` for proper bin probability computation
- Uses projection model for bin probabilities

---

## 3. Production Trading System

### 3.1 Kelly Config

**File**: `kelly/config.py` (~378 lines)

#### EdgeBufferConfig
Stake-based ROI model with asymmetric friction.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `required_roi` | `0.05` | 5% required ROI on stake |
| `friction_mid` | `0.01` | 1% friction for mid-range (10%-90%) |
| `friction_tail` | `0.02` | 2% friction for tails (≤10% or ≥90%) |
| `tail_threshold` | `0.10` | Threshold for tail zone |
| `min_perceived_prob` | `0.03` | Minimum model probability to trade |
| `min_market_price` | `0.01` | Minimum market price to trade |
| `require_two_sided_liquidity` | `True` | Both bid and ask required |
| `max_spread_ratio` | `2.0` | Max spread as ratio of bid |

**Edge formulas**:
- **Buy YES**: `p_f - p_m ≥ c + r * p_m` → max price `p_m ≤ (p_f - c) / (1 + r)`
- **Buy NO**: `p_m - p_f ≥ c + r * (1 - p_m)` → min price `p_m ≥ (p_f + c + r) / (1 + r)`
- Where c = friction (fees + slippage), r = required ROI on stake

#### RateLimitConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_orders_per_tick` | `10` | Per tick |
| `min_order_delay_seconds` | `1.0` | Dry-run min delay |
| `max_orders_per_minute` | `30` | Hard cap |
| `rate_limit_cooldown_seconds` | `60.0` | Cooldown after limit |
| `block_confirmation_timeout_seconds` | `30.0` | Wait for blockchain confirmation |
| `fak_failure_cooldown_seconds` | `60.0` | Cooldown after FAK failure |

#### AdaptiveDeltaConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_delta_usd` | `50.0` | Base chunk size in USD |
| `max_depth_fraction` | `0.3` | Max fraction of visible depth per trade |
| `min_delta_usd` | `5.0` | Minimum chunk size |

#### CollateralConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `c_event_max` | `500.0` | Max collateral per event (USD) |
| `c_bin_max_ratio` | `0.15` | Max per bin as ratio of event max (= $75) |

#### KellyConfig (Main)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `True` | Enable trading |
| `kappa` | `0.25` | Fractional Kelly (quarter Kelly) |
| `min_utility` | `0.00005` | Minimum utility gain threshold |
| `w_floor` | `1.0` | Minimum terminal wealth floor |
| `max_iters_per_tick` | `100` | Max iterations per tick |
| `t_stop_hours` | `1.0` | Trading cutoff before settlement |
| `renormalize_probabilities` | `True` | Renormalize after dead-bin removal |

#### EventTradingRulesConfig

Loaded from `config/event_trading_rules.yaml`. Three categories:

| Category | Duration | Rules |
|----------|----------|-------|
| **short** | 0-3 days | Must wait for counting to start; min 1h before settlement |
| **weekly** | 4-8 days | Can trade up to 72h before counting; min 1h before settlement |
| **monthly** | 9+ days | Only trade when ≤ 7 days remain; min 1h before settlement |

### 3.2 Kelly Math

**File**: `kelly/kelly_math.py` (~332 lines)

#### Terminal Wealth
For a multi-bin market, if bin j wins:
```
W_j = C + q_j^YES + Σ_{i≠j} q_i^NO
```
Where C = capital, q_j^YES = YES shares in winning bin (pay $1 each), q_i^NO = NO shares in losing bins (pay $1 each).

#### Normalizer
```
S = Σ_j (p_j / W_j)
```
Ensures reservation prices sum to 1. Uses `w_floor` to prevent extreme values.

#### Reservation Prices
```
c*_YES[i] = (p_i / W_i) / S
c*_NO[i] = 1 - c*_YES[i]
```
These represent fair values where Kelly is indifferent to trading.

#### Expected Log Utility
```
E[log(W)] = Σ_j p_j * log(W_j)
```
Kelly objective function to maximize.

#### Fractional Kelly
```
target = current + kappa * (optimal - current)
```
With kappa = 0.25 (quarter Kelly) for conservative production use.

#### Dead Bin Management
Bins where `upper_bound < current_count` are identified as dead. Their probabilities are zeroed and remaining bins renormalized.

### 3.3 Candidate Generation

**File**: `kelly/candidates.py` (~1079 lines)

**Constants**: `MIN_ORDER_SIZE = 15` shares, `MIN_ORDER_VALUE_USD = 1.0`

#### TradeAction Enum
`BUY_YES`, `SELL_YES`, `BUY_NO`, `SELL_NO`

#### TradeCandidate
```python
@dataclass
class TradeCandidate:
    bin_index: int
    action: TradeAction
    size: float         # Shares
    price: float        # VWAP execution price
    utility_gain: float # Expected utility improvement
    reservation_price: float  # Kelly fair price
    edge: float         # ROI on stake
```

#### Adaptive Delta Sizing
```
delta_usd = min(base_delta_usd, available_depth * max_depth_fraction)
delta_usd = max(delta_usd, min_delta_usd)
```

#### Candidate Generation Flow
1. Get Kelly reservation prices for all bins
2. For each live bin:
   - Generate BUY_YES candidate (if no conflicting NO position)
   - Generate SELL_YES candidate (if have YES position)
   - Generate BUY_NO candidate (if no conflicting YES position)
   - Generate SELL_NO candidate (if have NO position)
3. Sort: **sells first**, then buys by utility gain (descending)

**Buy validation chain**: Available depth > 0 → Best ask exists → Adaptive delta ≥ minimum → Capital available → Collateral limits → Min perceived probability → Min market price → Edge requirement → Utility gain ≥ min_utility

**Sell exit logic**: Uses `exit_threshold = min(model_prob, kelly_reservation)`. Allows exit when either edge disappeared or utility-based exit triggered. Blocks utility < 0 (selling below fair value).

### 3.4 Portfolio Management

**File**: `kelly/portfolio.py` (~472 lines)

#### BinPosition
```python
@dataclass
class BinPosition:
    bin_index: int
    yes_token_id: str
    no_token_id: Optional[str] = None
    yes_shares: float = 0.0
    no_shares: float = 0.0
    yes_avg_cost: float = 0.0
    no_avg_cost: float = 0.0
    collateral_used: float = 0.0
```

Methods: `add_yes(shares, price)`, `remove_yes(shares, price) -> realized_pnl`, and NO equivalents.

#### Portfolio
```python
@dataclass
class Portfolio:
    initial_capital: float
    capital: float                    # Current USDC balance
    positions: Dict[int, BinPosition]
    probabilities: List[float]
    num_bins: int
    bin_upper_bounds: List[int]
    dead_bins: List[int]
    external_capital_limit: Optional[float] = None
```

**Key property**: `available_capital = capital` (NOT `capital - collateral_used`; collateral_used is only for per-bin limit tracking).

**Non-mutating**: `simulate_buy_yes(bin, shares, price) -> Portfolio` — creates copy with trade applied.

**Mutating**: `execute_buy_yes(bin, shares, price, token_id)` — updates real portfolio.

### 3.5 Order Execution

**File**: `kelly/executor.py` (~1355 lines)

#### OrderExecutor
Places FAK (Fill and Kill) orders via Polymarket CLOB.

**Precision handling**:
- Price: rounded to 2 decimals (tick_size = $0.01)
- Size: floored to integer
- `maker_amount = size * price` must have ≤ 2 decimals

#### KellyExecutor

**Tick execution flow** (`run_tick`):
1. Check T_stop (stop trading `t_stop_hours` before settlement)
2. Get current orderbooks
3. Loop (max `max_iters_per_tick`):
   a. Check rate limits (per-minute sliding window)
   b. Sync portfolio from API on first iteration (trusts WebSocket after)
   c. Generate candidates
   d. Get best candidate
   e. Check utility threshold (sells execute unconditionally; buys require ≥ min_utility)
   f. Check FAK cooldown for bin
   g. Execute trade
   h. If success: track pending order → register with UserStreamClient → wait for confirmation (30s timeout) → API sync (2s wait)
   i. If failed: check for FAK failure → add to cooldown
   j. Refresh orderbooks for next iteration

**Portfolio update model**: Portfolio is **NOT** updated on order placement. Updates happen via API sync before each Kelly decision. This avoids double-counting from multiple WebSocket callbacks (MATCHED, MINED, CONFIRMED).

### 3.6 Integration (KellyTradingBot)

**File**: `kelly/integration.py` (~784 lines)

#### Setup
1. Store bin metadata
2. Create Portfolio
3. Create OrderbookManager with WebSocket config
4. Create OrderExecutor
5. Setup UserStreamClient (external for multi-event, or create own)
6. Create KellyExecutor with sync callback

#### Portfolio Sync
- Uses event budget (`c_event_max`), NOT wallet USDC balance
- `event_capital = budget - collateral_used`
- Updates from API only if API shows MORE shares than local (never clears local state)
- API has latency (seconds to minutes); WebSocket fills are primary real-time source

#### Tick Flow
```python
async def run_tick(current_count, hours_elapsed, hours_to_settlement, ...):
    1. Update probabilities (identify dead bins, renormalize)
    2. Fetch fresh orderbooks for live bins
    3. Call kelly_executor.run_tick()
```

### 3.7 Multi-Event Manager

**File**: `forecaster/multi_event_manager.py` (~600+ lines)

#### Event Lifecycle
`PENDING` → `ACTIVE` → `COMPLETED`

#### MultiEventConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_per_event` | `500.0` | Max capital per event |
| `tick_interval_seconds` | `300` | Optimization tick frequency (5 min) |
| `training_days` | `45` | History for probability models |
| `use_gas` | `True` | Use GAS forecaster |
| `posts_refresh_interval` | `150` | Refresh tweet data every 2.5 min |
| `max_data_age_seconds` | `300` | Skip trading if data > 5 min old |
| `count_validation_interval` | `900` | Validate counts every 15 min |
| `event_scan_interval` | `3600` | Check for new events every hour |
| `projection_model` | `"asymmetric"` | Projection model type |
| `min_hours_before_settlement` | `24.0` | Global minimum hours to trade |
| `max_event_store_days` | `60` | Keep last 60 days of data |

#### CapitalPool (`kelly/capital_pool.py`)
Thread-safe shared capital across concurrent events. Tracks allocations per event with `EventAllocation` dataclass. Enforces per-event limits.

#### Fill Routing
Global `UserStreamClient` routes fill events to correct bot using `_token_to_event` mapping.

### 3.8 WebSocket Clients

#### OrderbookManager (`kelly/orderbook.py`)
Maintains unified orderbook view per bin. `UnifiedOrderbook` stores YES bids/asks sorted. NO prices derived: `NO bid = 1 - YES ask`, `NO ask = 1 - YES bid`.

#### WebSocket Orderbook Client (`kelly/websocket_client.py`)
Real-time orderbook streaming from Polymarket WebSocket.

#### UserStreamClient (`kelly/user_stream.py`)
Authenticated WebSocket for user-specific events.

**OrderStatus progression**: `PENDING` → `MATCHED` → `MINED` → `CONFIRMED`

```python
@dataclass
class FillEvent:
    order_id: str
    token_id: str
    side: str        # "BUY" or "SELL"
    price: float
    size: float
    status: OrderStatus
    timestamp: datetime
```

---

## 4. Backtest Infrastructure

### 4.1 BacktestRunner (Unified)

**File**: `backtest/runner.py` (unified_runner.py)

Uses **identical Kelly trading logic** as production. Creates a `KellyExecutor` that operates on simulated orderbooks from historical price data.

#### UnifiedBacktestConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_capital` | `1000.0` | Starting capital |
| `spread` | `0.02` | Simulated bid-ask spread |
| `slippage` | `0.005` | Additional execution cost |
| `trading` | `KellyConfig` | Full Kelly configuration |
| `training_days` | `45` | Historical days for model fitting |
| `tick_interval_seconds` | `300` | Time between optimization ticks |
| `exit_hours_before_settlement` | `0.0` | Close positions X hours before settlement |
| `verbose` | `False` | Detailed trade logging |
| `projection_model` | `"asymmetric"` | Projection model type |
| `intraday_mode` | `"ridge"` | Intraday forecaster mode |
| `event_trading_rules` | `None` | Optional trading rules config |

#### BacktestResult
```python
@dataclass
class BacktestResult:
    event_name: str
    start_date, end_date: date
    winner_bin: int
    initial_capital, final_capital: float
    total_pnl, total_return: float
    num_trades, num_winning_trades, num_losing_trades: int
    win_rate: float
    settlement_pnl: float
    trades: List[Trade]
    pnl_by_bin: Dict[int, float]
    settlement_details: List[Dict]
```

#### Tick Loop
1. Create and fit forecaster from historical data
2. Separate posts into training (before counting start) and backtest (incremental)
3. For each sampled timestamp:
   a. Add new posts incrementally
   b. Run forecast at current time
   c. Compute bin probabilities using projection model
   d. Build simulated orderbooks from historical prices
   e. Run Kelly executor tick (same code as production)
   f. Record trades
4. Calculate settlement P&L
5. YES positions in winner bin pay $1; NO positions in loser bins pay $1

### 4.2 run_backtest.py CLI

**File**: `backtest/run_backtest.py` (648 lines)

```
python -m src.algo.musk_tweet_count.backtest.run_backtest [OPTIONS]
```

#### Event Selection
| Argument | Description |
|----------|-------------|
| `--event` | Specific event directory |
| `--start-date` | Filter events ending after this date |
| `--end-date` | Filter events ending before this date |
| `--list` | List available events and exit |
| `--duration` | Filter by counting window duration in days |

#### Trading Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--capital` | `1000` | Initial capital |
| `--spread` | `0.02` | Simulated spread |
| `--slippage` | `0.005` | Simulated slippage |
| `--roi` | `0.10` | Required ROI for trades |
| `--exit-hours` | `0.0` | Close positions X hours before settlement |
| `--stop-loss` | `0.0` | Stop-loss fraction (0 = disabled) |

#### Unified Mode (Kelly)
| Argument | Default | Description |
|----------|---------|-------------|
| `--unified` | — | Use production Kelly logic |
| `--kappa` | `0.25` | Fractional Kelly multiplier |
| `--min-utility` | `0.0001` | Minimum utility gain |
| `--t-stop` | from config | Stop trading hours before settlement |
| `--c-bin-max-ratio` | `0.15` | Max collateral per bin ratio |
| `--base-delta-usd` | `50.0` | Base trade size |
| `--max-orders` | `50` | Max orders per tick |
| `--quick` | — | Quick mode: $50 delta, 5 orders/tick |

#### Model Selection
| Argument | Default | Choices |
|----------|---------|---------|
| `--projection` | `asymmetric` | `asymmetric`, `normal`, `skew_normal`, `gamma` |
| `--intraday-mode` | `ridge` | `ridge`, `bucket` |
| `--event-rules` | None | Path to YAML rules file |

#### Edge Buffer Controls
| Argument | Default | Description |
|----------|---------|-------------|
| `--min-perceived-prob` | from config | Min model probability to trade |
| `--min-market-price` | from config | Min market price to trade |
| `--max-spread-ratio` | `2.0` | Max spread ratio |
| `--no-require-two-sided` | — | Disable two-sided liquidity requirement |

### 4.3 Data Provider

**File**: `backtest/data_provider.py` (541 lines)

#### HistoricalDataProvider
Loads scraped price history from disk. Each event directory contains `event_info.json` and `prices.csv`.

**Data structures**:
- `PricePoint`: `{timestamp, datetime, price}`
- `BinPriceHistory`: Per-bin price series with `get_price_at(ts)` (latest price at or before timestamp)
- `EventPriceData`: All bins for an event, with computed `all_timestamps`, `winner_bin_index`, `counting_start_date`/`counting_end_date` (parsed from short name)

**SimulatedOrderbook**: Creates bid/ask from mid-price ± spread/2:
```python
yes_bid = max(0.001, mid_price - spread/2)
yes_ask = min(0.999, mid_price + spread/2)
no_bid  = max(0.001, (1 - mid_price) - spread/2)
no_ask  = min(0.999, (1 - mid_price) + spread/2)
```

#### CachedPostsProvider
Caches XTracker posts to disk (`data/backtest_cache/posts_elonmusk.json`). Avoids repeated API calls across backtest runs. Extends cache if new date range is requested.

### 4.4 Trade Analysis Tool

**File**: `backtest/analyze_trades.py` (755 lines)

Per-trade P&L analysis using **FIFO matching** to determine whether each buy was closed early or held to settlement.

#### FIFO Matching
For each event's trades (sorted chronologically):
1. BUY trades push onto FIFO queue keyed by `(bin_index, side)`
2. SELL trades match against oldest open buy (FIFO):
   - `pnl = (sell_price - buy_price) * match_size`
   - Resolution: `"closed_early"`
3. Remaining open positions settle:
   - YES in winner bin: `pnl = (1.0 - buy_price) * size` → `"settlement_win"`
   - YES in loser bin: `pnl = -buy_price * size` → `"settlement_loss"`
   - NO in winner bin (loses): `pnl = -buy_price * size` → `"settlement_loss"`
   - NO in loser bin (wins): `pnl = (1.0 - buy_price) * size` → `"settlement_win"`

#### Price Buckets
```python
PRICE_BUCKETS = [
    (0.00, 0.03), (0.03, 0.05), (0.05, 0.10), (0.10, 0.30),
    (0.30, 0.50), (0.50, 0.70), (0.70, 0.90), (0.90, 1.00),
]
```

#### Output Tables
1. **Table 1**: Overall P&L by days-to-settlement
2. **Table 2**: Breakdown by resolution type (held-win, held-loss, closed)
3. **Table 3**: Overall P&L by buy-price bucket
4. **Table 4**: Price bucket breakdown by resolution type
5. **Table 5**: Days × price cross-table (ROI heatmap)
6. **Table 6**: Held-to-settlement ROI by days × price

---

## 5. Key Mathematical Formulas

### Contract-Day Calculation
```
contract_date = floor((timestamp_local - 12h) / 1 day)
τ (minutes) = (timestamp_local - contract_start) / 60
  where contract_start = contract_date at 12:00 ET
```

### Exponential Decay Weighting
```
w_d = exp(-days_ago / half_life * ln(2))
```
Used throughout for recency-weighted estimation.

### Intraday Progress Curve F(τ)
```
F(τ) = Σ(w_d * curve_d) / Σ(w_d)
  where curve_d[bin] = count(tweets with τ < bin_end) / total_tweets_day_d
```

### Ridge Regression Nowcast
```
features = [cum_so_far, τ, F(τ), implied_rate, count_15m, count_60m,
            count_180m, last_gap, in_session, max_burst_60m, is_weekend]
prediction = max(Ridge(α=1.0).predict(features), cum_so_far)
uncertainty = residual_std * sqrt(max(1 - F(τ), 0.01))
```

### Bucket Model Expected Progress
```
F(τ) = (Σ completed_bucket_means + partial_current_mean) / total_daily_mean
regime = clamp(observed / expected_so_far, 0.85, 1.15)
```

### EWMA Regime Model
```
λ̂_d = α * log(C_d + 1) + (1 - α) * λ̂_{d-1}        # α = 0.35
μ̄ = 0.01 * log(C + 1) + 0.99 * μ̄                    # Long-term mean (slow)
λ_h = μ̄ + (1 - ρ)^h * (λ̂ - μ̄)                      # h-step forecast, ρ = 0.05
intensity = exp(λ)
```

### Negative Binomial Dispersion
```
Var = μ + μ²/k
k = μ² / (Var - μ)                                    # Method of moments
Weighted: μ = Σ(w_i * C_i), Var = Σ(w_i * C_i²) - μ²
Bias correction: Var *= n_eff / (n_eff - 1)  where n_eff = 1 / Σ(w²)
```

### GAS Model
```
f_t = log(μ_t)                                        # Log-intensity state
s_t = (y_t - μ_t) / μ_t                               # Scaled Pearson score
f_{t+1} = ω + β·f_t + α_t·s_t                         # GAS recursion
Y_t ~ NegBin(μ_t = exp(f_t), k)

Multi-step: f_h = f̄·(1 - β^h) + β^h·f_t              # Mean reversion
f̄ = ω / (1 - β)                                       # Unconditional mean
```

### PIG Distribution
```
Y | Z ~ Poisson(μ·Z),  Z ~ IG(1, 1/σ²)
Mean: μ
Variance: μ + σ²·μ²

P(Y=y) = [μ^y √(φ/(2π)) / y!] · exp(φ) · 2 · (b/a)^((y-0.5)/2) · K_{y-0.5}(2√(ab))
  where φ = 1/σ², a = μ + φ/2, b = φ/2
```

### Bayesian Regime Adjustment (Kalman)
```
m_prior = log(λ_prior + 1)                            # Prior from interday
y_obs = log(cum_so_far / F(τ) + 1)                    # Implied full-day
σ²_prior = 0.35²
σ²_obs = 1.0² / F(τ)                                  # Scales with progress
w = σ²_prior / (σ²_prior + σ²_obs)                    # Kalman gain
m_post = (1-w)·m_prior + w·y_obs
adjustment = exp(m_post) / exp(m_prior)                # Clamped [0.9, 1.1]
decay across days: adj_h = 1 + (adj - 1) · 0.5^(h-1)
```

### Log-Normal Today Sampling
```
σ² = log(1 + (std/mean)²)
μ = log(mean) - σ²/2
sample = max(exp(N(μ, σ)), cum_so_far)
```

### Kelly Terminal Wealth & Reservation Prices
```
W_j = C + q_j^YES + Σ_{i≠j} q_i^NO                   # Wealth if bin j wins
S = Σ_j (p_j / W_j)                                   # Normalizer
c*_YES[i] = (p_i / W_i) / S                           # Reservation price YES
c*_NO[i] = 1 - c*_YES[i]                              # Reservation price NO
E[log(W)] = Σ_j p_j · log(W_j)                        # Kelly objective
```

### Edge Buffer (Stake-Based ROI)
```
Buy YES if: p_m ≤ (p_f - c) / (1 + r)
Buy NO  if: p_m ≥ (p_f + c + r) / (1 + r)
  where c = friction (mid or tail), r = required_roi
```

### Scoring
```
Log score  = log(P(actual bin))                        # Higher is better, max 0
Brier score = Σ_i (p_i - y_i)²                        # Lower is better, min 0
```

---

## 6. Configuration Reference

### Forecaster Defaults (All)

| Config | Parameter | Default |
|--------|-----------|---------|
| Intraday Curve | bin_size_minutes | 5 |
| Intraday Curve | half_life_days | 21.0 |
| Nowcast | ridge_alpha | 1.0 |
| Nowcast | training_window_days | 45 |
| Nowcast | weight_half_life_days | 14.0 |
| Bucket | n_buckets | 8 |
| Bucket | regime_min/max | 0.85 / 1.15 |
| Regime | ewma_alpha | 0.35 |
| Regime | mean_reversion_rate | 0.05 |
| Regime | max_log_intensity | 5.70 |
| Dispersion | estimation_window_days | 45 |
| Dispersion | min_k | 0.5 |
| Weekend | weekend_days | (5, 6) |
| Monte Carlo | n_simulations | 10000 |
| Monte Carlo | dispersion_inflation | 2.5 |
| Monte Carlo | today_std_inflation | 2.5 |
| Monte Carlo | regime_adj_decay | 0.5 |
| Monte Carlo | max_daily_forecast | 200 |
| GAS | cpd_threshold | 0.4 |
| GAS | cpd_alpha_multiplier | 3.0 |
| Ensemble | alpha_slow/fast | 0.2 / 0.5 |
| Ensemble | weight_slow/fast | 0.5 / 0.5 |

### Kelly Defaults (All)

| Config | Parameter | Default |
|--------|-----------|---------|
| Kelly | kappa | 0.25 |
| Kelly | min_utility | 0.00005 |
| Kelly | w_floor | 1.0 |
| Kelly | t_stop_hours | 1.0 |
| Edge | required_roi | 0.05 |
| Edge | friction_mid | 0.01 |
| Edge | friction_tail | 0.02 |
| Delta | base_delta_usd | 50.0 |
| Delta | max_depth_fraction | 0.3 |
| Collateral | c_event_max | 500.0 |
| Collateral | c_bin_max_ratio | 0.15 |
| Rate | max_orders_per_tick | 10 |
| Rate | max_orders_per_minute | 30 |

### Polymarket Bins (Default)

| Index | Range | Index | Range |
|-------|-------|-------|-------|
| 0 | 0-74 | 11 | 325-349 |
| 1 | 75-99 | 12 | 350-374 |
| 2 | 100-124 | 13 | 375-399 |
| 3 | 125-149 | 14 | 400-424 |
| 4 | 150-174 | 15 | 425-449 |
| 5 | 175-199 | 16 | 450-474 |
| 6 | 200-224 | 17 | 475-499 |
| 7 | 225-249 | 18 | 500-524 |
| 8 | 250-274 | 19 | 525-549 |
| 9 | 275-299 | 20 | 550+ |
| 10 | 300-324 | | |

---

## 7. Data Flow Diagrams

### Production Flow

```
XTracker API ──► EventStore ──► fit():
                                ├─ Historical timestamps ──► IntradayProgressCurve.fit(F(τ))
                                ├─ Historical events + counts ──► Nowcast.fit()
                                ├─ Historical counts ──► RegimeModel.initialize()
                                │                    ──► DispersionEstimator.estimate()
                                │                    ──► WeekendEffect.estimate()
                                └─ All components ──► MonteCarloForecaster.__init__()

    Periodic Tick:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 1. Refresh tweets from XTracker API                                    │
    │ 2. nowcast.predict(today_events) → (today_mean, today_std)             │
    │ 3. compute_regime_adjustment(cum_so_far, τ) → adj ∈ [0.9, 1.1]        │
    │ 4. Monte Carlo (10K samples):                                          │
    │    a. Today ~ LogNormal(mean, std')                                    │
    │    b. Days 1-6 ~ NegBin(μ_h * adj_h, k/2.5)                           │
    │    c. sums[i] = today + future_days                                    │
    │ 5. Projection → bin probabilities                                      │
    │ 6. Kelly optimizer:                                                    │
    │    a. Compute reservation prices                                       │
    │    b. Generate trade candidates (edge model)                           │
    │    c. Execute best candidate → CLOB FAK order                          │
    │    d. Wait for WebSocket confirmation                                  │
    │    e. Repeat until no profitable candidates                            │
    └─────────────────────────────────────────────────────────────────────────┘
```

### Backtest Flow

```
    Price History (scraped) ──► HistoricalDataProvider
    XTracker API ──────────► CachedPostsProvider

    For each event:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 1. Load event price data (bin prices over time)                        │
    │ 2. Load posts, separate into training + backtest                       │
    │ 3. Fit forecaster on training data                                     │
    │ 4. For each tick timestamp (every 300s):                               │
    │    a. Add new posts incrementally (simulate real-time)                 │
    │    b. Run forecast → bin probabilities                                 │
    │    c. Build simulated orderbook from historical mid-prices             │
    │    d. Run KellyExecutor.run_tick() (SAME code as production)           │
    │    e. Record trades                                                    │
    │ 5. Settlement: YES in winner bin pays $1, else $0                      │
    │ 6. Report P&L, trade statistics                                        │
    └─────────────────────────────────────────────────────────────────────────┘
```

### Order Execution Flow (Production)

```
    Kelly generates candidate
         │
    OrderExecutor.execute_candidate()
         │
    CLOB place_limit_order() (FAK)
         │ ─── price: round(2 decimals), size: floor(integer)
         │
    ExecutionResult { success, order_id }
         │
    Register pending with UserStreamClient
         │
    WebSocket: MATCHED → MINED → CONFIRMED
         │
    handle_fill(): log only, no portfolio update
         │
    Next tick: sync_portfolio() from API
         │
    Portfolio updated from official Polymarket state
```

---

*Generated from source code analysis. All file paths relative to `src/algo/musk_tweet_count/`.*
