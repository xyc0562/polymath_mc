# Elon Musk 7-Day Tweet Count Forecasting Model
## Implementation Plan with Statistical Background

**Document**: Implementation plan for probability forecasting model
**Date**: 2026-01-25
**Version**: 1.2
**Purpose**: Feed bin probabilities to Kelly trading optimizer

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Statistical Background Refresher](#2-statistical-background-refresher)
3. [Model Architecture](#3-model-architecture)
4. [Layer 1: Intraday Nowcast Model](#4-layer-1-intraday-nowcast-model)
5. [Layer 2: Interday Regime Model](#5-layer-2-interday-regime-model)
6. [Layer 3: Monte Carlo Aggregation](#6-layer-3-monte-carlo-aggregation)
7. [Parameter Summary](#7-parameter-summary)
8. [Data Source: XTracker API](#8-data-source-xtracker-api)
9. [Implementation Plan](#9-implementation-plan)
10. [File Structure](#10-file-structure)
11. [Integration with Kelly Optimizer](#11-integration-with-kelly-optimizer)
12. [Testing Strategy](#12-testing-strategy)
13. [Risks and Mitigations](#13-risks-and-mitigations)

---

## 1. Executive Summary

We need to forecast the probability distribution of Elon Musk's total tweet count over a 7-day period (noon-to-noon Eastern Time). This distribution feeds into our Kelly criterion optimizer, which determines optimal position sizes across Polymarket bins.

**Key insight**: The 7-day forecast decomposes into:
- **Today** (partial day): Use intraday patterns to estimate final count
- **Days 1-6** (future): Use regime/persistence modeling

**Output**: Probability for each Polymarket bin (e.g., P[total ∈ 0-74], P[total ∈ 75-99], etc.)

---

## 2. Statistical Background Refresher

### 2.1 Count Data and the Poisson Distribution

Tweet counts are **count data** (non-negative integers). The simplest model is the **Poisson distribution**:

```
P(X = k) = (λ^k × e^(-λ)) / k!
```

Where:
- **λ (lambda)** = expected count (mean = variance in Poisson)
- **k** = observed count (0, 1, 2, ...)

**Problem**: Real-world counts often have **variance > mean** (overdispersion). Elon's tweeting is bursty - some days 20 tweets, others 200.

### 2.2 Negative Binomial Distribution

The **Negative Binomial** handles overdispersion by adding a dispersion parameter:

```
Mean = μ
Variance = μ + μ²/k
```

Where:
- **μ** = expected count
- **k** = dispersion parameter (smaller k = more overdispersion)

As k → ∞, Negative Binomial → Poisson.

**Intuition**: Think of it as a Poisson where λ itself is random (Gamma-distributed). This captures day-to-day variability in Elon's "mood to tweet."

**Python**:
```python
from scipy.stats import nbinom

# Parameterization: n=k (dispersion), p=k/(k+μ)
def neg_binom_sample(mu, k, size=1):
    p = k / (k + mu)
    return nbinom.rvs(n=k, p=p, size=size)
```

### 2.3 Log Transformation for Counts

Count data is often right-skewed (long tail). Taking **log(count + 1)** makes it more symmetric and easier to model with linear methods.

```python
# Transform
log_count = np.log(count + 1)

# Inverse transform
count = np.exp(log_count) - 1
```

The "+1" avoids log(0) = -∞.

### 2.4 Exponentially Weighted Moving Average (EWMA)

EWMA gives more weight to recent observations:

```
EWMA_t = α × X_t + (1 - α) × EWMA_{t-1}
```

Where:
- **α** = smoothing factor (0 < α < 1)
- Higher α = more weight on recent data
- **Half-life** = log(0.5) / log(1-α)

**Example**: α = 0.20 → half-life ≈ 3.1 days

| Days ago | Weight (α=0.20) |
|----------|-----------------|
| 0 | 20.0% |
| 1 | 16.0% |
| 2 | 12.8% |
| 3 | 10.2% |
| 7 | 4.2% |
| 14 | 0.9% |

### 2.5 Exponential Decay Weighting

For averaging over history (like intraday curves), we use decay weights:

```python
weight_i = exp(-days_ago_i / half_life_days × ln(2))
```

With half-life = 21 days:

| Days ago | Weight |
|----------|--------|
| 0 | 100% |
| 7 | 79% |
| 14 | 63% |
| 21 | 50% |
| 42 | 25% |

### 2.6 Monte Carlo Simulation

When we can't compute a distribution analytically, we **simulate** many scenarios:

1. Draw random samples from each component distribution
2. Combine them (e.g., sum daily counts)
3. Histogram of results → empirical distribution

**Example**: To get P(7-day sum ∈ [75, 99]):
```python
samples = []
for _ in range(10000):
    c0 = sample_today()
    c1_6 = [sample_future_day(d) for d in range(1, 7)]
    total = c0 + sum(c1_6)
    samples.append(total)

prob_75_99 = np.mean((np.array(samples) >= 75) & (np.array(samples) <= 99))
```

### 2.7 Ridge Regression (Brief)

Regular regression minimizes squared error. **Ridge regression** adds a penalty on coefficient magnitude:

```
Loss = Σ(y - ŷ)² + λ × Σ(β²)
```

This prevents overfitting when you have many features. We'll use it for the intraday nowcast model.

---

## 3. Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Musk7DayForecaster                              │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Layer 1: Intraday Nowcast                     │   │
│  │                                                                   │   │
│  │  Inputs:                        Output:                          │   │
│  │  - cum_so_far (tweets today)    - Ĉ₀ (estimated final count)    │   │
│  │  - τ (minutes since noon)       - σ₀ (uncertainty)               │   │
│  │  - F(τ) (progress curve)                                         │   │
│  │  - burst features                                                 │   │
│  │  - is_weekend                                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Layer 2: Interday Regime                      │   │
│  │                                                                   │   │
│  │  State:                         Output:                          │   │
│  │  - λ̂ (latent intensity)        - Distribution params for        │   │
│  │  - k (dispersion)                 C₁, C₂, ..., C₆                │   │
│  │  - weekend effects                                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  Layer 3: Monte Carlo Aggregation                │   │
│  │                                                                   │   │
│  │  For each of 10,000 simulations:                                 │   │
│  │    S = Ĉ₀ + C₁ + C₂ + ... + C₆                                  │   │
│  │                                                                   │   │
│  │  Output: P[S ∈ bin_i] for each Polymarket bin                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Layer 1: Intraday Nowcast Model

### 4.1 Purpose

Given partial observations of today, estimate the final count for today.

**Example**: It's 6:00pm (τ = 360 minutes since noon), we've seen 45 tweets. How many total by tomorrow noon?

### 4.2 Contract-Day Definition

A "contract-day" runs from **12:00pm ET to 12:00pm ET next day**.

```python
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

def get_contract_date(timestamp: datetime) -> date:
    """
    Get the contract-day for a timestamp.

    12:00pm Jan 25 to 11:59am Jan 26 → contract_date = Jan 25
    """
    ts_et = timestamp.astimezone(ET)
    # Subtract 12 hours, then take the date
    adjusted = ts_et - timedelta(hours=12)
    return adjusted.date()

def get_tau(timestamp: datetime, contract_date: date) -> int:
    """
    Get minutes since contract-day start (noon).

    Returns τ ∈ [0, 1440)
    """
    ts_et = timestamp.astimezone(ET)
    contract_start = datetime.combine(contract_date, time(12, 0), tzinfo=ET)
    delta = ts_et - contract_start
    return int(delta.total_seconds() / 60)
```

**Edge case - Tweet exactly at noon boundary**:

The boundary is **left-inclusive, right-exclusive**: `[noon_d, noon_{d+1})`

```python
# Tweet at 12:00:00.000 on Jan 25 → belongs to Jan 25 contract-day
# Tweet at 11:59:59.999 on Jan 25 → belongs to Jan 24 contract-day
```

This is handled correctly by the subtraction-based approach above.

### 4.3 Intraday Progress Curve F(τ)

**Definition**: F(τ) = expected fraction of daily tweets by minute τ

**Construction**:
1. For each historical contract-day with total > 0:
   - Compute cumulative count at each 5-minute mark
   - Divide by that day's final total → get fraction at each τ
2. Average across days using exponential decay weights
3. Maintain separate curves for weekday vs weekend

**Parameters**:
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Bin size | 5 minutes | Matches update frequency |
| Half-life | 21 days | ~3 weeks of relevance |
| Exclude zero days | Yes | Avoid division by zero |
| Min tweets to include | 5 | Avoid distortion from sparse days |

**Expected shape**: S-curve starting slow (night), accelerating mid-day, flattening by evening.

**Edge case - Sparse days**: Days with very few tweets (e.g., 1 tweet at 11pm) would create
distorted curves (0% until late, then 100%). Exclude days with < 5 tweets from curve fitting.

### 4.4 Burst Features

Computed from official timestamps, looking backward from current time:

| Feature | Definition | Purpose |
|---------|------------|---------|
| `count_15m` | Tweets in last 15 min | Immediate burst detection |
| `count_60m` | Tweets in last 60 min | Session detection |
| `count_180m` | Tweets in last 3 hours | Extended activity |
| `last_gap_min` | Minutes since last tweet | Recency |
| `in_session` | last_gap_min ≤ 10 | Active session flag |
| `max_burst_60m` | Max tweets in any 10-min window within last 60 min | Peak intensity |

These features naturally cross the noon boundary - no special handling needed.

**Edge case - No tweets today**:

`last_gap_min` is undefined when there are 0 or 1 tweets in the lookback window.

```python
def compute_last_gap_min(tweets_today: List[datetime], tau: int) -> float:
    """
    Compute minutes since last tweet, handling edge cases.

    Args:
        tweets_today: List of tweet timestamps today (sorted)
        tau: Minutes since noon
    """
    n = len(tweets_today)

    if n == 0:
        # No tweets today: gap = time since noon (how long we've waited)
        return float(tau)
    elif n == 1:
        # One tweet: gap = time since that tweet
        tweet_tau = get_tau(tweets_today[0])
        return float(tau - tweet_tau)
    else:
        # Normal case: gap between last two tweets
        gap = tweets_today[-1] - tweets_today[-2]
        return gap.total_seconds() / 60.0

def compute_in_session(last_gap_min: float, num_tweets: int) -> bool:
    """Active session if recent tweet and not waiting too long."""
    if num_tweets == 0:
        return False  # Can't be in session with no tweets
    return last_gap_min <= 10
```

### 4.5 Nowcast Model

**Model type**: Ridge Regression

**Features**:
```python
features = [
    cum_so_far,                    # Tweets seen today
    tau,                           # Minutes since noon
    F_tau,                         # Expected progress at τ
    implied_rate,                  # See below for computation
    count_15m,
    count_60m,
    count_180m,
    last_gap_min,
    in_session,                    # 0 or 1
    max_burst_60m,
    is_weekend,                    # 0 or 1
]

# Implied rate computation (handles early-day instability)
# Only use when enough of the day has elapsed (F_tau > 5%)
if F_tau > 0.05:
    implied_rate = cum_so_far / F_tau
else:
    # Early in day: use historical mean as fallback
    # This avoids division by near-zero causing wild values
    implied_rate = historical_daily_mean
```

**Rationale for F(τ) > 0.05 threshold**: At the start of a contract-day (noon), F(τ) ≈ 0.
Dividing by near-zero would produce unstable implied rates (e.g., 3 tweets / 0.001 = 3000).
We wait until ~72 minutes (5% of day) before trusting this feature.

**Target**: Final count C₀ for that contract-day

**Training**:
- Use last 75 days of completed contract-days
- Exponential decay weighting (half-life 21 days)
- Retrain daily or when new day completes

**Parameters**:
| Parameter | Value |
|-----------|-------|
| Regularization (α) | 1.0 |
| Training window | 75 days |
| Weight half-life | 21 days |

**Output**:
- Point estimate Ĉ₀
- Residual standard deviation σ₀ (from training residuals)

**Edge case - Few training samples**:

If training window has < 10 completed days, residual std is unreliable.

```python
MIN_TRAINING_DAYS = 10

def get_nowcast_uncertainty(residuals: np.ndarray, historical_std: float) -> float:
    """
    Get uncertainty estimate, with fallback for sparse data.
    """
    if len(residuals) < MIN_TRAINING_DAYS:
        # Not enough data: use unconditional historical std
        return historical_std
    else:
        return np.std(residuals)
```

---

## 5. Layer 2: Interday Regime Model

### 5.1 Purpose

Forecast daily counts for days 1-6 ahead, capturing:
- **Persistence**: High-activity days tend to follow high-activity days
- **Overdispersion**: Large day-to-day variance

### 5.2 Latent Intensity State

We model a latent "posting intensity" λ that evolves smoothly:

```python
λ̂_d = α × log(C_d + 1) + (1 - α) × λ̂_{d-1}
```

**Parameters**:
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| α (smoothing) | 0.20 | Half-life ≈ 3 days, responsive but stable |
| Initialization | Mean of log(C+1) over last 45 days | Robust starting point |

### 5.3 Observation Model

Daily counts follow a **Negative Binomial**:

```
C_d ~ NegBinomial(μ = exp(λ̂_d + weekend_effect), k)
```

Where:
- **μ** = expected count for day d
- **k** = dispersion (estimated from data)
- **weekend_effect** = adjustment for weekend contract-days

**Edge case - Intensity overflow**:

If λ̂ grows unreasonably large (due to bugs or extreme data), exp(λ) could overflow
or produce unrealistic predictions. Cap at 300 tweets/day maximum.

```python
# ln(300) ≈ 5.70
MAX_LOG_INTENSITY = 5.70
MAX_TWEETS_PER_DAY = 300

def compute_mu(lambda_hat: float, weekend_effect: float) -> float:
    """
    Compute expected count with overflow protection.
    """
    log_intensity = lambda_hat + weekend_effect
    log_intensity_capped = min(log_intensity, MAX_LOG_INTENSITY)
    return np.exp(log_intensity_capped)
```

### 5.4 Dispersion Parameter k

**Estimation**: From residuals on rolling 90-day window

```python
def estimate_k(observed_counts, predicted_means):
    """
    Method of moments estimator for Negative Binomial k.

    Var = μ + μ²/k
    → k = μ² / (Var - μ)
    """
    residuals = observed_counts - predicted_means
    variance = np.var(residuals)
    mean = np.mean(predicted_means)

    # Avoid division by zero or negative k
    # Add 1% buffer to handle Var ≈ μ case
    if variance <= mean * 1.01:
        return 100.0  # Essentially Poisson (underdispersed or exact)

    k = mean**2 / (variance - mean)
    return max(k, 0.5)  # Floor at 0.5
```

**Parameters**:
| Parameter | Value |
|-----------|-------|
| Estimation window | 90 days |
| Update frequency | Daily |
| Minimum k | 0.5 |

**Typical value**: k ≈ 2-5 for Elon's tweeting (quite overdispersed)

### 5.5 Weekend Effect

Contract-days starting on Saturday or Sunday may have different patterns.

```python
def weekend_adjustment(contract_start_date: date) -> float:
    """
    Returns additive adjustment to log-intensity for weekends.

    Estimated from historical weekday vs weekend means.
    """
    weekday = contract_start_date.weekday()
    if weekday in (5, 6):  # Saturday, Sunday
        return weekend_effect  # Estimated from data, typically -0.1 to +0.1
    return 0.0
```

**Parameters**:
| Parameter | Value |
|-----------|-------|
| Weekend days | Saturday, Sunday (contract start day) |
| Effect estimation | From 90-day rolling comparison |

### 5.6 State Evolution for Forecasting

For forecasting days 1-6 ahead:

```python
def forecast_lambda(current_lambda, horizon, alpha=0.20):
    """
    Forecast latent intensity h days ahead.

    Simple approach: assume λ reverts toward long-term mean.
    """
    # Long-term mean (from initialization)
    long_term_mean = self.initial_lambda

    # Gradual reversion
    reversion_rate = 0.05  # Per day

    lambdas = []
    lam = current_lambda
    for h in range(1, horizon + 1):
        # Slight mean reversion
        lam = lam + reversion_rate * (long_term_mean - lam)
        lambdas.append(lam)

    return lambdas
```

---

## 6. Layer 3: Monte Carlo Aggregation

### 6.1 Purpose

Combine today's nowcast with future day forecasts to get 7-day sum distribution.

### 6.2 Simulation Procedure

```python
def sample_c0_lognormal(c0_mean: float, c0_std: float) -> float:
    """
    Sample today's count using log-normal distribution.

    Log-normal ensures positive values without clipping bias.
    Using max(0, normal) would bias toward 0 when std is large.
    """
    if c0_mean <= 0:
        return 0.0
    if c0_std <= 0:
        return c0_mean

    # Convert normal params to log-normal params
    # If X ~ LogNormal(μ_ln, σ_ln), then:
    #   E[X] = exp(μ_ln + σ_ln²/2)
    #   Var[X] = (exp(σ_ln²) - 1) * exp(2*μ_ln + σ_ln²)
    variance = c0_std ** 2
    mu_ln = np.log(c0_mean**2 / np.sqrt(c0_mean**2 + variance))
    sigma_ln = np.sqrt(np.log(1 + variance / c0_mean**2))

    return np.random.lognormal(mu_ln, sigma_ln)


def simulate_7day_total(n_simulations=10000):
    """
    Monte Carlo simulation of 7-day total.
    """
    samples = []

    # Today's estimate (Layer 1)
    c0_mean = nowcast_today()
    c0_std = get_nowcast_uncertainty()

    # Future lambdas (Layer 2)
    future_lambdas = forecast_lambda(current_lambda, horizon=6)

    for _ in range(n_simulations):
        # Sample today using log-normal (ensures positive, avoids clipping bias)
        c0 = sample_c0_lognormal(c0_mean, c0_std)

        # Sample future days (Negative Binomial)
        total = c0
        for h, lam in enumerate(future_lambdas):
            # Weekend adjustment
            future_date = current_contract_date + timedelta(days=h+1)
            adj = weekend_adjustment(future_date)

            # Apply intensity cap (max 300 tweets/day)
            mu = compute_mu(lam, adj)  # Uses MAX_LOG_INTENSITY cap
            c_h = neg_binom_sample(mu, k)
            total += c_h

        samples.append(total)

    return np.array(samples)
```

### 6.3 Bin Probability Extraction

```python
def get_bin_probabilities(samples, bins):
    """
    Convert samples to bin probabilities.

    bins: List of (lower, upper) tuples, e.g., [(0, 74), (75, 99), ...]
    """
    probs = []
    n = len(samples)

    for lower, upper in bins:
        count = np.sum((samples >= lower) & (samples <= upper))
        probs.append(count / n)

    return probs
```

### 6.4 Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_simulations | 10,000 | Good precision, fast enough |
| Random seed | Set per run for reproducibility | Optional |

---

## 7. Parameter Summary

### All Concrete Parameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| **Contract-Day** | Boundary | 12:00pm ET (left-inclusive) |
| | Timezone | America/New_York |
| **Intraday Curve** | Bin size | 5 minutes |
| | Half-life | 21 days |
| | Exclude zero days | Yes |
| | Min tweets to include day | 5 |
| **Burst Features** | Windows | 15m, 60m, 180m |
| | Session gap | 10 minutes |
| | Burst window | 10 min within 60 min |
| **Nowcast Model** | Type | Ridge Regression |
| | Regularization α | 1.0 |
| | Training window | 75 days |
| | Weight half-life | 21 days |
| | Implied rate threshold | F(τ) > 0.05 |
| | Min training days for σ₀ | 10 |
| **Regime Model** | EWMA α | 0.20 |
| | Initialization window | 45 days |
| | Mean reversion rate | 0.05 per day |
| | Max log intensity | 5.70 (300 tweets/day) |
| **Dispersion k** | Estimation window | 90 days |
| | Update frequency | Daily |
| | Minimum value | 0.5 |
| | Underdispersion buffer | 1.01× |
| **Weekend Effect** | Days | Sat, Sun (contract start) |
| | Estimation window | 90 days |
| **Monte Carlo** | Simulations | 10,000 |
| | C₀ sampling | Log-normal |
| **Update Trigger** | On new tweet | Yes (immediate) |
| | Periodic | Every 3-5 minutes |

---

## 8. Data Source: XTracker API

### 8.1 Official Data Endpoint

We fetch official Elon Musk post data from the XTracker API:

```
GET https://xtracker.polymarket.com/api/users/elonmusk/posts
    ?startDate={ISO_DATE}
    &endDate={ISO_DATE}
```

**Example**:
```
https://xtracker.polymarket.com/api/users/elonmusk/posts
    ?startDate=2025-11-01T17:00:00.000Z
    &endDate=2025-11-02T16:59:59.000Z
```

**Notes**:
- Dates are in **UTC** (ISO 8601 format)
- Data available from **October 31, 2025** onwards
- Includes original posts and retweets
- Response contains individual post timestamps

### 8.2 Contract-Day Query Pattern

To get all posts for a contract-day (noon ET to noon ET):

```python
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

def get_xtracker_query_params(contract_date: date) -> Tuple[str, str]:
    """
    Get startDate and endDate for XTracker API query.

    Contract-day runs from noon ET to noon ET next day.
    API expects UTC timestamps.
    """
    # Contract-day starts at noon ET
    start_et = datetime.combine(contract_date, time(12, 0), tzinfo=ET)
    # Ends at 11:59:59.999 ET next day (or noon next day exclusive)
    end_et = datetime.combine(contract_date + timedelta(days=1), time(11, 59, 59), tzinfo=ET)

    # Convert to UTC for API
    start_utc = start_et.astimezone(UTC)
    end_utc = end_et.astimezone(UTC)

    return (
        start_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        end_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    )
```

### 8.3 Historical Data Loading

For model training, we need ~90 days of history:

```python
def fetch_historical_posts(n_days: int = 90) -> List[TweetEvent]:
    """Fetch historical posts for training."""
    events = []
    today = get_current_contract_date()

    for days_ago in range(n_days):
        contract_date = today - timedelta(days=days_ago + 1)  # Completed days only
        start_date, end_date = get_xtracker_query_params(contract_date)

        response = requests.get(
            "https://xtracker.polymarket.com/api/users/elonmusk/posts",
            params={"startDate": start_date, "endDate": end_date}
        )

        for post in response.json():
            events.append(TweetEvent(
                timestamp=datetime.fromisoformat(post["timestamp"].replace("Z", "+00:00")),
                event_type=post.get("type", "tweet")
            ))

    return events
```

---

## 9. Implementation Plan

### Phase 1: Data Layer

**File**: `forecaster/data.py`

1. XTracker API client
2. Event storage and retrieval
3. Contract-day aggregation
4. Timezone handling with `zoneinfo`
5. Rolling window utilities

**Key classes**:
```python
@dataclass
class TweetEvent:
    timestamp: datetime
    event_type: str  # "tweet", "retweet"

class XTrackerClient:
    """Client for fetching official post data."""
    BASE_URL = "https://xtracker.polymarket.com/api"

    def get_posts(self, start_date: str, end_date: str) -> List[Dict]
    def get_contract_day_posts(self, contract_date: date) -> List[TweetEvent]
    def fetch_historical(self, n_days: int) -> List[TweetEvent]

class EventStore:
    def __init__(self, xtracker_client: XTrackerClient)
    def add_event(self, event: TweetEvent) -> None
    def get_events(self, start: datetime, end: datetime) -> List[TweetEvent]
    def get_contract_day_counts(self, n_days: int) -> Dict[date, int]
    def refresh_from_api(self) -> None
```

### Phase 2: Intraday Model

**File**: `forecaster/intraday.py`

1. Progress curve F(τ) computation
2. Burst feature extraction
3. Ridge regression nowcast
4. Uncertainty estimation

**Key classes**:
```python
class IntradayProgressCurve:
    def fit(self, historical_counts: Dict[date, List[datetime]]) -> None
    def get_expected_progress(self, tau: int, is_weekend: bool) -> float

class IntradayNowcast:
    def __init__(self, progress_curve: IntradayProgressCurve)
    def compute_burst_features(self, events: List[datetime], now: datetime) -> Dict
    def fit(self, historical_data) -> None
    def predict(self, cum_so_far, tau, features) -> Tuple[float, float]
```

### Phase 3: Interday Model

**File**: `forecaster/interday.py`

1. EWMA state tracking
2. Dispersion estimation
3. Weekend effect estimation
4. Future lambda forecasting

**Key classes**:
```python
class RegimeState:
    lambda_hat: float
    k: float
    weekend_effect: float
    last_updated: date

class InterdayModel:
    def __init__(self, alpha=0.20)
    def update_state(self, contract_day_count: int, contract_date: date) -> None
    def estimate_dispersion(self, history: Dict[date, int]) -> float
    def forecast_lambdas(self, horizon: int) -> List[float]
    def sample_future_day(self, lambda_h: float, is_weekend: bool) -> int
```

### Phase 4: Monte Carlo & Integration

**File**: `forecaster/monte_carlo.py`

1. Simulation engine
2. Bin probability computation
3. Caching layer

**Key classes**:
```python
class MonteCarloSimulator:
    def __init__(self, n_simulations=10000)
    def simulate_7day(self, c0_mean, c0_std, future_lambdas, k, weekend_flags) -> np.ndarray
    def get_bin_probabilities(self, samples, bins) -> List[float]

class DistributionCache:
    def __init__(self, ttl_seconds=180)
    def get_or_compute(self, key, compute_fn) -> List[float]
    def invalidate(self) -> None
```

### Phase 5: Main Forecaster

**File**: `forecaster/forecaster.py`

1. Orchestrates all components
2. Handles live updates
3. Provides clean interface for Kelly optimizer

**Key class**:
```python
class Musk7DayForecaster:
    def __init__(self, config: ForecasterConfig)

    # Data management
    def update_events(self, events: List[TweetEvent]) -> None
    def on_new_tweet(self, event: TweetEvent) -> None

    # Forecasting
    def nowcast_today(self) -> Tuple[float, float]  # (mean, std)
    def forecast_future_days(self) -> List[Tuple[float, float]]  # [(mu, k), ...]
    def forecast_7day_distribution(self, bins: List[Tuple]) -> List[float]

    # For Kelly integration
    def get_bin_probabilities(self) -> List[float]
```

### Phase 6: Backtesting

**File**: `forecaster/backtest.py`

1. Rolling window evaluation
2. Metrics computation
3. Ablation testing

---

## 10. File Structure

```
src/algo/musk_tweet_count/
├── kelly/                      # Already implemented
│   ├── __init__.py
│   ├── config.py
│   ├── kelly_math.py
│   ├── orderbook.py
│   ├── websocket_client.py
│   ├── portfolio.py
│   ├── candidates.py
│   ├── executor.py
│   └── integration.py
│
└── forecaster/                 # NEW: Probability model
    ├── __init__.py
    ├── config.py               # ForecasterConfig dataclass
    ├── data.py                 # Event storage, contract-day logic
    ├── intraday.py             # Progress curve, burst features, nowcast
    ├── interday.py             # Regime model, dispersion, weekend effects
    ├── monte_carlo.py          # Simulation, bin probabilities
    ├── forecaster.py           # Main Musk7DayForecaster class
    └── backtest.py             # Validation utilities
```

---

## 11. Integration with Kelly Optimizer

### 10.1 Interface

The Kelly optimizer expects a probability model function:

```python
def probability_model(current_count: int, hours_elapsed: float, hours_remaining: float) -> List[float]:
    """
    Returns probability for each bin.

    Args:
        current_count: Tweet count so far in this 7-day period
        hours_elapsed: Hours since market opened (7-day period started)
        hours_remaining: Hours until settlement

    Returns:
        List of probabilities, one per bin, summing to 1.0
    """
```

### 10.2 Implementation

```python
class ForecastingProbabilityModel:
    """
    Adapter between Musk7DayForecaster and Kelly optimizer.
    """

    def __init__(self, forecaster: Musk7DayForecaster, bins: List[Tuple[int, int]]):
        self.forecaster = forecaster
        self.bins = bins  # [(0, 74), (75, 99), (100, 124), ...]
        self._cache = None
        self._cache_count = None

    def __call__(self, current_count: int, hours_elapsed: float, hours_remaining: float) -> List[float]:
        # Invalidate cache if count changed
        if current_count != self._cache_count:
            self._cache = None
            self._cache_count = current_count

        # Return cached if available and recent
        if self._cache is not None:
            return self._cache

        # Compute fresh distribution
        probs = self.forecaster.get_bin_probabilities(self.bins)
        self._cache = probs

        return probs
```

### 10.3 Update Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Live Update Loop                           │
└─────────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌───────────────────┐                         ┌───────────────────┐
│  New Tweet Event  │                         │  Timer (3-5 min)  │
└───────────────────┘                         └───────────────────┘
        │                                               │
        ▼                                               │
┌───────────────────┐                                   │
│ forecaster.on_new │                                   │
│ _tweet(event)     │                                   │
└───────────────────┘                                   │
        │                                               │
        └───────────────────────┬───────────────────────┘
                                ▼
                  ┌───────────────────────────┐
                  │ forecaster.get_bin_       │
                  │ probabilities()           │
                  └───────────────────────────┘
                                │
                                ▼
                  ┌───────────────────────────┐
                  │ kelly_executor.run_tick() │
                  │ with new probabilities    │
                  └───────────────────────────┘
```

### 10.4 Configuration Example

```yaml
# config/musk_tweet_count.yaml

forecaster:
  # Data settings
  timezone: "America/New_York"
  contract_boundary_hour: 12  # noon

  # Intraday model
  intraday:
    curve_bin_minutes: 5
    curve_half_life_days: 21
    training_window_days: 75
    ridge_alpha: 1.0

  # Interday model
  interday:
    ewma_alpha: 0.20
    initialization_window_days: 45
    mean_reversion_rate: 0.05
    dispersion_window_days: 90
    min_dispersion_k: 0.5

  # Monte Carlo
  monte_carlo:
    n_simulations: 10000

  # Update triggers
  update_on_new_tweet: true
  periodic_update_seconds: 180  # 3 minutes

kelly:
  enabled: true
  kappa: 0.25
  # ... rest of kelly config
```

---

## 12. Testing Strategy

### 11.1 Unit Tests

| Component | Test |
|-----------|------|
| Contract-day | DST transition handling |
| Contract-day | Noon boundary (11:59am vs 12:01pm) |
| Progress curve | Zero-day exclusion |
| Burst features | Cross-boundary counting |
| EWMA | State initialization |
| Neg Binomial | Parameter edge cases (k → 0, k → ∞) |
| Monte Carlo | Distribution sum = 1.0 |

### 11.2 Integration Tests

1. End-to-end with synthetic data
2. Historical replay against known outcomes
3. Kelly integration smoke test

### 11.3 Backtest Metrics

| Metric | Target |
|--------|--------|
| 1-day MAE | < 20 tweets |
| 7-day MAE | < 50 tweets |
| Bin log-score | > -2.0 |
| Calibration | 50% interval contains true 45-55% of time |

---

## 13. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| DST bugs | Use `zoneinfo`, explicit tests at boundaries |
| Cold start | Require 75 days of history before live trading |
| Regime shift (Elon stops tweeting) | Monitor for extended gaps, alert if no tweets in 24h |
| Overflow in Monte Carlo | Cap daily counts at 1000 |
| Stale cache | TTL + invalidation on new tweets |

---

*Document created 2026-01-25. Ready for implementation.*
