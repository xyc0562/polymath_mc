"""
Configuration dataclasses for the forecasting model.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class IntradayCurveConfig:
    """Configuration for intraday progress curve F(τ)."""

    # Bin size in minutes for curve discretization
    bin_size_minutes: int = 5

    # Half-life for exponential decay weighting (days)
    half_life_days: float = 21.0

    # Exclude days with zero tweets from curve fitting
    exclude_zero_days: bool = True

    # Minimum tweets required to include a day in curve fitting
    min_tweets_per_day: int = 5


@dataclass
class BurstFeaturesConfig:
    """Configuration for burst feature extraction."""

    # Windows for count features (minutes)
    window_15m: int = 15
    window_60m: int = 60
    window_180m: int = 180

    # Gap threshold for "in session" detection (minutes)
    session_gap_threshold: int = 10

    # Window for max burst detection (minutes)
    burst_window: int = 10


@dataclass
class NowcastConfig:
    """Configuration for intraday nowcast model."""

    # Ridge regression regularization parameter
    ridge_alpha: float = 1.0

    # Training window in days - shorter for faster adaptation
    training_window_days: int = 45

    # Half-life for sample weighting (days) - shorter for more reactivity
    weight_half_life_days: float = 14.0

    # Threshold for implied rate feature (fraction of day)
    implied_rate_threshold: float = 0.05

    # Minimum training samples before using residual std
    min_training_days: int = 10


@dataclass
class BucketNowcastConfig:
    """Configuration for bucket-based intraday nowcast model."""

    # Number of buckets per day (8 = 3-hour buckets)
    n_buckets: int = 8

    # Training window in days
    training_window_days: int = 45

    # Half-life for sample weighting (days)
    weight_half_life_days: float = 14.0

    # Regime multiplier bounds (clamp to prevent extreme adjustments)
    regime_min: float = 0.85
    regime_max: float = 1.15

    # Minimum expected count to compute regime (avoid division by near-zero)
    min_expected_for_regime: float = 1.0

    # Sampling distribution: "negbin", "com_poisson", or "negbin_reflected"
    bucket_distribution: str = "com_poisson"

    # Minimum dispersion k (floor for Negative Binomial)
    min_dispersion_k: float = 0.5

    cmp_nu_scale: float = 1.0

    # Hawkes-style self-exciting impulse model
    # Shifted-linear excitation: rate_mult = clamp(1 + gain*(exc - expected*neutral_fraction), floor, ceiling)
    impulse_cutoff_minutes: float = 180.0              # Forward prediction window
    impulse_min_tweets_for_fit: int = 200              # Min tweets to fit rate curve
    impulse_rate_curve_sigma: float = 60.0             # Gaussian kernel σ for λ(τ)
    impulse_decay_halflife_minutes: float = 30.0       # Excitation decay half-life
    impulse_floor: float = 0.4                         # Minimum rate_mult (extended silence)
    impulse_silence_halflife_minutes: float = 90.0     # Forward decay halflife when in silence (rate_mult < 1)
    impulse_neutral_fraction: float = 0.3              # Fraction of expected excitation that's "neutral" (rate_mult=1.0)
    impulse_gain: float = 0.6                          # Linear sensitivity of rate_mult to shifted excitation
    impulse_ceiling: float = 3.0                       # Max rate_mult
    impulse_lookback_minutes: int = 360                # How far back for expected excitation calc
    impulse_min_expected_excitation: float = 0.3       # Below this, treat as low-data period
    impulse_overrides_path: str = "config/impulse_overrides.yaml"  # Time-of-day rate_mult overrides


@dataclass
class RegimeConfig:
    """Configuration for interday regime model."""

    # EWMA smoothing parameter
    # Higher α = faster adaptation to regime changes
    # α=0.20 → ~5 day memory, α=0.35 → ~3 day memory, α=0.5 → ~2 day memory
    ewma_alpha: float = 0.35

    # Window for initialization (days) - shorter to use recent data
    initialization_window_days: int = 21

    # Mean reversion rate (per day)
    mean_reversion_rate: float = 0.05

    # Maximum log intensity (prevents overflow)
    # ln(300) ≈ 5.70 for max 300 tweets/day
    max_log_intensity: float = 5.70

    # Half-life for recency weighting in initialization (days)
    # Shorter = more weight on recent days
    initialization_half_life_days: float = 7.0


@dataclass
class DispersionConfig:
    """Configuration for dispersion parameter estimation."""

    # Rolling window for k estimation (days)
    estimation_window_days: int = 45

    # Minimum k value (floor)
    min_k: float = 0.5

    # Buffer for underdispersion check (1.01 = 1% buffer)
    underdispersion_buffer: float = 1.01

    # Half-life for recency weighting (days)
    half_life_days: float = 14.0


@dataclass
class WeekendConfig:
    """Configuration for weekend effect estimation."""

    # Days considered weekend (0=Mon, 5=Sat, 6=Sun)
    weekend_days: Tuple[int, int] = (5, 6)

    # Window for effect estimation (days)
    estimation_window_days: int = 45

    # Half-life for recency weighting (days)
    half_life_days: float = 14.0


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""

    # Number of simulations
    n_simulations: int = 25000

    # Random seed (None for random)
    random_seed: int = None

    # Bayesian regime adjustment parameters
    # Minimum F(τ) to avoid division by near-zero
    regime_adj_f_min: float = 0.05

    # Prior standard deviation (how much regime can vary day-to-day)
    regime_adj_sigma_prior: float = 0.35

    # Observation noise parameter (higher = trust observation less)
    regime_adj_sigma0: float = 1.00

    # Adjustment clamp bounds
    regime_adj_min: float = 0.9
    regime_adj_max: float = 1.1

    # Minimum τ (minutes) before applying adjustment
    regime_adj_tau_gate: int = 360

    # Minimum F(τ) before applying adjustment
    regime_adj_f_gate: float = 0.20

    # Decay factor for regime adjustment across future days
    # Day h gets adjustment: 1 + (base_adj - 1) * decay^(h-1)
    # decay=0.5 means day 1 gets full adjustment, day 2 gets 50%, day 3 gets 25%, etc.
    regime_adj_decay: float = 0.5

    # Dispersion inflation factor for Negative Binomial sampling
    # k' = k / dispersion_inflation_factor
    # Higher values -> wider distribution -> better coverage
    #
    # Grid search results:
    #   - Unofficial data (560+ days): s=1.5 → Coverage90=92.5%
    #   - XTracker official EWMA (87 days): s=2.0 → Coverage90=78.4%
    #   - XTracker official GAS (89 days): s=2.5 → Coverage90=88.7%
    # Sampling distribution for future days: "negbin", "com_poisson", or "negbin_reflected"
    sampling_distribution: str = "negbin"

    dispersion_inflation_factor: float = 2.5

    cmp_nu_scale: float = 1.0

    # Standard deviation inflation factor for today's nowcast
    # today_std' = today_std * sqrt(today_std_inflation_factor)
    # Set to same as dispersion_inflation_factor for consistency
    today_std_inflation_factor: float = 2.5

    # Hard cap for individual future day samples (not applied to today's nowcast)
    # Based on historical analysis: only 1% of days exceeded 200
    max_daily_forecast: int = 200

    # Hard caps for multi-day sum forecasts, indexed by horizon (0=unused, 1-7=caps)
    # Based on ~p99 of historical data to clip ~1% of extreme forecasts
    # Set to 0 to disable cap for that horizon
    # Horizon 1 = 0 (disabled) because intraday nowcast should not be capped
    max_horizon_caps: Tuple[int, ...] = (0, 0, 350, 500, 650, 800, 950, 1100)


@dataclass
class UpdateConfig:
    """Configuration for update triggers."""

    # Update on new tweet
    update_on_new_tweet: bool = True

    # Periodic update interval (seconds)
    periodic_update_seconds: float = 180.0

    # Cache TTL (seconds)
    cache_ttl_seconds: float = 180.0


@dataclass
class GASConfig:
    """Configuration for NB-GAS (Negative Binomial GAS) regime model.

    Uses Pearson score s_t = (y_t - μ_t) / μ_t for stable regime tracking.
    Includes change point detection for faster adaptation to regime shifts.
    """

    # Initial parameter guesses for MLE
    omega_init: float = 0.1
    alpha_init: float = 0.05
    beta_init: float = 0.95

    # Parameter bounds for optimization
    beta_min: float = 0.5       # Minimum persistence
    beta_max: float = 0.999     # Maximum persistence
    alpha_min: float = 0.001    # Minimum score impact
    alpha_max: float = 0.5      # Maximum score impact

    # Max log-intensity (safety cap)
    # ln(300) ≈ 5.70 for max 300 tweets/day
    max_log_intensity: float = 5.70

    # Change point detection (symmetric for drops and surges)
    # If |2-day avg - 7-day avg| / 7-day avg > threshold, boost α
    cpd_threshold: float = 0.4      # 40% deviation triggers boost
    cpd_alpha_multiplier: float = 3.0  # Multiply α by this factor
    cpd_alpha_cap: float = 0.3      # Cap boosted α at this value


@dataclass
class EnsembleConfig:
    """Configuration for ensemble forecaster.

    Combines multiple EWMA-based interday forecasters with different
    adaptation speeds (α values) via sample-level pooling.
    """

    # EWMA alpha values for ensemble members
    # Slow (0.2): ~5 day memory, stable but slow to adapt
    # Fast (0.5): ~2 day memory, quick adaptation but noisy
    alpha_slow: float = 0.2
    alpha_fast: float = 0.5

    # Weights for each ensemble member [slow, fast]
    # Must sum to 1.0
    weight_slow: float = 0.5
    weight_fast: float = 0.5


@dataclass
class ForecasterConfig:
    """Main configuration for the forecasting model."""

    # Timezone for contract-day calculations
    timezone: str = "America/New_York"

    # Contract-day boundary hour (12 = noon)
    contract_boundary_hour: int = 12

    # Intraday forecaster mode: "ridge" (original) or "bucket" (new)
    intraday_mode: str = "ridge"

    # Interday forecaster mode: "ewma" (default), "gas", or "pig"
    interday_model: str = "ewma"

    # Component configs
    intraday_curve: IntradayCurveConfig = field(default_factory=IntradayCurveConfig)
    burst_features: BurstFeaturesConfig = field(default_factory=BurstFeaturesConfig)
    nowcast: NowcastConfig = field(default_factory=NowcastConfig)
    bucket_nowcast: BucketNowcastConfig = field(default_factory=BucketNowcastConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    dispersion: DispersionConfig = field(default_factory=DispersionConfig)
    weekend: WeekendConfig = field(default_factory=WeekendConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    update: UpdateConfig = field(default_factory=UpdateConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    gas: GASConfig = field(default_factory=GASConfig)

    # Polymarket bins (lower, upper) inclusive
    # Default bins for Musk tweet count market
    bins: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0, 74),
        (75, 99),
        (100, 124),
        (125, 149),
        (150, 174),
        (175, 199),
        (200, 224),
        (225, 249),
        (250, 274),
        (275, 299),
        (300, 324),
        (325, 349),
        (350, 374),
        (375, 399),
        (400, 424),
        (425, 449),
        (450, 474),
        (475, 499),
        (500, 524),
        (525, 549),
        (550, 10000),  # 550+ bin
    ])

    @classmethod
    def from_dict(cls, data: dict) -> "ForecasterConfig":
        """Create config from dictionary (e.g., from YAML)."""
        # Extract nested configs
        intraday_curve_data = data.pop("intraday_curve", data.pop("intraday", {}))
        burst_features_data = data.pop("burst_features", {})
        nowcast_data = data.pop("nowcast", {})
        regime_data = data.pop("regime", data.pop("interday", {}))
        dispersion_data = data.pop("dispersion", {})
        weekend_data = data.pop("weekend", {})
        monte_carlo_data = data.pop("monte_carlo", {})
        update_data = data.pop("update", {})
        ensemble_data = data.pop("ensemble", {})
        gas_data = data.pop("gas", {})

        # Handle bins
        bins_data = data.pop("bins", None)
        bins = None
        if bins_data:
            bins = [tuple(b) for b in bins_data]

        config = cls(
            intraday_curve=IntradayCurveConfig(**intraday_curve_data),
            burst_features=BurstFeaturesConfig(**burst_features_data),
            nowcast=NowcastConfig(**nowcast_data),
            regime=RegimeConfig(**regime_data),
            dispersion=DispersionConfig(**dispersion_data),
            weekend=WeekendConfig(**weekend_data),
            monte_carlo=MonteCarloConfig(**monte_carlo_data),
            update=UpdateConfig(**update_data),
            ensemble=EnsembleConfig(**ensemble_data),
            gas=GASConfig(**gas_data),
            **data,
        )

        if bins:
            config.bins = bins

        return config
