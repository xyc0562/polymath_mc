"""
Intraday model for nowcasting today's final tweet count.

Components:
- IntradayProgressCurve: F(τ) = expected fraction of day's tweets by minute τ
- BurstFeatureExtractor: Extract burst/session features from timestamps
- BaseIntradayForecaster: Abstract interface for intraday forecasters
- IntradayNowcast: Ridge regression model (original implementation)
- BucketIntradayForecaster: Bucket-based model with regime adjustment
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge

from .config import IntradayCurveConfig, BurstFeaturesConfig, NowcastConfig, BucketNowcastConfig
from .data import ContractDayUtils, TweetEvent

logger = logging.getLogger(__name__)


class IntradayProgressCurve:
    """
    Intraday progress curve F(τ).

    F(τ) = expected fraction of a day's total tweets completed by minute τ.
    Used to estimate implied daily rate from partial observations.
    """

    def __init__(self, config: IntradayCurveConfig, contract_utils: ContractDayUtils):
        """
        Initialize progress curve.

        Args:
            config: Curve configuration
            contract_utils: Contract-day utilities
        """
        self.config = config
        self.contract_utils = contract_utils

        # Number of bins (1440 minutes / bin_size)
        self.n_bins = 1440 // config.bin_size_minutes

        # Curves: weekday and weekend
        self._weekday_curve: Optional[np.ndarray] = None
        self._weekend_curve: Optional[np.ndarray] = None

        # Fallback curve (uniform progress)
        self._fallback_curve = np.linspace(0, 1, self.n_bins + 1)[1:]

    def fit(
        self,
        historical_timestamps: Dict[date, List[datetime]],
        as_of_date: Optional[date] = None,
    ) -> None:
        """
        Fit progress curves from historical data.

        Args:
            historical_timestamps: Dict mapping contract_date -> List[timestamp]
            as_of_date: Reference date for "today" (for backtesting). Default: actual today.
        """
        weekday_curves = []
        weekend_curves = []
        weekday_weights = []
        weekend_weights = []

        # Use as_of_date for backtesting, otherwise use actual current date
        today = as_of_date if as_of_date is not None else self.contract_utils.get_current_contract_date()

        for contract_date, timestamps in historical_timestamps.items():
            # Skip if not enough tweets
            if len(timestamps) < self.config.min_tweets_per_day:
                continue

            # Skip zero days if configured
            if self.config.exclude_zero_days and len(timestamps) == 0:
                continue

            # Compute curve for this day
            curve = self._compute_day_curve(contract_date, timestamps)

            # Compute weight (exponential decay)
            days_ago = (today - contract_date).days
            weight = np.exp(-days_ago / self.config.half_life_days * np.log(2))

            # Separate weekday vs weekend
            if self.contract_utils.is_weekend(contract_date):
                weekend_curves.append(curve)
                weekend_weights.append(weight)
            else:
                weekday_curves.append(curve)
                weekday_weights.append(weight)

        # Compute weighted averages
        if weekday_curves:
            self._weekday_curve = self._weighted_average(weekday_curves, weekday_weights)
        else:
            self._weekday_curve = self._fallback_curve.copy()

        if weekend_curves:
            self._weekend_curve = self._weighted_average(weekend_curves, weekend_weights)
        else:
            self._weekend_curve = self._fallback_curve.copy()

        logger.debug(
            f"Fitted progress curves: {len(weekday_curves)} weekdays, "
            f"{len(weekend_curves)} weekends"
        )

    def _compute_day_curve(
        self,
        contract_date: date,
        timestamps: List[datetime],
    ) -> np.ndarray:
        """
        Compute progress curve for a single day.

        Returns cumulative fraction at each bin boundary.
        """
        total = len(timestamps)
        if total == 0:
            return self._fallback_curve.copy()

        # Convert timestamps to τ values
        taus = [self.contract_utils.get_tau(ts, contract_date) for ts in timestamps]

        # Compute cumulative counts at each bin boundary
        curve = np.zeros(self.n_bins)
        for i in range(self.n_bins):
            bin_end = (i + 1) * self.config.bin_size_minutes
            count = sum(1 for tau in taus if tau < bin_end)
            curve[i] = count / total

        return curve

    def _weighted_average(
        self,
        curves: List[np.ndarray],
        weights: List[float],
    ) -> np.ndarray:
        """Compute weighted average of curves."""
        curves_arr = np.array(curves)
        weights_arr = np.array(weights)
        weights_arr = weights_arr / weights_arr.sum()

        return np.average(curves_arr, axis=0, weights=weights_arr)

    def get_expected_progress(self, tau: int, is_weekend: bool = False) -> float:
        """
        Get expected fraction of day's tweets completed by minute τ.

        Args:
            tau: Minutes since noon
            is_weekend: Whether it's a weekend contract-day

        Returns:
            Expected fraction (0 to 1)
        """
        # Clamp τ to valid range
        tau = max(0, min(tau, 1440 - 1))

        # Get bin index
        bin_idx = tau // self.config.bin_size_minutes
        bin_idx = min(bin_idx, self.n_bins - 1)

        # Get appropriate curve
        curve = self._weekend_curve if is_weekend else self._weekday_curve
        if curve is None:
            curve = self._fallback_curve

        return float(curve[bin_idx])

    def get_curve(self, is_weekend: bool = False) -> np.ndarray:
        """Get the full progress curve."""
        curve = self._weekend_curve if is_weekend else self._weekday_curve
        return curve if curve is not None else self._fallback_curve.copy()


@dataclass
class BurstFeatures:
    """Extracted burst features for a point in time."""

    count_15m: int
    count_60m: int
    count_180m: int
    last_gap_min: float
    in_session: bool
    max_burst_60m: int
    num_tweets_today: int


class BurstFeatureExtractor:
    """Extract burst/session features from tweet timestamps."""

    def __init__(self, config: BurstFeaturesConfig, contract_utils: ContractDayUtils):
        """
        Initialize feature extractor.

        Args:
            config: Feature configuration
            contract_utils: Contract-day utilities
        """
        self.config = config
        self.contract_utils = contract_utils

    def extract(
        self,
        events: List[TweetEvent],
        now: datetime,
        tau: int,
    ) -> BurstFeatures:
        """
        Extract burst features at a given time.

        Args:
            events: List of tweet events (today's events)
            now: Current timestamp
            tau: Minutes since noon

        Returns:
            BurstFeatures dataclass
        """
        # Filter to events before 'now'
        past_events = [e for e in events if e.timestamp < now]
        timestamps = [e.timestamp for e in past_events]
        num_tweets = len(timestamps)

        # Count features (look back from 'now')
        count_15m = self._count_in_window(timestamps, now, self.config.window_15m)
        count_60m = self._count_in_window(timestamps, now, self.config.window_60m)
        count_180m = self._count_in_window(timestamps, now, self.config.window_180m)

        # Last gap feature (with edge case handling)
        last_gap_min = self._compute_last_gap(timestamps, now, tau, num_tweets)

        # In session flag
        in_session = self._compute_in_session(last_gap_min, num_tweets)

        # Max burst in last 60 minutes
        max_burst_60m = self._compute_max_burst(
            timestamps, now,
            self.config.window_60m,
            self.config.burst_window
        )

        return BurstFeatures(
            count_15m=count_15m,
            count_60m=count_60m,
            count_180m=count_180m,
            last_gap_min=last_gap_min,
            in_session=in_session,
            max_burst_60m=max_burst_60m,
            num_tweets_today=num_tweets,
        )

    def _count_in_window(
        self,
        timestamps: List[datetime],
        now: datetime,
        window_minutes: int,
    ) -> int:
        """Count tweets in the last N minutes."""
        cutoff = now - timedelta(minutes=window_minutes)
        return sum(1 for ts in timestamps if ts >= cutoff)

    def _compute_last_gap(
        self,
        timestamps: List[datetime],
        now: datetime,
        tau: int,
        num_tweets: int,
    ) -> float:
        """
        Compute minutes since last tweet.

        Handles edge cases:
        - 0 tweets: return τ (time since noon)
        - 1 tweet: return time since that tweet
        - 2+ tweets: return gap between last two tweets
        """
        if num_tweets == 0:
            # No tweets today: gap = time since noon
            contract_date = self.contract_utils.get_contract_date(now)
            start_dt, _ = self.contract_utils.get_contract_day_bounds(contract_date)
            return self.contract_utils.minutes_between(start_dt, now)
        elif num_tweets == 1:
            # One tweet: gap = time since that tweet
            return self.contract_utils.minutes_between(timestamps[0], now)
        else:
            # Normal case: gap between last two tweets
            sorted_ts = sorted(timestamps)
            return self.contract_utils.minutes_between(sorted_ts[-2], sorted_ts[-1])

    def _compute_in_session(self, last_gap_min: float, num_tweets: int) -> bool:
        """Determine if currently in an active session."""
        if num_tweets == 0:
            return False
        return last_gap_min <= self.config.session_gap_threshold

    def _compute_max_burst(
        self,
        timestamps: List[datetime],
        now: datetime,
        lookback_minutes: int,
        burst_window_minutes: int,
    ) -> int:
        """
        Compute maximum tweets in any burst_window within lookback period.
        """
        cutoff = now - timedelta(minutes=lookback_minutes)
        recent = [ts for ts in timestamps if ts >= cutoff]

        if len(recent) < 2:
            return len(recent)

        # Sliding window
        max_count = 0
        for ts in recent:
            window_end = ts + timedelta(minutes=burst_window_minutes)
            count = sum(1 for t in recent if ts <= t < window_end)
            max_count = max(max_count, count)

        return max_count


class BaseIntradayForecaster(ABC):
    """
    Abstract base class for intraday forecasters.

    All intraday forecasters must implement:
    - fit(): Train the model on historical data
    - predict(): Predict final count and uncertainty for today
    - historical_mean, historical_std: Properties for fallback values
    """

    @abstractmethod
    def fit(
        self,
        historical_events: Dict[date, List[TweetEvent]],
        historical_counts: Dict[date, int],
        as_of_date: Optional[date] = None,
    ) -> None:
        """
        Fit the forecaster on historical data.

        Args:
            historical_events: Dict mapping contract_date -> events
            historical_counts: Dict mapping contract_date -> final count
            as_of_date: Reference date for "today" (for backtesting). Default: actual today.
        """
        pass

    @abstractmethod
    def predict(
        self,
        events: List[TweetEvent],
        contract_date: date,
        now: datetime,
    ) -> Tuple[float, float]:
        """
        Predict final count for today.

        Args:
            events: Today's events so far
            contract_date: Today's contract date
            now: Current timestamp

        Returns:
            Tuple of (predicted_count, uncertainty_std)
        """
        pass

    @abstractmethod
    def get_expected_progress(self, tau: int, is_weekend: bool = False) -> float:
        """
        Get expected fraction of day's activity completed by minute τ.

        This is used for regime adjustment calculations.

        Args:
            tau: Minutes since noon
            is_weekend: Whether it's a weekend contract-day

        Returns:
            Expected fraction (0 to 1)
        """
        pass

    def predict_samples(
        self,
        events: List[TweetEvent],
        contract_date: date,
        now: datetime,
        n_samples: int,
        rng: np.random.Generator,
        std_inflation_factor: float = 1.0,
    ) -> np.ndarray:
        """
        Draw n_samples of today's final count as discrete integers.

        Default implementation (used by Ridge path):
        - Calls predict() to get (mean, std)
        - Applies std_inflation_factor to widen variance
        - Moment-matches to NegBin (or Poisson if underdispersed)
        - Floors every sample at cum_so_far

        Subclasses (e.g. BucketIntradayForecaster) may override to
        produce samples directly from their internal bucket model.

        Args:
            events: Today's events so far
            contract_date: Today's contract date
            now: Current timestamp
            n_samples: Number of samples to draw
            rng: Caller-provided random number generator
            std_inflation_factor: Multiply variance by this factor (>=1)

        Returns:
            Integer array of shape (n_samples,)
        """
        mean, std = self.predict(events, contract_date, now)
        cum_so_far = len([e for e in events if e.timestamp < now])

        # Inflate std: var' = var * factor, so std' = std * sqrt(factor)
        if std_inflation_factor > 1.0:
            std = std * np.sqrt(std_inflation_factor)

        if mean <= 0 or std <= 0:
            return np.full(n_samples, max(int(round(mean)), cum_so_far), dtype=int)

        var = std ** 2

        if var > mean:
            # Overdispersed → Negative Binomial
            # Var = μ + μ²/k  →  k = μ² / (Var - μ)
            k = mean ** 2 / (var - mean)
            k = max(k, 0.1)
            p = k / (k + mean)
            samples = rng.negative_binomial(k, p, size=n_samples)
        else:
            # Underdispersed or equidispersed → Poisson
            samples = rng.poisson(mean, size=n_samples)

        # Floor at cum_so_far
        samples = np.maximum(samples, cum_so_far)
        return samples.astype(int)

    @property
    @abstractmethod
    def historical_mean(self) -> float:
        """Get historical daily mean."""
        pass

    @property
    @abstractmethod
    def historical_std(self) -> float:
        """Get historical daily std."""
        pass


class IntradayNowcast(BaseIntradayForecaster):
    """
    Ridge regression model to predict today's final tweet count.

    Original implementation using F(τ) progress curve and Ridge regression
    with burst features.
    """

    def __init__(
        self,
        config: NowcastConfig,
        progress_curve: IntradayProgressCurve,
        feature_extractor: BurstFeatureExtractor,
        contract_utils: ContractDayUtils,
    ):
        """
        Initialize nowcast model.

        Args:
            config: Nowcast configuration
            progress_curve: Fitted progress curve
            feature_extractor: Burst feature extractor
            contract_utils: Contract-day utilities
        """
        self.config = config
        self.progress_curve = progress_curve
        self.feature_extractor = feature_extractor
        self.contract_utils = contract_utils

        # Model
        self._model: Optional[Ridge] = None
        self._historical_mean: float = 50.0  # Default
        self._historical_std: float = 30.0  # Default
        self._residual_std: float = 30.0  # Default

    def fit(
        self,
        historical_events: Dict[date, List[TweetEvent]],
        historical_counts: Dict[date, int],
        as_of_date: Optional[date] = None,
    ) -> None:
        """
        Fit the nowcast model.

        Args:
            historical_events: Dict mapping contract_date -> events
            historical_counts: Dict mapping contract_date -> final count
            as_of_date: Reference date for "today" (for backtesting). Default: actual today.
        """
        X = []
        y = []
        weights = []

        # Use as_of_date for backtesting, otherwise use actual current date
        today = as_of_date if as_of_date is not None else self.contract_utils.get_current_contract_date()

        for contract_date, events in historical_events.items():
            final_count = historical_counts.get(contract_date, len(events))

            if final_count == 0:
                continue

            # Compute weight
            days_ago = (today - contract_date).days
            weight = np.exp(-days_ago / self.config.weight_half_life_days * np.log(2))

            # Generate training samples at various τ values
            # Sample at 25%, 50%, 75% of the day
            for frac in [0.25, 0.50, 0.75]:
                tau = int(1440 * frac)
                features = self._extract_features_at_tau(
                    events, contract_date, tau, final_count
                )
                if features is not None:
                    X.append(features)
                    y.append(final_count)
                    weights.append(weight)

        if len(X) < self.config.min_training_days:
            logger.warning(
                f"Not enough training samples ({len(X)}), "
                f"using fallback model"
            )
            self._historical_mean = np.mean(list(historical_counts.values())) if historical_counts else 50.0
            self._historical_std = np.std(list(historical_counts.values())) if len(historical_counts) > 1 else 30.0
            return

        X = np.array(X)
        y = np.array(y)
        weights = np.array(weights)

        # Fit Ridge regression
        self._model = Ridge(alpha=self.config.ridge_alpha)
        self._model.fit(X, y, sample_weight=weights)

        # Compute residual std
        y_pred = self._model.predict(X)
        residuals = y - y_pred
        self._residual_std = float(np.std(residuals))

        # Store historical stats
        self._historical_mean = float(np.mean(y))
        self._historical_std = float(np.std(y))

        logger.debug(
            f"Fitted nowcast model on {len(X)} samples, "
            f"residual_std={self._residual_std:.2f}"
        )

    def _extract_features_at_tau(
        self,
        events: List[TweetEvent],
        contract_date: date,
        tau: int,
        final_count: int,
    ) -> Optional[np.ndarray]:
        """Extract features at a specific τ for training."""
        # Simulate 'now' at this τ
        start_dt, _ = self.contract_utils.get_contract_day_bounds(contract_date)
        now = start_dt + timedelta(minutes=tau)

        # Filter events up to 'now'
        past_events = [e for e in events if e.timestamp < now]
        cum_so_far = len(past_events)

        # Get progress
        is_weekend = self.contract_utils.is_weekend(contract_date)
        F_tau = self.progress_curve.get_expected_progress(tau, is_weekend)

        # Compute implied rate (with threshold)
        if F_tau > self.config.implied_rate_threshold:
            implied_rate = cum_so_far / F_tau
        else:
            implied_rate = self._historical_mean

        # Extract burst features
        burst = self.feature_extractor.extract(past_events, now, tau)

        # Build feature vector
        features = [
            cum_so_far,
            tau,
            F_tau,
            implied_rate,
            burst.count_15m,
            burst.count_60m,
            burst.count_180m,
            burst.last_gap_min,
            1.0 if burst.in_session else 0.0,
            burst.max_burst_60m,
            1.0 if is_weekend else 0.0,
        ]

        return np.array(features)

    def predict(
        self,
        events: List[TweetEvent],
        contract_date: date,
        now: datetime,
    ) -> Tuple[float, float]:
        """
        Predict final count for today.

        Args:
            events: Today's events so far
            contract_date: Today's contract date
            now: Current timestamp

        Returns:
            Tuple of (predicted_count, uncertainty_std)
        """
        tau = self.contract_utils.get_tau(now, contract_date)
        cum_so_far = len([e for e in events if e.timestamp < now])

        # Get progress
        is_weekend = self.contract_utils.is_weekend(contract_date)
        F_tau = self.progress_curve.get_expected_progress(tau, is_weekend)

        # Compute implied rate
        if F_tau > self.config.implied_rate_threshold:
            implied_rate = cum_so_far / F_tau
        else:
            implied_rate = self._historical_mean

        # Extract burst features
        burst = self.feature_extractor.extract(events, now, tau)

        # Build feature vector
        features = np.array([
            cum_so_far,
            tau,
            F_tau,
            implied_rate,
            burst.count_15m,
            burst.count_60m,
            burst.count_180m,
            burst.last_gap_min,
            1.0 if burst.in_session else 0.0,
            burst.max_burst_60m,
            1.0 if is_weekend else 0.0,
        ]).reshape(1, -1)

        # Predict
        if self._model is not None:
            prediction = float(self._model.predict(features)[0])
        else:
            # Fallback: simple extrapolation
            if F_tau > 0.05:
                prediction = cum_so_far / F_tau
            else:
                prediction = self._historical_mean

        # Ensure prediction is at least cum_so_far
        prediction = max(prediction, cum_so_far)

        # Get uncertainty (scaled by remaining time)
        uncertainty = self._get_uncertainty(F_tau)

        return prediction, uncertainty

    def _get_uncertainty(self, F_tau: float = 0.0) -> float:
        """
        Get prediction uncertainty (std), scaled by remaining time.

        Uncertainty decays as more of the day is observed. The remaining
        variance is proportional to (1 - F_tau), so std scales as sqrt(1 - F_tau).

        At F_tau = 0 (start of day): full uncertainty
        At F_tau = 0.5 (midday): ~71% of full uncertainty
        At F_tau = 0.9 (90% done): ~32% of full uncertainty
        At F_tau = 0.97 (1.4h left): ~17% of full uncertainty

        Args:
            F_tau: Fractional progress through the day (0 to 1)

        Returns:
            Scaled uncertainty std
        """
        if self._model is not None and self._residual_std > 0:
            base_std = self._residual_std
        else:
            base_std = self._historical_std

        # Scale by remaining time fraction
        # Clip F_tau to [0, 0.99] to avoid zero std at end of day
        remaining_fraction = max(1.0 - F_tau, 0.01)
        return base_std * np.sqrt(remaining_fraction)

    def get_expected_progress(self, tau: int, is_weekend: bool = False) -> float:
        """
        Get expected fraction of day's activity completed by minute τ.

        Delegates to the progress curve.
        """
        return self.progress_curve.get_expected_progress(tau, is_weekend)

    @property
    def historical_mean(self) -> float:
        """Get historical daily mean."""
        return self._historical_mean

    @property
    def historical_std(self) -> float:
        """Get historical daily std."""
        return self._historical_std


@dataclass
class BucketDistribution:
    """Distribution parameters for a single time bucket."""

    bucket_idx: int
    start_tau: int  # Minutes since noon
    end_tau: int
    mean: float  # Historical mean count for this bucket
    std: float  # Historical std
    dispersion_k: float  # Negative Binomial dispersion parameter


@dataclass
class ImpulseOverrideWindow:
    """Time-of-day override for impulse rate_mult bounds."""

    name: str
    start: int   # tau start (minutes since noon ET)
    end: int     # tau end
    floor: Optional[float] = None
    ceiling: Optional[float] = None
    impulse_decay_halflife: Optional[float] = None  # Override excitation decay halflife (minutes)


@dataclass
class HistoricalSuffixProfile:
    """Observed intraday path for one historical contract day."""

    contract_date: date
    is_weekend: bool
    taus: np.ndarray
    final_count: int


@dataclass
class OvernightQuietProfile:
    """Learned overnight quiet-state priors and continuation tables."""

    tau_values: np.ndarray
    night_idle_prior: np.ndarray
    sleep_onset_success: np.ndarray
    sleep_onset_total: np.ndarray
    wake_success: np.ndarray
    wake_total: np.ndarray
    wake_relief_curve: np.ndarray
    prior_activity_edges: Tuple[float, float]


@dataclass
class OvernightFeatureSnapshot:
    """Runtime or historical snapshot of overnight activity state."""

    tau_now: int
    tau_bin_idx: int
    recent15: int
    recent60: int
    recent180: int
    silence_min: int
    last_session_tweets: int
    last_session_duration_min: int
    prior_activity_score: float
    prior_activity_bin: int
    silence_bin: int
    has_recent_activity: bool
    wake_session_tweets: int
    wake_session_bin: int
    recent15_bin: int
    has_wake_session: bool


@dataclass
class OvernightQuietState:
    """Soft overnight quiet-state diagnostics for the current tick."""

    state: str
    tau_now: int
    night_idle_prior: float
    sleep_onset_confidence: float
    wake_continuation_confidence: float
    quiet_strength: float
    regime_eff: float
    silence_min: int
    recent15: int
    recent60: int
    recent180: int
    prior_activity_score: float
    prior_activity_bin: int
    silence_bin: int
    wake_session_bin: Optional[int]
    idle_source: str
    onset_source: str
    wake_source: str


def _load_impulse_overrides(path: str) -> List[ImpulseOverrideWindow]:
    """Load impulse rate_mult overrides from YAML. Returns empty list if file not found."""
    import os
    if not os.path.exists(path):
        logger.warning(f"Impulse overrides file not found: {path}")
        return []
    try:
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        overrides = []
        for name, window in (data.get("overrides") or {}).items():
            overrides.append(ImpulseOverrideWindow(
                name=name,
                start=int(window["start"]),
                end=int(window["end"]),
                floor=window.get("floor"),
                ceiling=window.get("ceiling"),
                impulse_decay_halflife=window.get("impulse_decay_halflife"),
            ))
        logger.info(f"Loaded {len(overrides)} impulse override windows from {path}")
        return overrides
    except Exception as e:
        logger.error(f"Failed to load impulse overrides from {path}: {e}")
        return []


class BucketIntradayForecaster(BaseIntradayForecaster):
    """
    Bucket-based intraday forecaster.

    Divides the day into N buckets (default 8 = 3-hour windows) and learns
    the distribution of tweet counts for each bucket. At forecast time:

    1. Compute regime multiplier = observed / expected (clamped)
    2. Sample remaining buckets from Negative Binomial distributions
    3. Return: observed + sum(sampled remaining buckets)

    This approach respects intraday activity patterns (e.g., low activity
    during sleep hours, peak activity in evening).
    """

    def __init__(
        self,
        config: BucketNowcastConfig,
        contract_utils: ContractDayUtils,
    ):
        """
        Initialize bucket forecaster.

        Args:
            config: Bucket nowcast configuration
            contract_utils: Contract-day utilities
        """
        self.config = config
        self.contract_utils = contract_utils

        # Bucket parameters
        self.n_buckets = config.n_buckets
        self.bucket_size = 1440 // config.n_buckets  # Minutes per bucket

        # Learned distributions (separate for weekday/weekend)
        self._weekday_buckets: List[BucketDistribution] = []
        self._weekend_buckets: List[BucketDistribution] = []

        # Historical stats
        self._historical_mean: float = 50.0
        self._historical_std: float = 30.0

        # Bayesian impulse: kernel-smoothed rate curve λ(τ) at 1-min resolution
        self._rate_curve: Optional[np.ndarray] = None  # shape (1440,), tweets/min
        self._impulse_fitted: bool = False

        # Impulse overrides
        self._impulse_overrides: List[ImpulseOverrideWindow] = []

        # Last logged regime details for debugging/live diagnostics
        self._last_regime: Optional[Dict[str, float | int | None]] = None
        self._last_impulse: Optional[Dict[str, float | int | None]] = None
        self._last_bootstrap: Optional[Dict[str, float | int]] = None
        self._last_quiet: Optional[Dict[str, float | int | str | None]] = None

        # Historical suffix profiles for optional direct historical bootstrap
        self._historical_suffix_profiles: List[HistoricalSuffixProfile] = []
        self._weekday_overnight_profile: Optional[OvernightQuietProfile] = None
        self._weekend_overnight_profile: Optional[OvernightQuietProfile] = None

        # Fitted flag
        self._fitted = False

    def fit(
        self,
        historical_events: Dict[date, List[TweetEvent]],
        historical_counts: Dict[date, int],
        as_of_date: Optional[date] = None,
    ) -> None:
        """
        Fit bucket distributions from historical data.

        Args:
            historical_events: Dict mapping contract_date -> events
            historical_counts: Dict mapping contract_date -> final count
            as_of_date: Reference date for "today" (for backtesting). Default: actual today.
        """
        # Compute overall historical stats
        counts = list(historical_counts.values())
        if not counts:
            raise ValueError(
                "No historical counts provided to BucketIntradayForecaster.fit(). "
                "Cannot train without data."
            )
        self._historical_mean = float(np.mean(counts))
        self._historical_std = float(np.std(counts))

        # Separate weekday and weekend data
        weekday_bucket_counts: Dict[int, List[int]] = {i: [] for i in range(self.n_buckets)}
        weekend_bucket_counts: Dict[int, List[int]] = {i: [] for i in range(self.n_buckets)}

        # Use as_of_date for backtesting, otherwise use actual current date
        today = as_of_date if as_of_date is not None else self.contract_utils.get_current_contract_date()

        for contract_date, events in historical_events.items():
            # Skip if outside training window
            days_ago = (today - contract_date).days
            if days_ago > self.config.training_window_days:
                continue

            # Count tweets per bucket
            bucket_counts = self._count_events_per_bucket(events, contract_date)

            # Add to appropriate list
            is_weekend = self.contract_utils.is_weekend(contract_date)
            target = weekend_bucket_counts if is_weekend else weekday_bucket_counts

            for bucket_idx, count in enumerate(bucket_counts):
                target[bucket_idx].append(count)

        # Fit distributions for each bucket
        self._weekday_buckets = self._fit_bucket_distributions(weekday_bucket_counts)
        self._weekend_buckets = self._fit_bucket_distributions(weekend_bucket_counts)
        self._historical_suffix_profiles = self._build_historical_suffix_profiles(
            historical_events=historical_events,
            historical_counts=historical_counts,
            as_of_date=today,
        )
        (
            self._weekday_overnight_profile,
            self._weekend_overnight_profile,
        ) = self._build_overnight_quiet_profiles(
            historical_events=historical_events,
            historical_counts=historical_counts,
            as_of_date=today,
        )

        self._fitted = True

        logger.info(
            f"Fitted bucket forecaster: {self.n_buckets} buckets, "
            f"bucket_size={self.bucket_size}min, "
            f"weekday_days={len(weekday_bucket_counts[0])}, "
            f"weekend_days={len(weekend_bucket_counts[0])}"
        )

        # Fit impulse response from inter-tweet timing
        self._fit_impulse(historical_events, today)
        self._impulse_overrides = _load_impulse_overrides(self.config.impulse_overrides_path)

        # Log bucket means for debugging
        if self._weekday_buckets:
            means = [b.mean for b in self._weekday_buckets]
            logger.debug(f"Weekday bucket means: {[f'{m:.1f}' for m in means]}")
        if self._weekend_buckets:
            means = [b.mean for b in self._weekend_buckets]
            logger.debug(f"Weekend bucket means: {[f'{m:.1f}' for m in means]}")

    def _build_historical_suffix_profiles(
        self,
        historical_events: Dict[date, List[TweetEvent]],
        historical_counts: Dict[date, int],
        as_of_date: date,
    ) -> List[HistoricalSuffixProfile]:
        """Build per-day historical profiles for conditional suffix bootstrap."""
        profiles: List[HistoricalSuffixProfile] = []

        for contract_date, final_count in historical_counts.items():
            days_ago = (as_of_date - contract_date).days
            if days_ago > self.config.training_window_days:
                continue

            events = historical_events.get(contract_date, [])
            taus = sorted(
                self.contract_utils.get_tau(event.timestamp, contract_date)
                for event in events
            )
            profiles.append(
                HistoricalSuffixProfile(
                    contract_date=contract_date,
                    is_weekend=self.contract_utils.is_weekend(contract_date),
                    taus=np.asarray(taus, dtype=np.int16),
                    final_count=int(final_count),
                )
            )

        return profiles

    def _weighted_quantile(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        q: float,
    ) -> float:
        """Weighted quantile helper for learned activity-score bins."""
        if values.size == 0:
            return 0.0
        if values.size == 1:
            return float(values[0])

        order = np.argsort(values)
        sorted_values = values[order]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        total = float(cumulative[-1])
        if total <= 0:
            return float(np.quantile(sorted_values, q))
        target = q * total
        return float(np.interp(target, cumulative, sorted_values))

    def _tau_to_overnight_bin(self, tau_now: int) -> Optional[int]:
        if not self.config.use_overnight_quiet:
            return None
        if tau_now < self.config.overnight_quiet_start_tau or tau_now >= self.config.overnight_quiet_end_tau:
            return None
        return int((tau_now - self.config.overnight_quiet_start_tau) // 5)

    def _silence_bin(self, silence_min: int) -> int:
        if silence_min < 10:
            return 0
        if silence_min < 20:
            return 1
        if silence_min < 40:
            return 2
        if silence_min < 60:
            return 3
        if silence_min < 90:
            return 4
        return 5

    def _recent15_bin(self, recent15: int) -> int:
        if recent15 <= 0:
            return 0
        if recent15 == 1:
            return 1
        return 2

    def _wake_session_bin(self, wake_session_tweets: int) -> int:
        if wake_session_tweets <= 1:
            return 0
        if wake_session_tweets <= 3:
            return 1
        return 2

    def _prior_activity_bin(
        self,
        prior_activity_score: float,
        edges: Tuple[float, float],
    ) -> int:
        low_edge, high_edge = edges
        if prior_activity_score <= low_edge:
            return 0
        if prior_activity_score <= high_edge:
            return 1
        return 2

    def _sessionize_taus(
        self,
        taus: np.ndarray,
        tau_now: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Group past tweet taus into sessions as (start, end, count, gap_before)."""
        if taus.size == 0:
            return []

        hi = np.searchsorted(taus, tau_now, side="right")
        if hi <= 0:
            return []

        past = taus[:hi]
        gap_threshold = self.config.overnight_session_gap_minutes
        sessions: List[Tuple[int, int, int, int]] = []

        start = end = int(past[0])
        count = 1
        prev_tau = int(past[0])
        gap_before = start

        for raw_tau in past[1:]:
            tau_val = int(raw_tau)
            if tau_val - prev_tau > gap_threshold:
                sessions.append((start, end, count, gap_before))
                gap_before = tau_val - prev_tau
                start = end = tau_val
                count = 1
            else:
                end = tau_val
                count += 1
            prev_tau = tau_val

        sessions.append((start, end, count, gap_before))
        return sessions

    def _compute_prior_activity_score(
        self,
        recent180: int,
        last_session_tweets: int,
        last_session_duration_min: int,
    ) -> float:
        """Continuous recent-activity score used for learned prior-activity bins."""
        return (
            float(recent180)
            + 0.5 * float(last_session_tweets)
            + 0.1 * float(min(last_session_duration_min, 90))
        )

    def _compute_overnight_snapshot_from_taus(
        self,
        taus: np.ndarray,
        tau_now: int,
        edges: Tuple[float, float],
    ) -> OvernightFeatureSnapshot:
        recent15 = self._count_taus_in_window(taus, tau_now - 15, tau_now)
        recent60 = self._count_taus_in_window(taus, tau_now - 60, tau_now)
        recent180 = self._count_taus_in_window(taus, tau_now - 180, tau_now)
        silence_min = self._get_silence_minutes(taus, tau_now)

        sessions = self._sessionize_taus(taus, tau_now)
        active_session = bool(sessions and silence_min <= self.config.overnight_session_gap_minutes)
        last_completed = None
        if active_session and len(sessions) >= 2:
            last_completed = sessions[-2]
        elif not active_session and sessions:
            last_completed = sessions[-1]

        last_session_tweets = int(last_completed[2]) if last_completed else 0
        last_session_duration = int(max(1, last_completed[1] - last_completed[0])) if last_completed else 0

        prior_activity_score = self._compute_prior_activity_score(
            recent180=recent180,
            last_session_tweets=last_session_tweets,
            last_session_duration_min=last_session_duration,
        )
        prior_activity_bin = self._prior_activity_bin(prior_activity_score, edges)
        has_recent_activity = recent180 > 0 or last_session_tweets > 0

        wake_session_tweets = 0
        has_wake_session = False
        if sessions:
            last_session = sessions[-1]
            wake_gap_before = int(last_session[3])
            if wake_gap_before >= 20 and silence_min <= 15:
                wake_session_tweets = int(last_session[2])
                has_wake_session = True

        tau_bin_idx = self._tau_to_overnight_bin(tau_now)
        if tau_bin_idx is None:
            tau_bin_idx = -1

        return OvernightFeatureSnapshot(
            tau_now=tau_now,
            tau_bin_idx=tau_bin_idx,
            recent15=recent15,
            recent60=recent60,
            recent180=recent180,
            silence_min=silence_min,
            last_session_tweets=last_session_tweets,
            last_session_duration_min=last_session_duration,
            prior_activity_score=prior_activity_score,
            prior_activity_bin=prior_activity_bin,
            silence_bin=self._silence_bin(silence_min),
            has_recent_activity=has_recent_activity,
            wake_session_tweets=wake_session_tweets,
            wake_session_bin=self._wake_session_bin(wake_session_tweets),
            recent15_bin=self._recent15_bin(recent15),
            has_wake_session=has_wake_session,
        )

    def _build_overnight_quiet_profiles(
        self,
        historical_events: Dict[date, List[TweetEvent]],
        historical_counts: Dict[date, int],
        as_of_date: date,
    ) -> Tuple[Optional[OvernightQuietProfile], Optional[OvernightQuietProfile]]:
        """Build separate weekday/weekend overnight quiet profiles from historical tweets."""
        if not self.config.use_overnight_quiet:
            return None, None

        weekday_days: List[Tuple[np.ndarray, float]] = []
        weekend_days: List[Tuple[np.ndarray, float]] = []

        for contract_date, _final_count in historical_counts.items():
            days_ago = (as_of_date - contract_date).days
            if days_ago > self.config.training_window_days:
                continue

            events = historical_events.get(contract_date, [])
            taus = np.asarray(
                sorted(self.contract_utils.get_tau(event.timestamp, contract_date) for event in events),
                dtype=np.int16,
            )
            weight = float(np.exp(-days_ago / self.config.weight_half_life_days * np.log(2)))
            target = weekend_days if self.contract_utils.is_weekend(contract_date) else weekday_days
            target.append((taus, weight))

        return (
            self._fit_overnight_quiet_profile(weekday_days),
            self._fit_overnight_quiet_profile(weekend_days),
        )

    def _fit_overnight_quiet_profile(
        self,
        day_records: List[Tuple[np.ndarray, float]],
    ) -> Optional[OvernightQuietProfile]:
        """Fit one overnight quiet profile from historical day tau arrays."""
        if not day_records:
            return None

        start_tau = self.config.overnight_quiet_start_tau
        end_tau = self.config.overnight_quiet_end_tau
        tau_values = np.arange(start_tau, end_tau, 5, dtype=np.int16)
        n_tau = int(tau_values.size)
        alpha = self.config.overnight_profile_laplace_alpha
        min_samples = self.config.overnight_profile_min_weighted_samples

        training_records: List[Dict[str, float | int | bool]] = []
        prior_scores: List[float] = []
        prior_score_weights: List[float] = []
        idle_success = np.zeros(n_tau, dtype=float)
        idle_total = np.zeros(n_tau, dtype=float)
        wake_relief_counts = np.zeros(288, dtype=float)

        provisional_edges = (1.0, 3.0)
        for taus, weight in day_records:
            five_min_counts = np.zeros(288, dtype=float)
            for raw_tau in taus:
                idx = int(raw_tau) // 5
                if 0 <= idx < five_min_counts.size:
                    five_min_counts[idx] += 1.0
            wake_relief_counts += five_min_counts * weight

            for tau_now in tau_values:
                tau_int = int(tau_now)
                snapshot = self._compute_overnight_snapshot_from_taus(taus, tau_int, provisional_edges)
                idle_label = 1 if self._count_taus_in_window(
                    taus, tau_int, tau_int + self.config.overnight_idle_horizon_minutes,
                ) == 0 else 0
                onset_label = 1 if self._count_taus_in_window(
                    taus, tau_int, tau_int + self.config.overnight_sleep_onset_horizon_minutes,
                ) == 0 else 0
                wake_label = 1 if self._count_taus_in_window(
                    taus, tau_int, tau_int + self.config.overnight_wake_horizon_minutes,
                ) >= self.config.overnight_sustained_wake_min_tweets else 0

                tau_idx = snapshot.tau_bin_idx
                idle_success[tau_idx] += weight * idle_label
                idle_total[tau_idx] += weight

                record = {
                    "tau_idx": tau_idx,
                    "prior_activity_score": snapshot.prior_activity_score,
                    "silence_bin": snapshot.silence_bin,
                    "has_recent_activity": snapshot.has_recent_activity,
                    "wake_session_bin": snapshot.wake_session_bin,
                    "recent15_bin": snapshot.recent15_bin,
                    "has_wake_session": snapshot.has_wake_session,
                    "idle_label": idle_label,
                    "onset_label": onset_label,
                    "wake_label": wake_label,
                    "weight": weight,
                }
                training_records.append(record)

                if snapshot.has_recent_activity:
                    prior_scores.append(snapshot.prior_activity_score)
                    prior_score_weights.append(weight)

        if prior_scores:
            score_values = np.asarray(prior_scores, dtype=float)
            score_weights = np.asarray(prior_score_weights, dtype=float)
            low_edge = self._weighted_quantile(score_values, score_weights, 1.0 / 3.0)
            high_edge = self._weighted_quantile(score_values, score_weights, 2.0 / 3.0)
            if high_edge <= low_edge:
                high_edge = low_edge + 1e-3
            prior_edges = (float(low_edge), float(high_edge))
        else:
            prior_edges = provisional_edges

        onset_success = np.zeros((n_tau, 3, 6), dtype=float)
        onset_total = np.zeros((n_tau, 3, 6), dtype=float)
        wake_success = np.zeros((n_tau, 3, 3), dtype=float)
        wake_total = np.zeros((n_tau, 3, 3), dtype=float)

        for record in training_records:
            tau_idx = int(record["tau_idx"])
            weight = float(record["weight"])
            if bool(record["has_recent_activity"]):
                prior_bin = self._prior_activity_bin(float(record["prior_activity_score"]), prior_edges)
                silence_bin = int(record["silence_bin"])
                onset_success[tau_idx, prior_bin, silence_bin] += weight * float(record["onset_label"])
                onset_total[tau_idx, prior_bin, silence_bin] += weight

            if bool(record["has_wake_session"]):
                wake_bin = int(record["wake_session_bin"])
                recent15_bin = int(record["recent15_bin"])
                wake_success[tau_idx, wake_bin, recent15_bin] += weight * float(record["wake_label"])
                wake_total[tau_idx, wake_bin, recent15_bin] += weight

        global_idle = float((idle_success.sum() + alpha) / (idle_total.sum() + 2.0 * alpha))
        night_idle_prior = np.full(n_tau, global_idle, dtype=float)
        for idx in range(n_tau):
            total = float(idle_total[idx])
            if total <= 0:
                continue
            tau_prob = float((idle_success[idx] + alpha) / (total + 2.0 * alpha))
            if total < min_samples:
                shrink = max(0.0, min(1.0, total / min_samples))
                tau_prob = shrink * tau_prob + (1.0 - shrink) * global_idle
            night_idle_prior[idx] = tau_prob

        ref_start = 900 // 5   # 03:00 ET
        ref_end = 1200 // 5    # 08:00 ET
        ref_max = float(np.max(wake_relief_counts[ref_start:ref_end])) if ref_end > ref_start else 0.0
        if ref_max <= 0:
            wake_relief_5m = np.ones(288, dtype=float)
        else:
            wake_relief_5m = np.clip(wake_relief_counts / ref_max, 0.15, 1.0)
        wake_relief_5m[ref_end:] = 1.0
        wake_relief_curve = np.repeat(wake_relief_5m, 5)[:1440]

        return OvernightQuietProfile(
            tau_values=tau_values,
            night_idle_prior=night_idle_prior,
            sleep_onset_success=onset_success,
            sleep_onset_total=onset_total,
            wake_success=wake_success,
            wake_total=wake_total,
            wake_relief_curve=wake_relief_curve,
            prior_activity_edges=prior_edges,
        )

    def _get_overnight_profile(self, is_weekend: bool) -> Optional[OvernightQuietProfile]:
        return self._weekend_overnight_profile if is_weekend else self._weekday_overnight_profile

    def _estimate_binary_probability(
        self,
        success: float,
        total: float,
        fallback_prob: float,
        source_name: str,
        fallback_source: str,
    ) -> Tuple[float, str]:
        """Estimate Bernoulli probability with shrinkage toward a fallback prior."""
        alpha = self.config.overnight_profile_laplace_alpha
        min_samples = self.config.overnight_profile_min_weighted_samples
        mix = self.config.overnight_fallback_mix
        if total <= 0:
            return float(fallback_prob), fallback_source

        prob = float((success + alpha) / (total + 2.0 * alpha))
        if total < min_samples:
            shrink = max(0.0, min(1.0, total / min_samples))
            prob = shrink * prob + (1.0 - shrink) * fallback_prob
            return prob, f"{source_name}->backoff({fallback_source})"

        prob = (1.0 - mix) * prob + mix * fallback_prob
        return float(prob), source_name

    def _lookup_sleep_onset_probability(
        self,
        profile: OvernightQuietProfile,
        snapshot: OvernightFeatureSnapshot,
    ) -> Tuple[float, str]:
        if not snapshot.has_recent_activity:
            return 0.0, "no_recent_activity"

        tau_idx = snapshot.tau_bin_idx
        idle_prior = float(profile.night_idle_prior[tau_idx])

        prior_success = float(profile.sleep_onset_success[tau_idx, snapshot.prior_activity_bin, :].sum())
        prior_total = float(profile.sleep_onset_total[tau_idx, snapshot.prior_activity_bin, :].sum())
        prior_prob, prior_source = self._estimate_binary_probability(
            success=prior_success,
            total=prior_total,
            fallback_prob=idle_prior,
            source_name="tau+prior",
            fallback_source="idle_tau",
        )

        silence_success = float(profile.sleep_onset_success[tau_idx, :, snapshot.silence_bin].sum())
        silence_total = float(profile.sleep_onset_total[tau_idx, :, snapshot.silence_bin].sum())
        silence_prob, silence_source = self._estimate_binary_probability(
            success=silence_success,
            total=silence_total,
            fallback_prob=idle_prior,
            source_name="tau+silence",
            fallback_source="idle_tau",
        )

        combined_weight = prior_total + silence_total
        if combined_weight > 0:
            marginal_prob = ((prior_prob * prior_total) + (silence_prob * silence_total)) / combined_weight
            marginal_source = f"marginal({prior_source}|{silence_source})"
        else:
            marginal_prob = idle_prior
            marginal_source = "idle_tau"

        full_success = float(profile.sleep_onset_success[
            tau_idx, snapshot.prior_activity_bin, snapshot.silence_bin
        ])
        full_total = float(profile.sleep_onset_total[
            tau_idx, snapshot.prior_activity_bin, snapshot.silence_bin
        ])
        return self._estimate_binary_probability(
            success=full_success,
            total=full_total,
            fallback_prob=float(marginal_prob),
            source_name="tau+prior+silence",
            fallback_source=marginal_source,
        )

    def _lookup_wake_probability(
        self,
        profile: OvernightQuietProfile,
        snapshot: OvernightFeatureSnapshot,
    ) -> Tuple[float, str]:
        if not snapshot.has_wake_session:
            return 0.0, "no_wake_session"

        tau_idx = snapshot.tau_bin_idx
        base_prob = max(0.0, 1.0 - float(profile.night_idle_prior[tau_idx]))

        wake_success = float(profile.wake_success[tau_idx, snapshot.wake_session_bin, :].sum())
        wake_total = float(profile.wake_total[tau_idx, snapshot.wake_session_bin, :].sum())
        wake_prob, wake_source = self._estimate_binary_probability(
            success=wake_success,
            total=wake_total,
            fallback_prob=base_prob,
            source_name="tau+wake",
            fallback_source="1-idle_tau",
        )

        recent_success = float(profile.wake_success[tau_idx, :, snapshot.recent15_bin].sum())
        recent_total = float(profile.wake_total[tau_idx, :, snapshot.recent15_bin].sum())
        recent_prob, recent_source = self._estimate_binary_probability(
            success=recent_success,
            total=recent_total,
            fallback_prob=base_prob,
            source_name="tau+recent15",
            fallback_source="1-idle_tau",
        )

        combined_weight = wake_total + recent_total
        if combined_weight > 0:
            marginal_prob = ((wake_prob * wake_total) + (recent_prob * recent_total)) / combined_weight
            marginal_source = f"marginal({wake_source}|{recent_source})"
        else:
            marginal_prob = base_prob
            marginal_source = "1-idle_tau"

        full_success = float(profile.wake_success[
            tau_idx, snapshot.wake_session_bin, snapshot.recent15_bin
        ])
        full_total = float(profile.wake_total[
            tau_idx, snapshot.wake_session_bin, snapshot.recent15_bin
        ])
        return self._estimate_binary_probability(
            success=full_success,
            total=full_total,
            fallback_prob=float(marginal_prob),
            source_name="tau+wake+recent15",
            fallback_source=marginal_source,
        )

    def _compute_overnight_quiet_state_from_taus(
        self,
        taus: np.ndarray,
        tau_now: int,
        is_weekend: bool,
        regime: float = 1.0,
    ) -> OvernightQuietState:
        profile = self._get_overnight_profile(is_weekend)
        tau_idx = self._tau_to_overnight_bin(tau_now)
        if not self.config.use_overnight_quiet or profile is None or tau_idx is None:
            return OvernightQuietState(
                state="inactive",
                tau_now=tau_now,
                night_idle_prior=0.0,
                sleep_onset_confidence=0.0,
                wake_continuation_confidence=0.0,
                quiet_strength=0.0,
                regime_eff=float(regime),
                silence_min=self._get_silence_minutes(taus, tau_now),
                recent15=self._count_taus_in_window(taus, tau_now - 15, tau_now),
                recent60=self._count_taus_in_window(taus, tau_now - 60, tau_now),
                recent180=self._count_taus_in_window(taus, tau_now - 180, tau_now),
                prior_activity_score=0.0,
                prior_activity_bin=0,
                silence_bin=self._silence_bin(self._get_silence_minutes(taus, tau_now)),
                wake_session_bin=None,
                idle_source="inactive",
                onset_source="inactive",
                wake_source="inactive",
            )

        snapshot = self._compute_overnight_snapshot_from_taus(taus, tau_now, profile.prior_activity_edges)
        night_idle_prior = float(profile.night_idle_prior[tau_idx])
        sleep_onset_confidence, onset_source = self._lookup_sleep_onset_probability(profile, snapshot)
        wake_continuation_confidence, wake_source = self._lookup_wake_probability(profile, snapshot)
        inferred_quiet_strength = max(night_idle_prior, sleep_onset_confidence) * (1.0 - wake_continuation_confidence)
        inferred_quiet_strength = float(np.clip(inferred_quiet_strength, 0.0, 1.0))
        quiet_strength = (
            inferred_quiet_strength
            if self.config.use_overnight_quiet_runtime_mask
            else 0.0
        )
        regime_eff = (
            float(1.0 + (regime - 1.0) * (1.0 - sleep_onset_confidence))
            if self.config.use_overnight_quiet_regime_damping
            else float(regime)
        )

        if wake_continuation_confidence >= 0.5:
            state = "night_wake"
        elif inferred_quiet_strength >= 0.55 or sleep_onset_confidence >= 0.55:
            state = "overnight_quiet"
        elif snapshot.recent15 >= 2 or snapshot.recent60 >= 4:
            state = "awake_active"
        else:
            state = "awake_quiet"

        return OvernightQuietState(
            state=state,
            tau_now=tau_now,
            night_idle_prior=night_idle_prior,
            sleep_onset_confidence=float(np.clip(sleep_onset_confidence, 0.0, 1.0)),
            wake_continuation_confidence=float(np.clip(wake_continuation_confidence, 0.0, 1.0)),
            quiet_strength=quiet_strength,
            regime_eff=regime_eff,
            silence_min=snapshot.silence_min,
            recent15=snapshot.recent15,
            recent60=snapshot.recent60,
            recent180=snapshot.recent180,
            prior_activity_score=snapshot.prior_activity_score,
            prior_activity_bin=snapshot.prior_activity_bin,
            silence_bin=snapshot.silence_bin,
            wake_session_bin=snapshot.wake_session_bin if snapshot.has_wake_session else None,
            idle_source="idle_tau",
            onset_source=onset_source,
            wake_source=wake_source,
        )

    def _compute_overnight_quiet_state(
        self,
        events: List[TweetEvent],
        contract_date: date,
        now: datetime,
        regime: float = 1.0,
    ) -> OvernightQuietState:
        event_taus = np.asarray(
            sorted(
                self.contract_utils.get_tau(event.timestamp, contract_date)
                for event in events
                if event.timestamp < now
            ),
            dtype=np.int16,
        )
        tau_now = self.contract_utils.get_tau(now, contract_date)
        is_weekend = self.contract_utils.is_weekend(contract_date)
        return self._compute_overnight_quiet_state_from_taus(
            event_taus,
            tau_now=int(tau_now),
            is_weekend=is_weekend,
            regime=regime,
        )

    def _quiet_mask_at_tau(
        self,
        tau_value: int,
        profile: Optional[OvernightQuietProfile],
        quiet_state: Optional[OvernightQuietState],
    ) -> float:
        if quiet_state is None or quiet_state.quiet_strength <= 0.0 or profile is None:
            return 1.0
        tau_idx = max(0, min(int(tau_value), profile.wake_relief_curve.size - 1))
        wake_relief = float(profile.wake_relief_curve[tau_idx])
        return float(1.0 - quiet_state.quiet_strength * (1.0 - wake_relief))

    def _avg_quiet_mult_for_slice(
        self,
        tau_now: int,
        rel_start: float,
        rel_end: float,
        profile: Optional[OvernightQuietProfile],
        quiet_state: Optional[OvernightQuietState],
    ) -> float:
        if quiet_state is None or quiet_state.quiet_strength <= 0.0 or profile is None:
            return 1.0
        span = rel_end - rel_start
        if span <= 0:
            return 1.0
        start_min = int(math.floor(rel_start))
        end_min = int(math.ceil(rel_end))
        if end_min <= start_min:
            return self._quiet_mask_at_tau(int(tau_now + rel_start), profile, quiet_state)

        vals = [
            self._quiet_mask_at_tau(tau_now + minute, profile, quiet_state)
            for minute in range(start_min, end_min)
        ]
        return float(np.mean(vals)) if vals else 1.0

    def _count_events_per_bucket(
        self,
        events: List[TweetEvent],
        contract_date: date,
    ) -> List[int]:
        """Count events in each bucket for a given day."""
        counts = [0] * self.n_buckets

        for event in events:
            tau = self.contract_utils.get_tau(event.timestamp, contract_date)
            bucket_idx = min(tau // self.bucket_size, self.n_buckets - 1)
            counts[bucket_idx] += 1

        return counts

    def _fit_bucket_distributions(
        self,
        bucket_counts: Dict[int, List[int]],
    ) -> List[BucketDistribution]:
        """Fit Negative Binomial distribution for each bucket."""
        distributions = []

        for bucket_idx in range(self.n_buckets):
            counts = bucket_counts[bucket_idx]

            if not counts:
                raise ValueError(
                    f"No data for bucket {bucket_idx} (τ={bucket_idx * self.bucket_size}-"
                    f"{(bucket_idx + 1) * self.bucket_size}). "
                    f"Insufficient training data - need more historical days."
                )

            else:
                mean = float(np.mean(counts))
                std = float(np.std(counts))
                var = std ** 2

                # Fit Negative Binomial dispersion k
                # Var = μ + μ²/k  →  k = μ² / (Var - μ)
                if var > mean and mean > 0:
                    k = mean ** 2 / (var - mean)
                    k = max(k, self.config.min_dispersion_k)
                else:
                    # Underdispersed or zero mean - use high k (approaches Poisson)
                    k = 100.0

                dist = BucketDistribution(
                    bucket_idx=bucket_idx,
                    start_tau=bucket_idx * self.bucket_size,
                    end_tau=(bucket_idx + 1) * self.bucket_size,
                    mean=mean,
                    std=std,
                    dispersion_k=k,
                )

            distributions.append(dist)

        return distributions

    def _fit_impulse(
        self,
        historical_events: Dict[date, List[TweetEvent]],
        as_of_date: date,
    ) -> None:
        """
        Build kernel-smoothed rate curve λ(τ) at 1-min resolution.

        For each minute τ ∈ [0, 1440), compute the recency-weighted average
        tweet rate across training days, then smooth with a Gaussian kernel.
        """
        from scipy.ndimage import gaussian_filter1d

        # Collect per-day 1-min count arrays with recency weights
        day_counts: List[np.ndarray] = []
        day_weights: List[float] = []
        total_tweets = 0

        for contract_date, events in historical_events.items():
            days_ago = (as_of_date - contract_date).days
            if days_ago > self.config.training_window_days:
                continue

            # 1-min resolution count array for this day
            counts = np.zeros(1440)
            for e in events:
                tau = self.contract_utils.get_tau(e.timestamp, contract_date)
                tau = max(0, min(tau, 1439))
                counts[tau] += 1
                total_tweets += 1

            # Recency weight (same half_life as bucket fitting)
            weight = np.exp(-days_ago / self.config.weight_half_life_days * np.log(2))
            day_counts.append(counts)
            day_weights.append(weight)

        if total_tweets < self.config.impulse_min_tweets_for_fit:
            logger.warning(
                f"Insufficient tweets for rate curve ({total_tweets} < "
                f"{self.config.impulse_min_tweets_for_fit}). Impulse disabled."
            )
            self._impulse_fitted = False
            return

        # Weighted average across days → raw rate λ_raw(τ)
        weights_arr = np.array(day_weights)
        weights_arr /= weights_arr.sum()
        raw_rate = np.zeros(1440)
        for counts, w in zip(day_counts, weights_arr):
            raw_rate += counts * w

        # Smooth with Gaussian kernel (σ = impulse_rate_curve_sigma minutes)
        sigma = self.config.impulse_rate_curve_sigma
        self._rate_curve = gaussian_filter1d(raw_rate, sigma=sigma, mode="wrap")

        # Floor at small epsilon to avoid division by zero
        self._rate_curve = np.maximum(self._rate_curve, 1e-6)
        self._impulse_fitted = True

        total_rate = float(self._rate_curve.sum())
        peak_tau = int(np.argmax(self._rate_curve))
        peak_rate = float(self._rate_curve[peak_tau])
        logger.info(
            f"Rate curve fitted: total={total_rate:.1f} tweets/day, "
            f"peak={peak_rate:.4f} tweets/min at τ={peak_tau}, "
            f"σ={sigma:.0f}min (from {total_tweets} tweets, {len(day_counts)} days)"
        )

    def _get_last_tweet_timestamp(
        self,
        events: List[TweetEvent],
        now: datetime,
    ) -> Optional[datetime]:
        """Get timestamp of the most recent tweet before now, or None."""
        past_events = [e for e in events if e.timestamp < now]
        if not past_events:
            return None
        last = max(past_events, key=lambda e: e.timestamp)
        return last.timestamp

    def predict(
        self,
        events: List[TweetEvent],
        contract_date: date,
        now: datetime,
        settlement_tau: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Predict final count using impulse response + bucket Monte Carlo.

        Args:
            events: Today's events so far
            contract_date: Today's contract date
            now: Current timestamp
            settlement_tau: Optional τ of settlement (caps forecast window)

        Returns:
            Tuple of (predicted_count, uncertainty_std)
        """
        if not self._fitted:
            raise RuntimeError(
                "BucketIntradayForecaster.predict() called before fit(). "
                "Must call fit() with historical data first."
            )

        n_simulations = 1000
        samples = self._sample_with_impulse(
            events=events,
            contract_date=contract_date,
            now=now,
            n_simulations=n_simulations,
            settlement_tau=settlement_tau,
        )

        mean = float(np.mean(samples))
        std = float(np.std(samples))

        # Ensure prediction is at least observed
        observed = len([e for e in events if e.timestamp < now])
        mean = max(mean, observed)

        return mean, std

    def predict_samples(
        self,
        events: List[TweetEvent],
        contract_date: date,
        now: datetime,
        n_samples: int,
        rng: np.random.Generator,
        std_inflation_factor: float = 1.0,
        settlement_tau: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draw n_samples of today's final count using impulse + bucket model.

        Returns the raw discrete samples.
        """
        if not self._fitted:
            raise RuntimeError(
                "BucketIntradayForecaster.predict_samples() called before fit()."
            )

        return self._sample_with_impulse(
            events=events,
            contract_date=contract_date,
            now=now,
            n_simulations=n_samples,
            rng=rng,
            std_inflation_factor=std_inflation_factor,
            settlement_tau=settlement_tau,
        ).astype(int)

    def _avg_rate_mult_for_slice(
        self,
        rate_mult: float,
        forward_decay: float,
        rel_start: float,
        rel_end: float,
    ) -> float:
        """Compute average rate_mult over a time slice [rel_start, rel_end] from now.

        Uses analytical integral of 1 + (rate_mult - 1) * exp(-fd * t) over the slice.
        Returns 1.0 if slice is beyond effective range or has zero span.
        """
        span = rel_end - rel_start
        if span <= 0:
            return 1.0
        delta = rate_mult - 1.0
        if abs(delta) < 1e-6 or forward_decay <= 0:
            return 1.0
        integral = (delta / forward_decay) * (
            math.exp(-forward_decay * rel_start) - math.exp(-forward_decay * rel_end)
        )
        return 1.0 + integral / span

    def _count_taus_in_window(self, taus: np.ndarray, start_tau: int, end_tau: int) -> int:
        """Count tweets with start_tau < tau <= end_tau."""
        if taus.size == 0 or end_tau <= start_tau:
            return 0
        lo = np.searchsorted(taus, start_tau, side="right")
        hi = np.searchsorted(taus, end_tau, side="right")
        return int(max(0, hi - lo))

    def _get_silence_minutes(self, taus: np.ndarray, tau_now: int) -> int:
        """Minutes since last tweet before tau_now, or tau_now if none."""
        if taus.size == 0:
            return tau_now
        idx = np.searchsorted(taus, tau_now, side="left") - 1
        if idx < 0:
            return tau_now
        return int(max(0, tau_now - int(taus[idx])))

    def _gaussian_weight(self, diff: float, sigma: float) -> float:
        """Gaussian similarity weight with defensive sigma handling."""
        if sigma <= 0:
            return 1.0 if abs(diff) < 1e-9 else 0.0
        z = diff / sigma
        return float(math.exp(-0.5 * z * z))

    def _compute_historical_bootstrap_alpha(self, hours_left: float, n_eff: float) -> float:
        """Blend weight for historical bootstrap based on time left and sample size."""
        if not self.config.use_historical_bootstrap or hours_left > self.config.bootstrap_start_hours:
            return 0.0

        if self.config.bootstrap_start_hours <= self.config.bootstrap_full_hours:
            time_alpha = 1.0
        else:
            span = self.config.bootstrap_start_hours - self.config.bootstrap_full_hours
            time_alpha = (self.config.bootstrap_start_hours - hours_left) / span
            time_alpha = float(np.clip(time_alpha, 0.0, 1.0))

        if self.config.bootstrap_full_effective_n <= self.config.bootstrap_min_effective_n:
            sample_alpha = 1.0
        else:
            sample_alpha = (
                (n_eff - self.config.bootstrap_min_effective_n)
                / (self.config.bootstrap_full_effective_n - self.config.bootstrap_min_effective_n)
            )
            sample_alpha = float(np.clip(sample_alpha, 0.0, 1.0))

        return float(self.config.bootstrap_max_blend * time_alpha * sample_alpha)

    def _sample_historical_bootstrap_suffix(
        self,
        observed: int,
        event_taus: np.ndarray,
        tau_now: int,
        end_tau: int,
        is_weekend: bool,
        expected_so_far: float,
        n_simulations: int,
        rng: np.random.Generator,
        current_quiet_state: Optional[OvernightQuietState] = None,
    ) -> Optional[np.ndarray]:
        """Sample remaining final counts from weighted historical suffix analogs."""
        hours_left = max(0.0, end_tau - tau_now) / 60.0
        self._last_bootstrap = None

        if not self.config.use_historical_bootstrap:
            return None
        if hours_left > self.config.bootstrap_start_hours:
            return None
        if not self._historical_suffix_profiles:
            return None

        current_regime = observed / expected_so_far if expected_so_far >= self.config.min_expected_for_regime else 1.0
        current_regime = max(current_regime, 1e-3)
        current_recent60 = self._count_taus_in_window(event_taus, tau_now - 60, tau_now)
        current_recent180 = self._count_taus_in_window(event_taus, tau_now - 180, tau_now)
        current_silence = self._get_silence_minutes(event_taus, tau_now)

        remaining_values: List[int] = []
        weights: List[float] = []

        for profile in self._historical_suffix_profiles:
            if profile.is_weekend != is_weekend:
                continue

            hist_observed = int(np.searchsorted(profile.taus, tau_now, side="right"))
            hist_end_count = int(np.searchsorted(profile.taus, end_tau, side="right"))
            hist_remaining = max(0, hist_end_count - hist_observed)
            hist_recent60 = self._count_taus_in_window(profile.taus, tau_now - 60, tau_now)
            hist_recent180 = self._count_taus_in_window(profile.taus, tau_now - 180, tau_now)
            hist_silence = self._get_silence_minutes(profile.taus, tau_now)
            hist_regime = hist_observed / expected_so_far if expected_so_far >= self.config.min_expected_for_regime else 1.0
            hist_regime = max(hist_regime, 1e-3)

            weight = 1.0
            weight *= self._gaussian_weight(
                math.log(current_regime) - math.log(hist_regime),
                self.config.bootstrap_regime_sigma,
            )
            weight *= self._gaussian_weight(
                current_recent60 - hist_recent60,
                self.config.bootstrap_recent60_sigma,
            )
            weight *= self._gaussian_weight(
                current_recent180 - hist_recent180,
                self.config.bootstrap_recent180_sigma,
            )
            weight *= self._gaussian_weight(
                current_silence - hist_silence,
                self.config.bootstrap_silence_sigma_minutes,
            )
            if (
                current_quiet_state is not None
                and self.config.use_overnight_quiet_bootstrap_matching
            ):
                hist_quiet_state = self._compute_overnight_quiet_state_from_taus(
                    taus=profile.taus,
                    tau_now=tau_now,
                    is_weekend=is_weekend,
                    regime=1.0,
                )
                weight *= self._gaussian_weight(
                    current_quiet_state.sleep_onset_confidence - hist_quiet_state.sleep_onset_confidence,
                    self.config.bootstrap_quiet_sigma,
                )
                if (
                    current_quiet_state.wake_continuation_confidence > 0.0
                    or hist_quiet_state.wake_continuation_confidence > 0.0
                ):
                    weight *= self._gaussian_weight(
                        current_quiet_state.wake_continuation_confidence - hist_quiet_state.wake_continuation_confidence,
                        self.config.bootstrap_quiet_sigma,
                    )

            if weight <= 0.0:
                continue

            remaining_values.append(hist_remaining)
            weights.append(weight)

        if not weights:
            return None

        weights_arr = np.asarray(weights, dtype=float)
        weights_sum = float(weights_arr.sum())
        if weights_sum <= 0:
            return None
        probs = weights_arr / weights_sum
        n_eff = float((weights_sum ** 2) / np.square(weights_arr).sum())
        alpha = self._compute_historical_bootstrap_alpha(hours_left, n_eff)

        remaining_arr = np.asarray(remaining_values, dtype=int)
        weighted_mean = float(np.dot(probs, remaining_arr))
        weighted_var = float(np.dot(probs, np.square(remaining_arr - weighted_mean)))

        self._last_bootstrap = {
            "hours_left": round(hours_left, 2),
            "n_hist": int(len(remaining_values)),
            "n_eff": round(n_eff, 2),
            "alpha": round(alpha, 3),
            "remaining_mean": round(weighted_mean, 2),
            "remaining_std": round(math.sqrt(max(weighted_var, 0.0)), 2),
        }

        if alpha <= 0.0:
            return None

        sampled_remaining = rng.choice(remaining_arr, size=n_simulations, replace=True, p=probs)
        return observed + sampled_remaining

    def _sample_with_impulse(
        self,
        events: List[TweetEvent],
        contract_date: date,
        now: datetime,
        n_simulations: int,
        rng: Optional[np.random.Generator] = None,
        std_inflation_factor: float = 1.0,
        settlement_tau: Optional[int] = None,
    ) -> np.ndarray:
        """
        Core sampling using Hawkes excitation impulse + bucket model.

        Shifted-linear excitation model:
          1. Compute actual excitation (Hawkes decay sum of recent tweets)
          2. Compute expected excitation from rate curve lookback
          3. shifted = excitation - expected * neutral_fraction
          4. rate_mult = clamp(1 + gain * shifted, floor, ceiling)

        Bucket scaling (replaces impulse window):
          - Buckets run from tau_now to end_tau (no more impulse/bucket handoff)
          - Each bucket mean is multiplied by avg_rate_mult for that time slice
          - rate_mult decays toward 1.0 with asymmetric forward decay
          - All sampling uses NegBin via buckets (no more Poisson from rate curve)
        """
        if rng is None:
            rng = np.random.default_rng()

        tau = self.contract_utils.get_tau(now, contract_date)
        is_weekend = self.contract_utils.is_weekend(contract_date)
        buckets = self._weekend_buckets if is_weekend else self._weekday_buckets

        if not buckets:
            raise RuntimeError(
                f"No bucket distributions for {'weekend' if is_weekend else 'weekday'}."
            )

        event_taus = np.asarray(
            sorted(
                self.contract_utils.get_tau(event.timestamp, contract_date)
                for event in events
                if event.timestamp < now
            ),
            dtype=np.int16,
        )
        observed = int(event_taus.size)
        end_tau = min(settlement_tau, 1440) if settlement_tau is not None else 1440

        samples = np.full(n_simulations, float(observed))

        # --- Hawkes excitation with shifted-linear model ---
        use_impulse = (
            self._impulse_fitted
            and self._rate_curve is not None
            and tau > 10
        )

        rate_mult = 1.0
        forward_decay = 0.0

        if use_impulse:
            tau_now = int(tau)
            halflife = self.config.impulse_decay_halflife_minutes

            # Check if a time-of-day override changes the decay halflife
            active_override_name = None
            active_override = None
            for ov in self._impulse_overrides:
                if ov.start <= tau_now < ov.end:
                    active_override = ov
                    active_override_name = ov.name
                    if ov.impulse_decay_halflife is not None:
                        halflife = ov.impulse_decay_halflife
                    break

            decay = math.log(2) / halflife

            # Actual excitation: each tweet adds a decaying boost
            excitation = 0.0
            for e in events:
                if e.timestamp < now:
                    age_min = self.contract_utils.minutes_between(e.timestamp, now)
                    if age_min > 0:
                        excitation += math.exp(-decay * age_min)

            # Expected excitation from rate curve lookback
            lookback = self.config.impulse_lookback_minutes
            expected_excitation = 0.0
            for t in range(1, lookback + 1):
                past_time = now - timedelta(minutes=t)
                past_tau = self.contract_utils.get_tau(past_time, contract_date)
                if 0 <= past_tau < 1440:
                    expected_excitation += self._rate_curve[past_tau] * math.exp(-decay * t)

            # Shifted-linear mapping
            min_expected = self.config.impulse_min_expected_excitation
            if expected_excitation < min_expected:
                if excitation > min_expected:
                    # Use floor for neutral baseline
                    neutral = min_expected * self.config.impulse_neutral_fraction
                else:
                    # Both near zero → no signal
                    neutral = excitation  # forces shifted=0, rate_mult=1.0
            else:
                neutral = expected_excitation * self.config.impulse_neutral_fraction

            shifted = excitation - neutral
            rate_mult = 1.0 + self.config.impulse_gain * shifted
            rate_mult = max(self.config.impulse_floor, min(self.config.impulse_ceiling, rate_mult))

            # Apply time-of-day floor/ceiling overrides
            if active_override is not None:
                ov_floor = active_override.floor if active_override.floor is not None else self.config.impulse_floor
                ov_ceiling = active_override.ceiling if active_override.ceiling is not None else self.config.impulse_ceiling
                rate_mult = max(ov_floor, min(ov_ceiling, rate_mult))

            # Asymmetric forward decay
            if rate_mult >= 1.0:
                forward_decay = decay  # boost: halflife matches current decay
            else:
                forward_decay = math.log(2) / self.config.impulse_silence_halflife_minutes  # 90 min

            # Logging
            last_tweet_ts = self._get_last_tweet_timestamp(events, now)
            if last_tweet_ts is not None:
                silence_min = self.contract_utils.minutes_between(last_tweet_ts, now)
            else:
                start_dt, _ = self.contract_utils.get_contract_day_bounds(contract_date)
                silence_min = self.contract_utils.minutes_between(start_dt, now)

            self._last_impulse = {
                "silence_min": silence_min,
                "excitation": round(excitation, 2),
                "expected": round(expected_excitation, 2),
                "shifted": round(shifted, 2),
                "rate_mult_now": round(rate_mult, 2),
                "override": active_override_name,
            }
        else:
            self._last_impulse = None

        # --- Bucket component (from tau_now to end_tau, scaled by rate_mult) ---
        tau_now = int(tau)
        quiet_profile = self._get_overnight_profile(is_weekend)
        quiet_state: Optional[OvernightQuietState] = None
        if tau_now < end_tau:
            # Compute regime from observed vs expected at tau_now
            full_bucket_idx = min(tau // self.bucket_size, self.n_buckets - 1)
            full_partial = (tau % self.bucket_size) / self.bucket_size
            expected_so_far = sum(b.mean for b in buckets[:full_bucket_idx])
            expected_so_far += buckets[full_bucket_idx].mean * full_partial

            if expected_so_far >= self.config.min_expected_for_regime:
                raw_regime = observed / expected_so_far
                regime = np.clip(raw_regime, self.config.regime_min, self.config.regime_max)
            else:
                raw_regime = None
                regime = 1.0

            quiet_state = self._compute_overnight_quiet_state_from_taus(
                taus=event_taus,
                tau_now=tau_now,
                is_weekend=is_weekend,
                regime=float(regime),
            )

            self._last_regime = {
                "tau_now": tau_now,
                "bucket_idx": int(full_bucket_idx),
                "observed": int(observed),
                "expected_so_far": round(expected_so_far, 2),
                "raw_regime": round(raw_regime, 3) if raw_regime is not None else None,
                "regime": round(float(regime), 3),
                "min_expected": self.config.min_expected_for_regime,
            }
            self._last_quiet = {
                "state": quiet_state.state,
                "night_idle_prior": round(quiet_state.night_idle_prior, 3),
                "sleep_onset_confidence": round(quiet_state.sleep_onset_confidence, 3),
                "wake_continuation_confidence": round(quiet_state.wake_continuation_confidence, 3),
                "quiet_strength": round(quiet_state.quiet_strength, 3),
                "regime_eff": round(quiet_state.regime_eff, 3),
                "silence_min": quiet_state.silence_min,
                "recent15": quiet_state.recent15,
                "recent60": quiet_state.recent60,
                "recent180": quiet_state.recent180,
                "prior_activity_score": round(quiet_state.prior_activity_score, 2),
                "prior_activity_bin": quiet_state.prior_activity_bin,
                "silence_bin": quiet_state.silence_bin,
                "wake_session_bin": quiet_state.wake_session_bin,
                "idle_source": quiet_state.idle_source,
                "onset_source": quiet_state.onset_source,
                "wake_source": quiet_state.wake_source,
            }

            # Determine which bucket tau_now falls in
            bucket_start_idx = min(tau_now // self.bucket_size, self.n_buckets - 1)
            cutoff_minutes = self.config.impulse_cutoff_minutes

            # Sample remaining portion of the bucket containing tau_now
            b = buckets[bucket_start_idx]
            bucket_end = min(b.end_tau, end_tau)
            fraction = (bucket_end - tau_now) / self.bucket_size
            if fraction > 0:
                # Compute avg rate_mult for this slice
                rel_start = 0.0
                rel_end = float(bucket_end - tau_now)
                if use_impulse and rel_start < cutoff_minutes:
                    avg_mult = self._avg_rate_mult_for_slice(
                        rate_mult, forward_decay,
                        rel_start, min(rel_end, cutoff_minutes),
                    )
                    if rel_end > cutoff_minutes:
                        # Blend: part under impulse, part at 1.0
                        impulse_span = cutoff_minutes - rel_start
                        rest_span = rel_end - cutoff_minutes
                        avg_mult = (avg_mult * impulse_span + 1.0 * rest_span) / (rel_end - rel_start)
                else:
                    avg_mult = 1.0
                quiet_avg_mult = self._avg_quiet_mult_for_slice(
                    tau_now=tau_now,
                    rel_start=rel_start,
                    rel_end=rel_end,
                    profile=quiet_profile,
                    quiet_state=quiet_state,
                )

                remaining_mean = b.mean * fraction * quiet_state.regime_eff * avg_mult * quiet_avg_mult
                k = b.dispersion_k
                if std_inflation_factor > 1.0:
                    k = k / std_inflation_factor
                if remaining_mean > 0:
                    samples += self._sample_negative_binomial(
                        mean=remaining_mean, k=k, size=n_simulations, rng=rng,
                    )

            # Sample full remaining buckets
            for bucket_idx in range(bucket_start_idx + 1, self.n_buckets):
                b = buckets[bucket_idx]
                if b.start_tau >= end_tau:
                    break
                bucket_end = min(b.end_tau, end_tau)
                fraction = (bucket_end - b.start_tau) / self.bucket_size

                # Compute avg rate_mult for this bucket slice
                rel_start = float(b.start_tau - tau_now)
                rel_end = float(bucket_end - tau_now)
                if use_impulse and rel_start < cutoff_minutes:
                    avg_mult = self._avg_rate_mult_for_slice(
                        rate_mult, forward_decay,
                        rel_start, min(rel_end, cutoff_minutes),
                    )
                    if rel_end > cutoff_minutes:
                        impulse_span = cutoff_minutes - rel_start
                        rest_span = rel_end - cutoff_minutes
                        avg_mult = (avg_mult * impulse_span + 1.0 * rest_span) / (rel_end - rel_start)
                else:
                    avg_mult = 1.0
                quiet_avg_mult = self._avg_quiet_mult_for_slice(
                    tau_now=tau_now,
                    rel_start=rel_start,
                    rel_end=rel_end,
                    profile=quiet_profile,
                    quiet_state=quiet_state,
                )

                bucket_mean = b.mean * fraction * quiet_state.regime_eff * avg_mult * quiet_avg_mult
                k = b.dispersion_k
                if std_inflation_factor > 1.0:
                    k = k / std_inflation_factor
                if bucket_mean > 0:
                    samples += self._sample_negative_binomial(
                        mean=bucket_mean, k=k, size=n_simulations, rng=rng,
                    )

            bootstrap_samples = self._sample_historical_bootstrap_suffix(
                observed=observed,
                event_taus=event_taus,
                tau_now=tau_now,
                end_tau=end_tau,
                is_weekend=is_weekend,
                expected_so_far=expected_so_far,
                n_simulations=n_simulations,
                rng=rng,
                current_quiet_state=quiet_state,
            )
            if bootstrap_samples is not None and self._last_bootstrap:
                alpha = self._last_bootstrap["alpha"]
                use_bootstrap = rng.random(n_simulations) < alpha
                samples = np.where(use_bootstrap, bootstrap_samples, samples)
        else:
            self._last_regime = None
            self._last_bootstrap = None
            self._last_quiet = None

        return samples

    def _sample_negative_binomial(
        self,
        mean: float,
        k: float,
        size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample using configured distribution (negbin, com_poisson, or negbin_reflected)."""
        from .distributions import sample_negbin, sample_com_poisson, sample_negbin_reflected

        dist = self.config.bucket_distribution
        if dist == "com_poisson":
            return sample_com_poisson(mean, k, size, rng, nu_scale=self.config.cmp_nu_scale)
        elif dist == "negbin_reflected":
            return sample_negbin_reflected(mean, k, size, rng)
        else:
            return sample_negbin(mean, k, size, rng)

    def get_expected_progress(self, tau: int, is_weekend: bool = False) -> float:
        """
        Get expected fraction of day's activity completed by minute τ.

        Computed from bucket means: sum of completed bucket means / total daily mean.
        """
        if not self._fitted:
            raise RuntimeError(
                "BucketIntradayForecaster.get_expected_progress() called before fit(). "
                "Must call fit() with historical data first."
            )

        buckets = self._weekend_buckets if is_weekend else self._weekday_buckets

        if not buckets:
            raise RuntimeError(
                f"No bucket distributions available for {'weekend' if is_weekend else 'weekday'}. "
                f"fit() may have failed or data was insufficient."
            )

        # Total expected daily count
        total_mean = sum(b.mean for b in buckets)
        if total_mean <= 0:
            raise RuntimeError(
                f"Total bucket mean is {total_mean} (expected > 0). "
                f"fit() may have failed or data was invalid."
            )

        # Compute cumulative expected by tau
        current_bucket_idx = min(tau // self.bucket_size, self.n_buckets - 1)
        partial_fraction = (tau % self.bucket_size) / self.bucket_size

        expected_so_far = sum(b.mean for b in buckets[:current_bucket_idx])
        expected_so_far += buckets[current_bucket_idx].mean * partial_fraction

        return min(expected_so_far / total_mean, 1.0)

    def get_bucket_stats(self, is_weekend: bool = False) -> List[Dict]:
        """Get bucket statistics for debugging/display."""
        buckets = self._weekend_buckets if is_weekend else self._weekday_buckets
        return [
            {
                "bucket": b.bucket_idx,
                "start_tau": b.start_tau,
                "end_tau": b.end_tau,
                "mean": b.mean,
                "std": b.std,
                "k": b.dispersion_k,
            }
            for b in buckets
        ]

    @property
    def historical_mean(self) -> float:
        """Get historical daily mean."""
        return self._historical_mean

    @property
    def historical_std(self) -> float:
        """Get historical daily std."""
        return self._historical_std
