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
from datetime import datetime, date, timedelta, timezone
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
class DecaySegment:
    """One segment of a piecewise decay schedule with uniform parameters."""

    start_tau: int
    end_tau: int
    halflife: float
    decay: float
    floor: float
    ceiling: float


def _validate_impulse_overrides(
    overrides: List[ImpulseOverrideWindow],
) -> List[ImpulseOverrideWindow]:
    """Normalize, sort, and validate impulse override windows."""
    validated: List[ImpulseOverrideWindow] = []

    for override in sorted(overrides, key=lambda ov: (ov.start, ov.end, ov.name)):
        start = int(override.start)
        end = int(override.end)
        floor = None if override.floor is None else float(override.floor)
        ceiling = None if override.ceiling is None else float(override.ceiling)
        halflife = (
            None
            if override.impulse_decay_halflife is None
            else float(override.impulse_decay_halflife)
        )

        if start < 0 or end > 1440:
            raise ValueError(
                f"Impulse override '{override.name}' must stay within [0, 1440], got {start}-{end}"
            )
        if start >= end:
            raise ValueError(
                f"Impulse override '{override.name}' must satisfy start < end, got {start}-{end}"
            )
        if floor is not None and floor < 0:
            raise ValueError(
                f"Impulse override '{override.name}' floor must be >= 0, got {floor}"
            )
        if ceiling is not None and ceiling <= 0:
            raise ValueError(
                f"Impulse override '{override.name}' ceiling must be > 0, got {ceiling}"
            )
        if floor is not None and ceiling is not None and floor > ceiling:
            raise ValueError(
                f"Impulse override '{override.name}' floor {floor} exceeds ceiling {ceiling}"
            )
        if halflife is not None and halflife <= 0:
            raise ValueError(
                f"Impulse override '{override.name}' impulse_decay_halflife must be > 0, got {halflife}"
            )
        if validated and start < validated[-1].end:
            raise ValueError(
                f"Impulse override '{override.name}' overlaps with '{validated[-1].name}'"
            )

        validated.append(
            ImpulseOverrideWindow(
                name=override.name,
                start=start,
                end=end,
                floor=floor,
                ceiling=ceiling,
                impulse_decay_halflife=halflife,
            )
        )

    return validated


def _resolve_decay_params(
    tau: int,
    overrides: List[ImpulseOverrideWindow],
    default_halflife: float,
    default_floor: float,
    default_ceiling: float,
) -> Tuple[Optional[str], float, float, float]:
    """Resolve decay and clamp params active at a specific tau."""
    active_override = None
    for override in overrides:
        if override.start <= tau < override.end:
            active_override = override
            break

    halflife = (
        active_override.impulse_decay_halflife
        if active_override is not None and active_override.impulse_decay_halflife is not None
        else default_halflife
    )
    floor = (
        active_override.floor
        if active_override is not None and active_override.floor is not None
        else default_floor
    )
    ceiling = (
        active_override.ceiling
        if active_override is not None and active_override.ceiling is not None
        else default_ceiling
    )
    return (active_override.name if active_override is not None else None, halflife, floor, ceiling)


def _build_decay_schedule(
    tau_start: int,
    tau_end: int,
    overrides: List[ImpulseOverrideWindow],
    default_halflife: float,
    default_floor: float,
    default_ceiling: float,
) -> List[DecaySegment]:
    """Build a disjoint piecewise decay schedule covering [tau_start, tau_end)."""
    tau_start = max(0, int(tau_start))
    tau_end = min(1440, int(tau_end))
    if tau_end <= tau_start:
        return []

    boundaries = {tau_start, tau_end}
    for override in overrides:
        if tau_start < override.start < tau_end:
            boundaries.add(override.start)
        if tau_start < override.end < tau_end:
            boundaries.add(override.end)

    ordered = sorted(boundaries)
    schedule: List[DecaySegment] = []
    for start_tau, end_tau in zip(ordered, ordered[1:]):
        _, halflife, floor, ceiling = _resolve_decay_params(
            start_tau,
            overrides,
            default_halflife,
            default_floor,
            default_ceiling,
        )
        schedule.append(
            DecaySegment(
                start_tau=start_tau,
                end_tau=end_tau,
                halflife=halflife,
                decay=math.log(2) / halflife,
                floor=floor,
                ceiling=ceiling,
            )
        )

    return schedule


def _piecewise_decay_factor_tau(
    tau_start: int,
    tau_end: int,
    schedule: List[DecaySegment],
) -> float:
    """Compute piecewise exponential decay over integer tau overlaps."""
    if tau_end <= tau_start:
        return 1.0

    total_decay = 0.0
    for segment in schedule:
        overlap_start = max(int(tau_start), segment.start_tau)
        overlap_end = min(int(tau_end), segment.end_tau)
        if overlap_end > overlap_start:
            total_decay += segment.decay * (overlap_end - overlap_start)

    return math.exp(-total_decay)


def _piecewise_decay_factor_exact(
    start_dt: datetime,
    end_dt: datetime,
    contract_date: date,
    schedule: List[DecaySegment],
    contract_utils: ContractDayUtils,
) -> float:
    """Compute piecewise exponential decay over exact timestamp overlaps."""
    if end_dt <= start_dt:
        return 1.0

    day_start_utc, day_end_utc = contract_utils.get_contract_day_bounds_utc(contract_date)
    start_utc = max(start_dt.astimezone(timezone.utc), day_start_utc)
    end_utc = min(end_dt.astimezone(timezone.utc), day_end_utc)
    if end_utc <= start_utc:
        return 1.0

    total_decay = 0.0
    for segment in schedule:
        segment_start = day_start_utc + timedelta(minutes=segment.start_tau)
        segment_end = day_start_utc + timedelta(minutes=segment.end_tau)
        overlap_start = max(start_utc, segment_start)
        overlap_end = min(end_utc, segment_end)
        if overlap_end > overlap_start:
            overlap_minutes = (overlap_end - overlap_start).total_seconds() / 60.0
            total_decay += segment.decay * overlap_minutes

    return math.exp(-total_decay)


def _load_impulse_overrides(path: str) -> List[ImpulseOverrideWindow]:
    """Load impulse rate_mult overrides from YAML. Returns empty list if file not found."""
    import os
    if not os.path.exists(path):
        logger.warning(f"Impulse overrides file not found: {path}")
        return []
    import yaml

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        raise ValueError(f"Failed to parse impulse overrides from {path}: {exc}") from exc

    raw_overrides = data.get("overrides") or {}
    if not isinstance(raw_overrides, dict):
        raise ValueError(f"Impulse overrides in {path} must be a mapping")

    overrides = []
    try:
        for name, window in raw_overrides.items():
            if not isinstance(window, dict):
                raise ValueError(f"Impulse override '{name}' must be a mapping")
            overrides.append(
                ImpulseOverrideWindow(
                    name=str(name),
                    start=int(window["start"]),
                    end=int(window["end"]),
                    floor=window.get("floor"),
                    ceiling=window.get("ceiling"),
                    impulse_decay_halflife=window.get("impulse_decay_halflife"),
                )
            )
    except KeyError as exc:
        raise ValueError(
            f"Impulse override in {path} is missing required key: {exc.args[0]}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric values in impulse overrides {path}: {exc}") from exc

    validated = _validate_impulse_overrides(overrides)
    logger.info(f"Loaded {len(validated)} impulse override windows from {path}")
    return validated


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

        # Historical suffix profiles for optional direct historical bootstrap
        self._historical_suffix_profiles: List[HistoricalSuffixProfile] = []

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

    def _avg_rate_mult_for_slice_piecewise(
        self,
        rate_mult_now: float,
        tau_now: int,
        abs_start: int,
        abs_end: int,
        forward_schedule: List[DecaySegment],
        silence_halflife: float,
    ) -> float:
        """Average forward rate_mult over a slice using piecewise decay/clamps."""
        span = abs_end - abs_start
        if span <= 0:
            return 1.0

        delta = rate_mult_now - 1.0
        if abs(delta) < 1e-6:
            return 1.0

        if delta < 0:
            if silence_halflife <= 0:
                raise ValueError("impulse_silence_halflife_minutes must be > 0")
            silence_decay = math.log(2) / silence_halflife
        else:
            silence_decay = 0.0

        def unclamped_rate(delta_start: float, seg_decay: float, offset: float) -> float:
            if abs(delta_start) < 1e-12:
                return 1.0
            if seg_decay <= 0:
                return 1.0 + delta_start
            return 1.0 + delta_start * math.exp(-seg_decay * offset)

        def integral_unclamped(
            delta_start: float,
            seg_decay: float,
            start_offset: float,
            end_offset: float,
        ) -> float:
            if end_offset <= start_offset:
                return 0.0
            if abs(delta_start) < 1e-12:
                return end_offset - start_offset
            if seg_decay <= 0:
                return (1.0 + delta_start) * (end_offset - start_offset)
            return (
                end_offset - start_offset
                + (delta_start / seg_decay)
                * (math.exp(-seg_decay * start_offset) - math.exp(-seg_decay * end_offset))
            )

        def crossing_offset(delta_start: float, seg_decay: float, threshold: float) -> Optional[float]:
            if abs(delta_start) < 1e-12 or seg_decay <= 0:
                return None
            ratio = (threshold - 1.0) / delta_start
            if ratio <= 0.0:
                return None
            return -math.log(ratio) / seg_decay

        total_integral = 0.0
        delta_at_segment_start = delta

        for segment in forward_schedule:
            seg_start = max(segment.start_tau, tau_now)
            seg_end = segment.end_tau
            if seg_end <= seg_start:
                continue

            seg_decay = segment.decay if delta >= 0 else silence_decay

            lo = max(float(abs_start), float(seg_start))
            hi = min(float(abs_end), float(seg_end))
            if hi > lo:
                local_lo = lo - seg_start
                local_hi = hi - seg_start
                boundaries = [local_lo, local_hi]
                for threshold in (segment.floor, segment.ceiling):
                    cross = crossing_offset(delta_at_segment_start, seg_decay, threshold)
                    if cross is not None and local_lo < cross < local_hi:
                        boundaries.append(cross)

                boundaries = sorted(set(boundaries))
                for start_offset, end_offset in zip(boundaries, boundaries[1:]):
                    midpoint = 0.5 * (start_offset + end_offset)
                    raw_mid = unclamped_rate(delta_at_segment_start, seg_decay, midpoint)
                    clamped_mid = max(segment.floor, min(segment.ceiling, raw_mid))
                    if math.isclose(raw_mid, clamped_mid, rel_tol=1e-9, abs_tol=1e-9):
                        total_integral += integral_unclamped(
                            delta_at_segment_start,
                            seg_decay,
                            start_offset,
                            end_offset,
                        )
                    else:
                        total_integral += clamped_mid * (end_offset - start_offset)

            seg_span = seg_end - seg_start
            if abs(delta_at_segment_start) >= 1e-12 and seg_decay > 0:
                delta_at_segment_start *= math.exp(-seg_decay * seg_span)

        return total_integral / span

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

        tau_now = int(tau)
        rate_mult = 1.0
        actual_schedule: List[DecaySegment] = []
        expected_schedule: List[DecaySegment] = []
        forward_schedule: List[DecaySegment] = []

        if use_impulse:
            active_override_name, _, active_floor, active_ceiling = _resolve_decay_params(
                tau_now,
                self._impulse_overrides,
                self.config.impulse_decay_halflife_minutes,
                self.config.impulse_floor,
                self.config.impulse_ceiling,
            )
            actual_schedule_end = min(1440, int(math.ceil(tau)))
            actual_schedule = _build_decay_schedule(
                0,
                actual_schedule_end,
                self._impulse_overrides,
                self.config.impulse_decay_halflife_minutes,
                self.config.impulse_floor,
                self.config.impulse_ceiling,
            )
            lookback = self.config.impulse_lookback_minutes
            expected_schedule = _build_decay_schedule(
                max(0, tau_now - lookback),
                tau_now,
                self._impulse_overrides,
                self.config.impulse_decay_halflife_minutes,
                self.config.impulse_floor,
                self.config.impulse_ceiling,
            )
            forward_schedule = _build_decay_schedule(
                tau_now,
                end_tau,
                self._impulse_overrides,
                self.config.impulse_decay_halflife_minutes,
                self.config.impulse_floor,
                self.config.impulse_ceiling,
            )

            # Actual excitation: each tweet adds a decaying boost
            excitation = 0.0
            for event in events:
                if event.timestamp < now:
                    excitation += _piecewise_decay_factor_exact(
                        event.timestamp,
                        now,
                        contract_date,
                        actual_schedule,
                        self.contract_utils,
                    )

            # Expected excitation from rate curve lookback
            expected_excitation = 0.0
            for t in range(1, lookback + 1):
                past_time = now - timedelta(minutes=t)
                past_tau = self.contract_utils.get_tau(past_time, contract_date)
                if 0 <= past_tau < 1440:
                    expected_excitation += self._rate_curve[past_tau] * _piecewise_decay_factor_tau(
                        past_tau,
                        tau_now,
                        expected_schedule,
                    )

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
            rate_mult = max(active_floor, min(active_ceiling, rate_mult))

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
                "n_actual_decay_segments": len(actual_schedule),
                "n_expected_decay_segments": len(expected_schedule),
                "n_forward_decay_segments": len(forward_schedule),
            }
        else:
            self._last_impulse = None

        # --- Bucket component (from tau_now to end_tau, scaled by rate_mult) ---
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

            self._last_regime = {
                "tau_now": tau_now,
                "bucket_idx": int(full_bucket_idx),
                "observed": int(observed),
                "expected_so_far": round(expected_so_far, 2),
                "raw_regime": round(raw_regime, 3) if raw_regime is not None else None,
                "regime": round(float(regime), 3),
                "min_expected": self.config.min_expected_for_regime,
            }

            # Determine which bucket tau_now falls in
            bucket_start_idx = min(tau_now // self.bucket_size, self.n_buckets - 1)
            cutoff_minutes = self.config.impulse_cutoff_minutes
            cutoff_tau = tau_now + int(cutoff_minutes)

            # Sample remaining portion of the bucket containing tau_now
            b = buckets[bucket_start_idx]
            bucket_end = min(b.end_tau, end_tau)
            fraction = (bucket_end - tau_now) / self.bucket_size
            if fraction > 0:
                abs_start = tau_now
                abs_end = bucket_end
                if use_impulse and abs_start < cutoff_tau:
                    effective_end = min(abs_end, cutoff_tau)
                    avg_mult = self._avg_rate_mult_for_slice_piecewise(
                        rate_mult_now=rate_mult,
                        tau_now=tau_now,
                        abs_start=abs_start,
                        abs_end=effective_end,
                        forward_schedule=forward_schedule,
                        silence_halflife=self.config.impulse_silence_halflife_minutes,
                    )
                    if abs_end > cutoff_tau:
                        impulse_span = effective_end - abs_start
                        rest_span = abs_end - cutoff_tau
                        avg_mult = (
                            (avg_mult * impulse_span + 1.0 * rest_span)
                            / (abs_end - abs_start)
                        )
                else:
                    avg_mult = 1.0

                remaining_mean = b.mean * fraction * regime * avg_mult
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

                abs_start = b.start_tau
                abs_end = bucket_end
                if use_impulse and abs_start < cutoff_tau:
                    effective_end = min(abs_end, cutoff_tau)
                    avg_mult = self._avg_rate_mult_for_slice_piecewise(
                        rate_mult_now=rate_mult,
                        tau_now=tau_now,
                        abs_start=abs_start,
                        abs_end=effective_end,
                        forward_schedule=forward_schedule,
                        silence_halflife=self.config.impulse_silence_halflife_minutes,
                    )
                    if abs_end > cutoff_tau:
                        impulse_span = effective_end - abs_start
                        rest_span = abs_end - cutoff_tau
                        avg_mult = (
                            (avg_mult * impulse_span + 1.0 * rest_span)
                            / (abs_end - abs_start)
                        )
                else:
                    avg_mult = 1.0

                bucket_mean = b.mean * fraction * regime * avg_mult
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
            )
            if bootstrap_samples is not None and self._last_bootstrap:
                alpha = self._last_bootstrap["alpha"]
                use_bootstrap = rng.random(n_simulations) < alpha
                samples = np.where(use_bootstrap, bootstrap_samples, samples)
        else:
            self._last_regime = None
            self._last_bootstrap = None

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
