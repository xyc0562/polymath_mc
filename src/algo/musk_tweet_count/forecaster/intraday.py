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
        last_gap_min = self._compute_last_gap(timestamps, tau, num_tweets)

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
            return float(tau)
        elif num_tweets == 1:
            # One tweet: gap = time since that tweet
            # But we need the current tau vs that tweet's tau
            # For simplicity, use time since that tweet to 'now'
            return float(tau)  # Approximate
        else:
            # Normal case: gap between last two tweets
            sorted_ts = sorted(timestamps)
            gap = sorted_ts[-1] - sorted_ts[-2]
            return gap.total_seconds() / 60.0

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

        self._fitted = True

        logger.info(
            f"Fitted bucket forecaster: {self.n_buckets} buckets, "
            f"bucket_size={self.bucket_size}min, "
            f"weekday_days={len(weekday_bucket_counts[0])}, "
            f"weekend_days={len(weekend_bucket_counts[0])}"
        )

        # Fit impulse response from inter-tweet timing
        self._fit_impulse(historical_events, today)

        # Log bucket means for debugging
        if self._weekday_buckets:
            means = [b.mean for b in self._weekday_buckets]
            logger.debug(f"Weekday bucket means: {[f'{m:.1f}' for m in means]}")
        if self._weekend_buckets:
            means = [b.mean for b in self._weekend_buckets]
            logger.debug(f"Weekend bucket means: {[f'{m:.1f}' for m in means]}")

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

    def _get_last_tweet_tau(
        self,
        events: List[TweetEvent],
        contract_date: date,
        now: datetime,
    ) -> Optional[int]:
        """Get τ of the most recent tweet, or None if no tweets today."""
        past_events = [e for e in events if e.timestamp < now]
        if not past_events:
            return None
        last = max(past_events, key=lambda e: e.timestamp)
        return self.contract_utils.get_tau(last.timestamp, contract_date)

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

        observed = len([e for e in events if e.timestamp < now])
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
            decay = math.log(2) / halflife

            # Actual excitation: each tweet adds a decaying boost
            excitation = 0.0
            for e in events:
                e_tau = self.contract_utils.get_tau(e.timestamp, contract_date)
                if e_tau < tau_now:
                    excitation += math.exp(-decay * (tau_now - e_tau))

            # Expected excitation from rate curve lookback
            lookback = self.config.impulse_lookback_minutes
            expected_excitation = 0.0
            for t in range(1, lookback + 1):
                past_tau = tau_now - t
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

            # Asymmetric forward decay
            if rate_mult >= 1.0:
                forward_decay = decay  # boost: halflife = 30 min
            else:
                forward_decay = math.log(2) / self.config.impulse_silence_halflife_minutes  # 90 min

            # Logging
            last_tweet_tau = self._get_last_tweet_tau(events, contract_date, now)
            silence_min = (tau_now - last_tweet_tau) if last_tweet_tau is not None else tau_now

            self._last_impulse = {
                "silence_min": silence_min,
                "excitation": round(excitation, 2),
                "expected": round(expected_excitation, 2),
                "shifted": round(shifted, 2),
                "rate_mult_now": round(rate_mult, 2),
            }
        else:
            self._last_impulse = None

        # --- Bucket component (from tau_now to end_tau, scaled by rate_mult) ---
        tau_now = int(tau)
        if tau_now < end_tau:
            # Compute regime from observed vs expected at tau_now
            full_bucket_idx = min(tau // self.bucket_size, self.n_buckets - 1)
            full_partial = (tau % self.bucket_size) / self.bucket_size
            expected_so_far = sum(b.mean for b in buckets[:full_bucket_idx])
            expected_so_far += buckets[full_bucket_idx].mean * full_partial

            if expected_so_far >= self.config.min_expected_for_regime:
                regime = observed / expected_so_far
                regime = np.clip(regime, self.config.regime_min, self.config.regime_max)
            else:
                regime = 1.0

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

                bucket_mean = b.mean * fraction * regime * avg_mult
                k = b.dispersion_k
                if std_inflation_factor > 1.0:
                    k = k / std_inflation_factor
                if bucket_mean > 0:
                    samples += self._sample_negative_binomial(
                        mean=bucket_mean, k=k, size=n_simulations, rng=rng,
                    )

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
            return sample_com_poisson(mean, k, size, rng,
                                      nu_scale=self.config.cmp_nu_scale)
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
