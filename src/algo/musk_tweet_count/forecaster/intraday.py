"""
Intraday model for nowcasting today's final tweet count.

Components:
- IntradayProgressCurve: F(τ) = expected fraction of day's tweets by minute τ
- BurstFeatureExtractor: Extract burst/session features from timestamps
- IntradayNowcast: Ridge regression model to predict final daily count
"""

import logging
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge

from .config import IntradayCurveConfig, BurstFeaturesConfig, NowcastConfig
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

    def fit(self, historical_timestamps: Dict[date, List[datetime]]) -> None:
        """
        Fit progress curves from historical data.

        Args:
            historical_timestamps: Dict mapping contract_date -> List[timestamp]
        """
        weekday_curves = []
        weekend_curves = []
        weekday_weights = []
        weekend_weights = []

        today = self.contract_utils.get_current_contract_date()

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


class IntradayNowcast:
    """
    Ridge regression model to predict today's final tweet count.
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
    ) -> None:
        """
        Fit the nowcast model.

        Args:
            historical_events: Dict mapping contract_date -> events
            historical_counts: Dict mapping contract_date -> final count
        """
        X = []
        y = []
        weights = []

        today = self.contract_utils.get_current_contract_date()

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

    @property
    def historical_mean(self) -> float:
        """Get historical daily mean."""
        return self._historical_mean

    @property
    def historical_std(self) -> float:
        """Get historical daily std."""
        return self._historical_std
