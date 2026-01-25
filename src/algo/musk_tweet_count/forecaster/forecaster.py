"""
Main Musk 7-Day Tweet Count Forecaster.

Integrates all components:
- Data layer (XTracker API, event storage)
- Intraday model (progress curve, burst features, nowcast)
- Interday model (regime, dispersion, weekend effects)
- Monte Carlo simulation (7-day distribution)
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import ForecasterConfig
from .data import ContractDayUtils, EventStore, TweetEvent, XTrackerClient
from .intraday import IntradayProgressCurve, BurstFeatureExtractor, IntradayNowcast
from .interday import InterdayForecaster
from .monte_carlo import MonteCarloForecaster, ForecastResult

logger = logging.getLogger(__name__)


class Musk7DayForecaster:
    """
    Complete forecasting system for Musk tweet count markets.

    This class provides the main interface for:
    - Fetching and managing tweet data
    - Nowcasting today's final count
    - Forecasting 7-day sum distribution
    - Generating bin probabilities for trading
    """

    def __init__(self, config: Optional[ForecasterConfig] = None):
        """
        Initialize forecaster.

        Args:
            config: Forecaster configuration (uses defaults if None)
        """
        self.config = config or ForecasterConfig()

        # Contract-day utilities
        self.contract_utils = ContractDayUtils(
            timezone=self.config.timezone,
            boundary_hour=self.config.contract_boundary_hour,
        )

        # Data layer
        self.xtracker = XTrackerClient()
        self.event_store = EventStore(self.contract_utils, self.xtracker)

        # Intraday components
        self.progress_curve = IntradayProgressCurve(
            self.config.intraday_curve,
            self.contract_utils,
        )
        self.burst_extractor = BurstFeatureExtractor(
            self.config.burst_features,
            self.contract_utils,
        )
        self.nowcast = IntradayNowcast(
            self.config.nowcast,
            self.progress_curve,
            self.burst_extractor,
            self.contract_utils,
        )

        # Interday components
        self.interday = InterdayForecaster(
            self.config.regime,
            self.config.dispersion,
            self.config.weekend,
            self.contract_utils,
        )

        # Monte Carlo
        self.monte_carlo: Optional[MonteCarloForecaster] = None

        # State
        self._fitted = False
        self._last_update: Optional[datetime] = None
        self._cached_forecast: Optional[ForecastResult] = None
        self._cache_time: Optional[datetime] = None

    def fit(self, n_days: int = 90) -> None:
        """
        Fit all model components from historical data.

        Fetches data from XTracker API and trains all models.

        Args:
            n_days: Number of historical days to use
        """
        logger.info(f"Fitting forecaster with {n_days} days of history")

        # Fetch historical data
        self.event_store.refresh_from_api(n_days)

        # Get historical data in required formats
        historical_timestamps = self.event_store.get_historical_timestamps(n_days)
        historical_counts = self.event_store.get_contract_day_counts(n_days)

        # Fit progress curve
        self.progress_curve.fit(historical_timestamps)

        # Fit nowcast model
        historical_events = {
            d: self.event_store.get_contract_day_events(d)
            for d in historical_counts.keys()
        }
        self.nowcast.fit(historical_events, historical_counts)

        # Fit interday model
        self.interday.fit(historical_counts)

        # Initialize Monte Carlo
        self.monte_carlo = MonteCarloForecaster(
            self.config,
            self.nowcast,
            self.interday,
            self.contract_utils,
        )

        self._fitted = True
        logger.info("Forecaster fitted successfully")

    def fit_from_data(
        self,
        events_by_date: Dict[date, List[TweetEvent]],
        counts_by_date: Optional[Dict[date, int]] = None,
    ) -> None:
        """
        Fit from pre-loaded data (useful for backtesting).

        Args:
            events_by_date: Dict mapping contract_date -> events
            counts_by_date: Optional explicit counts (otherwise computed from events)
        """
        # Load events into store
        for contract_date, events in events_by_date.items():
            for event in events:
                self.event_store.add_event(event)

        # Compute counts if not provided
        if counts_by_date is None:
            counts_by_date = {d: len(events) for d, events in events_by_date.items()}

        # Get timestamps
        historical_timestamps = {
            d: [e.timestamp for e in events]
            for d, events in events_by_date.items()
        }

        # Fit progress curve
        self.progress_curve.fit(historical_timestamps)

        # Fit nowcast
        self.nowcast.fit(events_by_date, counts_by_date)

        # Fit interday
        self.interday.fit(counts_by_date)

        # Initialize Monte Carlo
        self.monte_carlo = MonteCarloForecaster(
            self.config,
            self.nowcast,
            self.interday,
            self.contract_utils,
        )

        self._fitted = True
        logger.info(f"Forecaster fitted from {len(events_by_date)} days of data")

    def update_events(self, events: List[TweetEvent]) -> None:
        """
        Update with new events.

        Args:
            events: New events to add
        """
        for event in events:
            self.event_store.add_event(event)

        self._last_update = datetime.now(self.contract_utils.tz)

        # Invalidate cache
        self._cached_forecast = None

    def update_from_api(self) -> int:
        """
        Fetch latest events from XTracker API.

        Returns:
            Number of new events added
        """
        today = self.contract_utils.get_current_contract_date()

        # Fetch today's events
        events = self.xtracker.get_contract_day_posts(
            today,
            self.contract_utils,
        )

        # Count new events
        existing = set(
            e.event_id for e in self.event_store.get_contract_day_events(today)
            if e.event_id
        )
        new_events = [e for e in events if e.event_id not in existing]

        # Update store
        self.update_events(new_events)

        if new_events:
            logger.info(f"Added {len(new_events)} new events for {today}")

        return len(new_events)

    def nowcast_today(
        self,
        now: Optional[datetime] = None,
    ) -> Tuple[float, float]:
        """
        Nowcast today's final tweet count.

        Args:
            now: Current timestamp (default: now)

        Returns:
            Tuple of (predicted_count, uncertainty_std)
        """
        if not self._fitted:
            raise RuntimeError("Forecaster not fitted")

        if now is None:
            now = datetime.now(self.contract_utils.tz)

        contract_date = self.contract_utils.get_contract_date(now)
        events = self.event_store.get_contract_day_events(contract_date)

        return self.nowcast.predict(events, contract_date, now)

    def forecast_future_days(
        self,
        n_days: int = 6,
        base_date: Optional[date] = None,
    ) -> List[Tuple[float, float]]:
        """
        Forecast future daily counts.

        Args:
            n_days: Number of days to forecast (default: 6)
            base_date: Base date (default: today)

        Returns:
            List of (mean, k) tuples for each day
        """
        if not self._fitted:
            raise RuntimeError("Forecaster not fitted")

        if base_date is None:
            base_date = self.contract_utils.get_current_contract_date()

        horizons = list(range(1, n_days + 1))
        return self.interday.get_forecast_params(horizons, base_date)

    def forecast_7day_distribution(
        self,
        now: Optional[datetime] = None,
        n_simulations: Optional[int] = None,
        use_cache: bool = True,
    ) -> ForecastResult:
        """
        Generate full 7-day sum distribution via Monte Carlo.

        Args:
            now: Current timestamp (default: now)
            n_simulations: Number of simulations (default from config)
            use_cache: Whether to use cached result if available

        Returns:
            ForecastResult with distribution and bin probabilities
        """
        if not self._fitted:
            raise RuntimeError("Forecaster not fitted")

        if now is None:
            now = datetime.now(self.contract_utils.tz)

        # Check cache
        if use_cache and self._cached_forecast is not None:
            cache_age = (now - self._cache_time).total_seconds()
            if cache_age < self.config.update.cache_ttl_seconds:
                return self._cached_forecast

        contract_date = self.contract_utils.get_contract_date(now)
        events = self.event_store.get_contract_day_events(contract_date)

        # Run simulation
        result = self.monte_carlo.simulate(
            events,
            contract_date,
            now,
            n_simulations,
        )

        # Update cache
        self._cached_forecast = result
        self._cache_time = now

        return result

    def get_bin_probabilities(
        self,
        now: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Get probabilities for each bin (for trading).

        Args:
            now: Current timestamp

        Returns:
            Dict mapping bin label -> probability
        """
        forecast = self.forecast_7day_distribution(now)
        return {bp.label: bp.probability for bp in forecast.bin_probabilities}

    def get_state_summary(self) -> Dict:
        """
        Get summary of current forecaster state.

        Returns:
            Dict with state information
        """
        now = datetime.now(self.contract_utils.tz)
        today = self.contract_utils.get_contract_date(now)
        tau = self.contract_utils.get_tau(now, today)

        summary = {
            "current_time": now.isoformat(),
            "contract_date": today.isoformat(),
            "tau_minutes": tau,
            "tau_hours": tau / 60,
            "fitted": self._fitted,
        }

        if self._fitted:
            # Today's data
            today_events = self.event_store.get_contract_day_events(today)
            summary["tweets_today_so_far"] = len(today_events)

            # Nowcast
            mean, std = self.nowcast_today(now)
            summary["nowcast_mean"] = mean
            summary["nowcast_std"] = std

            # Regime state
            regime_state = self.interday.current_regime_state
            summary["regime_log_intensity"] = regime_state.log_intensity
            summary["regime_intensity"] = regime_state.intensity

            # Dispersion
            summary["dispersion_k"] = self.interday.dispersion.k

            # Weekend effect
            summary["weekend_effect"] = self.interday.weekend.effect

        return summary

    def refresh_and_forecast(
        self,
        n_simulations: Optional[int] = None,
    ) -> ForecastResult:
        """
        Convenience method: update from API and return forecast.

        Args:
            n_simulations: Number of simulations

        Returns:
            ForecastResult
        """
        self.update_from_api()
        return self.forecast_7day_distribution(n_simulations=n_simulations, use_cache=False)

    def should_update(self) -> bool:
        """
        Check if an update is due based on config.

        Returns:
            True if update is recommended
        """
        if self._last_update is None:
            return True

        now = datetime.now(self.contract_utils.tz)
        elapsed = (now - self._last_update).total_seconds()

        return elapsed >= self.config.update.periodic_update_seconds

    # Methods for backtesting compatibility

    def _forecast_at_time(
        self,
        events_by_date: Dict[date, List[TweetEvent]],
        forecast_date: date,
        tau: int,
        training_dates: List[date],
    ) -> ForecastResult:
        """
        Generate forecast at a specific point in time (for backtesting).

        Args:
            events_by_date: All available events
            forecast_date: Date to forecast from
            tau: Minutes since noon
            training_dates: Dates to use for training

        Returns:
            ForecastResult
        """
        # Filter training data
        training_events = {
            d: events_by_date[d] for d in training_dates
            if d in events_by_date
        }
        training_counts = {d: len(events) for d, events in training_events.items()}

        # Fit on training data
        self.fit_from_data(training_events, training_counts)

        # Construct 'now' timestamp
        start_dt, _ = self.contract_utils.get_contract_day_bounds(forecast_date)
        now = start_dt + timedelta(minutes=tau)

        # Get today's events up to 'now'
        today_events = events_by_date.get(forecast_date, [])

        # Run forecast
        return self.monte_carlo.simulate(
            today_events,
            forecast_date,
            now,
        )


def create_forecaster(
    timezone: str = "America/New_York",
    n_simulations: int = 10000,
    **config_overrides,
) -> Musk7DayForecaster:
    """
    Factory function to create a configured forecaster.

    Args:
        timezone: Timezone for calculations
        n_simulations: Number of Monte Carlo simulations
        **config_overrides: Additional config overrides

    Returns:
        Configured Musk7DayForecaster instance
    """
    from .config import MonteCarloConfig

    config = ForecasterConfig(
        timezone=timezone,
        monte_carlo=MonteCarloConfig(n_simulations=n_simulations),
    )

    return Musk7DayForecaster(config)
