"""
Tweet Count Forecaster.

Integrates all components:
- Data layer (XTracker API, event storage)
- Intraday model (progress curve, burst features, nowcast)
- Interday model (regime, dispersion, weekend effects)
- Monte Carlo simulation (generates probability distributions)

Supports variable-length events (2-day, 7-day, monthly, etc.).
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import ForecasterConfig
from .data import ContractDayUtils, EventStore, TweetEvent, XTrackerClient
from .intraday import (
    IntradayProgressCurve,
    BurstFeatureExtractor,
    BaseIntradayForecaster,
    IntradayNowcast,
    BucketIntradayForecaster,
)
from .interday import InterdayForecaster, GASInterdayForecaster, PIGInterdayForecaster
from .monte_carlo import MonteCarloForecaster, ForecastResult
from .projection import ProjectionModel, AsymmetricProjection, create_projection_model

logger = logging.getLogger(__name__)


class TweetCountForecaster:
    """
    Complete forecasting system for Musk tweet count markets.

    This class provides the main interface for:
    - Fetching and managing tweet data
    - Nowcasting today's final count
    - Forecasting sum distribution for variable-length event windows
    - Generating bin probabilities for trading
    """

    def __init__(
        self,
        config: Optional[ForecasterConfig] = None,
        event_store: Optional[EventStore] = None,
        projection_model: Optional[ProjectionModel] = None,
    ):
        """
        Initialize forecaster.

        Args:
            config: Forecaster configuration (uses defaults if None)
            event_store: Optional shared EventStore (creates own if None)
            projection_model: Model for computing bin probabilities (default: AsymmetricProjection)
        """
        self.config = config or ForecasterConfig()
        self.projection = projection_model or AsymmetricProjection()

        # Contract-day utilities
        self.contract_utils = ContractDayUtils(
            timezone=self.config.timezone,
            boundary_hour=self.config.contract_boundary_hour,
        )

        # Data layer - use shared or create own
        if event_store is not None:
            self.event_store = event_store
            self.xtracker = event_store.xtracker_client
            self._uses_shared_store = True
        else:
            self.xtracker = XTrackerClient()
            self.event_store = EventStore(self.contract_utils, self.xtracker)
            self._uses_shared_store = False

        # Intraday components - choose based on config
        if self.config.intraday_mode == "bucket":
            # Bucket-based intraday forecaster (new)
            self.nowcast: BaseIntradayForecaster = BucketIntradayForecaster(
                self.config.bucket_nowcast,
                self.contract_utils,
            )
            # Progress curve and burst extractor not needed for bucket mode
            self.progress_curve = None
            self.burst_extractor = None
            logger.info("Using BUCKET intraday forecaster")
        else:
            # Ridge-based intraday forecaster (original)
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
            logger.info("Using RIDGE intraday forecaster")

        # Interday components
        interday_model = self.config.interday_model.lower()
        if interday_model == "gas":
            self.interday = GASInterdayForecaster(
                self.config.gas,
                self.config.dispersion,
                self.config.weekend,
                self.contract_utils,
            )
            logger.info("Using GAS interday forecaster")
        elif interday_model == "pig":
            self.interday = PIGInterdayForecaster(
                self.config.gas,
                self.config.dispersion,
                self.config.weekend,
                self.contract_utils,
            )
            logger.info("Using PIG interday forecaster")
        elif interday_model == "ewma":
            self.interday = InterdayForecaster(
                self.config.regime,
                self.config.dispersion,
                self.config.weekend,
                self.contract_utils,
            )
            logger.info("Using EWMA interday forecaster")
        else:
            raise ValueError(
                f"Unsupported interday_model={self.config.interday_model!r}. "
                "Expected one of: ewma, gas, pig."
            )

        # Monte Carlo
        self.monte_carlo: Optional[MonteCarloForecaster] = None

        # State
        self._fitted = False
        self._last_update: Optional[datetime] = None
        self._cached_forecast: Optional[ForecastResult] = None
        self._cache_time: Optional[datetime] = None

    def fit(
        self,
        n_days: int = 90,
        skip_fetch: bool = False,
        as_of_date: Optional[date] = None,
    ) -> None:
        """
        Fit all model components from historical data.

        Fetches data from XTracker API and trains all models.

        Args:
            n_days: Number of historical days to use
            skip_fetch: If True, skip API fetch (use when sharing EventStore)
            as_of_date: Reference date for "today" (for backtesting). Default: actual today.
        """
        logger.info(f"Fitting forecaster with {n_days} days of history")

        # Fetch historical data (skip if using shared store that's already populated)
        if not skip_fetch:
            self.event_store.refresh_from_api(n_days)

        # Get historical data in required formats
        historical_timestamps = self.event_store.get_historical_timestamps(
            n_days, as_of_date=as_of_date
        )
        historical_counts = self.event_store.get_contract_day_counts(
            n_days, as_of_date=as_of_date
        )

        # Fit progress curve (only for Ridge mode)
        if self.progress_curve is not None:
            self.progress_curve.fit(historical_timestamps, as_of_date=as_of_date)

        # Fit nowcast model
        historical_events = {
            d: self.event_store.get_contract_day_events(d)
            for d in historical_counts.keys()
        }
        self.nowcast.fit(historical_events, historical_counts, as_of_date=as_of_date)

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
        as_of_date: Optional[date] = None,
    ) -> None:
        """
        Fit from pre-loaded data (useful for backtesting).

        Args:
            events_by_date: Dict mapping contract_date -> events
            counts_by_date: Optional explicit counts (otherwise computed from events)
            as_of_date: Reference date for "today" (for backtesting). Default: actual today.
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

        # Fit progress curve (only for Ridge mode)
        if self.progress_curve is not None:
            self.progress_curve.fit(historical_timestamps, as_of_date=as_of_date)

        # Fit nowcast
        self.nowcast.fit(events_by_date, counts_by_date, as_of_date=as_of_date)

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

    # ========== Methods for live trading integration ==========

    def get_current_count(self, contract_date: Optional[date] = None) -> int:
        """
        Get current tweet count for today's contract day.

        Args:
            contract_date: Contract date (default: today)

        Returns:
            Number of tweets so far today
        """
        if contract_date is None:
            contract_date = self.contract_utils.get_current_contract_date()

        events = self.event_store.get_contract_day_events(contract_date)
        return len(events)

    def get_7day_cumulative_count(
        self,
        now: Optional[datetime] = None,
    ) -> int:
        """
        Get cumulative 7-day tweet count from previous 6 days + today so far.

        This is the "running total" for the current 7-day contract window.

        Args:
            now: Current timestamp (default: now)

        Returns:
            Total tweets in the 7-day window so far
        """
        if now is None:
            now = datetime.now(self.contract_utils.tz)

        contract_date = self.contract_utils.get_contract_date(now)
        total = 0

        # Add previous 6 completed days
        for days_ago in range(1, 7):
            past_date = contract_date - timedelta(days=days_ago)
            total += self.event_store.get_contract_day_count(past_date)

        # Add today's count so far
        total += self.get_current_count(contract_date)

        return total

    def get_dead_bins(
        self,
        cumulative_count: Optional[int] = None,
    ) -> List[int]:
        """
        Get indices of bins that are impossible (count already exceeded upper bound).

        Args:
            cumulative_count: Current 7-day sum (default: computed from data)

        Returns:
            List of dead bin indices
        """
        if cumulative_count is None:
            cumulative_count = self.get_7day_cumulative_count()

        dead = []
        for i, (lower, upper) in enumerate(self.config.bins):
            if upper < cumulative_count:
                dead.append(i)

        return dead

    def get_settlement_timing(
        self,
        settlement_date: date,
        now: Optional[datetime] = None,
    ) -> Tuple[float, float]:
        """
        Get timing information for a 7-day contract.

        Args:
            settlement_date: The settlement date (last day of 7-day window)
            now: Current timestamp (default: now)

        Returns:
            Tuple of (hours_elapsed, hours_remaining) since contract start
        """
        if now is None:
            now = datetime.now(self.contract_utils.tz)

        # Contract starts 7 days before settlement at noon
        # E.g., for Jan 27 - Feb 3 market, settlement_date = Feb 3
        # contract_start_date = Feb 3 - 7 = Jan 27
        contract_start_date = settlement_date - timedelta(days=7)
        start_dt, _ = self.contract_utils.get_contract_day_bounds(contract_start_date)

        # Settlement is at noon on settlement date (start of that contract day)
        # E.g., for Feb 3, settlement is at noon Feb 3 (not noon Feb 4)
        settlement_dt, _ = self.contract_utils.get_contract_day_bounds(settlement_date)

        # Calculate elapsed and remaining
        hours_elapsed = (now - start_dt).total_seconds() / 3600
        hours_remaining = (settlement_dt - now).total_seconds() / 3600

        return max(0, hours_elapsed), max(0, hours_remaining)

    def forecast_for_event_window(
        self,
        market_start_date: date,
        settlement_date: date,
        now: Optional[datetime] = None,
        n_simulations: Optional[int] = None,
    ) -> ForecastResult:
        """
        Generate forecast for a specific event's 7-day window.

        Unlike forecast_7day_distribution() which always forecasts today + 6 days,
        this method accounts for the actual event window by:
        1. Using actual counts from completed days in the window
        2. Forecasting only the remaining days until settlement

        Args:
            market_start_date: First day of the 7-day counting window
            settlement_date: Settlement date (day after the 7th counting day)
            now: Current timestamp (default: now)
            n_simulations: Number of simulations (default from config)

        Returns:
            ForecastResult with proper distribution for the event window
        """
        if not self._fitted:
            raise RuntimeError("Forecaster not fitted")

        if now is None:
            now = datetime.now(self.contract_utils.tz)

        today = self.contract_utils.get_contract_date(now)
        today_events = self.event_store.get_contract_day_events(today)

        # Calculate days remaining until settlement
        # The 7 counting days are: market_start_date through (settlement_date - 1 day)
        # Example: Jan 27-Feb 3 market has counting days Jan 27, 28, 29, 30, 31, Feb 1, 2
        # Settlement at noon Feb 3
        last_counting_day = settlement_date - timedelta(days=1)

        # Count actual tweets from completed days in the window
        past_count = 0
        past_dates = []
        current_date = market_start_date
        while current_date < today and current_date <= last_counting_day:
            day_count = self.event_store.get_contract_day_count(current_date)
            past_count += day_count
            past_dates.append(current_date)
            current_date += timedelta(days=1)

        # Calculate remaining forecast horizon (in days)
        # If today is Jan 31 and settlement is Feb 3:
        # - Remaining counting days: Jan 31, Feb 1, Feb 2 = 3 days
        if today < market_start_date:
            # Before the counting window starts - forecast entire window
            remaining_days = (last_counting_day - market_start_date).days + 1
        elif today > last_counting_day:
            # We're past the counting window
            remaining_days = 0
        else:
            # Within the counting window
            # Full days remaining after today + today
            days_after_today = (last_counting_day - today).days
            remaining_days = days_after_today + 1

        logger.debug(
            f"[forecast_for_event_window] {market_start_date} - {settlement_date}: "
            f"past_count={past_count} from {len(past_dates)} days ({past_dates}), "
            f"today={today}, remaining_days={remaining_days}"
        )

        if remaining_days <= 0:
            # Past settlement, return actual count as the forecast
            total = past_count + len(today_events)
            return ForecastResult(
                mean=float(total),
                median=float(total),
                std=0.0,
                p5=float(total),
                p25=float(total),
                p75=float(total),
                p95=float(total),
                bin_probabilities=self.monte_carlo._compute_bin_probabilities(
                    np.array([total])
                ),
                today_estimate=float(len(today_events)),
                future_days_estimate=0.0,
                n_simulations=1,
                simulation_time_ms=0.0,
            )

        # Run Monte Carlo simulation for remaining days
        # Always request samples so we can shift them properly for past_count
        if today < market_start_date:
            # Before counting window starts - use pure interday forecast
            # No intraday nowcast since we have no data from the counting window yet
            # base_date = market_start_date ensures correct weekend effects
            forecast = self.monte_carlo.simulate_horizon_pure_interday(
                base_date=market_start_date,
                horizon=remaining_days,
                n_simulations=n_simulations,
                return_samples=True,
            )
            logger.debug(
                f"[forecast_for_event_window] Pre-counting window: using pure interday "
                f"forecast from {market_start_date} for {remaining_days} days"
            )
        else:
            # Within counting window - use normal simulation with intraday nowcast
            forecast = self.monte_carlo.simulate_horizon(
                events=today_events,
                contract_date=today,
                now=now,
                horizon=remaining_days,
                n_simulations=n_simulations,
                return_samples=True,
            )

        logger.debug(
            f"[forecast_for_event_window] horizon={remaining_days}, "
            f"forecast.mean={forecast.mean:.1f}, past_count={past_count}, "
            f"total={forecast.mean + past_count:.1f}"
        )

        # Add past_count to the forecast distribution
        # This shifts the entire distribution (including raw samples) up by past_count
        shifted_samples = None
        if forecast.samples is not None:
            shifted_samples = forecast.samples + past_count

        return ForecastResult(
            mean=forecast.mean + past_count,
            median=forecast.median + past_count,
            std=forecast.std,  # Uncertainty doesn't change
            p5=forecast.p5 + past_count,
            p25=forecast.p25 + past_count,
            p75=forecast.p75 + past_count,
            p95=forecast.p95 + past_count,
            bin_probabilities=self._shift_bin_probabilities(forecast, past_count),
            today_estimate=forecast.today_estimate,
            future_days_estimate=forecast.future_days_estimate,
            regime_adjustment=forecast.regime_adjustment,
            n_simulations=forecast.n_simulations,
            simulation_time_ms=forecast.simulation_time_ms,
            today_interday_estimate=forecast.today_interday_estimate,
            future_days_pure=forecast.future_days_pure,
            samples=shifted_samples,
        )

    def _shift_bin_probabilities(
        self,
        forecast: ForecastResult,
        shift: int,
    ) -> List:
        """
        Recompute bin probabilities after shifting the distribution.

        Uses the configured projection model to compute probabilities.
        Default (AsymmetricProjection) preserves the actual sample distribution.
        """
        from .monte_carlo import BinProbability

        # Use projection model to compute probabilities
        probs = self.projection.compute_bin_probabilities(
            forecast=forecast,
            bins=self.config.bins,
            shift=shift,
        )

        # Convert to BinProbability objects
        return [
            BinProbability(lower=lower, upper=upper, probability=prob)
            for (lower, upper), prob in zip(self.config.bins, probs)
        ]

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
) -> TweetCountForecaster:
    """
    Factory function to create a configured forecaster.

    Args:
        timezone: Timezone for calculations
        n_simulations: Number of Monte Carlo simulations
        **config_overrides: Additional config overrides

    Returns:
        Configured TweetCountForecaster instance
    """
    from .config import MonteCarloConfig

    config = ForecasterConfig(
        timezone=timezone,
        monte_carlo=MonteCarloConfig(n_simulations=n_simulations),
    )

    return TweetCountForecaster(config)
