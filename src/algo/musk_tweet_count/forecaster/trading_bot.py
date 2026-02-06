"""
Trading bot integrating GAS forecaster with Kelly optimizer.

Orchestrates the forecaster and Kelly optimizer for live trading
on the Musk tweet count market.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional

from py_clob_client.client import ClobClient

from .config import ForecasterConfig
from .forecaster import TweetCountForecaster
from .projection import ProjectionModel, AsymmetricProjection, create_projection_model
from ..kelly.config import KellyConfig
from ..kelly.integration import KellyTradingBot
from ..kelly.executor import TickResult
from ..kelly.user_stream import UserStreamClient

if TYPE_CHECKING:
    from .data import EventStore
    from ..kelly.orderbook import UnifiedOrderbook

logger = logging.getLogger(__name__)


def _get_wallet_for_positions() -> Optional[str]:
    """
    Get wallet address for position queries.

    When using proxy wallet (signature_type 1 or 2), positions are held
    by the proxy wallet (POLY_FUNDER), not the main wallet.
    """
    funder = os.getenv("POLY_FUNDER")
    signature_type = int(os.getenv("POLY_SIGNATURE_TYPE", "0"))

    if funder and signature_type in (1, 2):
        return funder

    return os.getenv("WALLET_ADDRESS")


@dataclass
class TradingBotConfig:
    """Configuration for the GAS-Kelly trading bot."""

    # Slow loop interval in seconds (5 minutes default)
    # This is when we recompute Monte Carlo forecasts (if fresh data available)
    slow_tick_interval_seconds: int = 300

    # Fast loop interval in seconds (30 seconds default)
    # This checks for trade opportunities using cached probabilities
    fast_tick_interval_seconds: int = 30

    # Whether to run in dry-run mode (no real orders)
    dry_run: bool = True

    # Settlement date for current contract
    settlement_date: Optional[date] = None

    # Initial capital in USD
    initial_capital: float = 1000.0

    # Number of days of history to use for model fitting
    training_days: int = 45

    # Whether to use GAS model (vs EWMA)
    use_gas: bool = True

    # Whether to disable WebSocket (use REST polling only)
    disable_websocket: bool = False

    # Forecast cache timeout in seconds (should be > slow_interval to avoid recomputing mid-cycle)
    forecast_cache_timeout: int = 165

    # Event name for logging (e.g., "Feb 03 - Feb 10")
    event_name: Optional[str] = None

    # Projection model type: "asymmetric" (default) or "normal"
    # - asymmetric: Uses actual Monte Carlo samples (preserves right-skew)
    # - normal: Approximates with Normal distribution (symmetric)
    projection_model: str = "asymmetric"

    # Legacy alias for slow_tick_interval_seconds
    @property
    def tick_interval_seconds(self) -> int:
        return self.slow_tick_interval_seconds


class GASKellyTradingBot:
    """
    Trading bot combining GAS forecaster with Kelly optimizer.

    Main loop (every tick_interval):
    1. Fetch current tweet count (XTracker API)
    2. Update forecaster with new observation
    3. Generate bin probabilities
    4. Pass to Kelly optimizer
    5. Execute trades (max N per tick, with delay between)
    6. Sleep until next tick
    """

    def __init__(
        self,
        clob_client: ClobClient,
        kelly_config: KellyConfig,
        forecaster_config: ForecasterConfig,
        bot_config: TradingBotConfig,
        event_store: "EventStore",
        user_stream: Optional[UserStreamClient] = None,
    ):
        """
        Initialize the trading bot.

        Args:
            clob_client: Authenticated Polymarket CLOB client
            kelly_config: Configuration for Kelly optimizer
            forecaster_config: Configuration for forecaster
            bot_config: Bot-level configuration
            event_store: Shared EventStore from MultiEventManager (required)
            user_stream: Optional global UserStreamClient for fill confirmations
                        (if None, bot will create its own if API credentials available)
        """
        self._external_user_stream = user_stream
        self.clob_client = clob_client
        self.kelly_config = kelly_config
        self.forecaster_config = forecaster_config
        self.bot_config = bot_config

        # Initialize forecaster with shared EventStore
        # Bot relies on upstream MultiEventManager for data (no standalone mode)
        self.forecaster = TweetCountForecaster(forecaster_config, event_store=event_store)

        # Kelly bot (initialized during setup)
        self.kelly_bot: Optional[KellyTradingBot] = None

        # State
        self._running = False
        self._setup_complete = False
        self._last_tick_time: Optional[datetime] = None
        self._tick_count = 0

        # Settlement date (set during setup or from config)
        self.settlement_date: Optional[date] = bot_config.settlement_date

        # Market start date (7 days before settlement)
        self.market_start_date: Optional[date] = None

        # Market bin boundaries (set during setup)
        # List of (lower, upper) tuples
        self._market_bins: List[tuple] = []

        # XTracker count from market API (authoritative source)
        # Used for dead bin detection to avoid betting on impossible bins
        self._authoritative_count: Optional[int] = None

        # Last known good count (for detecting regression/stale data)
        self._last_known_count: Optional[int] = None

        # Stop event for graceful shutdown
        self._stop_event: Optional[asyncio.Event] = None

        # Cached probabilities (set by SharedMarketData for fast loop)
        self._cached_probabilities: Optional[List[float]] = None

        # Lock to prevent concurrent tick execution
        self._tick_lock = asyncio.Lock()
        self._tick_in_progress = False

        # WebSocket-triggered trading
        self._pending_ws_tick = False  # Flag to schedule a tick from WS callback
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # WebSocket callback tracking
        self._last_ws_callback_time: Optional[datetime] = None
        self._ws_callback_count: int = 0
        self._last_ws_log_time: Optional[datetime] = None

        # Projection model for computing bin probabilities
        self._projection: ProjectionModel = create_projection_model(bot_config.projection_model)

        # Cached forecast for table display (avoid recomputing every tick)
        self._cached_forecast_mean: Optional[float] = None
        self._cached_forecast_std: Optional[float] = None
        self._cached_forecast_breakdown: Optional[Dict] = None
        self._cached_forecast_time: Optional[datetime] = None

        # Fresh data notification (set by upstream MultiEventManager)
        # When True, next slow tick will recompute Monte Carlo
        # Start True so first tick computes Monte Carlo forecast
        self._fresh_data_available: bool = True
        self._last_monte_carlo_time: Optional[datetime] = None

        # Data freshness checker callback (set by MultiEventManager)
        # Returns (is_fresh, age_seconds). If None, freshness check is skipped.
        self._data_freshness_checker: Optional[callable] = None

    def set_data_freshness_checker(self, checker: callable) -> None:
        """
        Set the data freshness checker callback.

        Args:
            checker: Callable that returns (is_fresh: bool, age_seconds: float)
        """
        self._data_freshness_checker = checker

    def set_authoritative_count(self, count: int) -> None:
        """
        Set the authoritative count from XTracker trackings API.

        This is used for dead bin detection to avoid betting on bins
        that are impossible based on the authoritative count, even if
        the posts-based count is lower.

        Args:
            count: Authoritative count from XTracker trackings API
        """
        if self._authoritative_count is not None and count < self._authoritative_count:
            logger.warning(
                f"Authoritative count regressed: {self._authoritative_count} -> {count}. "
                f"Keeping higher value."
            )
            return
        self._authoritative_count = count
        logger.debug(f"Updated authoritative count: {count}")

    def get_effective_count_for_dead_bins(self, computed_count: int) -> int:
        """
        Get the effective count for dead bin detection.

        Uses max(computed_count, authoritative_count) to ensure we don't
        bet on bins that are impossible from the authoritative perspective.

        Args:
            computed_count: Count computed from posts

        Returns:
            Effective count for dead bin detection
        """
        if self._authoritative_count is not None:
            effective = max(computed_count, self._authoritative_count)
            if effective > computed_count:
                logger.debug(
                    f"Using authoritative count ({self._authoritative_count}) > "
                    f"computed count ({computed_count}) for dead bin detection"
                )
            return effective
        return computed_count

    def notify_data_refreshed(self) -> None:
        """
        Notify the bot that fresh data is available from the upstream manager.

        Called by MultiEventManager after it refreshes the shared EventStore.
        This signals that the next slow tick should recompute Monte Carlo.
        """
        self._fresh_data_available = True
        logger.debug("Bot notified of fresh data from upstream manager")

    def _get_probabilities(
        self,
        current_count: int,
        hours_elapsed: float,
        hours_remaining: float,
        use_cache: bool = True,
    ) -> List[float]:
        """
        Adapter: forecaster -> Kelly probability format.

        If cached probabilities are available (from SharedMarketData), uses those.
        Otherwise computes probabilities using Monte Carlo simulation.

        Args:
            current_count: Current 7-day cumulative tweet count
            hours_elapsed: Hours since contract start
            hours_remaining: Hours until settlement
            use_cache: If True, use cached probabilities if available

        Returns:
            List of probabilities for each bin, ordered by bin index
        """
        # Use cached probabilities if available (from slow loop)
        if use_cache and self._cached_probabilities:
            return self._cached_probabilities

        # Otherwise compute fresh probabilities
        return self._compute_probabilities(current_count)

    def _compute_probabilities(self, current_count: int) -> List[float]:
        """
        Compute probabilities using the configured projection model.

        Called by slow loop to update cached probabilities.

        Uses forecast_for_event_window() which properly accounts for:
        1. Actual tweet counts from completed days in the event window
        2. Forecasts only the remaining days until settlement

        The projection model determines how bin probabilities are computed:
        - AsymmetricProjection: Uses actual MC samples (preserves right-skew)
        - NormalProjection: Approximates with Normal (symmetric)
        """
        # Get forecast for this specific event's window
        # This accounts for past actual counts + remaining forecast
        if self.market_start_date and self.settlement_date:
            result = self.forecaster.forecast_for_event_window(
                market_start_date=self.market_start_date,
                settlement_date=self.settlement_date,
            )
        else:
            # Fallback to generic 7-day forecast if no event window specified
            result = self.forecaster.forecast_7day_distribution(use_cache=False)

        # Use projection model to compute bin probabilities
        # The projection handles:
        # - Using actual samples (asymmetric) or Normal approximation
        # - Applying floor at current_count
        try:
            probabilities = self._projection.compute_bin_probabilities(
                forecast=result,
                bins=self._market_bins,
                shift=0,  # No shift - forecast already includes past counts
                floor=current_count,  # Samples can't be below current count
            )
        except RuntimeError as e:
            # Asymmetric projection requires samples - log and fall back
            logger.error(f"Projection failed: {e}")
            if self._cached_probabilities:
                return self._cached_probabilities
            num_bins = len(self._market_bins)
            return [1.0 / num_bins] * num_bins if num_bins > 0 else []

        # Renormalize to sum to 1.0
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            live_bins = [i for i, (lower, upper) in enumerate(self._market_bins) if upper >= current_count]
            if live_bins:
                probabilities = [0.0] * len(self._market_bins)
                for i in live_bins:
                    probabilities[i] = 1.0 / len(live_bins)

        # Update cache
        self._cached_probabilities = probabilities

        logger.debug(f"Computed probabilities using {self._projection.name} projection")

        return probabilities

    def _on_orderbook_update(
        self,
        token_id: str,
        bin_index: int,
        orderbook: "UnifiedOrderbook",
    ) -> None:
        """
        Callback for significant WebSocket orderbook updates.

        This is called synchronously from the WebSocket handler.
        Schedules an async tick if not already in progress.
        """
        # Track callback for monitoring
        now = datetime.now(self.forecaster.contract_utils.tz)
        self._last_ws_callback_time = now
        self._ws_callback_count += 1

        # Log periodically (every 5 seconds) to confirm WS is working
        should_log = (
            self._last_ws_log_time is None or
            (now - self._last_ws_log_time).total_seconds() >= 5.0
        )
        if should_log:
            self._last_ws_log_time = now
            bid_str = f"{orderbook.best_yes_bid:.2f}" if orderbook.best_yes_bid is not None else "None"
            ask_str = f"{orderbook.best_yes_ask:.2f}" if orderbook.best_yes_ask is not None else "None"
            logger.info(
                f"[WS] Callback #{self._ws_callback_count} for bin {bin_index} "
                f"(best_bid={bid_str}, best_ask={ask_str})"
            )

        if self._tick_in_progress:
            return

        # Flag that we should run a tick
        self._pending_ws_tick = True

        # Schedule the tick on the event loop if we have one
        if self._event_loop and self._event_loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(
                    self._run_ws_triggered_tick(),
                    self._event_loop
                )
            except Exception as e:
                logger.debug(f"Failed to schedule WS tick: {e}")

    async def _run_ws_triggered_tick(self) -> None:
        """Run a tick triggered by WebSocket update (with debouncing)."""
        if not self._pending_ws_tick:
            return

        self._pending_ws_tick = False

        # Small delay to batch rapid updates
        await asyncio.sleep(0.1)

        # Run the tick (uses cached probabilities, minimal logging)
        await self.run_tick(log_header=False)

    async def setup(self, bins: List[Dict]) -> None:
        """
        Setup the trading bot.

        Args:
            bins: List of bin definitions with {"upper_bound": int, "token_id": str}
        """
        logger.info("Setting up GAS-Kelly trading bot...")

        # Capture event loop for WebSocket callbacks
        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._event_loop = None

        # Extract market bin boundaries
        # Bins should be sorted by upper_bound (inf will naturally sort last)
        sorted_bins = sorted(bins, key=lambda x: x.get("upper_bound", 0))
        self._market_bins = []
        prev_upper = -1
        for b in sorted_bins:
            upper = b.get("upper_bound", 0)
            # Lower bound is previous upper + 1 (or 0 for first bin)
            lower = prev_upper + 1
            self._market_bins.append((lower, upper))
            # For inf upper bounds, don't update prev_upper (shouldn't matter as it's last)
            if upper != float('inf'):
                prev_upper = upper

        # Log bins with readable format
        first_bin = self._market_bins[0]
        last_bin = self._market_bins[-1]
        last_upper_str = "inf" if last_bin[1] == float('inf') else str(last_bin[1])
        logger.info(f"Market bins: {len(self._market_bins)} bins from {first_bin} to ({last_bin[0]}, {last_upper_str})")

        # Fit forecaster on historical data
        # Always skip fetch - data is managed by upstream MultiEventManager
        logger.info(f"Fitting forecaster with {self.bot_config.training_days} days of history")
        self.forecaster.fit(
            n_days=self.bot_config.training_days,
            skip_fetch=True,
        )

        # Log WebSocket status
        if self.bot_config.disable_websocket:
            logger.info("WebSocket disabled (--no-ws flag)")
        else:
            logger.info("WebSocket enabled for real-time orderbook updates")

        # Create Kelly bot with wallet address and optional external user stream
        # If external user stream is provided (from MultiEventManager), use it
        # Otherwise, KellyTradingBot will create its own if API credentials available
        # Use proxy wallet if configured (positions are held there)
        self.kelly_bot = KellyTradingBot(
            clob_client=self.clob_client,
            config=self.kelly_config,
            probability_model=self._get_probabilities,
            dry_run=self.bot_config.dry_run,
            wallet_address=_get_wallet_for_positions(),
            api_key=os.getenv("CLOB_API_KEY"),
            api_secret=os.getenv("CLOB_API_SECRET"),
            api_passphrase=os.getenv("CLOB_API_PASSPHRASE"),
            external_user_stream=self._external_user_stream,
            disable_websocket=self.bot_config.disable_websocket,
            event_name=self.bot_config.event_name,
        )

        # Setup Kelly bot
        await self.kelly_bot.setup(
            initial_capital=self.bot_config.initial_capital,
            bins=bins,
        )

        # Register WebSocket callback for real-time trading (if enabled)
        if (self.kelly_bot.orderbook_manager and
            self.kelly_bot.orderbook_manager.config.enabled):
            self.kelly_bot.orderbook_manager.on_significant_update = self._on_orderbook_update
            logger.info("Registered WebSocket callback for real-time orderbook updates")

        # Determine settlement date if not set
        if self.settlement_date is None:
            # Default: assume 7-day contract ending at next noon
            today = self.forecaster.contract_utils.get_current_contract_date()
            self.settlement_date = today + timedelta(days=6)
            logger.info(f"Using default settlement date: {self.settlement_date}")

        # Compute market start date
        # For a Jan 27 - Feb 3 market: start is Jan 27, settlement is Feb 3
        # The 7 counting days are: Jan 27, 28, 29, 30, 31, Feb 1, Feb 2
        # Settlement happens at noon Feb 3 (end of the Feb 2 contract day)
        self.market_start_date = self.settlement_date - timedelta(days=7)
        logger.info(f"Market counting period: {self.market_start_date} to {self.settlement_date}")

        self._setup_complete = True
        logger.info("GAS-Kelly trading bot setup complete")

    async def shutdown(self) -> None:
        """Shutdown the trading bot."""
        logger.info("Shutting down GAS-Kelly trading bot...")
        self._running = False

        if self.kelly_bot:
            await self.kelly_bot.shutdown()

        self._setup_complete = False
        logger.info("GAS-Kelly trading bot shutdown complete")

    def _get_market_cumulative_count(self) -> int:
        """
        Get cumulative count for the market's specific date range.

        Uses the forecaster's event store to count tweets from market_start_date
        to now, which aligns with the market's counting period.

        Returns:
            Total tweets in the market's counting period so far
        """
        if self.market_start_date is None:
            # Fallback to forecaster's default method
            return self.forecaster.get_7day_cumulative_count()

        now = datetime.now(self.forecaster.contract_utils.tz)
        today = self.forecaster.contract_utils.get_contract_date(now)

        total = 0

        # Count from market start date to today
        current_date = self.market_start_date
        while current_date <= today:
            if current_date == today:
                # For today, only count events so far
                total += self.forecaster.get_current_count(current_date)
            else:
                # For past days, count all events
                total += self.forecaster.event_store.get_contract_day_count(current_date)
            current_date += timedelta(days=1)

        return total

    def _get_timing(self) -> tuple[float, float]:
        """
        Get current timing information.

        Returns:
            Tuple of (hours_elapsed, hours_remaining)
        """
        return self.forecaster.get_settlement_timing(self.settlement_date)

    def _compute_forecast_breakdown(self) -> Optional[Dict]:
        """
        Compute breakdown of forecast into actual past + forecasted remaining.

        Returns:
            Dict with breakdown info, or None if not available
        """
        if not self.market_start_date or not self.settlement_date:
            return None

        now = datetime.now(self.forecaster.contract_utils.tz)
        today = self.forecaster.contract_utils.get_contract_date(now)

        # Last counting day is one day before settlement
        last_counting_day = self.settlement_date - timedelta(days=1)

        # Count completed days and their tweets
        past_count = 0
        past_days = 0
        current_date = self.market_start_date
        while current_date < today and current_date <= last_counting_day:
            past_count += self.forecaster.event_store.get_contract_day_count(current_date)
            past_days += 1
            current_date += timedelta(days=1)

        # Calculate remaining days
        if today < self.market_start_date:
            # Before counting window starts - use window start, not current date
            remaining_days = (last_counting_day - self.market_start_date).days + 1
        elif today > last_counting_day:
            remaining_days = 0
        else:
            remaining_days = (last_counting_day - today).days + 1

        return {
            "past_count": past_count,
            "past_days": past_days,
            "remaining_days": remaining_days,
        }

    async def run_tick(self, log_header: bool = True) -> Optional[TickResult]:
        """
        Run a single optimization tick.

        Thread-safe: uses lock to prevent concurrent tick execution.

        Args:
            log_header: If True, log tick header and detailed comparison table.
                       Set to False for fast ticks to reduce log noise.

        Returns:
            TickResult from Kelly optimizer, or None if not ready or skipped
        """
        if not self._setup_complete:
            logger.warning("Bot not setup, skipping tick")
            return None

        # Try to acquire lock without blocking
        if self._tick_lock.locked():
            logger.debug("Tick already in progress, skipping")
            return None

        async with self._tick_lock:
            self._tick_in_progress = True
            try:
                return await self._run_tick_impl(log_header=log_header)
            finally:
                self._tick_in_progress = False

    async def _run_tick_impl(self, log_header: bool = True) -> Optional[TickResult]:
        """
        Internal tick implementation (called with lock held).

        Uses cached probabilities - does NOT recompute Monte Carlo.
        Monte Carlo is recomputed in _run_slow_tick() when fresh data arrives.
        """
        tick_start = datetime.now(self.forecaster.contract_utils.tz)

        try:
            # 0. Check data freshness before trading
            if self._data_freshness_checker is not None:
                is_fresh, age_seconds = self._data_freshness_checker()
                if not is_fresh:
                    logger.warning(
                        f"Data is stale ({age_seconds:.0f}s old), skipping trading tick. "
                        f"Waiting for fresh data..."
                    )
                    return TickResult(
                        tick_start_time=tick_start,
                        num_candidates=0,
                        num_executed=0,
                        total_utility_gain=0.0,
                        executions=[],
                        elapsed_seconds=0.0,
                    )

            # 1. Get timing info
            hours_elapsed, hours_remaining = self._get_timing()

            # 2. Get current cumulative count for the market's date range
            current_count = self._get_market_cumulative_count()

            # 2b. Check for count regression (impossible for tweets - indicates stale data)
            if self._last_known_count is not None and current_count < self._last_known_count:
                logger.warning(
                    f"Count regressed from {self._last_known_count} to {current_count}! "
                    f"This indicates stale/corrupted data. Skipping trading tick."
                )
                return TickResult(
                    tick_start_time=tick_start,
                    num_candidates=0,
                    num_executed=0,
                    total_utility_gain=0.0,
                    executions=[],
                    elapsed_seconds=0.0,
                )

            # Update last known count
            if current_count > (self._last_known_count or 0):
                self._last_known_count = current_count

            # 2c. Get effective count for dead bin detection (uses max of computed and authoritative)
            effective_count = self.get_effective_count_for_dead_bins(current_count)

            # 3. Log header info (only for slow ticks / explicit requests)
            if log_header:
                logger.info(f"Timing: {hours_elapsed:.1f}h elapsed, {hours_remaining:.1f}h remaining")
                if effective_count > current_count:
                    logger.info(
                        f"Current count: {current_count} (posts), {effective_count} (authoritative)"
                    )
                else:
                    logger.info(f"Current 7-day cumulative count: {current_count}")

                # Log dead bins using effective count
                dead_bins = [i for i, (lower, upper) in enumerate(self._market_bins) if upper < effective_count]
                if dead_bins:
                    logger.info(f"Dead bins: {dead_bins}")

            # 4. Check if forecast is available - skip trading if not
            if self._cached_forecast_mean is None:
                logger.warning("Forecast not available, skipping trading tick")
                return TickResult(
                    tick_start_time=tick_start,
                    num_candidates=0,
                    num_executed=0,
                    total_utility_gain=0.0,
                    executions=[],
                    elapsed_seconds=0.0,
                )

            # 5. Fetch orderbooks for table display (slow ticks only)
            if log_header and self.kelly_bot and self.kelly_bot.orderbook_manager:
                dead_bins = set(i for i, (lower, upper) in enumerate(self._market_bins) if upper < effective_count)
                for bin_idx, token_id in self.kelly_bot.bin_token_ids.items():
                    if bin_idx in dead_bins:
                        continue
                    if not self.kelly_bot.orderbook_manager.get_orderbook(token_id):
                        await self.kelly_bot.orderbook_manager.fetch_orderbook(token_id, bin_idx)

            # 6. Log probability table BEFORE trading (slow ticks only)
            if log_header:
                probabilities = self._get_probabilities(
                    current_count=current_count,
                    hours_elapsed=hours_elapsed,
                    hours_remaining=hours_remaining,
                )
                # Update portfolio probabilities so table shows correct Kelly c* values
                # (otherwise get_reservation_prices() uses stale probabilities)
                if self.kelly_bot and self.kelly_bot.portfolio:
                    self.kelly_bot.update_probabilities(
                        current_count=effective_count,
                        hours_elapsed=hours_elapsed,
                        hours_remaining=hours_remaining,
                    )
                self._log_probability_comparison(
                    probabilities=probabilities,
                    current_count=effective_count,  # Use effective count for dead bin display
                    hours_elapsed=hours_elapsed,
                    hours_remaining=hours_remaining,
                    forecast_mean=self._cached_forecast_mean,
                    forecast_std=self._cached_forecast_std,
                    forecast_breakdown=self._cached_forecast_breakdown,
                    forecast_time=self._cached_forecast_time,
                )

            # 7. Run Kelly optimization tick
            # Use effective_count for dead bin detection to avoid betting on impossible bins
            # Pass verbose=True on slow ticks to show rejection reasons
            result = await self.kelly_bot.run_tick(
                current_count=effective_count,
                hours_elapsed=hours_elapsed,
                hours_to_settlement=hours_remaining,
                forecast_mean=self._cached_forecast_mean,
                forecast_std=self._cached_forecast_std or 0,
                verbose=log_header,  # verbose on slow ticks only
            )

            # 8. Log results
            if log_header or result.num_executed > 0:
                self._log_tick_result(result)

            self._last_tick_time = tick_start
            return result

        except Exception as e:
            logger.error(f"Error in tick: {e}", exc_info=True)
            return None

    def _log_tick_result(self, result: TickResult) -> None:
        """Log the result of a tick."""
        logger.info(
            f"Tick result: candidates={result.num_candidates}, "
            f"executed={result.num_executed}, "
            f"utility_gain={result.total_utility_gain:.6f}, "
            f"elapsed={result.elapsed_seconds:.2f}s"
        )

        for execution in result.executions:
            if execution.success:
                # Note: size/price are from candidate, not fill (fill comes via WebSocket)
                logger.info(
                    f"  Order placed: {execution.candidate.action.value} "
                    f"bin={execution.candidate.bin_index} "
                    f"size={execution.candidate.size:.2f} @ {execution.candidate.price:.4f} "
                    f"(pending fill)"
                )
            else:
                logger.warning(f"  Order failed: {execution.error}")

    def _log_probability_comparison(
        self,
        probabilities: List[float],
        current_count: int,
        hours_elapsed: float,
        hours_remaining: float,
        forecast_mean: Optional[float],
        forecast_std: Optional[float],
        forecast_breakdown: Optional[Dict] = None,
        forecast_time: Optional[datetime] = None,
    ) -> None:
        """
        Log comparison of forecast probabilities vs market-implied probabilities.

        Shows a table with: bin range, forecast prob, market prices, thresholds, signals.
        Uses stake-based ROI model for thresholds:
        - Y.Thr = max YES price to buy YES = (p_f - c) / (1 + r)
        - N.Thr = max NO price to buy NO = (1 - p_f - c) / (1 + r)
        """
        if not self.kelly_bot or not self.kelly_bot.orderbook_manager:
            return

        dead_bins = set(i for i, (lower, upper) in enumerate(self._market_bins) if upper < current_count)
        num_dead = len(dead_bins)
        num_live = len(self._market_bins) - num_dead

        # Import edge buffer calculation for threshold display
        from ..kelly.candidates import compute_buy_yes_threshold
        edge_config = self.kelly_config.edge_buffer

        # Get event name from settlement date
        event_name = "Unknown"
        if self.market_start_date and self.settlement_date:
            start_str = self.market_start_date.strftime("%b %d")
            end_str = self.settlement_date.strftime("%b %d")
            event_name = f"{start_str} - {end_str}"

        # Format time remaining
        days_remaining = hours_remaining / 24
        if days_remaining >= 1:
            time_str = f"{days_remaining:.1f}d"
        else:
            time_str = f"{hours_remaining:.1f}h"

        # Build forecast breakdown string
        if forecast_mean is None:
            forecast_str = "N/A"
            std_str = ""
        elif forecast_breakdown:
            past_count = forecast_breakdown["past_count"]
            past_days = forecast_breakdown["past_days"]
            remaining_days = forecast_breakdown["remaining_days"]
            forecast_remaining = forecast_mean - past_count
            forecast_str = f"{past_count} ({past_days}d actual) + {forecast_remaining:.0f} ({remaining_days}d forecast) = {forecast_mean:.0f}"
            std_str = f"  (std: {forecast_std:.0f})" if forecast_std else ""
        else:
            forecast_str = f"{forecast_mean:.0f}"
            std_str = f" ± {forecast_std:.0f}" if forecast_std else ""

        # Use forecast generation time (local timezone) for "as of" timestamp
        if forecast_time:
            time_stamp = forecast_time.strftime("%H:%M:%S")
        else:
            time_stamp = datetime.now().strftime("%H:%M:%S")

        # Get Kelly reservation prices (c*) from portfolio
        kelly_yes_prices = {}
        kelly_no_prices = {}
        if self.kelly_bot and self.kelly_bot.portfolio:
            try:
                yes_prices, no_prices = self.kelly_bot.portfolio.get_reservation_prices(
                    self.kelly_config.w_floor
                )
                for bin_idx in range(len(yes_prices)):
                    kelly_yes_prices[bin_idx] = yes_prices[bin_idx]
                    kelly_no_prices[bin_idx] = no_prices[bin_idx]
            except Exception as e:
                logger.debug(f"Could not get Kelly c* prices: {e}")

        # Header with context
        logger.info("")
        logger.info("=" * 120)
        logger.info(f"  Event: {event_name}  |  Count: {current_count}  |  Time Left: {time_str}  |  Forecast @{time_stamp}")
        logger.info(f"  Forecast: {forecast_str}{std_str}")
        logger.info(f"  Bins: {num_live} live, {num_dead} dead  |  Edge: r={edge_config.required_roi:.0%}, c_mid={edge_config.friction_mid:.0%}, c_tail={edge_config.friction_tail:.0%}")
        logger.info("-" * 120)
        logger.info(f"{'Bin':<4} {'Range':<9} {'Model':>6} {'c*_Y':>6} {'Y.Ask':>6} {'Y.Thr':>6} {'c*_N':>6} {'N.Ask':>6} {'N.Thr':>6} {'Y.Pos':>7} {'N.Pos':>7} {'Signal':>8}")
        logger.info("-" * 120)

        for bin_idx, (lower, upper) in enumerate(self._market_bins):
            if bin_idx in dead_bins:
                continue  # Skip dead bins

            # Get model probability
            model_prob = probabilities[bin_idx] if bin_idx < len(probabilities) else 0.0

            # Get Kelly reservation prices (c*)
            kelly_yes = kelly_yes_prices.get(bin_idx, model_prob)
            kelly_no = kelly_no_prices.get(bin_idx, 1.0 - model_prob)

            # Get orderbook for this bin
            token_id = self.kelly_bot.bin_token_ids.get(bin_idx)
            if not token_id:
                continue

            orderbook = self.kelly_bot.orderbook_manager.get_orderbook(token_id)

            # Get ask prices
            yes_ask = orderbook.best_yes_ask if orderbook else None
            no_ask = orderbook.best_no_ask if orderbook else None

            # Format bin range
            if upper == float('inf'):
                range_str = f"{lower}+"
            else:
                range_str = f"{lower}-{upper}"

            # Calculate thresholds using Kelly c* (matches actual trading logic)
            # Y.Thr: max YES price to pay for BUY_YES (based on Kelly c*_YES)
            yes_threshold = compute_buy_yes_threshold(kelly_yes, edge_config)
            # N.Thr: max NO price to pay for BUY_NO (based on Kelly c*_NO)
            no_threshold = compute_buy_yes_threshold(kelly_no, edge_config)

            # Determine signal: trade if market ask <= threshold
            signal = ""
            if no_ask is not None and no_ask <= no_threshold:
                signal = "BUY_NO"
            elif yes_ask is not None and yes_ask <= yes_threshold:
                signal = "BUY_YES"

            # Get current positions for this bin
            yes_pos = 0.0
            no_pos = 0.0
            if self.kelly_bot and self.kelly_bot.portfolio:
                position = self.kelly_bot.portfolio.get_position(bin_idx)
                if position:
                    yes_pos = position.yes_shares
                    no_pos = position.no_shares

            # Format output - all prices as percentages
            kelly_yes_str = f"{kelly_yes * 100:5.1f}%"
            kelly_no_str = f"{kelly_no * 100:5.1f}%"
            yes_ask_str = f"{yes_ask * 100:5.1f}%" if yes_ask is not None else "  N/A "
            yes_thr_str = f"{yes_threshold * 100:5.1f}%"
            no_ask_str = f"{no_ask * 100:5.1f}%" if no_ask is not None else "  N/A "
            no_thr_str = f"{no_threshold * 100:5.1f}%"
            yes_pos_str = f"{yes_pos:6.0f}" if yes_pos > 0 else "     -"
            no_pos_str = f"{no_pos:6.0f}" if no_pos > 0 else "     -"

            logger.info(
                f"{bin_idx:<4} {range_str:<9} {model_prob:>5.1%} "
                f"{kelly_yes_str:>6} {yes_ask_str:>6} {yes_thr_str:>6} "
                f"{kelly_no_str:>6} {no_ask_str:>6} {no_thr_str:>6} "
                f"{yes_pos_str:>7} {no_pos_str:>7} {signal:>8}"
            )

        logger.info("-" * 120)

        # Log portfolio summary
        summary = self.kelly_bot.get_portfolio_summary()
        logger.info(
            f"  Portfolio: capital=${summary.get('capital', 0):.2f}, "
            f"invested=${summary.get('total_collateral', 0):.2f}"
        )
        logger.info("")

    async def run(self) -> None:
        """
        Main trading loop with two-pronged approach:

        1. Slow loop (every slow_tick_interval_seconds, default 5 min):
           - Recomputes Monte Carlo forecast IF fresh data is available
           - Updates cached probabilities
           - Runs Kelly optimization tick

        2. Fast loop (every fast_tick_interval_seconds, default 30 sec):
           - Uses cached probabilities (no Monte Carlo recomputation)
           - Runs Kelly optimization tick
           - Catches opportunities between slow ticks

        3. WebSocket callback (on orderbook changes):
           - Uses cached probabilities
           - Runs Kelly optimization tick immediately

        When using shared EventStore (from MultiEventManager):
        - Does NOT fetch data from API (manager handles that)
        - Waits for notify_data_refreshed() signal before recomputing Monte Carlo
        """
        if not self._setup_complete:
            raise RuntimeError("Bot not setup. Call setup() first.")

        self._running = True
        self._stop_event = asyncio.Event()
        slow_interval = self.bot_config.slow_tick_interval_seconds
        fast_interval = self.bot_config.fast_tick_interval_seconds

        logger.info(
            f"Starting trading loop: slow_interval={slow_interval}s, "
            f"fast_interval={fast_interval}s, dry_run={self.bot_config.dry_run}"
        )

        # Track last slow tick time
        last_slow_tick = datetime.now(self.forecaster.contract_utils.tz)

        # Run initial slow tick to establish baseline probabilities
        await self._run_slow_tick()

        while self._running:
            try:
                # Check if we're past settlement
                hours_elapsed, hours_remaining = self._get_timing()
                if hours_remaining <= 0:
                    logger.info("Past settlement time, stopping trading loop")
                    break

                now = datetime.now(self.forecaster.contract_utils.tz)

                # Check if it's time for a slow tick
                time_since_slow = (now - last_slow_tick).total_seconds()
                if time_since_slow >= slow_interval:
                    await self._run_slow_tick()
                    last_slow_tick = now
                else:
                    # Run fast tick (uses cached probabilities)
                    await self._run_fast_tick()

                # Sleep until next fast tick (interruptible by stop_event)
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=fast_interval
                    )
                    # If we get here, stop was requested
                    logger.info("Stop event received")
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, continue loop
                    pass

            except asyncio.CancelledError:
                logger.info("Trading loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                # Continue after error, with short backoff
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=min(fast_interval, 30)
                    )
                    break
                except asyncio.TimeoutError:
                    pass

        self._running = False
        logger.info("Trading loop stopped")

    async def _run_slow_tick(self) -> Optional[TickResult]:
        """
        Run a slow tick that may recompute Monte Carlo forecasts.

        Called every slow_tick_interval_seconds (default 5 min).
        Only recomputes Monte Carlo if fresh data is available
        (signaled by notify_data_refreshed() from upstream MultiEventManager).
        """
        tick_start = datetime.now(self.forecaster.contract_utils.tz)
        self._tick_count += 1

        logger.info(f"=== Slow Tick {self._tick_count} at {tick_start.isoformat()} ===")

        # Recompute Monte Carlo only if upstream manager signaled fresh data
        if self._fresh_data_available:
            logger.info("Fresh data available from upstream manager")
            self._fresh_data_available = False  # Reset flag
            self._recompute_monte_carlo()
        else:
            logger.debug("No fresh data, using cached probabilities")

        # Run the tick (will use cached probabilities)
        return await self.run_tick()

    async def _run_fast_tick(self) -> Optional[TickResult]:
        """
        Run a fast tick using cached probabilities.

        Called every fast_tick_interval_seconds (default 30 sec).
        Does NOT recompute Monte Carlo - uses cached probabilities.
        """
        if not self._cached_probabilities:
            logger.debug("No cached probabilities, skipping fast tick")
            return None

        # Run tick with cached probabilities (no logging of tick number)
        return await self.run_tick(log_header=False)

    def _recompute_monte_carlo(self) -> None:
        """
        Recompute Monte Carlo forecast and update cached probabilities.
        """
        try:
            # Get current count for dead bin detection
            current_count = self._get_market_cumulative_count()

            # Recompute forecast
            if self.market_start_date and self.settlement_date:
                forecast = self.forecaster.forecast_for_event_window(
                    market_start_date=self.market_start_date,
                    settlement_date=self.settlement_date,
                )
                forecast_breakdown = self._compute_forecast_breakdown()
            else:
                forecast = self.forecaster.forecast_7day_distribution(use_cache=False)
                forecast_breakdown = None

            # Update forecast cache
            self._cached_forecast_mean = forecast.mean
            self._cached_forecast_std = forecast.std
            self._cached_forecast_breakdown = forecast_breakdown
            self._cached_forecast_time = datetime.now()

            # Recompute bin probabilities
            self._cached_probabilities = self._compute_probabilities(current_count)

            self._last_monte_carlo_time = datetime.now(self.forecaster.contract_utils.tz)

            logger.info(
                f"Monte Carlo recomputed: mean={forecast.mean:.1f}, "
                f"std={forecast.std:.1f}, "
                f"90% CI=[{forecast.p5:.0f}, {forecast.p95:.0f}]"
            )

        except Exception as e:
            logger.error(f"Error recomputing Monte Carlo: {e}", exc_info=True)

    def stop(self) -> None:
        """Signal the trading loop to stop."""
        self._running = False
        if hasattr(self, '_stop_event') and self._stop_event:
            self._stop_event.set()

    def get_state_summary(self) -> Dict:
        """
        Get summary of current bot state.

        Returns:
            Dict with state information
        """
        hours_elapsed, hours_remaining = self._get_timing()

        summary = {
            "running": self._running,
            "setup_complete": self._setup_complete,
            "tick_count": self._tick_count,
            "dry_run": self.bot_config.dry_run,
            "settlement_date": self.settlement_date.isoformat() if self.settlement_date else None,
            "hours_elapsed": hours_elapsed,
            "hours_remaining": hours_remaining,
        }

        # Add forecaster state
        if self._setup_complete:
            summary["forecaster"] = self.forecaster.get_state_summary()
            summary["portfolio"] = self.kelly_bot.get_portfolio_summary()

            # Add bot status including pending orders
            kelly_status = self.kelly_bot.get_status()
            summary["pending_orders"] = kelly_status.get("pending_orders", {})
            summary["user_stream"] = kelly_status.get("user_stream", {})

        return summary
