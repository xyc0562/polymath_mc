"""
Multi-event manager for concurrent trading across multiple events.

Orchestrates multiple GASKellyTradingBot instances, each trading on a different
event, all drawing from a shared capital pool.

Key features:
- Shared capital pool across all events
- Shared EventStore for tweet data (avoids duplicate API calls)
- Pre-fetches all tweet data, refreshes periodically (especially at noon ET)
- Validates computed count against XTracker authoritative count
- State reconstruction on restart from Polymarket API
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from zoneinfo import ZoneInfo

import requests
from py_clob_client.client import ClobClient

# Polymarket Data API for fetching positions
POLYMARKET_DATA_API = "https://data-api.polymarket.com"

from .config import ForecasterConfig
from .trading_bot import GASKellyTradingBot, TradingBotConfig
from .data import EventStore, ContractDayUtils, XTrackerClient as PostsXTrackerClient
from ..musk_tweet_count import XTrackerClient as TrackingsXTrackerClient
from ..kelly.config import KellyConfig, EventTradingRulesConfig
from ..kelly.capital_pool import CapitalPool, CapitalPoolConfig
from ..kelly.user_stream import UserStreamClient, FillEvent

logger = logging.getLogger(__name__)

# Count validation threshold - warn if computed vs API count differs by more than this
COUNT_MISMATCH_THRESHOLD = 5


@dataclass
class EventInfo:
    """Information about a tradeable event."""

    event_id: str
    title: str
    short_name: str
    settlement_date: date
    market_start_date: date  # First counting day
    bins: List[Dict]  # Bin definitions with token_ids
    condition_id: Optional[str] = None


@dataclass
class ActiveEvent:
    """Tracks an actively trading event."""

    info: EventInfo
    bot: GASKellyTradingBot
    task: asyncio.Task
    started_at: datetime
    allocated_capital: float


@dataclass
class MultiEventConfig:
    """Configuration for multi-event manager."""

    # Capital pool configuration
    capital_pool: CapitalPoolConfig = field(default_factory=CapitalPoolConfig)

    # Maximum capital per event (from KellyConfig.collateral.c_event_max)
    max_per_event: float = 500.0

    # Trading bot configuration (shared across events)
    tick_interval_seconds: int = 300
    dry_run: bool = True

    # Forecaster configuration
    training_days: int = 45
    use_gas: bool = True

    # How often to refresh posts data (seconds)
    # XTracker posts update ~every 5 minutes, so 2.5 min refresh is safe
    posts_refresh_interval: int = 150  # 2.5 minutes

    # Maximum age of data before it's considered stale (seconds)
    # If data is older than this, skip trading until refreshed
    max_data_age_seconds: int = 300  # 5 minutes

    # How often to validate counts against XTracker API (seconds)
    count_validation_interval: int = 900  # 15 minutes

    # How often to check for new events (seconds)
    event_scan_interval: int = 3600  # 1 hour

    # Projection model type: "asymmetric" (default) or "normal"
    # - asymmetric: Uses actual Monte Carlo samples (preserves right-skew)
    # - normal: Approximates with Normal distribution (symmetric)
    projection_model: str = "asymmetric"

    # Start trading this many hours before settlement
    # (events closer to settlement than this won't be started)
    # NOTE: This is a global minimum; per-category rules may have stricter limits
    min_hours_before_settlement: float = 24.0

    # Event trading rules configuration
    # Controls when trading is allowed based on event duration and counting status
    # If None, uses EventTradingRulesConfig.default()
    event_trading_rules: Optional[EventTradingRulesConfig] = None

    # Maximum completed events to keep in history (for memory management)
    max_completed_events: int = 100

    # Maximum days of tweet data to keep in EventStore
    max_event_store_days: int = 60

    # How often to log health status (seconds)
    health_log_interval: int = 3600  # Every hour


class MultiEventManager:
    """
    Manages concurrent trading across multiple events.

    Features:
    - Shared capital pool across all events
    - Automatic event discovery and lifecycle management
    - Shared EventStore for tweet data (avoids duplicate API calls)
    - Graceful shutdown and capital return

    Usage:
        manager = MultiEventManager(
            clob_client=clob_client,
            kelly_config=kelly_config,
            forecaster_config=forecaster_config,
            config=MultiEventConfig(
                capital_pool=CapitalPoolConfig(total_capital=5000),
            ),
        )

        # Add events to trade
        await manager.add_event(event_info)

        # Run all events concurrently
        await manager.run()
    """

    def __init__(
        self,
        clob_client: ClobClient,
        kelly_config: KellyConfig,
        forecaster_config: ForecasterConfig,
        config: MultiEventConfig,
        wallet_address: str,
        event_discovery_callback: Optional[Any] = None,
        initial_events: Optional[List[EventInfo]] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        api_passphrase: Optional[str] = None,
    ):
        """
        Initialize multi-event manager.

        Args:
            clob_client: Authenticated Polymarket CLOB client
            kelly_config: Kelly configuration (shared across events)
            forecaster_config: Forecaster configuration
            config: Multi-event manager configuration
            wallet_address: Wallet address for fetching positions on restart
            event_discovery_callback: Optional async function that returns List[EventInfo]
                                      Called periodically to discover new events.
            initial_events: Optional list of EventInfo objects to trade.
                           If provided, these events will be used instead of discovery.
            api_key: Polymarket API key for user stream (auto-detects from env if not provided)
            api_secret: Polymarket API secret for user stream
            api_passphrase: Polymarket API passphrase for user stream
        """
        import os

        self.wallet_address = wallet_address
        self.event_discovery_callback = event_discovery_callback
        self.initial_events = initial_events or []
        self.clob_client = clob_client
        self.kelly_config = kelly_config
        self.forecaster_config = forecaster_config
        self.config = config

        # API credentials for user stream (auto-detect from env if not provided)
        self._api_key = api_key or os.getenv("CLOB_API_KEY", "")
        self._api_secret = api_secret or os.getenv("CLOB_API_SECRET", "")
        self._api_passphrase = api_passphrase or os.getenv("CLOB_API_PASSPHRASE", "")

        # Capital pool
        self.capital_pool = CapitalPool(config.capital_pool, max_per_event=config.max_per_event)

        # Contract-day utilities (shared across all components)
        self.contract_utils = ContractDayUtils(
            timezone=forecaster_config.timezone,
            boundary_hour=forecaster_config.contract_boundary_hour,
        )

        # Timezone for scheduling
        self._tz = ZoneInfo(forecaster_config.timezone)

        # Shared XTracker client for /posts endpoint (single session)
        self.posts_xtracker_client = PostsXTrackerClient()

        # Shared XTracker client for /trackings endpoint (authoritative counts)
        self.trackings_xtracker_client = TrackingsXTrackerClient()

        # Shared EventStore for all bots (avoids duplicate tweet fetches)
        # All events use the same tweet data, just different date ranges
        self.shared_event_store = EventStore(
            contract_utils=self.contract_utils,
            xtracker_client=self.posts_xtracker_client,
        )

        # Track if data has been pre-fetched
        self._data_prefetched = False

        # Track last refresh time for smart refresh scheduling
        self._last_full_refresh: Optional[datetime] = None
        self._last_refresh_contract_date: Optional[date] = None

        # Track when data was last successfully refreshed (for freshness check)
        self._last_data_refresh_time: Optional[datetime] = None

        # Global UserStreamClient for fill confirmations (shared across all bots)
        # This is more efficient than one UserStreamClient per bot since
        # all bots use the same wallet and receive the same fill events
        # Enabled even in dry_run mode to monitor fills from manual trades
        self.user_stream: Optional[UserStreamClient] = None
        if self._api_key and self._api_secret:
            self.user_stream = UserStreamClient(
                api_key=self._api_key,
                api_secret=self._api_secret,
                api_passphrase=self._api_passphrase,
            )
            # Wire up fill handler to route to correct bot
            self.user_stream.on_fill = self._handle_global_fill
            logger.info("Global UserStreamClient created for fill confirmations")
        else:
            logger.warning("No API credentials for user stream - fill confirmations disabled")

        # Mapping from token_id to (event_id, bin_index) for fill routing
        self._token_to_event: Dict[str, Tuple[str, int]] = {}

        # Active events: event_id -> ActiveEvent
        self._active_events: Dict[str, ActiveEvent] = {}

        # Pending events (discovered but not yet started)
        self._pending_events: Dict[str, EventInfo] = {}

        # Completed events (for history)
        self._completed_events: List[str] = []

        # Lock for event dictionaries (asyncio lock for async methods)
        self._events_lock = asyncio.Lock()

        # Control
        self._running = False
        self._stop_event: Optional[asyncio.Event] = None

        # Health tracking
        self._start_time: Optional[datetime] = None
        self._last_cleanup_time: Optional[datetime] = None
        self._last_health_log_time: Optional[datetime] = None
        self._total_events_started: int = 0
        self._total_events_completed: int = 0
        self._total_trades_executed: int = 0
        self._errors_count: int = 0

        if config.capital_pool.total_capital > 0:
            logger.info(
                f"MultiEventManager initialized with capital pool: "
                f"${config.capital_pool.total_capital:.2f}"
            )
        else:
            logger.info(
                "MultiEventManager initialized (capital will be auto-detected from API)"
            )

    @property
    def num_active_events(self) -> int:
        """Number of events currently trading."""
        return len(self._active_events)

    async def _cleanup_old_data(self) -> None:
        """
        Clean up old data to prevent memory leaks over long runs.

        Called periodically in main loop.
        """
        # 1. Trim completed events list
        async with self._events_lock:
            if len(self._completed_events) > self.config.max_completed_events:
                # Keep only the most recent events
                excess = len(self._completed_events) - self.config.max_completed_events
                self._completed_events = self._completed_events[excess:]
                logger.debug(f"Trimmed {excess} old completed events from history")

        # 2. Clean up old EventStore data
        self._cleanup_event_store()

        # 3. Check for zombie active events (tasks that died without cleanup)
        await self._check_for_zombie_events()

        # 4. Record cleanup time
        self._last_cleanup_time = datetime.now(self._tz)

    def _cleanup_event_store(self) -> None:
        """Remove old tweet data from EventStore to prevent memory growth."""
        removed = self.shared_event_store.cleanup_old_data(
            keep_days=self.config.max_event_store_days
        )
        if removed > 0:
            logger.info(f"Cleaned up {removed} old days from EventStore")

    async def _check_for_zombie_events(self) -> None:
        """
        Check for and clean up zombie events (active events with dead tasks).

        This handles cases where a bot task crashed without proper cleanup.
        """
        zombies = []

        async with self._events_lock:
            for event_id, active in self._active_events.items():
                if active.task.done():
                    # Task finished but event still in active - it's a zombie
                    zombies.append((event_id, active))

        for event_id, active in zombies:
            logger.warning(f"Found zombie event {event_id}, cleaning up...")
            self._errors_count += 1  # Zombie events indicate something went wrong
            try:
                # Try to get the exception if any
                if active.task.exception():
                    logger.error(f"Zombie event {event_id} died with: {active.task.exception()}")
            except (asyncio.CancelledError, asyncio.InvalidStateError):
                pass

            # Force cleanup
            await self._cleanup_event(event_id, active.bot)

    async def _initialize_capital_from_api(self) -> None:
        """
        Initialize capital pool from Polymarket API.

        Fetches USDC balance and existing positions to determine total capital.
        Called automatically if capital_pool.total_capital is 0.
        """
        logger.info("Auto-detecting capital from Polymarket API...")

        # Fetch USDC balance
        usdc_balance = await self.fetch_usdc_balance()

        # Fetch existing positions with actual values from API
        all_positions = await self.fetch_all_positions()
        position_value = 0.0

        # Use actual values from API (cost basis / initialValue)
        for token_id, pos_info in all_positions.items():
            position_value += pos_info["cost_basis"]

        total_capital = usdc_balance + position_value

        # Set capital in pool
        await self.capital_pool.set_total_from_api(total_capital)

        logger.info(
            f"Capital auto-detected: ${usdc_balance:.2f} USDC + "
            f"${position_value:.2f} positions = ${total_capital:.2f} total"
        )

    async def prefetch_shared_data(self, n_days: Optional[int] = None) -> None:
        """
        Pre-fetch tweet data that will be shared across all events.

        This should be called once before starting any events. All bots will
        use this shared data instead of making duplicate API calls.

        Args:
            n_days: Number of days of history to fetch (default: config.training_days)
        """
        if self._data_prefetched:
            logger.debug("Data already prefetched, skipping")
            return

        if n_days is None:
            n_days = self.config.training_days

        logger.info(f"Pre-fetching {n_days} days of tweet data for all events...")

        # Fetch all tweet history once
        self.shared_event_store.refresh_from_api(n_days)

        # Get data range for logging
        date_range = self.shared_event_store.get_date_range()
        if date_range[0] and date_range[1]:
            logger.info(
                f"Pre-fetched tweet data from {date_range[0]} to {date_range[1]}"
            )

        self._data_prefetched = True
        self._last_data_refresh_time = datetime.now(self._tz)

    def is_data_fresh(self) -> Tuple[bool, float]:
        """
        Check if data is fresh enough for trading.

        Returns:
            Tuple of (is_fresh, age_seconds)
            is_fresh is True if data is within max_data_age_seconds
        """
        if self._last_data_refresh_time is None:
            return False, float('inf')

        now = datetime.now(self._tz)
        age = (now - self._last_data_refresh_time).total_seconds()
        is_fresh = age <= self.config.max_data_age_seconds

        return is_fresh, age

    async def refresh_shared_data(self) -> int:
        """
        Refresh shared tweet data with latest from API.

        Smart refresh logic:
        - If contract day changed (noon ET passed), do full refresh and refit models
        - Otherwise, do incremental refresh for recent days

        After refresh, notifies all active bots that fresh data is available.

        Returns:
            Number of new events added
        """
        now = datetime.now(self._tz)
        current_contract_date = self.contract_utils.get_current_contract_date()

        # Check if contract day changed (noon ET boundary crossed)
        contract_day_changed = (
            self._last_refresh_contract_date is not None and
            current_contract_date != self._last_refresh_contract_date
        )

        if contract_day_changed:
            logger.info(
                f"Contract day changed: {self._last_refresh_contract_date} -> {current_contract_date}. "
                f"Triggering full refresh and model refit."
            )
            # Do full refresh to capture complete previous day
            new_events = await self._do_full_refresh()

            # Refit models for all active events
            await self._refit_active_event_models()
        else:
            # Incremental refresh - just recent data
            new_events = await self._do_incremental_refresh()

        self._last_refresh_contract_date = current_contract_date
        self._last_full_refresh = now

        # Only update freshness timestamp if we actually got data
        # (new_events can be 0 if fetch succeeded but no new posts)
        # We check for >= 0 because 0 is valid (no new posts), but the fetch
        # would return early with 0 if it failed
        if new_events >= 0:
            self._last_data_refresh_time = now  # Track successful refresh for freshness check

            # Notify all active bots that fresh data is available
            # This signals them to recompute Monte Carlo on their next slow tick
            await self._notify_bots_of_fresh_data()

        return new_events

    async def _notify_bots_of_fresh_data(self) -> None:
        """
        Notify all active trading bots that fresh data is available.

        Bots will use this signal to recompute Monte Carlo forecasts
        on their next slow tick.
        """
        # Get snapshot of active bots under lock
        async with self._events_lock:
            active_bots = [active.bot for active in self._active_events.values()]

        for bot in active_bots:
            try:
                bot.notify_data_refreshed()
            except Exception as e:
                logger.debug(f"Error notifying bot of fresh data: {e}")

    def _handle_global_fill(self, fill_event: FillEvent) -> None:
        """
        Route fill event from global UserStreamClient to the correct bot.

        Uses token_id to identify which event/bot should receive the fill.
        """
        token_id = fill_event.token_id

        # Look up which event owns this token
        event_mapping = self._token_to_event.get(token_id)
        if not event_mapping:
            logger.debug(f"Fill for unknown token {token_id[:16]}... (may be from different market)")
            return

        event_id, bin_index = event_mapping

        # Get the active event (no await needed, just direct access under lock)
        active = self._active_events.get(event_id)
        if not active:
            logger.warning(f"Fill for inactive event {event_id} token {token_id[:16]}...")
            return

        # Route fill to the bot's Kelly executor
        try:
            if active.bot.kelly_bot and active.bot.kelly_bot.kelly_executor:
                active.bot.kelly_bot.kelly_executor.handle_fill(fill_event)
                logger.info(
                    f"[FILL ROUTED] event={active.info.short_name} bin={bin_index} | "
                    f"size={fill_event.size:.1f} @ {fill_event.price:.3f}"
                )
        except Exception as e:
            logger.error(f"Error routing fill to event {event_id}: {e}", exc_info=True)

    def _register_event_tokens(self, event_info: EventInfo) -> None:
        """
        Register token_ids for an event to enable fill routing.

        Called when an event is started.
        Registers both YES and NO token_ids for each bin.
        """
        registered_count = 0
        for i, bin_def in enumerate(event_info.bins):
            # Register YES token
            yes_token_id = bin_def.get("token_id")
            if yes_token_id:
                self._token_to_event[yes_token_id] = (event_info.event_id, i)
                registered_count += 1

            # Register NO token (for BUY_NO / SELL_NO fills)
            no_token_id = bin_def.get("no_token_id")
            if no_token_id:
                self._token_to_event[no_token_id] = (event_info.event_id, i)
                registered_count += 1

        logger.debug(f"Registered {registered_count} tokens for event {event_info.event_id}")

    def _unregister_event_tokens(self, event_info: EventInfo) -> None:
        """
        Unregister token_ids for an event.

        Called when an event is cleaned up.
        Unregisters both YES and NO token_ids.
        """
        for bin_def in event_info.bins:
            # Unregister YES token
            yes_token_id = bin_def.get("token_id")
            if yes_token_id:
                self._token_to_event.pop(yes_token_id, None)

            # Unregister NO token
            no_token_id = bin_def.get("no_token_id")
            if no_token_id:
                self._token_to_event.pop(no_token_id, None)

        logger.debug(f"Unregistered tokens for event {event_info.event_id}")

    def _get_event_trading_rules(self) -> EventTradingRulesConfig:
        """Get event trading rules config, using default if not configured."""
        if self.config.event_trading_rules is not None:
            return self.config.event_trading_rules
        return EventTradingRulesConfig.default()

    def _check_event_trading_rules(
        self,
        event_info: EventInfo,
        rules: "EventCategoryRules",
        now: datetime,
        hours_to_settlement: float,
    ) -> Tuple[bool, str]:
        """
        Check if trading is allowed for an event based on its rules.

        Args:
            event_info: Event information
            rules: Trading rules for this event's category
            now: Current time (UTC)
            hours_to_settlement: Hours until settlement

        Returns:
            Tuple of (should_skip, reason) - if should_skip is True, don't trade yet
        """
        from ..kelly.config import EventCategoryRules

        event_duration_days = (event_info.settlement_date - event_info.market_start_date).days

        # Calculate counting start time (noon ET on market_start_date)
        counting_start_dt = datetime.combine(
            event_info.market_start_date,
            datetime.min.time().replace(hour=17)  # 12:00 ET = 17:00 UTC
        )
        hours_until_counting = (counting_start_dt - now).total_seconds() / 3600
        counting_started = hours_until_counting <= 0

        # Check min_hours_before_settlement (from rules)
        if hours_to_settlement <= rules.min_hours_before_settlement:
            return True, (
                f"Too close to settlement: {hours_to_settlement:.1f}h remaining, "
                f"rule requires > {rules.min_hours_before_settlement:.1f}h"
            )

        # Check require_counting_started
        if rules.require_counting_started and not counting_started:
            return True, (
                f"Waiting for counting to start ({rules.name} event, {event_duration_days}d duration). "
                f"Counting starts in {hours_until_counting:.1f}h"
            )

        # Check max_hours_before_counting (if not requiring counting to start)
        if not rules.require_counting_started and rules.max_hours_before_counting is not None:
            if hours_until_counting > rules.max_hours_before_counting:
                return True, (
                    f"Too early before counting ({rules.name} event). "
                    f"Can trade {rules.max_hours_before_counting:.0f}h before counting, "
                    f"currently {hours_until_counting:.1f}h before"
                )

        # Check max_days_before_settlement
        if rules.max_days_before_settlement is not None:
            max_hours = rules.max_days_before_settlement * 24
            if hours_to_settlement > max_hours:
                return True, (
                    f"Waiting for T-{max_hours:.0f}h ({rules.name} event, {event_duration_days}d duration). "
                    f"Currently T-{hours_to_settlement:.1f}h"
                )

        # All checks passed
        return False, ""

    async def _do_full_refresh(self) -> int:
        """Do a full refresh of tweet data."""
        n_days = self.config.training_days

        logger.info(f"Full refresh: fetching {n_days} days of tweet data...")

        # Clear and refetch (returns False if rejected due to regression)
        success = self.shared_event_store.refresh_from_api(n_days)
        if not success:
            logger.warning("Full refresh rejected due to data regression")
            return 0

        date_range = self.shared_event_store.get_date_range()
        if date_range[0] and date_range[1]:
            total_events = sum(
                len(self.shared_event_store.get_contract_day_events(d))
                for d in self._date_range_iter(date_range[0], date_range[1])
            )
            logger.info(
                f"Full refresh complete: {total_events} events from "
                f"{date_range[0]} to {date_range[1]}"
            )
            return total_events

        return 0

    async def _do_incremental_refresh(self) -> int:
        """
        Do incremental refresh of recent tweet data.

        Fetches posts for the last 2-3 calendar days to ensure we capture all posts
        that might fall into the current or previous contract days.

        Note: Contract days use noon ET boundary, but XTracker API uses UTC calendar dates.
        So if it's before noon ET, we need to fetch today's calendar date to capture
        posts from midnight UTC to now.

        Returns:
            Number of new events compared to before refresh
        """
        contract_today = self.contract_utils.get_current_contract_date()
        calendar_today = datetime.now(self._tz).date()

        # Fetch from yesterday's contract day to today's calendar date
        # This ensures we capture:
        # 1. Late-arriving posts from yesterday's contract day
        # 2. Today's contract day (which may span 2 calendar days around noon ET)
        # 3. Posts from the current calendar day even if before noon ET
        yesterday = contract_today - timedelta(days=1)

        # Determine which contract days we're updating
        contract_days_to_update = [yesterday, contract_today]

        # Get count before refresh for comparison
        initial_count = sum(
            len(self.shared_event_store.get_contract_day_events(d))
            for d in contract_days_to_update
        )

        # Fetch all posts in date range - use calendar_today to catch all posts
        # even if before noon ET (contract day boundary)
        events = self.posts_xtracker_client.fetch_all_events(
            start_date=yesterday,
            end_date=calendar_today,  # Use calendar date, not contract date
        )

        # IMPORTANT: Don't replace cache if fetch failed or returned empty
        # This prevents count regression when network is unavailable
        if not events:
            logger.warning(
                f"Incremental refresh: no events fetched, keeping cached data. "
                f"Initial count: {initial_count}"
            )
            return 0

        # Group events by contract day
        events_by_day: Dict[date, List] = {d: [] for d in contract_days_to_update}
        for event in events:
            event_contract_date = self.contract_utils.get_contract_date(event.timestamp)
            if event_contract_date in events_by_day:
                events_by_day[event_contract_date].append(event)

        # Sanity check: count should never decrease (monotonic tweets)
        new_total = sum(len(day_events) for day_events in events_by_day.values())
        if new_total < initial_count:
            logger.warning(
                f"Incremental refresh: new count ({new_total}) < old count ({initial_count}). "
                f"This is suspicious - keeping cached data to avoid regression."
            )
            return 0

        # Replace data for each day (not append - avoids duplicates)
        for contract_date, day_events in events_by_day.items():
            self.shared_event_store.set_contract_day_events(contract_date, day_events)

        # Count after refresh
        final_count = sum(
            len(self.shared_event_store.get_contract_day_events(d))
            for d in contract_days_to_update
        )

        new_events = final_count - initial_count
        if new_events != 0:
            logger.info(
                f"Incremental refresh: {'+' if new_events > 0 else ''}{new_events} events "
                f"(fetched {yesterday} to {calendar_today}, contract days: {contract_days_to_update})"
            )

        return max(0, new_events)

    def _date_range_iter(self, start: date, end: date):
        """Iterate over dates in range."""
        current = start
        while current <= end:
            yield current
            current += timedelta(days=1)

    async def _refit_active_event_models(self) -> None:
        """Refit forecaster models for all active events after data refresh."""
        # Get snapshot of active events under lock
        async with self._events_lock:
            active_snapshot = [(eid, active.bot) for eid, active in self._active_events.items()]

        for event_id, bot in active_snapshot:
            try:
                logger.info(f"Refitting model for event {event_id}")
                # Refit with shared data (skip_fetch=True since data is already in store)
                bot.forecaster.fit(
                    n_days=self.config.training_days,
                    skip_fetch=True,
                )
            except Exception as e:
                logger.error(f"Failed to refit model for event {event_id}: {e}")

    async def get_authoritative_count(
        self,
        event_info: EventInfo,
    ) -> Tuple[Optional[int], Optional[datetime]]:
        """
        Get authoritative tweet count from XTracker trackings API.

        This is the "official" count that should match the market settlement.

        Args:
            event_info: Event to get count for

        Returns:
            Tuple of (count, updated_at) or (None, None) if not available
        """
        # Convert dates to datetime for the trackings API
        # Counting period: market_start_date noon ET to settlement_date noon ET
        start_dt = datetime.combine(
            event_info.market_start_date,
            time(12, 0),  # Noon ET
            tzinfo=self._tz,
        )
        end_dt = datetime.combine(
            event_info.settlement_date,
            time(12, 0),  # Noon ET
            tzinfo=self._tz,
        )

        try:
            count, updated_at = self.trackings_xtracker_client.get_current_count(
                start_date=start_dt,
                end_date=end_dt,
            )
            return count, updated_at
        except Exception as e:
            logger.warning(f"Failed to get authoritative count for {event_info.short_name}: {e}")
            return None, None

    def compute_count_from_posts(self, event_info: EventInfo) -> int:
        """
        Compute tweet count from cached posts data.

        Args:
            event_info: Event to compute count for

        Returns:
            Computed count from posts
        """
        now = datetime.now(self._tz)
        today = self.contract_utils.get_contract_date(now)

        total = 0
        current_date = event_info.market_start_date

        # Count days in the event window up to today
        while current_date <= today and current_date < event_info.settlement_date:
            count = self.shared_event_store.get_contract_day_count(current_date)
            total += count
            current_date += timedelta(days=1)

        return total

    async def validate_counts_for_event(
        self,
        event_info: EventInfo,
    ) -> Tuple[int, Optional[int], bool]:
        """
        Validate computed count against authoritative XTracker count.

        Args:
            event_info: Event to validate

        Returns:
            Tuple of (computed_count, api_count, is_valid)
            is_valid is False if counts differ by more than COUNT_MISMATCH_THRESHOLD
        """
        computed = self.compute_count_from_posts(event_info)
        api_count, _ = await self.get_authoritative_count(event_info)

        if api_count is None:
            # Can't validate without API count
            logger.debug(f"No API count available for {event_info.short_name}, skipping validation")
            return computed, None, True

        diff = abs(computed - api_count)
        is_valid = diff <= COUNT_MISMATCH_THRESHOLD

        if not is_valid:
            logger.warning(
                f"COUNT MISMATCH for {event_info.short_name}: "
                f"computed={computed}, API={api_count}, diff={diff}. "
                f"Threshold={COUNT_MISMATCH_THRESHOLD}. Check data integrity!"
            )
        else:
            logger.debug(
                f"Count validation OK for {event_info.short_name}: "
                f"computed={computed}, API={api_count}, diff={diff}"
            )

        return computed, api_count, is_valid

    async def validate_all_event_counts(self) -> Dict[str, Tuple[int, Optional[int], bool]]:
        """
        Validate counts for all active events and update authoritative counts on bots.

        This is called periodically (every count_validation_interval seconds) and:
        1. Validates computed count against authoritative XTracker count
        2. Updates the trading bot with the authoritative count for dead bin detection

        Returns:
            Dict mapping event_id -> (computed, api_count, is_valid)
        """
        # Get snapshot of active events under lock
        async with self._events_lock:
            active_snapshot = [
                (eid, active.info, active.bot)
                for eid, active in self._active_events.items()
            ]

        results = {}
        for event_id, event_info, bot in active_snapshot:
            computed, api_count, is_valid = await self.validate_counts_for_event(event_info)
            results[event_id] = (computed, api_count, is_valid)

            # Update trading bot with authoritative count for dead bin detection
            if api_count is not None:
                bot.set_authoritative_count(api_count)

        return results

    async def add_event(self, event_info: EventInfo) -> bool:
        """
        Add an event to be traded.

        The event will be started when:
        1. Capital is available
        2. Event hasn't settled yet
        3. Event is within tradeable time window

        Args:
            event_info: Event information

        Returns:
            True if event was added (may be pending), False if rejected
        """
        event_id = event_info.event_id

        async with self._events_lock:
            # Check if already active or pending
            if event_id in self._active_events:
                logger.warning(f"Event {event_id} already active")
                return False

            if event_id in self._pending_events:
                logger.warning(f"Event {event_id} already pending")
                return False

            if event_id in self._completed_events:
                logger.warning(f"Event {event_id} already completed")
                return False

            # Check if event is still tradeable
            now = datetime.utcnow()
            settlement_dt = datetime.combine(
                event_info.settlement_date,
                datetime.min.time().replace(hour=17)  # Noon ET = 17:00 UTC
            )

            hours_to_settlement = (settlement_dt - now).total_seconds() / 3600

            if hours_to_settlement <= self.kelly_config.t_stop_hours:
                logger.info(
                    f"Event {event_id} too close to settlement "
                    f"({hours_to_settlement:.1f}h remaining)"
                )
                return False

            # Note: Long-duration events are added to pending but only started
            # when remaining time < max_event_duration_days (checked in _try_start_pending_events)

            # Add to pending
            self._pending_events[event_id] = event_info
            logger.info(f"Added event {event_id} ({event_info.short_name}) to pending queue")

        # Try to start immediately if running (outside lock to avoid deadlock)
        if self._running:
            await self._try_start_pending_events()

        return True

    async def _discover_and_add_events(self) -> int:
        """
        Discover new events using the discovery callback and add them.

        Returns:
            Number of new events added to pending queue
        """
        if self.event_discovery_callback is None:
            return 0

        try:
            logger.info("Discovering new events...")
            discovered_events = await self.event_discovery_callback()

            if not discovered_events:
                logger.debug("No new events discovered")
                return 0

            added_count = 0
            for event_info in discovered_events:
                # add_event() handles duplicate checking
                if await self.add_event(event_info):
                    added_count += 1

            if added_count > 0:
                logger.info(f"Discovered and added {added_count} new events")

            return added_count

        except Exception as e:
            logger.error(f"Error discovering events: {e}", exc_info=True)
            self._errors_count += 1
            return 0

    async def _try_start_pending_events(self) -> None:
        """Try to start any pending events that have available capital."""
        # Get snapshot of pending events under lock
        async with self._events_lock:
            pending_snapshot = list(self._pending_events.items())

        for event_id, event_info in pending_snapshot:
            # Check timing again
            now = datetime.utcnow()
            settlement_dt = datetime.combine(
                event_info.settlement_date,
                datetime.min.time().replace(hour=17)
            )
            hours_to_settlement = (settlement_dt - now).total_seconds() / 3600

            if hours_to_settlement <= self.kelly_config.t_stop_hours:
                logger.info(f"Event {event_id} expired, removing from pending")
                async with self._events_lock:
                    self._pending_events.pop(event_id, None)
                continue

            # Get event trading rules for this event's duration
            event_duration_days = (event_info.settlement_date - event_info.market_start_date).days
            trading_rules = self._get_event_trading_rules()
            rules = trading_rules.get_rules_for_event(event_duration_days)

            # Check if trading is allowed based on rules
            should_skip, skip_reason = self._check_event_trading_rules(
                event_info, rules, now, hours_to_settlement
            )
            if should_skip:
                logger.info(
                    f"Event {event_id} ({event_info.short_name}) pending: {skip_reason}"
                )
                continue

            # Request capital (CapitalPool has its own lock)
            allocated = await self.capital_pool.request_capital(event_id)

            if allocated <= 0:
                logger.debug(
                    f"No capital available for event {event_id}, "
                    f"will retry later"
                )
                continue

            # Start the event
            try:
                await self._start_event(event_info, allocated)
                async with self._events_lock:
                    self._pending_events.pop(event_id, None)
            except Exception as e:
                logger.error(f"Failed to start event {event_id}: {e}")
                # Return capital on failure
                await self.capital_pool.return_capital(event_id, allocated)

    async def _start_event(
        self,
        event_info: EventInfo,
        allocated_capital: float,
    ) -> None:
        """
        Start trading on an event.

        Args:
            event_info: Event information
            allocated_capital: Capital allocated from pool
        """
        event_id = event_info.event_id

        logger.info(
            f"Starting event {event_id} ({event_info.short_name}) "
            f"with ${allocated_capital:.2f}"
        )

        # Register token_ids for fill routing
        self._register_event_tokens(event_info)

        # Create bot config
        bot_config = TradingBotConfig(
            slow_tick_interval_seconds=self.config.tick_interval_seconds,
            dry_run=self.config.dry_run,
            settlement_date=event_info.settlement_date,
            initial_capital=allocated_capital,
            training_days=self.config.training_days,
            use_gas=self.config.use_gas,
            event_name=event_info.short_name,
            projection_model=self.config.projection_model,
        )

        # Create bot with shared EventStore
        bot = GASKellyTradingBot(
            clob_client=self.clob_client,
            kelly_config=self.kelly_config,
            forecaster_config=self.forecaster_config,
            bot_config=bot_config,
            event_store=self.shared_event_store,
            user_stream=self.user_stream,  # Pass global user stream
        )

        # Setup bot (expensive, do outside lock)
        await bot.setup(event_info.bins)

        # CRITICAL: Sync existing positions from API into the Kelly portfolio
        # This ensures collateral tracking works correctly for restored positions
        # Do this even in dry_run mode - we need accurate collateral tracking
        if bot.kelly_bot:
            try:
                await bot.kelly_bot.sync_positions_from_api(self.wallet_address)
                collateral = bot.kelly_bot.portfolio.total_collateral_used if bot.kelly_bot.portfolio else 0
                logger.info(f"Synced existing positions into Kelly portfolio: invested=${collateral:.2f}")
            except Exception as e:
                logger.warning(f"Failed to sync existing positions: {e}", exc_info=True)

        # Set market dates
        bot.market_start_date = event_info.market_start_date
        bot.settlement_date = event_info.settlement_date

        # Set data freshness checker callback
        bot.set_data_freshness_checker(self.is_data_fresh)

        # Start trading task
        task = asyncio.create_task(
            self._run_event(event_id, bot),
            name=f"event_{event_id}"
        )

        # Track active event (protected by lock)
        async with self._events_lock:
            self._active_events[event_id] = ActiveEvent(
                info=event_info,
                bot=bot,
                task=task,
                started_at=datetime.utcnow(),
                allocated_capital=allocated_capital,
            )
            num_active = len(self._active_events)
            self._total_events_started += 1

        logger.info(
            f"Event {event_id} started. "
            f"Active events: {num_active}, "
            f"Pool available: ${self.capital_pool.available_capital:.2f}"
        )

    async def _run_event(
        self,
        event_id: str,
        bot: GASKellyTradingBot,
    ) -> None:
        """
        Run a single event until settlement or stop.

        Args:
            event_id: Event identifier
            bot: Trading bot for this event
        """
        try:
            await bot.run()
        except asyncio.CancelledError:
            logger.info(f"Event {event_id} cancelled")
        except Exception as e:
            logger.error(f"Event {event_id} error: {e}", exc_info=True)
            self._errors_count += 1
        finally:
            await self._cleanup_event(event_id, bot)

    async def _cleanup_event(
        self,
        event_id: str,
        bot: GASKellyTradingBot,
    ) -> None:
        """
        Clean up after event completes or fails.

        Args:
            event_id: Event identifier
            bot: Trading bot for this event
        """
        final_value = 0.0

        # Unregister token_ids for fill routing
        async with self._events_lock:
            if event_id in self._active_events:
                self._unregister_event_tokens(self._active_events[event_id].info)

        try:
            # Get the original allocation for this event
            async with self._events_lock:
                allocated_capital = 0.0
                if event_id in self._active_events:
                    allocated_capital = self._active_events[event_id].allocated_capital

            # Calculate final value based on the event's collateral, NOT portfolio.capital
            # (portfolio.capital may include USDC from other events or unallocated funds)
            if bot.kelly_bot and bot.kelly_bot.portfolio:
                portfolio = bot.kelly_bot.portfolio
                # Use collateral_used as basis (what this event actually has deployed)
                # For unsettled events, return the collateral value
                position_value = 0.0
                for pos in portfolio.positions.values():
                    # Use cost basis for unsettled positions
                    position_value += pos.yes_shares * pos.yes_avg_cost
                    position_value += pos.no_shares * pos.no_avg_cost

                # Final value = original allocation
                # (In reality, P&L only happens at settlement when positions resolve)
                # For early termination, just return the allocated amount
                final_value = allocated_capital if allocated_capital > 0 else position_value

                logger.debug(
                    f"Event {event_id} cleanup: allocated=${allocated_capital:.2f}, "
                    f"position_value=${position_value:.2f}, returning=${final_value:.2f}"
                )
            else:
                # Fallback: return initial allocation
                final_value = allocated_capital

            # Return capital to pool (CapitalPool has its own lock)
            await self.capital_pool.return_capital(event_id, final_value)

            # Shutdown bot
            await bot.shutdown()

        except Exception as e:
            logger.error(f"Error cleaning up event {event_id}: {e}")

        finally:
            # Remove from active events and add to completed (protected by lock)
            async with self._events_lock:
                self._active_events.pop(event_id, None)
                self._completed_events.append(event_id)
                num_active = len(self._active_events)
                self._total_events_completed += 1

            logger.info(
                f"Event {event_id} completed. "
                f"Active events: {num_active}, "
                f"Pool available: ${self.capital_pool.available_capital:.2f}"
            )

            # Try to start pending events with freed capital
            if self._running:
                await self._try_start_pending_events()

    async def run(self) -> None:
        """
        Run the multi-event manager.

        Manages all active events concurrently and handles event lifecycle.
        Pre-fetches shared tweet data before starting any events.

        Main loop timing:
        - posts_refresh_interval (5 min): Refresh posts for intraday modeling
        - count_validation_interval (15 min): Validate counts against XTracker API
        - event_scan_interval (1 hour): Check for new events to trade
        - health_log_interval (1 hour): Log health status
        """
        self._running = True
        self._stop_event = asyncio.Event()
        self._start_time = datetime.now(self._tz)

        logger.info("Starting MultiEventManager")

        # Auto-initialize capital from API if total_capital is 0
        if self.capital_pool.needs_initialization:
            await self._initialize_capital_from_api()

        # Start global user stream for fill confirmations
        if self.user_stream:
            await self.user_stream.start()
            logger.info("Global UserStreamClient started")

        # Pre-fetch shared tweet data (once for all events)
        await self.prefetch_shared_data()

        # Initial validation
        await self.validate_all_event_counts()

        # Start any pending events
        await self._try_start_pending_events()

        # Track last execution times for different periodic tasks
        last_posts_refresh = datetime.now(self._tz)
        last_count_validation = datetime.now(self._tz)
        last_event_scan = datetime.now(self._tz)
        self._last_health_log_time = datetime.now(self._tz)

        # Use shortest interval for loop timing
        loop_interval = min(
            self.config.posts_refresh_interval,
            self.config.count_validation_interval,
            self.config.event_scan_interval,
            60,  # At least check every minute
        )

        logger.info(
            f"Main loop started with intervals: "
            f"posts_refresh={self.config.posts_refresh_interval}s, "
            f"count_validation={self.config.count_validation_interval}s, "
            f"event_scan={self.config.event_scan_interval}s, "
            f"health_log={self.config.health_log_interval}s"
        )

        # Log initial health status
        self._log_health()

        # Main loop: monitor events and periodically run tasks
        while self._running:
            try:
                # Wait for stop signal or loop interval
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=loop_interval
                    )
                    # Stop requested
                    break
                except asyncio.TimeoutError:
                    pass

                now = datetime.now(self._tz)

                # Task 1: Refresh posts data (most frequent - for intraday modeling)
                if (now - last_posts_refresh).total_seconds() >= self.config.posts_refresh_interval:
                    await self.refresh_shared_data()
                    last_posts_refresh = now

                # Task 2: Validate counts against XTracker API
                if (now - last_count_validation).total_seconds() >= self.config.count_validation_interval:
                    await self.validate_all_event_counts()
                    last_count_validation = now

                # Task 3: Discover new events and start pending (least frequent)
                if (now - last_event_scan).total_seconds() >= self.config.event_scan_interval:
                    await self._discover_and_add_events()
                    await self._try_start_pending_events()
                    await self._cleanup_old_data()  # Cleanup old data to prevent memory leaks
                    last_event_scan = now

                # Task 4: Log health status periodically
                if (now - self._last_health_log_time).total_seconds() >= self.config.health_log_interval:
                    self._log_health()
                    self._last_health_log_time = now

                # Always update capital values and log status
                await self._update_capital_values()
                self._log_status()

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                self._errors_count += 1
                await asyncio.sleep(60)  # Back off on error

        # Shutdown
        await self._shutdown()

    async def _update_capital_values(self) -> None:
        """Update capital pool with current portfolio values."""
        # Get snapshot of active events under lock
        async with self._events_lock:
            active_snapshot = [
                (eid, active.bot.kelly_bot)
                for eid, active in self._active_events.items()
            ]

        for event_id, kelly_bot in active_snapshot:
            try:
                if kelly_bot and kelly_bot.portfolio:
                    portfolio = kelly_bot.portfolio
                    current_value = portfolio.capital
                    # Include position values (at cost basis for simplicity)
                    for pos in portfolio.positions.values():
                        current_value += pos.yes_shares * pos.yes_avg_cost
                        current_value += pos.no_shares * pos.no_avg_cost

                    await self.capital_pool.update_value(event_id, current_value)
            except Exception as e:
                logger.debug(f"Error updating value for {event_id}: {e}")

    def _log_status(self) -> None:
        """Log current status."""
        summary = self.capital_pool.get_summary()
        logger.info(
            f"Status: {summary['num_active_events']} active events, "
            f"${summary['available_capital']:.2f} available, "
            f"${summary['allocated_capital']:.2f} allocated, "
            f"total value ${summary['total_value']:.2f}"
        )

    async def _shutdown(self) -> None:
        """Shutdown all active events."""
        logger.info("Shutting down MultiEventManager")

        # Get snapshot and cancel all active event tasks
        async with self._events_lock:
            active_snapshot = list(self._active_events.items())

        tasks_to_wait = []
        for event_id, active in active_snapshot:
            logger.info(f"Stopping event {event_id}")
            active.bot.stop()
            active.task.cancel()
            tasks_to_wait.append(active.task)

        # Wait for all tasks to complete
        if tasks_to_wait:
            await asyncio.gather(*tasks_to_wait, return_exceptions=True)

        # Stop global user stream
        if self.user_stream:
            await self.user_stream.stop()
            logger.info("Global UserStreamClient stopped")

        self._running = False
        logger.info("MultiEventManager shutdown complete")

        # Log final health status
        self._log_health()

    def stop(self) -> None:
        """Signal the manager to stop."""
        self._running = False
        if self._stop_event:
            self._stop_event.set()

    def get_status(self) -> Dict[str, Any]:
        """Get current status for monitoring."""
        return {
            "running": self._running,
            "capital_pool": self.capital_pool.get_summary(),
            "performance": self.capital_pool.get_performance_summary(),
            "active_events": {
                event_id: {
                    "short_name": active.info.short_name,
                    "settlement_date": active.info.settlement_date.isoformat(),
                    "allocated_capital": active.allocated_capital,
                    "started_at": active.started_at.isoformat(),
                }
                for event_id, active in self._active_events.items()
            },
            "pending_events": list(self._pending_events.keys()),
            "completed_events": self._completed_events,
        }

    def get_health(self) -> Dict[str, Any]:
        """
        Get health status for long-running monitoring.

        Returns comprehensive health metrics for production monitoring.
        """
        import sys
        import gc

        now = datetime.now(self._tz)

        # Calculate uptime
        uptime_seconds = 0.0
        if self._start_time:
            uptime_seconds = (now - self._start_time).total_seconds()

        uptime_hours = uptime_seconds / 3600
        uptime_days = uptime_hours / 24

        # Memory usage (approximate)
        try:
            import resource
            memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
            # On macOS, ru_maxrss is in bytes; on Linux it's in KB
            if sys.platform == "darwin":
                memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
            else:
                memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            memory_mb = None

        # EventStore stats
        event_store_dates = 0
        event_store_total_events = 0
        with self.shared_event_store._lock:
            event_store_dates = len(self.shared_event_store._events)
            event_store_total_events = sum(
                len(events) for events in self.shared_event_store._events.values()
            )

        # Capital pool stats
        pool_summary = self.capital_pool.get_summary()
        pool_performance = self.capital_pool.get_performance_summary()

        # Time since last cleanup
        time_since_cleanup = None
        if self._last_cleanup_time:
            time_since_cleanup = (now - self._last_cleanup_time).total_seconds() / 3600

        return {
            # Overall health
            "status": "healthy" if self._running else "stopped",
            "timestamp": now.isoformat(),

            # Uptime
            "uptime_seconds": uptime_seconds,
            "uptime_hours": round(uptime_hours, 2),
            "uptime_days": round(uptime_days, 2),
            "start_time": self._start_time.isoformat() if self._start_time else None,

            # Event counts
            "active_events": len(self._active_events),
            "pending_events": len(self._pending_events),
            "completed_events_in_history": len(self._completed_events),
            "total_events_started": self._total_events_started,
            "total_events_completed": self._total_events_completed,

            # Resource usage
            "memory_mb": round(memory_mb, 2) if memory_mb else None,
            "event_store_days": event_store_dates,
            "event_store_total_tweets": event_store_total_events,

            # Capital
            "total_capital": pool_summary.get("total_capital", 0),
            "available_capital": pool_summary.get("available_capital", 0),
            "allocated_capital": pool_summary.get("allocated_capital", 0),
            "total_pnl": pool_performance.get("total_pnl", 0),

            # Maintenance
            "last_cleanup_hours_ago": round(time_since_cleanup, 2) if time_since_cleanup else None,
            "errors_count": self._errors_count,

            # Active event details
            "active_event_names": [
                active.info.short_name for active in self._active_events.values()
            ],

            # User stream status - get detailed status if available
            "user_stream": self._get_user_stream_status(),
        }

    def _get_user_stream_status(self) -> Dict[str, Any]:
        """Get detailed user stream status for health reporting."""
        if not self.user_stream:
            return {
                "enabled": False,
                "connected": False,
                "pending_orders": 0,
                "fill_count": 0,
                "message_count": 0,
                "last_message_age_seconds": None,
            }

        # Use the new get_connection_status method
        status = self.user_stream.get_connection_status()
        return {
            "enabled": True,
            "connected": status.get("connected", False),
            "pending_orders": status.get("pending_orders", 0),
            "fill_count": status.get("fill_count", 0),
            "message_count": status.get("message_count", 0),
            "last_message_age_seconds": status.get("last_message_age_seconds"),
            "on_fill_callback_set": status.get("on_fill_callback_set", False),
        }

    def _log_health(self) -> None:
        """Log health status in a human-readable format."""
        health = self.get_health()

        logger.info("=" * 70)
        logger.info("HEALTH CHECK REPORT")
        logger.info("=" * 70)
        logger.info(f"  Status: {health['status'].upper()}")
        logger.info(f"  Timestamp: {health['timestamp']}")
        logger.info(f"  Uptime: {health['uptime_days']:.1f} days ({health['uptime_hours']:.1f} hours)")
        logger.info("")
        logger.info("  EVENTS:")
        logger.info(f"    Active: {health['active_events']}")
        logger.info(f"    Pending: {health['pending_events']}")
        logger.info(f"    Total started: {health['total_events_started']}")
        logger.info(f"    Total completed: {health['total_events_completed']}")
        if health['active_event_names']:
            logger.info(f"    Active names: {', '.join(health['active_event_names'])}")
        logger.info("")
        logger.info("  CAPITAL:")
        logger.info(f"    Total: ${health['total_capital']:.2f}")
        logger.info(f"    Available: ${health['available_capital']:.2f}")
        logger.info(f"    Allocated: ${health['allocated_capital']:.2f}")
        logger.info(f"    Total P&L: ${health['total_pnl']:.2f}")
        logger.info("")
        logger.info("  RESOURCES:")
        if health['memory_mb']:
            logger.info(f"    Memory: {health['memory_mb']:.1f} MB")
        logger.info(f"    EventStore: {health['event_store_days']} days, {health['event_store_total_tweets']} tweets")
        logger.info(f"    Completed history: {health['completed_events_in_history']} events")
        if health['last_cleanup_hours_ago']:
            logger.info(f"    Last cleanup: {health['last_cleanup_hours_ago']:.1f} hours ago")
        logger.info(f"    Errors: {health['errors_count']}")
        logger.info("")
        logger.info("  USER STREAM:")
        user_stream = health.get('user_stream', {})
        logger.info(f"    Enabled: {user_stream.get('enabled', False)}")
        logger.info(f"    Connected: {user_stream.get('connected', False)}")
        logger.info(f"    Pending orders: {user_stream.get('pending_orders', 0)}")
        logger.info(f"    Total fills: {user_stream.get('fill_count', 0)}")
        logger.info(f"    Total messages: {user_stream.get('message_count', 0)}")
        last_msg = user_stream.get('last_message_age_seconds')
        if last_msg is not None:
            logger.info(f"    Last message: {last_msg:.0f}s ago")
        logger.info("=" * 70)

    async def fetch_all_positions(self) -> Dict[str, dict]:
        """
        Fetch all positions from Polymarket Data API.

        Returns:
            Dict mapping token_id -> {shares, value, avg_price}
        """
        try:
            response = requests.get(
                f"{POLYMARKET_DATA_API}/positions",
                params={"user": self.wallet_address.lower()},
                timeout=30,
            )
            response.raise_for_status()
            positions_data = response.json()

            # Debug: log raw response structure
            logger.info(f"Positions API raw response type: {type(positions_data).__name__}, len={len(positions_data) if hasattr(positions_data, '__len__') else 'N/A'}")
            if positions_data:
                if isinstance(positions_data, list) and len(positions_data) > 0:
                    sample = positions_data[0]
                    logger.info(f"Positions API first item type: {type(sample).__name__}")
                    logger.info(f"Positions API first item: {str(sample)[:500]}")
                elif isinstance(positions_data, dict):
                    logger.info(f"Positions API dict keys: {list(positions_data.keys())[:10]}")

            positions = {}

            def parse_position(pos: dict) -> tuple:
                """Parse a position dict, returns (token_id, position_info) or (None, None)."""
                # Extract token_id - 'asset' can be a string (token_id) or a dict
                asset = pos.get("asset")
                if isinstance(asset, str):
                    token_id = asset
                elif isinstance(asset, dict):
                    token_id = asset.get("id")
                else:
                    token_id = pos.get("token_id") or pos.get("asset_id")

                size = float(pos.get("size", 0))
                if not token_id or size <= 0:
                    return None, None

                # Extract both cost basis and current market value
                avg_price = float(pos.get("avgPrice", 0))
                initial_value = float(pos.get("initialValue", 0))
                current_value = float(pos.get("currentValue", 0))

                # Cost basis: what we actually spent (for allocation limit tracking)
                cost_basis = initial_value if initial_value > 0 else (size * avg_price)
                # Current value: what it's worth now (for portfolio value tracking)
                market_value = current_value if current_value > 0 else (size * avg_price)

                return token_id, {
                    "shares": size,
                    "cost_basis": cost_basis,
                    "current_value": market_value,
                    "avg_price": avg_price,
                }

            # Handle different response formats
            if isinstance(positions_data, list):
                for pos in positions_data:
                    if isinstance(pos, dict):
                        token_id, pos_info = parse_position(pos)
                        if token_id:
                            positions[token_id] = pos_info
                            logger.debug(f"Found position: token={token_id[:20]}..., shares={pos_info['shares']:.2f}, cost=${pos_info['cost_basis']:.2f}")
                    elif isinstance(pos, str):
                        logger.warning(f"Unexpected position format (string): {pos[:100]}")
            elif isinstance(positions_data, dict):
                # Alternative format: dict with positions key
                pos_list = positions_data.get("positions", positions_data.get("data", []))
                for pos in pos_list:
                    if isinstance(pos, dict):
                        token_id, pos_info = parse_position(pos)
                        if token_id:
                            positions[token_id] = pos_info

            logger.info(f"Fetched {len(positions)} positions from Polymarket API (wallet: {self.wallet_address[:10]}...)")
            return positions

        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}", exc_info=True)
            return {}

    async def fetch_usdc_balance(self, max_retries: int = 3) -> float:
        """
        Fetch current USDC balance with retry logic.

        Args:
            max_retries: Maximum number of retry attempts

        Returns:
            USDC balance
        """
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)

        for attempt in range(max_retries):
            try:
                balance_info = self.clob_client.get_balance_allowance(params)
                # USDC has 6 decimals, so divide by 1e6
                balance = float(balance_info.get("balance", 0)) / 1e6
                logger.info(f"Fetched USDC balance: ${balance:.2f}")
                return balance
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"Failed to fetch USDC balance (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Failed to fetch USDC balance after {max_retries} attempts: {e}")
                    return 0.0

        return 0.0

    async def reconstruct_state_from_api(self) -> None:
        """
        Reconstruct manager state from Polymarket API on restart.

        This should be called before run() to recover state after a crash/restart.

        Steps:
        1. Get events (from initial_events or discovery callback)
        2. Fetch all positions from Polymarket Data API
        3. Fetch USDC balance
        4. Map positions to events by token_id
        5. Compute capital per event from positions
        6. Restore capital pool allocations
        7. Add events with positions FIRST (priority), then events without
        """
        logger.info("=" * 60)
        logger.info("RECONSTRUCTING STATE FROM POLYMARKET API")
        logger.info("=" * 60)

        # 1. Get events - prefer initial_events if provided, otherwise use discovery
        if self.initial_events:
            discovered_events = self.initial_events
            logger.info(f"Using {len(discovered_events)} provided events")
        elif self.event_discovery_callback:
            discovered_events = await self.event_discovery_callback()
            logger.info(f"Discovered {len(discovered_events)} active events")
        else:
            discovered_events = []
            logger.warning("No initial events or discovery callback provided")

        # 2. Fetch all positions
        all_positions = await self.fetch_all_positions()

        # 3. Fetch USDC balance
        usdc_balance = await self.fetch_usdc_balance()

        # 4. Build token_id -> event mapping (include BOTH YES and NO tokens)
        token_to_event: Dict[str, EventInfo] = {}
        for event_info in discovered_events:
            yes_count = 0
            no_count = 0
            for bin_def in event_info.bins:
                # Map YES token
                token_id = bin_def.get("token_id")
                if token_id:
                    token_to_event[token_id] = event_info
                    yes_count += 1
                # Map NO token (positions can be YES or NO)
                no_token_id = bin_def.get("no_token_id")
                if no_token_id:
                    token_to_event[no_token_id] = event_info
                    no_count += 1
            logger.debug(f"Event {event_info.short_name}: mapped {yes_count} YES tokens, {no_count} NO tokens")

        # Debug: check if our position token is in the mapping
        for pos_token, pos_info in all_positions.items():
            if pos_token in token_to_event:
                logger.info(f"Position token {pos_token[:20]}... matches event {token_to_event[pos_token].short_name}")
            else:
                logger.warning(f"Position token {pos_token[:20]}... NOT FOUND in any event!")

        # 5. Compute capital per event from positions
        event_positions: Dict[str, Dict[str, dict]] = {}  # event_id -> {token_id: pos_info}
        event_values: Dict[str, float] = {}  # event_id -> total value

        for token_id, pos_info in all_positions.items():
            event_info = token_to_event.get(token_id)
            if event_info:
                event_id = event_info.event_id
                if event_id not in event_positions:
                    event_positions[event_id] = {}
                    event_values[event_id] = 0.0

                event_positions[event_id][token_id] = pos_info
                # Use actual value from API (cost basis / initialValue)
                event_values[event_id] += pos_info["cost_basis"]

                logger.info(
                    f"Position: {pos_info['shares']:.2f} shares @ ${pos_info['avg_price']:.4f} = ${pos_info['cost_basis']:.2f} "
                    f"of {event_info.short_name} (token {token_id[:16]}...)"
                )

        # 6. Compute total capital and restore allocations
        total_position_value = sum(event_values.values())
        total_capital = usdc_balance + total_position_value

        await self.capital_pool.set_total_from_api(total_capital)

        for event_id, value in event_values.items():
            await self.capital_pool.restore_allocation(event_id, value)

        # 7. Add events - PRIORITY: events with positions FIRST, then events without
        # This ensures events with existing positions get capital allocation priority
        events_with_positions = []
        events_without_positions = []

        for event_info in discovered_events:
            if event_info.event_id in event_positions:
                events_with_positions.append(event_info)
            else:
                events_without_positions.append(event_info)

        # First: Add events WITH positions (they have restored allocations)
        for event_info in events_with_positions:
            event_id = event_info.event_id
            async with self._events_lock:
                self._pending_events[event_id] = event_info
            logger.info(f"[PRIORITY] Restored event {event_info.short_name} with ${event_values[event_id]:.2f} position value")

        # Second: Add events WITHOUT positions (they need new allocations)
        for event_info in events_without_positions:
            await self.add_event(event_info)
            logger.info(f"Added event {event_info.short_name} (no existing positions)")

        logger.info("=" * 60)
        logger.info(f"STATE RECONSTRUCTION COMPLETE")
        logger.info(f"  Total capital: ${total_capital:.2f}")
        logger.info(f"  USDC balance: ${usdc_balance:.2f}")
        logger.info(f"  Position value: ${total_position_value:.2f}")
        logger.info(f"  Events with positions: {len(event_positions)}")
        logger.info(f"  Events discovered: {len(discovered_events)}")
        logger.info("=" * 60)
