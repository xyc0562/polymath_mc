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
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from zoneinfo import ZoneInfo

import requests
from py_clob_client.client import ClobClient

# Polymarket Data API for fetching positions
POLYMARKET_DATA_API = "https://data-api.polymarket.com"

from .config import ForecasterConfig
from .trading_bot import GASKellyTradingBot, TradingBotConfig
from .data import EventStore, ContractDayUtils, XTrackerClient as PostsXTrackerClient
from .realtime_tracker import RealtimeTweetTracker, RealtimePollResult
from ..notifications import SlackNotifier
from ..musk_tweet_count import XTrackerClient as TrackingsXTrackerClient
from ..kelly.config import KellyConfig, EventTradingRulesConfig
from ..kelly.capital_pool import CapitalPool, CapitalPoolConfig
from ..kelly.user_stream import UserStreamClient, FillEvent, PendingOrder, OrderStatus
from ..kelly.executor import BalanceAllowanceErrorContext

try:
    from src.twitter_scraper import get_cookies_path as get_default_twitter_cookies_path
except Exception:  # pragma: no cover - fallback for stripped environments
    def get_default_twitter_cookies_path() -> Path:
        return Path("config/twitter_cookies.json")

logger = logging.getLogger(__name__)

# Count validation threshold - warn if computed vs API count differs by more than this
COUNT_MISMATCH_THRESHOLD = 5

TICK_SOURCE_PRIORITY = {
    "startup": 0,
    "realtime": 1,
    "xtracker": 2,
    "authoritative_rebase": 3,
}


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
    stabilization_until: float = 0.0
    stable_sync_count: int = 0
    last_position_fingerprint: Optional[Tuple[Any, ...]] = None
    startup_tick_pending: bool = False


@dataclass
class RefreshOutcome:
    """Outcome from applying authoritative XTracker data."""

    effective_changed: bool = False
    new_events: int = 0
    affected_days: Set[date] = field(default_factory=set)
    authoritative_rebase: bool = False


@dataclass
class FillNotification:
    """Buffered fill notification for Slack summaries."""

    event_short_name: str
    bin_index: int
    bin_range: str
    side: str
    status: str
    size: float
    price: float
    notional: float
    timestamp: datetime


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

    # How often to poll lastSync for XTracker data updates (seconds)
    sync_poll_interval: float = 5.0

    # Debounce window: after detecting a lastSync change, keep polling for
    # this many seconds to absorb subsequent updates from the same sync cycle
    # (XTracker's sync is not atomic — lastSync can update 2+ times per cycle)
    sync_debounce_seconds: float = 15.0

    # Soft deadline: log warning if sync→trade cycle exceeds this (seconds)
    # This is NOT a hard cutoff — the cycle always runs to completion
    sync_trade_deadline_seconds: float = 10.0

    # How often to refresh posts data (seconds) — used as fallback
    # In sync-driven mode, posts are refreshed on lastSync change instead
    posts_refresh_interval: int = 150  # 2.5 minutes

    # Realtime twikit polling for provisional edge detection
    realtime_tracker_enabled: bool = True
    realtime_poll_interval_seconds: float = 20.0
    realtime_fetch_count: int = 40
    realtime_late_tweet_grace_seconds: float = 120.0
    realtime_cookies_path: Optional[str] = None

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

    # Event duration filter (inclusive, in days)
    # Only auto-discover events within this range
    # Events with existing positions bypass this filter
    min_event_duration_days: int = 7
    max_event_duration_days: int = 7

    # Maximum completed events to keep in history (for memory management)
    max_completed_events: int = 100

    # Maximum days of tweet data to keep in EventStore
    max_event_store_days: int = 60

    # How often to log health status (seconds)
    health_log_interval: int = 3600  # Every hour

    # How often to re-sync capital pool from on-chain USDC balance (seconds)
    capital_sync_interval: int = 3600  # 1 hour

    # After restart, events restored with existing positions briefly wait for
    # one additional stable API sync before trading. This is event-local only.
    restart_stabilization_seconds: float = 45.0
    restart_required_stable_syncs: int = 2


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
        slack_notifier: Optional[SlackNotifier] = None,
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
            slack_notifier: Optional async Slack notifier for state updates
        """
        import os

        self.wallet_address = wallet_address
        self.event_discovery_callback = event_discovery_callback
        self.initial_events = initial_events or []
        self.clob_client = clob_client
        self.kelly_config = kelly_config
        self.forecaster_config = forecaster_config
        self.config = config
        self.slack_notifier = slack_notifier

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
        self._data_update_lock = asyncio.Lock()
        self.shared_data_version: int = 0

        # Sync-driven polling: last known XTracker sync timestamp
        self._last_known_sync: Optional[datetime] = None
        # Consecutive sync poll error counter (for backoff)
        self._sync_poll_errors: int = 0

        # Track if data has been pre-fetched
        self._data_prefetched = False

        # Track last refresh time for smart refresh scheduling
        self._last_full_refresh: Optional[datetime] = None
        self._last_refresh_contract_date: Optional[date] = None

        # Track when data was last successfully refreshed (for freshness check)
        self._last_data_refresh_time: Optional[datetime] = None
        self._last_data_refresh_source: Optional[str] = None
        self._last_xtracker_refresh_time: Optional[datetime] = None

        self.realtime_tracker: Optional[RealtimeTweetTracker] = None
        if self.config.realtime_tracker_enabled:
            cookies_path = Path(self.config.realtime_cookies_path) if self.config.realtime_cookies_path else get_default_twitter_cookies_path()
            self.realtime_tracker = RealtimeTweetTracker(
                event_store=self.shared_event_store,
                contract_utils=self.contract_utils,
                cookies_path=cookies_path,
                poll_interval=self.config.realtime_poll_interval_seconds,
                fetch_count=self.config.realtime_fetch_count,
                late_tweet_grace_seconds=self.config.realtime_late_tweet_grace_seconds,
            )

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
            # Wire up fill and stale order handlers to route to correct bot
            self.user_stream.on_fill = self._handle_global_fill
            self.user_stream.on_stale_order = self._handle_global_stale_order
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
        self._last_slack_health_notification: Optional[datetime] = None
        self._last_slack_health_signature: Optional[Tuple[Any, ...]] = None
        self._pending_fill_notifications: List[FillNotification] = []
        self._first_pending_fill_at: Optional[datetime] = None
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

    @staticmethod
    def _fmt_usd(value: float) -> str:
        return f"${value:.2f}"

    def _notify_slack(
        self,
        level: str,
        title: str,
        lines: Optional[List[str]] = None,
        *,
        dedupe_key: Optional[str] = None,
        cooldown_seconds: float = 0.0,
        mention: bool = False,
    ) -> None:
        if not self.slack_notifier:
            return
        if level == "error":
            self.slack_notifier.notify_error(
                title,
                lines,
                mention=mention,
                dedupe_key=dedupe_key,
                cooldown_seconds=cooldown_seconds,
            )
        elif level == "warning":
            self.slack_notifier.notify_warning(
                title,
                lines,
                mention=mention,
                dedupe_key=dedupe_key,
                cooldown_seconds=cooldown_seconds,
            )
        else:
            self.slack_notifier.notify_info(
                title,
                lines,
                mention=mention,
                dedupe_key=dedupe_key,
                cooldown_seconds=cooldown_seconds,
            )

    def _notify_startup(self) -> None:
        health = self.get_health()
        lines = [
            f"mode={'DRY RUN' if self.config.dry_run else 'LIVE'} active={health['active_events']} pending={health['pending_events']}",
            f"capital available={self._fmt_usd(health['available_capital'])} allocated={self._fmt_usd(health['allocated_capital'])} pnl={self._fmt_usd(health['total_pnl'])}",
            f"data source={health.get('last_data_refresh_source') or 'unknown'} version={health.get('shared_data_version')} refreshed_at={health.get('last_data_refresh_time') or 'n/a'}",
        ]
        if health["active_event_names"]:
            lines.append("events=" + ", ".join(health["active_event_names"][:5]))
        self._notify_slack("info", "Multi-event manager started", lines)
        self._last_slack_health_notification = datetime.now(self._tz)
        self._last_slack_health_signature = self._health_digest_signature(health)

    @staticmethod
    def _health_digest_signature(health: Dict[str, Any]) -> Tuple[Any, ...]:
        """Build a coarse health signature for change-driven Slack digests."""
        user_stream = health.get("user_stream", {})
        realtime = health.get("realtime_tracker", {})
        return (
            health.get("status"),
            health.get("errors_count"),
            user_stream.get("connected", False),
            realtime.get("enabled", False),
            bool(realtime.get("backoff_until")),
            min(int(realtime.get("consecutive_errors", 0) or 0), 3),
        )

    def _maybe_notify_health(self, *, force: bool = False) -> None:
        if not self.slack_notifier:
            return
        interval = max(0, self.slack_notifier.health_interval_seconds)

        now = datetime.now(self._tz)
        health = self.get_health()
        signature = self._health_digest_signature(health)
        changed = signature != self._last_slack_health_signature
        interval_elapsed = (
            self._last_slack_health_notification is None or
            (
                interval > 0 and
                (now - self._last_slack_health_notification).total_seconds() >= interval
            )
        )

        if not force:
            if self.slack_notifier.health_on_change_only:
                if not changed and not interval_elapsed:
                    return
            elif interval <= 0 or not interval_elapsed:
                return

        lines = [
            f"status={health['status']} uptime={health['uptime_hours']:.1f}h active={health['active_events']} pending={health['pending_events']}",
            f"capital available={self._fmt_usd(health['available_capital'])} allocated={self._fmt_usd(health['allocated_capital'])} pnl={self._fmt_usd(health['total_pnl'])}",
            f"errors={health['errors_count']} data source={health.get('last_data_refresh_source') or 'unknown'} version={health.get('shared_data_version')}",
        ]
        user_stream = health.get("user_stream", {})
        lines.append(
            f"user_stream connected={user_stream.get('connected', False)} pending_orders={user_stream.get('pending_orders', 0)} fills={user_stream.get('fill_count', 0)}"
        )
        realtime = health.get("realtime_tracker", {})
        if realtime.get("enabled", False):
            lines.append(
                f"realtime errors={realtime.get('consecutive_errors', 0)} last_poll={realtime.get('last_poll_time') or 'n/a'}"
            )
        if health["active_event_names"]:
            lines.append("events=" + ", ".join(health["active_event_names"][:5]))

        self._notify_slack("info", "Health digest", lines)
        self._last_slack_health_notification = now
        self._last_slack_health_signature = signature

    def _notify_event_started(
        self,
        event_info: EventInfo,
        allocated_capital: float,
        num_active: int,
    ) -> None:
        self._notify_slack(
            "info",
            f"Event started: {event_info.short_name}",
            [
                f"event_id={event_info.event_id}",
                f"dates={event_info.market_start_date} -> {event_info.settlement_date}",
                f"allocated={self._fmt_usd(allocated_capital)} active_events={num_active}",
                f"pool_available={self._fmt_usd(self.capital_pool.available_capital)}",
            ],
        )

    def _notify_event_completed(
        self,
        event_info: EventInfo,
        final_value: float,
        num_active: int,
    ) -> None:
        self._notify_slack(
            "info",
            f"Event completed: {event_info.short_name}",
            [
                f"event_id={event_info.event_id}",
                f"returned={self._fmt_usd(final_value)} active_events={num_active}",
                f"pool_available={self._fmt_usd(self.capital_pool.available_capital)}",
            ],
        )

    def _notify_fill(
        self,
        event_info: EventInfo,
        bin_index: int,
        fill_event: FillEvent,
        *,
        bin_range: str = "",
    ) -> None:
        if fill_event.status != OrderStatus.CONFIRMED:
            return
        self._pending_fill_notifications.append(
            FillNotification(
                event_short_name=event_info.short_name,
                bin_index=bin_index,
                bin_range=bin_range,
                side=fill_event.side,
                status=fill_event.status.value,
                size=fill_event.size,
                price=fill_event.price,
                notional=fill_event.size * fill_event.price,
                timestamp=fill_event.timestamp,
            )
        )
        if self._first_pending_fill_at is None:
            self._first_pending_fill_at = datetime.now(self._tz)

    def _maybe_flush_fill_summaries(self, *, force: bool = False) -> None:
        if not self.slack_notifier or not self._pending_fill_notifications:
            return

        now = datetime.now(self._tz)
        interval = self.slack_notifier.fill_summary_interval_seconds
        started_at = self._first_pending_fill_at or now
        if not force and interval > 0 and (now - started_at).total_seconds() < interval:
            return

        fills = self._pending_fill_notifications
        self._pending_fill_notifications = []
        self._first_pending_fill_at = None

        event_totals: Dict[str, Tuple[int, float]] = {}
        for fill in fills:
            count, total_notional = event_totals.get(fill.event_short_name, (0, 0.0))
            event_totals[fill.event_short_name] = (count + 1, total_notional + fill.notional)

        lines = [
            f"window={started_at.isoformat()} -> {now.isoformat()} fills={len(fills)} total_notional={self._fmt_usd(sum(fill.notional for fill in fills))} events={len(event_totals)}",
        ]
        for event_name, (count, total_notional) in sorted(
            event_totals.items(),
            key=lambda item: (-item[1][0], item[0]),
        )[: self.slack_notifier.fill_summary_max_examples]:
            lines.append(
                f"{event_name}: fills={count} notional={self._fmt_usd(total_notional)}"
            )

        lines.append("samples:")
        for fill in fills[: self.slack_notifier.fill_summary_max_examples]:
            bin_label = (
                f"{fill.bin_index} ({fill.bin_range})"
                if fill.bin_range else str(fill.bin_index)
            )
            lines.append(
                f"{fill.event_short_name}: {fill.side} bin={bin_label} size={fill.size:.2f} price={fill.price:.4f} notional={self._fmt_usd(fill.notional)}"
            )

        self._notify_slack("info", "Fill summary", lines)

    def _notify_balance_allowance_error(
        self,
        event_info: EventInfo,
        context: BalanceAllowanceErrorContext,
    ) -> None:
        bin_label = (
            f"{context.bin_index} ({context.bin_range})"
            if context.bin_range else str(context.bin_index)
        )
        clob_balance = (
            f"{context.clob_available_shares:.2f}"
            if context.clob_available_shares is not None else "unknown"
        )
        lines = [
            f"event_id={event_info.event_id} action={context.action} side={context.side} token_type={context.token_type}",
            f"bin={bin_label} token={context.token_id}",
            f"requested_size={context.requested_size:.2f} requested_price={context.requested_price:.4f} limit_price={context.requested_limit_price:.4f} notional={self._fmt_usd(context.requested_notional)}",
            f"fair={context.reservation_price:.4f} edge={context.edge:+.2%} utility={context.utility_gain:.6f}",
            f"local_yes={context.local_yes_shares:.2f} @ {context.local_yes_avg_cost:.4f} local_no={context.local_no_shares:.2f} @ {context.local_no_avg_cost:.4f}",
            f"portfolio_available={self._fmt_usd(context.portfolio_available_capital or 0.0)} collateral={self._fmt_usd(context.portfolio_total_collateral or 0.0)} pending_orders={context.pending_orders_count}",
            f"clob_balance={clob_balance} raw_balance={context.raw_balance or 'n/a'} nonzero_allowances={context.nonzero_allowances if context.nonzero_allowances is not None else 'unknown'}",
            f"allowances={context.allowances if context.allowances is not None else 'n/a'}",
            f"error={context.error}",
        ]
        self._notify_slack(
            "warning",
            f"Balance / allowance rejection: {event_info.short_name}",
            lines,
            dedupe_key=f"balance_allowance:{event_info.event_id}:{context.bin_index}:{context.token_id}",
            cooldown_seconds=(
                self.slack_notifier.balance_allowance_cooldown_seconds
                if self.slack_notifier else 0.0
            ),
            mention=True,
        )

    def _build_position_fingerprint(self, bot: GASKellyTradingBot) -> Optional[Tuple[Any, ...]]:
        """Build a compact fingerprint of the bot's synced portfolio state."""
        if not bot.kelly_bot or not bot.kelly_bot.portfolio:
            return None

        portfolio = bot.kelly_bot.portfolio
        positions = []
        for bin_idx, pos in sorted(portfolio.positions.items()):
            if pos.yes_shares <= 0.01 and pos.no_shares <= 0.01:
                continue
            positions.append(
                (
                    bin_idx,
                    round(pos.yes_shares, 2),
                    round(pos.no_shares, 2),
                    round(pos.yes_avg_cost, 4),
                    round(pos.no_avg_cost, 4),
                )
            )

        return (
            round(portfolio.capital, 2),
            round(portfolio.total_collateral_used, 2),
            tuple(positions),
        )

    async def _maybe_wait_for_event_stabilization(self, event_id: str, active: ActiveEvent) -> bool:
        """
        Event-local restart stabilization.

        Returns True when the event is ready to trade. While stabilizing, only this
        event is skipped; all others continue normally.
        """
        if active.stabilization_until <= 0.0 or not active.bot.kelly_bot:
            return True

        loop = asyncio.get_running_loop()
        now = loop.time()

        try:
            await active.bot.kelly_bot.sync_positions_from_api(self.wallet_address)
        except Exception as e:
            if now >= active.stabilization_until:
                logger.warning(
                    f"[{active.info.short_name}] Startup stabilization timed out after sync error: {e}. "
                    "Trading will resume."
                )
                active.stabilization_until = 0.0
                return True
            logger.info(
                f"[{active.info.short_name}] Startup stabilization: sync failed ({e}), "
                "skipping this event until next trigger"
            )
            return False

        fingerprint = self._build_position_fingerprint(active.bot)
        if fingerprint == active.last_position_fingerprint:
            active.stable_sync_count += 1
        else:
            active.last_position_fingerprint = fingerprint
            active.stable_sync_count = 1

        required = max(1, self.config.restart_required_stable_syncs)
        if active.stable_sync_count >= required:
            logger.info(
                f"[{active.info.short_name}] Startup stabilization complete after "
                f"{active.stable_sync_count} matching position syncs"
            )
            active.stabilization_until = 0.0
            return True

        if now >= active.stabilization_until:
            logger.warning(
                f"[{active.info.short_name}] Startup stabilization timed out after "
                f"{active.stable_sync_count} sync(s). Trading will resume."
            )
            active.stabilization_until = 0.0
            return True

        logger.info(
            f"[{active.info.short_name}] Startup stabilization: "
            f"{active.stable_sync_count}/{required} stable syncs, skipping this event"
        )
        return False

    async def _trigger_event_startup_tick(self, event_id: str) -> None:
        """
        Trigger the first sync-driven tick for a specific event.

        In sync-driven mode, the manager owns startup tick timing so restored
        events can finish stabilization first.
        """
        while True:
            async with self._events_lock:
                active = self._active_events.get(event_id)
                if not active or not active.startup_tick_pending:
                    return
                delay = 0.0
                if active.stabilization_until > 0.0:
                    delay = max(0.0, active.stabilization_until - asyncio.get_running_loop().time())

            if delay > 0:
                await asyncio.sleep(delay)

            async with self._events_lock:
                active = self._active_events.get(event_id)
                if not active or not active.startup_tick_pending:
                    return

            if not await self._maybe_wait_for_event_stabilization(event_id, active):
                await asyncio.sleep(1.0)
                continue

            async with self._events_lock:
                active = self._active_events.get(event_id)
                if not active or not active.startup_tick_pending:
                    return
                active.startup_tick_pending = False
                bot = active.bot
                bot_name = active.info.short_name

            try:
                logger.info(f"[{bot_name}] Queueing startup tick after stabilization")
                bot.request_sync_tick(
                    source="startup",
                    data_version=self.shared_data_version,
                    allow_authoritative_rebase=False,
                )
            except Exception as e:
                logger.error(f"[{bot_name}] Error in startup tick: {e}", exc_info=True)
            return

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
        if not self.shared_event_store.refresh_from_api(n_days):
            logger.warning("Pre-fetch from XTracker returned no usable data")
            return

        # Get data range for logging
        date_range = self.shared_event_store.get_date_range()
        if date_range[0] and date_range[1]:
            logger.info(
                f"Pre-fetched tweet data from {date_range[0]} to {date_range[1]}"
            )

        self._data_prefetched = True
        now = datetime.now(self._tz)
        self._last_refresh_contract_date = self.contract_utils.get_current_contract_date()
        self._mark_data_fresh("xtracker", now)
        self.shared_data_version += 1

        if self.realtime_tracker:
            self.realtime_tracker.seed_from_official_store(self._last_xtracker_refresh_time)

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

    def _mark_data_fresh(self, source: str, timestamp: Optional[datetime] = None) -> None:
        ts = timestamp or datetime.now(self._tz)
        self._last_data_refresh_time = ts
        self._last_data_refresh_source = source
        if source == "xtracker":
            self._last_xtracker_refresh_time = ts

    def _next_shared_data_version(self) -> int:
        self.shared_data_version += 1
        return self.shared_data_version

    def _event_overlaps_days(self, event_info: EventInfo, days: Set[date]) -> bool:
        if not days:
            return False
        for contract_day in days:
            if event_info.market_start_date <= contract_day < event_info.settlement_date:
                return True
        return False

    async def _notify_bots_of_fresh_data(self) -> None:
        """
        Notify all active trading bots that fresh data is available.

        Bots use this in legacy mode and as a cache invalidation hint.
        """
        async with self._events_lock:
            active_bots = [active.bot for active in self._active_events.values()]

        for bot in active_bots:
            try:
                bot.notify_data_refreshed()
            except Exception as e:
                logger.debug(f"Error notifying bot of fresh data: {e}")

    def _collect_recent_refresh_days(self) -> Set[date]:
        contract_today = self.contract_utils.get_current_contract_date()
        return {contract_today - timedelta(days=1), contract_today}

    def _apply_authoritative_refresh(
        self,
        events_by_day: Dict[date, List],
        affected_days: Set[date],
    ) -> RefreshOutcome:
        outcome = RefreshOutcome(affected_days=set(affected_days))

        for contract_day in sorted(affected_days):
            before = self.shared_event_store.get_contract_day_events(contract_day)
            before_count = len(before)
            replaced = self.shared_event_store.replace_official_day(
                contract_day,
                events_by_day.get(contract_day, []),
            )
            cleared = self.shared_event_store.clear_provisional_day(contract_day)
            after = self.shared_event_store.get_contract_day_events(contract_day)
            after_count = len(after)
            if replaced or cleared or before != after:
                outcome.effective_changed = True
            if after_count < before_count:
                outcome.authoritative_rebase = True
            outcome.new_events += (after_count - before_count)

        return outcome

    async def _do_full_refresh(self) -> RefreshOutcome:
        """Do a full authoritative refresh of official tweet data."""
        n_days = self.config.training_days
        logger.info(f"Full refresh: fetching {n_days} days of authoritative tweet data...")

        history = self.posts_xtracker_client.fetch_historical(n_days, self.contract_utils)
        if not history:
            logger.warning("Full refresh returned no authoritative data")
            return RefreshOutcome()

        cutoff = self.contract_utils.get_current_contract_date() - timedelta(days=n_days)
        affected_days = {
            contract_day
            for contract_day in (
                set(self.shared_event_store._official_events) |
                set(self.shared_event_store._provisional_events) |
                set(history)
            )
            if contract_day >= cutoff
        }
        outcome = self._apply_authoritative_refresh(history, affected_days)

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

        return outcome

    async def _do_incremental_refresh(self) -> RefreshOutcome:
        """
        Refresh current and previous contract days from authoritative XTracker data.
        """
        contract_today = self.contract_utils.get_current_contract_date()
        api_end_date = contract_today + timedelta(days=1)
        yesterday = contract_today - timedelta(days=1)
        contract_days_to_update = {yesterday, contract_today}

        events = self.posts_xtracker_client.fetch_all_events(
            start_date=yesterday,
            end_date=api_end_date,
        )
        if not events:
            logger.warning("Incremental refresh returned no authoritative data; keeping provisional overlay")
            return RefreshOutcome()

        events_by_day: Dict[date, List] = {d: [] for d in contract_days_to_update}
        for event in events:
            event_contract_date = self.contract_utils.get_contract_date(event.timestamp)
            if event_contract_date in events_by_day:
                events_by_day[event_contract_date].append(event)

        outcome = self._apply_authoritative_refresh(events_by_day, contract_days_to_update)
        if outcome.effective_changed:
            logger.info(
                "Incremental refresh applied for %s (delta=%+d, rebase=%s)",
                sorted(contract_days_to_update),
                outcome.new_events,
                outcome.authoritative_rebase,
            )

        return outcome

    async def refresh_shared_data(self) -> RefreshOutcome:
        """
        Refresh shared tweet data with latest from API.

        Smart refresh logic:
        - If contract day changed (noon ET passed), do full refresh and refit models
        - Otherwise, do incremental refresh for recent days

        After refresh, notifies all active bots that fresh data is available.

        Returns:
            RefreshOutcome describing whether the effective snapshot changed.
        """
        now = datetime.now(self._tz)
        current_contract_date = self.contract_utils.get_current_contract_date()

        async with self._data_update_lock:
            contract_day_changed = (
                self._last_refresh_contract_date is not None and
                current_contract_date != self._last_refresh_contract_date
            )
            if contract_day_changed:
                logger.info(
                    f"Contract day changed: {self._last_refresh_contract_date} -> {current_contract_date}. "
                    "Triggering full authoritative refresh and model refit."
                )
                outcome = await self._do_full_refresh()
            else:
                outcome = await self._do_incremental_refresh()

            self._last_refresh_contract_date = current_contract_date
            self._last_full_refresh = now
            self._mark_data_fresh("xtracker", now)
            if outcome.effective_changed:
                self._next_shared_data_version()

        if contract_day_changed and outcome.affected_days:
            await self._refit_active_event_models()

        await self._notify_bots_of_fresh_data()
        return outcome

    async def _queue_sync_tick_requests(
        self,
        source: str,
        affected_days: Set[date],
        *,
        allow_authoritative_rebase: bool = False,
    ) -> None:
        if not affected_days:
            return

        queue_source = "authoritative_rebase" if allow_authoritative_rebase else source

        async with self._events_lock:
            active_snapshot = list(self._active_events.items())

        async def _queue_for_event(event_id: str, active: ActiveEvent) -> None:
            if not self._event_overlaps_days(active.info, affected_days):
                return
            if not await self._maybe_wait_for_event_stabilization(event_id, active):
                return
            active.startup_tick_pending = False
            active.bot.request_sync_tick(
                source=queue_source,
                data_version=self.shared_data_version,
                allow_authoritative_rebase=allow_authoritative_rebase,
            )

        await asyncio.gather(*[_queue_for_event(event_id, active) for event_id, active in active_snapshot])

    async def ingest_realtime_events(self, events: List) -> Tuple[int, Set[date], Optional[RefreshOutcome]]:
        """
        Add provisional realtime events and queue affected ticks.

        Returns:
            Tuple of (inserted_count, affected_days, rollover_refresh_outcome)
        """
        inserted = 0
        affected_days: Set[date] = set()
        rollover_outcome: Optional[RefreshOutcome] = None
        now = datetime.now(self._tz)

        async with self._data_update_lock:
            current_contract_date = self.contract_utils.get_current_contract_date()
            contract_day_changed = (
                self._last_refresh_contract_date is not None and
                current_contract_date != self._last_refresh_contract_date
            )
            if contract_day_changed:
                logger.info("Realtime ingest detected contract-day rollover; refreshing authoritative data first")
                rollover_outcome = await self._do_full_refresh()
                self._last_refresh_contract_date = current_contract_date
                self._last_full_refresh = now
                self._mark_data_fresh("xtracker", now)
                if rollover_outcome.effective_changed:
                    self._next_shared_data_version()

            for event in events:
                if self.shared_event_store.add_provisional_event(event):
                    inserted += 1
                    affected_days.add(self.contract_utils.get_contract_date(event.timestamp))

            if inserted > 0:
                self._mark_data_fresh("twikit", now)
                self._next_shared_data_version()

        if rollover_outcome and rollover_outcome.affected_days:
            await self._refit_active_event_models()
            await self._notify_bots_of_fresh_data()
            if rollover_outcome.effective_changed:
                await self._queue_sync_tick_requests(
                    "xtracker",
                    rollover_outcome.affected_days,
                    allow_authoritative_rebase=rollover_outcome.authoritative_rebase,
                )

        if inserted > 0:
            await self._notify_bots_of_fresh_data()
            await self._queue_sync_tick_requests("realtime", affected_days)

        return inserted, affected_days, rollover_outcome

    async def poll_realtime_tracker(self) -> RealtimePollResult:
        """Poll the realtime tracker once and apply any provisional events."""
        if not self.realtime_tracker:
            return RealtimePollResult(events=[])

        result = await self.realtime_tracker.poll_once()

        cookie_err = self.realtime_tracker.last_cookie_error
        if cookie_err:
            cookie_label = self.realtime_tracker._last_cookie_error_label or "unknown"
            self._notify_slack(
                "error",
                f"Twitter cookie error: {cookie_err}",
                [
                    f"Cookie '{cookie_label}' failed with: {cookie_err}.",
                    f"Consecutive errors: {self.realtime_tracker.consecutive_errors}.",
                    "Check/refresh cookies in config/twitter_cookies.json.",
                ],
                dedupe_key=f"realtime_cookie_error_{cookie_label}",
                cooldown_seconds=3600.0,
                mention=True,
            )

        if result.gap_detected:
            logger.warning("Realtime gap detected; forcing authoritative XTracker refresh")
            self._notify_slack(
                "warning",
                "Realtime gap detected",
                ["Forcing authoritative XTracker refresh."],
                dedupe_key="realtime_gap_detected",
                cooldown_seconds=300.0,
                mention=True,
            )
            refresh_outcome = await self.refresh_shared_data()
            if refresh_outcome.effective_changed:
                await self._queue_sync_tick_requests(
                    "xtracker",
                    refresh_outcome.affected_days,
                    allow_authoritative_rebase=refresh_outcome.authoritative_rebase,
                )
            return RealtimePollResult(events=[], gap_detected=True, newest_timestamp=result.newest_timestamp)

        if result.events:
            inserted, affected_days, _ = await self.ingest_realtime_events(result.events)
            if inserted > 0:
                logger.info(
                    "Realtime tracker inserted %d provisional tweet(s) across %s",
                    inserted,
                    sorted(affected_days),
                )

        return result

    async def _enforce_integrity_deadlines(self) -> None:
        """
        Enforce per-event integrity deadlines on wall-clock time.

        This runs outside normal trading ticks so frozen events still recover
        when no sync-driven tick is triggered for that event.
        """
        async with self._events_lock:
            active_snapshot = list(self._active_events.items())

        async def _enforce(event_id: str, active: ActiveEvent) -> None:
            bot = active.bot
            bot_name = getattr(bot.bot_config, "event_name", None) or active.info.short_name or event_id
            try:
                recovered = await bot.enforce_integrity_deadline()
                if recovered:
                    logger.info(f"[{bot_name}] Integrity deadline enforced outside trading tick")
            except Exception as e:
                logger.error(f"[{bot_name}] Error enforcing integrity deadline: {e}", exc_info=True)

        await asyncio.gather(*[_enforce(event_id, active) for event_id, active in active_snapshot])

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
            logger.warning(f"Fill for inactive event {event_id} bin={bin_index} token {token_id[:16]}...")
            return

        # Route fill to the bot's Kelly executor
        try:
            if active.bot.kelly_bot and active.bot.kelly_bot.kelly_executor:
                kelly_executor = active.bot.kelly_bot.kelly_executor
                kelly_executor.handle_fill(fill_event)
                logger.info(
                    f"[{active.info.short_name}][FILL ROUTED] bin={bin_index} | "
                    f"size={fill_event.size:.1f} @ {fill_event.price:.3f}"
                )
                self._notify_fill(
                    active.info,
                    bin_index,
                    fill_event,
                    bin_range=kelly_executor._bin_range(bin_index),
                )
        except Exception as e:
            logger.error(f"[{active.info.short_name}] Error routing fill bin={bin_index}: {e}", exc_info=True)
            self._notify_slack(
                "error",
                f"Fill routing failed: {active.info.short_name}",
                [f"bin={bin_index}", str(e)],
                dedupe_key=f"fill_routing_error:{event_id}:{bin_index}",
                cooldown_seconds=300.0,
                mention=True,
            )

    async def _handle_global_stale_order(self, pending: PendingOrder) -> None:
        """
        Route stale order from global UserStreamClient to the correct bot.

        Uses token_id to identify which event/bot should handle the cancellation.
        For FAK orders the unfilled remainder is already killed by the exchange,
        so this mainly cleans up local tracking in the executor.
        """
        token_id = pending.token_id

        event_mapping = self._token_to_event.get(token_id)
        if not event_mapping:
            logger.debug(f"Stale order for unknown token {token_id[:16]}...")
            return

        event_id, bin_index = event_mapping

        active = self._active_events.get(event_id)
        if not active:
            logger.warning(f"Stale order for inactive event {event_id} bin={bin_index} token {token_id[:16]}...")
            return

        try:
            if active.bot.kelly_bot and active.bot.kelly_bot.kelly_executor:
                await active.bot.kelly_bot.kelly_executor.handle_stale_order(pending)
        except Exception as e:
            logger.error(f"[{active.info.short_name}] Error handling stale order bin={bin_index}: {e}", exc_info=True)
            self._notify_slack(
                "error",
                f"Stale order handling failed: {active.info.short_name}",
                [f"bin={bin_index}", str(e)],
                dedupe_key=f"stale_order_error:{event_id}:{bin_index}",
                cooldown_seconds=300.0,
            )

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

    @staticmethod
    def _extract_user_stream_market_ids(event_info: EventInfo) -> List[str]:
        """
        Extract market ids for Polymarket user-channel subscriptions.

        Multi-bin events expose per-bin condition ids. Those are the most
        specific market ids we have for routing fill and order events, so prefer
        them and fall back to the event-level condition id when necessary.
        """
        market_ids: List[str] = []
        seen = set()

        for bin_def in event_info.bins:
            condition_id = bin_def.get("condition_id")
            if condition_id and condition_id not in seen:
                seen.add(condition_id)
                market_ids.append(condition_id)

        if not market_ids and event_info.condition_id:
            market_ids.append(event_info.condition_id)

        return market_ids

    async def _sync_user_stream_markets(self) -> None:
        """Keep the shared user stream aligned with known pending/active markets."""
        if not self.user_stream:
            return

        async with self._events_lock:
            event_infos = [
                *(active.info for active in self._active_events.values()),
                *self._pending_events.values(),
            ]

        market_ids: List[str] = []
        for event_info in event_infos:
            market_ids.extend(self._extract_user_stream_market_ids(event_info))

        await self.user_stream.set_markets(market_ids)

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
        counting_start_dt, _ = self.contract_utils.get_contract_day_bounds_utc(
            event_info.market_start_date
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
            now = datetime.now(timezone.utc)
            settlement_dt, _ = self.contract_utils.get_contract_day_bounds_utc(
                event_info.settlement_date
            )

            hours_to_settlement = (settlement_dt - now).total_seconds() / 3600

            # Use event-specific t_stop from trading rules
            event_duration_days = (event_info.settlement_date - event_info.market_start_date).days
            if self.config.event_trading_rules:
                rules = self.config.event_trading_rules.get_rules_for_event(event_duration_days)
                t_stop = rules.min_hours_before_settlement
            else:
                t_stop = self.kelly_config.t_stop_hours

            if hours_to_settlement <= t_stop:
                logger.info(
                    f"Event {event_id} too close to settlement "
                    f"({hours_to_settlement:.1f}h remaining, t_stop={t_stop}h)"
                )
                return False

            # Note: Long-duration events are added to pending but only started
            # when remaining time < max_event_duration_days (checked in _try_start_pending_events)

            # Add to pending
            self._pending_events[event_id] = event_info
            logger.info(f"Added event {event_id} ({event_info.short_name}) to pending queue")

        await self._sync_user_stream_markets()

        # Try to start immediately if running (outside lock to avoid deadlock)
        if self._running:
            await self._try_start_pending_events()

        return True

    def _event_passes_duration_filter(self, event_info: EventInfo) -> bool:
        """Check if event duration is within configured min/max range (inclusive)."""
        duration = (event_info.settlement_date - event_info.market_start_date).days
        return self.config.min_event_duration_days <= duration <= self.config.max_event_duration_days

    async def _discover_and_add_events(self) -> int:
        """
        Discover new events using the discovery callback and add them.

        Only adds events that pass the duration filter (min/max_event_duration_days).

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
                # Filter by duration
                duration = (event_info.settlement_date - event_info.market_start_date).days
                if not self._event_passes_duration_filter(event_info):
                    logger.debug(
                        f"Skipping {event_info.short_name} ({duration}d) - "
                        f"outside duration filter [{self.config.min_event_duration_days}, {self.config.max_event_duration_days}]"
                    )
                    continue

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
            now = datetime.now(timezone.utc)
            settlement_dt, _ = self.contract_utils.get_contract_day_bounds_utc(
                event_info.settlement_date
            )
            hours_to_settlement = (settlement_dt - now).total_seconds() / 3600

            # Use event-specific t_stop from trading rules
            event_duration_days = (event_info.settlement_date - event_info.market_start_date).days
            if self.config.event_trading_rules:
                rules = self.config.event_trading_rules.get_rules_for_event(event_duration_days)
                t_stop = rules.min_hours_before_settlement
            else:
                t_stop = self.kelly_config.t_stop_hours

            if hours_to_settlement <= t_stop:
                logger.info(f"Event {event_id} expired (t_stop={t_stop}h), removing from pending")
                async with self._events_lock:
                    self._pending_events.pop(event_id, None)
                await self._sync_user_stream_markets()
                # Return any restored capital allocation (e.g., from reconstruct_state_from_api)
                # Without this, concluded events with positions leave orphaned allocations
                allocation = await self.capital_pool.get_allocation(event_id)
                if allocation:
                    await self.capital_pool.return_capital(event_id, allocation.current_value)
                    logger.info(f"Returned ${allocation.current_value:.2f} from expired event {event_id}")
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
                await self._sync_user_stream_markets()
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

        # Create bot config (sync-driven: bot idles, manager triggers ticks)
        bot_config = TradingBotConfig(
            slow_tick_interval_seconds=self.config.tick_interval_seconds,
            dry_run=self.config.dry_run,
            settlement_date=event_info.settlement_date,
            initial_capital=allocated_capital,
            training_days=self.config.training_days,
            use_gas=self.config.use_gas,
            event_name=event_info.short_name,
            projection_model=self.config.projection_model,
            sync_driven=True,
        )

        # Create per-event kelly_config with event-specific t_stop_hours
        # from the event trading rules (min_hours_before_settlement)
        event_kelly_config = self.kelly_config
        if self.config.event_trading_rules:
            event_duration_days = (event_info.settlement_date - event_info.market_start_date).days
            rules = self.config.event_trading_rules.get_rules_for_event(event_duration_days)
            from dataclasses import replace
            event_kelly_config = replace(self.kelly_config, t_stop_hours=rules.min_hours_before_settlement)
            logger.info(
                f"Event {event_info.short_name}: t_stop={rules.min_hours_before_settlement}h "
                f"(from '{rules.name}' rules, {event_duration_days}d event)"
            )

        # Create bot with shared EventStore
        bot = GASKellyTradingBot(
            clob_client=self.clob_client,
            kelly_config=event_kelly_config,
            forecaster_config=self.forecaster_config,
            bot_config=bot_config,
            event_store=self.shared_event_store,
            user_stream=self.user_stream,  # Pass global user stream
        )

        # Setup bot (expensive, do outside lock)
        await bot.setup(event_info.bins)
        if (
            bot.kelly_bot and
            bot.kelly_bot.kelly_executor is not None
        ):
            bot.kelly_bot.kelly_executor.on_balance_allowance_error = (
                lambda context, event_info=event_info: self._notify_balance_allowance_error(
                    event_info,
                    context,
                )
            )

        # CRITICAL: Sync existing positions from API into the Kelly portfolio
        # This ensures collateral tracking works correctly for restored positions
        # Do this even in dry_run mode - we need accurate collateral tracking
        stabilization_until = 0.0
        stable_sync_count = 0
        last_position_fingerprint: Optional[Tuple[Any, ...]] = None
        if bot.kelly_bot:
            try:
                await bot.kelly_bot.sync_positions_from_api(self.wallet_address)
                collateral = bot.kelly_bot.portfolio.total_collateral_used if bot.kelly_bot.portfolio else 0
                logger.info(f"Synced existing positions into Kelly portfolio: invested=${collateral:.2f}")
                if collateral > 0.01 and self.config.restart_stabilization_seconds > 0:
                    loop = asyncio.get_running_loop()
                    stabilization_until = loop.time() + self.config.restart_stabilization_seconds
                    stable_sync_count = 1
                    last_position_fingerprint = self._build_position_fingerprint(bot)
                    logger.info(
                        f"[{event_info.short_name}] Startup stabilization enabled for "
                        f"{self.config.restart_stabilization_seconds:.0f}s after restoring positions"
                    )
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
                stabilization_until=stabilization_until,
                stable_sync_count=stable_sync_count,
                last_position_fingerprint=last_position_fingerprint,
                startup_tick_pending=True,
            )
            num_active = len(self._active_events)
            self._total_events_started += 1

        asyncio.create_task(
            self._trigger_event_startup_tick(event_id),
            name=f"startup_tick_{event_id}",
        )

        logger.info(
            f"Event {event_id} started. "
            f"Active events: {num_active}, "
            f"Pool available: ${self.capital_pool.available_capital:.2f}"
        )
        self._notify_event_started(event_info, allocated_capital, num_active)

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
            event_name = getattr(bot.bot_config, "event_name", event_id)
            self._notify_slack(
                "error",
                f"Event runtime error: {event_name}",
                [str(e)],
                dedupe_key=f"event_runtime_error:{event_id}",
                cooldown_seconds=300.0,
                mention=True,
            )
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
        event_info: Optional[EventInfo] = None

        # Unregister token_ids for fill routing
        async with self._events_lock:
            if event_id in self._active_events:
                event_info = self._active_events[event_id].info
                self._unregister_event_tokens(event_info)

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

            await self._sync_user_stream_markets()

            logger.info(
                f"Event {event_id} completed. "
                f"Active events: {num_active}, "
                f"Pool available: ${self.capital_pool.available_capital:.2f}"
            )
            if event_info is not None:
                self._notify_event_completed(event_info, final_value, num_active)

            # Try to start pending events with freed capital
            if self._running:
                await self._try_start_pending_events()

    async def run(self) -> None:
        """
        Run the multi-event manager with sync-driven trading.

        Instead of refreshing posts on a timer and trading on a separate timer,
        this loop polls XTracker's lastSync timestamp every ~5 seconds.
        When lastSync changes (meaning XTracker just synced posts from X/Twitter):
          1. Immediately refresh posts from XTracker /posts API
          2. Trigger immediate trading on all bots (recompute MC + Kelly + execute)

        Between sync events, the bots are completely idle (no fast ticks).

        Other periodic tasks (count validation, event scan, health) remain timer-based.
        """
        self._running = True
        self._stop_event = asyncio.Event()
        self._start_time = datetime.now(self._tz)

        logger.info("Starting MultiEventManager (sync-driven mode)")

        # Auto-initialize capital from API if total_capital is 0
        if self.capital_pool.needs_initialization:
            await self._initialize_capital_from_api()

        # Start global user stream for fill confirmations
        if self.user_stream:
            await self._sync_user_stream_markets()
            await self.user_stream.start()
            logger.info("Global UserStreamClient started")

        # Pre-fetch shared tweet data (once for all events)
        await self.prefetch_shared_data()

        # Initial validation
        await self.validate_all_event_counts()

        # Start any pending events
        await self._try_start_pending_events()

        # Track last execution times for periodic tasks
        last_count_validation = datetime.now(self._tz)
        last_event_scan = datetime.now(self._tz)
        last_capital_sync = datetime.now(self._tz)
        self._last_health_log_time = datetime.now(self._tz)

        logger.info(
            f"Main loop started (sync-driven): "
            f"sync_poll={self.config.sync_poll_interval}s, "
            f"count_validation={self.config.count_validation_interval}s, "
            f"event_scan={self.config.event_scan_interval}s, "
            f"health_log={self.config.health_log_interval}s, "
            f"capital_sync={self.config.capital_sync_interval}s"
        )

        # Log initial health status
        self._log_health()
        self._notify_startup()

        # Main loop: poll lastSync + periodic tasks
        while self._running:
            try:
                # --- Sync-driven polling: check lastSync ---
                try:
                    current_sync = self.trackings_xtracker_client.get_last_sync()
                except Exception as e:
                    current_sync = None
                    self._sync_poll_errors += 1
                    if self._sync_poll_errors <= 3 or self._sync_poll_errors % 10 == 0:
                        logger.warning(
                            f"Failed to poll lastSync (error #{self._sync_poll_errors}): {e}"
                        )

                if current_sync is not None:
                    self._sync_poll_errors = 0  # Reset on success

                    if current_sync != self._last_known_sync:
                        # Sync changed — but XTracker's sync is not atomic,
                        # so lastSync may update 2+ times per cycle.
                        # Debounce: keep polling until lastSync stabilizes.
                        logger.info(
                            f"XTracker sync change detected: {current_sync} "
                            f"(prev: {self._last_known_sync}), debouncing..."
                        )
                        self._last_known_sync = current_sync
                        debounce_start = datetime.now(self._tz)

                        while (datetime.now(self._tz) - debounce_start).total_seconds() < self.config.sync_debounce_seconds:
                            try:
                                await asyncio.wait_for(
                                    self._stop_event.wait(),
                                    timeout=self.config.sync_poll_interval
                                )
                                break  # Stop requested during debounce
                            except asyncio.TimeoutError:
                                pass

                            if not self._running:
                                break

                            try:
                                latest = self.trackings_xtracker_client.get_last_sync()
                            except Exception:
                                latest = None

                            if latest is not None and latest != self._last_known_sync:
                                logger.info(f"XTracker sync updated during debounce: {latest}")
                                self._last_known_sync = latest

                        if not self._running:
                            break

                        # Debounce complete — now act on the final sync value
                        sync_detected_at = datetime.now(self._tz)
                        logger.info(
                            f"XTracker sync settled: {self._last_known_sync} "
                            f"(debounced {(sync_detected_at - debounce_start).total_seconds():.1f}s)"
                        )

                        refresh_outcome = await self.refresh_shared_data()
                        if refresh_outcome.effective_changed:
                            await self._queue_sync_tick_requests(
                                "xtracker",
                                refresh_outcome.affected_days,
                                allow_authoritative_rebase=refresh_outcome.authoritative_rebase,
                            )
                        else:
                            logger.info("Authoritative refresh made no effective event-store changes")

                        # 2. Log cycle timing (soft deadline — always completes)
                        elapsed = (datetime.now(self._tz) - sync_detected_at).total_seconds()
                        logger.info(f"Sync→trade cycle completed in {elapsed:.1f}s")
                        if elapsed > self.config.sync_trade_deadline_seconds:
                            logger.warning(
                                f"Sync→trade took {elapsed:.1f}s "
                                f"(soft deadline: {self.config.sync_trade_deadline_seconds}s)"
                            )

                # --- Periodic tasks (timer-based, unchanged) ---
                now = datetime.now(self._tz)

                if self.realtime_tracker and self.realtime_tracker.should_poll(now):
                    await self.poll_realtime_tracker()
                    now = datetime.now(self._tz)

                # Count validation (15 min)
                if (now - last_count_validation).total_seconds() >= self.config.count_validation_interval:
                    await self.validate_all_event_counts()
                    last_count_validation = now

                # Event discovery + cleanup (1 hour)
                if (now - last_event_scan).total_seconds() >= self.config.event_scan_interval:
                    await self._discover_and_add_events()
                    await self._try_start_pending_events()
                    await self._cleanup_old_data()
                    last_event_scan = now

                # Capital re-sync from API (1 hour)
                if (now - last_capital_sync).total_seconds() >= self.config.capital_sync_interval:
                    await self._sync_capital_from_api()
                    await self._try_start_pending_events()
                    last_capital_sync = now

                # Health logging (1 hour)
                if (now - self._last_health_log_time).total_seconds() >= self.config.health_log_interval:
                    self._log_health()
                    self._last_health_log_time = now

                # Integrity deadlines (every poll cycle)
                await self._enforce_integrity_deadlines()

                # Capital values + status (every poll cycle)
                await self._update_capital_values()
                self._log_status()
                self._maybe_notify_health()
                self._maybe_flush_fill_summaries()

                # --- Sleep until next poll (interruptible by stop_event) ---
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.config.sync_poll_interval
                    )
                    # Stop requested
                    break
                except asyncio.TimeoutError:
                    pass

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                self._errors_count += 1
                self._notify_slack(
                    "error",
                    "Main loop error",
                    [str(e)],
                    dedupe_key="multi_event_manager_main_loop",
                    cooldown_seconds=300.0,
                    mention=True,
                )
                await asyncio.sleep(60)  # Back off on error

        # Shutdown
        await self._shutdown()

    async def _sync_capital_from_api(self) -> None:
        """Re-sync capital pool total from on-chain USDC balance + positions."""
        try:
            await self._initialize_capital_from_api()
        except Exception as e:
            logger.warning(f"Capital re-sync failed: {e}")
            self._notify_slack(
                "warning",
                "Capital re-sync failed",
                [str(e)],
                dedupe_key="capital_resync_failed",
                cooldown_seconds=1800.0,
                mention=True,
            )

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

        self._maybe_flush_fill_summaries(force=True)
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
        active_events = {}
        for event_id, active in self._active_events.items():
            integrity = {}
            try:
                integrity = active.bot.get_state_summary().get("integrity", {})
            except Exception:
                integrity = {}
            active_events[event_id] = {
                "short_name": active.info.short_name,
                "settlement_date": active.info.settlement_date.isoformat(),
                "allocated_capital": active.allocated_capital,
                "started_at": active.started_at.isoformat(),
                "integrity": integrity,
            }

        return {
            "running": self._running,
            "capital_pool": self.capital_pool.get_summary(),
            "performance": self.capital_pool.get_performance_summary(),
            "shared_data_version": self.shared_data_version,
            "last_data_refresh_time": (
                self._last_data_refresh_time.isoformat()
                if self._last_data_refresh_time else None
            ),
            "last_data_refresh_source": self._last_data_refresh_source,
            "last_xtracker_refresh_time": (
                self._last_xtracker_refresh_time.isoformat()
                if self._last_xtracker_refresh_time else None
            ),
            "active_events": active_events,
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
        event_store_official_events = 0
        event_store_provisional_events = 0
        event_store_effective_events = 0
        with self.shared_event_store._lock:
            all_dates = (
                set(self.shared_event_store._official_events) |
                set(self.shared_event_store._provisional_events)
            )
            event_store_dates = len(all_dates)
            event_store_official_events = sum(
                len(events) for events in self.shared_event_store._official_events.values()
            )
            event_store_provisional_events = sum(
                len(events) for events in self.shared_event_store._provisional_events.values()
            )
            event_store_effective_events = sum(
                len(self.shared_event_store._merged_events_for_day(contract_date))
                for contract_date in all_dates
            )

        realtime_status = None
        if self.realtime_tracker is not None:
            realtime_status = {
                "enabled": True,
                "last_poll_time": (
                    self.realtime_tracker.last_poll_time.isoformat()
                    if self.realtime_tracker.last_poll_time else None
                ),
                "watermark": (
                    self.realtime_tracker.watermark.isoformat()
                    if self.realtime_tracker.watermark else None
                ),
                "consecutive_errors": self.realtime_tracker.consecutive_errors,
                "backoff_until": (
                    self.realtime_tracker.backoff_until.isoformat()
                    if self.realtime_tracker.backoff_until else None
                ),
            }
        else:
            realtime_status = {"enabled": False}

        # Capital pool stats
        pool_summary = self.capital_pool.get_summary()
        pool_performance = self.capital_pool.get_performance_summary()

        # Time since last cleanup
        time_since_cleanup = None
        if self._last_cleanup_time:
            time_since_cleanup = (now - self._last_cleanup_time).total_seconds() / 3600

        frozen_event_names = []
        frozen_events = 0
        for active in self._active_events.values():
            try:
                integrity = active.bot.get_state_summary().get("integrity", {})
            except Exception:
                integrity = {}
            if integrity.get("frozen"):
                frozen_events += 1
                frozen_event_names.append(active.info.short_name)

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
            "event_store_official_tweets": event_store_official_events,
            "event_store_provisional_tweets": event_store_provisional_events,
            "event_store_total_tweets": event_store_effective_events,

            # Capital
            "total_capital": pool_summary.get("total_capital", 0),
            "available_capital": pool_summary.get("available_capital", 0),
            "allocated_capital": pool_summary.get("allocated_capital", 0),
            "total_pnl": pool_performance.get("total_pnl", 0),

            # Maintenance
            "last_cleanup_hours_ago": round(time_since_cleanup, 2) if time_since_cleanup else None,
            "errors_count": self._errors_count,
            "shared_data_version": self.shared_data_version,
            "last_data_refresh_time": (
                self._last_data_refresh_time.isoformat()
                if self._last_data_refresh_time else None
            ),
            "last_data_refresh_source": self._last_data_refresh_source,
            "last_xtracker_refresh_time": (
                self._last_xtracker_refresh_time.isoformat()
                if self._last_xtracker_refresh_time else None
            ),

            # Active event details
            "active_event_names": [
                active.info.short_name for active in self._active_events.values()
            ],
            "frozen_events": frozen_events,
            "frozen_event_names": frozen_event_names,

            # User stream status - get detailed status if available
            "user_stream": self._get_user_stream_status(),
            "realtime_tracker": realtime_status,
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
        logger.info(
            "    EventStore: %s days, %s effective tweets (%s official, %s provisional)",
            health['event_store_days'],
            health['event_store_total_tweets'],
            health['event_store_official_tweets'],
            health['event_store_provisional_tweets'],
        )
        logger.info(f"    Completed history: {health['completed_events_in_history']} events")
        if health['last_cleanup_hours_ago']:
            logger.info(f"    Last cleanup: {health['last_cleanup_hours_ago']:.1f} hours ago")
        logger.info(f"    Errors: {health['errors_count']}")
        logger.info(
            "    Data freshness: source=%s version=%s refreshed_at=%s",
            health.get('last_data_refresh_source'),
            health.get('shared_data_version'),
            health.get('last_data_refresh_time'),
        )
        if health.get('last_xtracker_refresh_time'):
            logger.info(f"    Last XTracker refresh: {health['last_xtracker_refresh_time']}")
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
        logger.info("")
        logger.info("  REALTIME TRACKER:")
        realtime = health.get('realtime_tracker', {})
        logger.info(f"    Enabled: {realtime.get('enabled', False)}")
        if realtime.get('enabled', False):
            logger.info(f"    Last poll: {realtime.get('last_poll_time')}")
            logger.info(f"    Watermark: {realtime.get('watermark')}")
            logger.info(f"    Consecutive errors: {realtime.get('consecutive_errors')}")
            if realtime.get('backoff_until'):
                logger.info(f"    Backoff until: {realtime.get('backoff_until')}")
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

        await self._sync_user_stream_markets()

        # Second: Add events WITHOUT positions (they need new allocations)
        # Apply duration filter — only auto-start events within configured range
        for event_info in events_without_positions:
            duration = (event_info.settlement_date - event_info.market_start_date).days
            if not self._event_passes_duration_filter(event_info):
                logger.info(
                    f"Skipping {event_info.short_name} ({duration}d, no positions) - "
                    f"outside duration filter [{self.config.min_event_duration_days}, {self.config.max_event_duration_days}]"
                )
                continue
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
