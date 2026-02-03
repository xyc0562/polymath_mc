"""
Entry point for live trading with GAS-Kelly bot.

Usage:
    # Paper trading (dry run) - trades ALL active 7-day events
    python3 -m src.algo.musk_tweet_count.forecaster.run_trading --dry-run

    # Trade specific event only
    python3 -m src.algo.musk_tweet_count.forecaster.run_trading --event "Jan 27 - Feb 3"

    # List available events without trading
    python3 -m src.algo.musk_tweet_count.forecaster.run_trading --list-events

    # Live trading with small position limit
    python3 -m src.algo.musk_tweet_count.forecaster.run_trading --live --max-position-usd 10
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds

from src.algo.musk_tweet_count.forecaster.config import ForecasterConfig
from src.algo.musk_tweet_count.forecaster.trading_bot import (
    GASKellyTradingBot,
    TradingBotConfig,
)
from src.algo.musk_tweet_count.kelly.config import KellyConfig
from src.utils.crypto_utils import load_private_key

# Setup logging (force=True to override any handlers set during imports)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s[%(levelname).1s]: %(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

# Suppress verbose httpx logging (set to WARNING to hide repetitive request logs)
logging.getLogger("httpx").setLevel(logging.WARNING)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_clob_client() -> ClobClient:
    """
    Create authenticated CLOB client from environment variables.

    Required environment variables:
    - CLOB_API_KEY: Polymarket API key
    - CLOB_API_SECRET: Polymarket API secret
    - CLOB_API_PASSPHRASE: Polymarket API passphrase

    Private key can be set via (in order of priority):
    - POLYMARKET_PRIVATE_KEY: Plain private key (less secure)
    - ENCRYPTED_POLYMARKET_PRIVATE_KEY: Encrypted key string
    - ENCRYPTED_POLYMARKET_PRIVATE_KEY_FILE: Path to encrypted key file

    For encrypted keys, password is obtained from:
    - PK_PWD environment variable
    - Interactive prompt (hidden input)
    """
    # Load .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("CLOB_API_KEY")
    api_secret = os.environ.get("CLOB_API_SECRET")
    passphrase = os.environ.get("CLOB_API_PASSPHRASE")

    if not all([api_key, api_secret, passphrase]):
        raise ValueError(
            "Missing required environment variables. "
            "Set CLOB_API_KEY, CLOB_API_SECRET, CLOB_API_PASSPHRASE"
        )

    # Load private key (supports encrypted keys)
    private_key = load_private_key()

    creds = ApiCreds(
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=passphrase,
    )

    # Use mainnet by default
    host = os.environ.get("POLY_HOST", "https://clob.polymarket.com")
    chain_id = int(os.environ.get("POLY_CHAIN_ID", "137"))

    return ClobClient(
        host=host,
        chain_id=chain_id,
        key=private_key,
        creds=creds,
    )


@dataclass
class EventInfo:
    """Information about a 7-day Musk tweet event."""
    title: str
    short_name: str  # e.g., "Jan 27 - Feb 3"
    start: Optional[datetime]
    end: Optional[datetime]
    xtracker_count: Optional[int]
    bins: List[Dict]
    settlement_date: Optional[date]
    status: str  # "ACTIVE", "NOT_STARTED", "ENDED"
    hours_remaining: float


class SharedMarketData:
    """
    Manages shared data across all trading events.

    Optimizations:
    1. Single EventStore shared across all forecasters (one API call for posts)
    2. Cached probability tables updated on slow loop (every 2-2.5 min)
    3. Fast loop uses cached probabilities without recalculation

    Architecture:
        Slow Loop (2-2.5 min):
            1. Fetch posts from XTracker (ONCE)
            2. Update all forecasters
            3. Recalculate probability tables

        Fast Loop (WebSocket + polling):
            1. Use cached probabilities
            2. Compare with orderbook prices
            3. Execute trades
    """

    def __init__(
        self,
        forecaster_config: "ForecasterConfig",
        training_days: int = 45,
    ):
        """
        Initialize shared market data.

        Args:
            forecaster_config: Configuration for forecasters
            training_days: Days of history for model fitting
        """
        from .config import ForecasterConfig
        from .data import ContractDayUtils, EventStore, XTrackerClient

        self.forecaster_config = forecaster_config
        self.training_days = training_days

        # Shared data layer
        self.contract_utils = ContractDayUtils(
            timezone=forecaster_config.timezone,
            boundary_hour=forecaster_config.contract_boundary_hour,
        )
        self.xtracker = XTrackerClient()
        self.event_store = EventStore(self.contract_utils, self.xtracker)

        # Per-event forecasters (share the same EventStore)
        self._forecasters: Dict[str, "Musk7DayForecaster"] = {}

        # Cached probability tables: event_name -> List[float]
        self._probability_tables: Dict[str, List[float]] = {}

        # Market bins per event: event_name -> List[(lower, upper)]
        self._market_bins: Dict[str, List[tuple]] = {}

        # Current cumulative counts per event
        self._cumulative_counts: Dict[str, int] = {}

        # Timing
        self._last_posts_update: Optional[datetime] = None
        self._last_probability_update: Optional[datetime] = None
        self._initialized = False

    async def initialize(self, n_days: int = 45) -> None:
        """
        Initialize by fetching historical data.

        Args:
            n_days: Days of history to fetch
        """
        logger.info(f"[SharedMarketData] Fetching {n_days} days of historical posts...")

        # Fetch posts once
        self.event_store.refresh_from_api(n_days)

        self._last_posts_update = datetime.now(self.contract_utils.tz)
        self._initialized = True

        logger.info(f"[SharedMarketData] Initialized with {len(self.event_store._events)} contract-days of data")

    def register_event(
        self,
        event_name: str,
        market_bins: List[tuple],
        settlement_date: date,
        market_start_date: Optional[date] = None,
    ) -> "Musk7DayForecaster":
        """
        Register an event and create its forecaster.

        Args:
            event_name: Event identifier (e.g., "Jan 27 - Feb 3")
            market_bins: List of (lower, upper) bin boundaries
            settlement_date: Event settlement date
            market_start_date: First day of counting window (default: settlement - 7 days)

        Returns:
            The forecaster for this event
        """
        from .forecaster import Musk7DayForecaster

        if event_name in self._forecasters:
            return self._forecasters[event_name]

        # Create forecaster with shared EventStore
        forecaster = Musk7DayForecaster(
            config=self.forecaster_config,
            event_store=self.event_store,
        )

        # Fit the forecaster (skip fetch since we already have data)
        forecaster.fit(n_days=self.training_days, skip_fetch=True)

        self._forecasters[event_name] = forecaster
        self._market_bins[event_name] = market_bins
        self._probability_tables[event_name] = []
        self._cumulative_counts[event_name] = 0

        # Store event dates for proper forecasting
        if not hasattr(self, '_settlement_dates'):
            self._settlement_dates: Dict[str, date] = {}
        if not hasattr(self, '_market_start_dates'):
            self._market_start_dates: Dict[str, date] = {}

        self._settlement_dates[event_name] = settlement_date
        self._market_start_dates[event_name] = market_start_date or (settlement_date - timedelta(days=7))

        logger.info(f"[SharedMarketData] Registered event: {event_name} ({self._market_start_dates[event_name]} - {settlement_date})")

        return forecaster

    def unregister_event(self, event_name: str) -> None:
        """Remove an event."""
        self._forecasters.pop(event_name, None)
        self._market_bins.pop(event_name, None)
        self._probability_tables.pop(event_name, None)
        self._cumulative_counts.pop(event_name, None)
        if hasattr(self, '_settlement_dates'):
            self._settlement_dates.pop(event_name, None)
        if hasattr(self, '_market_start_dates'):
            self._market_start_dates.pop(event_name, None)

    async def update_posts(self) -> int:
        """
        Fetch new posts from XTracker API.

        Called by slow loop. Fetches incrementally since last update.

        Returns:
            Total events in store after refresh
        """
        # Track count before refresh
        count_before = sum(len(events) for events in self.event_store._events.values())

        # Fetch posts for recent days only (incremental update)
        self.event_store.refresh_from_api(n_days=3)

        # Track count after refresh
        count_after = sum(len(events) for events in self.event_store._events.values())
        new_count = count_after - count_before

        self._last_posts_update = datetime.now(self.contract_utils.tz)

        logger.info(f"[SharedMarketData] Refreshed posts: {new_count} new, {count_after} total")
        return count_after

    def update_probabilities(self) -> None:
        """
        Recalculate probability tables for all events.

        Called by slow loop after posts are updated.
        Uses event-specific forecasting that accounts for:
        1. Actual counts from completed days in the event window
        2. Forecasts only the remaining days until settlement
        """
        import numpy as np

        for event_name, forecaster in self._forecasters.items():
            market_bins = self._market_bins.get(event_name, [])
            if not market_bins:
                continue

            # Get event dates for proper window-based forecasting
            settlement_date = getattr(self, '_settlement_dates', {}).get(event_name)
            market_start_date = getattr(self, '_market_start_dates', {}).get(event_name)

            if settlement_date and market_start_date:
                # Use event-window forecast (proper handling of past + remaining days)
                result = forecaster.forecast_for_event_window(
                    market_start_date=market_start_date,
                    settlement_date=settlement_date,
                )
                logger.debug(
                    f"[{event_name}] Event-window forecast: mean={result.mean:.1f}, "
                    f"std={result.std:.1f}"
                )
            else:
                # Fallback to generic 7-day forecast
                result = forecaster.forecast_7day_distribution(use_cache=False)
                logger.debug(
                    f"[{event_name}] Generic 7-day forecast (no event dates): "
                    f"mean={result.mean:.1f}, std={result.std:.1f}"
                )

            # Get current cumulative count for dead bin detection
            # This needs to be set externally based on market's date range
            current_count = self._cumulative_counts.get(event_name, 0)

            # Generate samples and compute bin probabilities
            n_samples = 10000
            samples = np.random.normal(result.mean, result.std, n_samples)
            samples = np.maximum(samples, current_count)

            probabilities = []
            for lower, upper in market_bins:
                if upper < current_count:
                    prob = 0.0  # Dead bin
                else:
                    count = np.sum((samples >= lower) & (samples <= upper))
                    prob = count / n_samples
                probabilities.append(prob)

            # Renormalize
            total = sum(probabilities)
            if total > 0:
                probabilities = [p / total for p in probabilities]

            self._probability_tables[event_name] = probabilities

        self._last_probability_update = datetime.now(self.contract_utils.tz)
        logger.info(f"[SharedMarketData] Updated probabilities for {len(self._forecasters)} events")

    def set_cumulative_count(self, event_name: str, count: int) -> None:
        """Update the cumulative count for an event."""
        self._cumulative_counts[event_name] = count

    def get_probabilities(self, event_name: str) -> List[float]:
        """Get cached probability table for an event."""
        return self._probability_tables.get(event_name, [])

    def get_forecaster(self, event_name: str) -> Optional["Musk7DayForecaster"]:
        """Get the forecaster for an event."""
        return self._forecasters.get(event_name)

    def get_status(self) -> Dict:
        """Get status summary."""
        return {
            "initialized": self._initialized,
            "num_events": len(self._forecasters),
            "events": list(self._forecasters.keys()),
            "last_posts_update": self._last_posts_update.isoformat() if self._last_posts_update else None,
            "last_probability_update": self._last_probability_update.isoformat() if self._last_probability_update else None,
            "total_contract_days": len(self.event_store._events),
        }


class CapitalManager:
    """
    Manages a shared capital pool across multiple trading events.

    Each event can draw from the pool up to its per-event limit (c_event_max).
    Capital is released when positions are closed or events settle.

    Example:
        Total: $1000, c_event_max: $200
        Event A: $150 collateral (can add $50 more)
        Event B: $100 collateral (can add $100 more)
        Available: $750
    """

    def __init__(self, total_capital: float, c_event_max: float):
        """
        Initialize the capital manager.

        Args:
            total_capital: Total capital available across all events
            c_event_max: Maximum collateral allowed per event
        """
        self.total_capital = total_capital
        self.c_event_max = c_event_max
        self._available = total_capital
        self._event_collateral: Dict[str, float] = {}  # event_name -> collateral
        self._realized_pnl: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def available(self) -> float:
        """Capital available for new trades."""
        return self._available

    @property
    def total_collateral(self) -> float:
        """Total collateral across all events."""
        return sum(self._event_collateral.values())

    @property
    def realized_pnl(self) -> float:
        """Cumulative realized P&L."""
        return self._realized_pnl

    def get_event_collateral(self, event_name: str) -> float:
        """Get current collateral for an event."""
        return self._event_collateral.get(event_name, 0.0)

    def get_available_for_event(self, event_name: str) -> float:
        """
        Get how much more capital this event can use.

        Returns min of:
        - Per-event limit minus current event collateral
        - Pool available capital
        """
        current = self._event_collateral.get(event_name, 0.0)
        event_headroom = max(0, self.c_event_max - current)
        return min(event_headroom, self._available)

    async def request_allocation(self, event_name: str, amount: float) -> float:
        """
        Request capital allocation for a trade.

        Args:
            event_name: Event identifier
            amount: Requested collateral amount

        Returns:
            Actual amount granted (may be less than requested)
        """
        async with self._lock:
            available_for_event = self.get_available_for_event(event_name)
            granted = min(amount, available_for_event)

            if granted > 0:
                self._available -= granted
                self._event_collateral[event_name] = (
                    self._event_collateral.get(event_name, 0.0) + granted
                )

            return granted

    async def release_collateral(self, event_name: str, amount: float, pnl: float = 0.0) -> None:
        """
        Release collateral back to the pool (when closing a position).

        Args:
            event_name: Event identifier
            amount: Collateral being released
            pnl: Realized P&L from the trade (positive = profit)
        """
        async with self._lock:
            current = self._event_collateral.get(event_name, 0.0)
            release = min(amount, current)

            self._event_collateral[event_name] = current - release
            self._available += release + pnl
            self._realized_pnl += pnl

    async def record_settlement(self, event_name: str, payout: float) -> None:
        """
        Record an event settlement.

        Args:
            event_name: Event identifier
            payout: Total payout received from settlement
        """
        async with self._lock:
            collateral = self._event_collateral.get(event_name, 0.0)
            pnl = payout - collateral

            # Release all collateral and add payout
            self._event_collateral[event_name] = 0.0
            self._available += payout
            self._realized_pnl += pnl

            logger.info(
                f"[{event_name}] Settlement: collateral=${collateral:.2f}, "
                f"payout=${payout:.2f}, pnl=${pnl:+.2f}"
            )

    async def sync_event_collateral(self, event_name: str, actual_collateral: float) -> None:
        """
        Sync the manager's view with actual collateral from a bot.

        Called after each tick to reconcile any differences.
        """
        async with self._lock:
            old_collateral = self._event_collateral.get(event_name, 0.0)
            delta = actual_collateral - old_collateral

            self._event_collateral[event_name] = actual_collateral
            self._available -= delta  # If collateral increased, available decreases

    def register_event(self, event_name: str) -> None:
        """Register a new event (starts with 0 collateral)."""
        if event_name not in self._event_collateral:
            self._event_collateral[event_name] = 0.0
            logger.info(f"[CapitalManager] Registered event: {event_name}")

    def unregister_event(self, event_name: str) -> None:
        """Unregister an event (should have 0 collateral)."""
        if event_name in self._event_collateral:
            remaining = self._event_collateral.pop(event_name)
            if remaining > 0:
                logger.warning(
                    f"[CapitalManager] Unregistered {event_name} with "
                    f"${remaining:.2f} collateral still allocated"
                )

    def get_summary(self) -> Dict:
        """Get a summary of the capital state."""
        return {
            "total_capital": self.total_capital,
            "available": self._available,
            "total_collateral": self.total_collateral,
            "realized_pnl": self._realized_pnl,
            "c_event_max": self.c_event_max,
            "events": {
                name: {
                    "collateral": coll,
                    "headroom": max(0, self.c_event_max - coll),
                }
                for name, coll in self._event_collateral.items()
            },
        }

    def display_status(self) -> None:
        """Display current capital status."""
        print()
        print("=" * 70)
        print("Capital Pool Status")
        print("=" * 70)
        print(f"Total Capital: ${self.total_capital:.2f}")
        print(f"Available:     ${self._available:.2f}")
        print(f"In Positions:  ${self.total_collateral:.2f}")
        print(f"Realized P&L:  ${self._realized_pnl:+.2f}")
        print(f"Per-Event Max: ${self.c_event_max:.2f}")
        print("-" * 70)

        if self._event_collateral:
            print(f"{'Event':<25} {'Collateral':>12} {'Headroom':>12} {'Usage':>10}")
            print("-" * 70)
            for name, coll in sorted(self._event_collateral.items()):
                headroom = max(0, self.c_event_max - coll)
                usage_pct = (coll / self.c_event_max * 100) if self.c_event_max > 0 else 0
                print(f"{name:<25} ${coll:>10.2f} ${headroom:>10.2f} {usage_pct:>9.1f}%")
        else:
            print("No events registered")

        print("=" * 70)
        print()


def get_all_7day_events() -> List[EventInfo]:
    """
    Fetch all 7-day Musk tweet events from Polymarket API.

    Returns:
        List of EventInfo for each 7-day event.
    """
    from src.algo.musk_tweet_count.musk_tweet_count import (
        GammaAPIClient,
        XTrackerClient,
    )
    import json
    import re
    from datetime import datetime, timezone

    logger.info("Fetching market data from Polymarket API...")

    gamma = GammaAPIClient()
    xtracker = XTrackerClient()

    # Fetch markets (only 7-day events)
    markets = gamma.get_musk_tweet_markets(xtracker_client=xtracker)

    if not markets:
        logger.warning("No active 7-day Musk tweet markets found via API")
        return []

    now = datetime.now(timezone.utc)

    # Group by event
    events_dict = {}
    for m in markets:
        event_title = m.get("eventTitle", "Unknown")
        if event_title not in events_dict:
            events_dict[event_title] = {
                "title": event_title,
                "start": m.get("countingStartDate"),
                "end": m.get("countingEndDate"),
                "xtracker_count": m.get("xtrackerCount"),
                "markets": []
            }
        events_dict[event_title]["markets"].append(m)

    # Convert to EventInfo objects
    events = []
    for event_title, event_data in events_dict.items():
        start = event_data["start"]
        end = event_data["end"]

        if not start or not end:
            continue

        # Determine status
        if now < start:
            status = "NOT_STARTED"
            hours_remaining = (end - start).total_seconds() / 3600  # Full duration
        elif now >= end:
            status = "ENDED"
            hours_remaining = 0
        else:
            status = "ACTIVE"
            hours_remaining = (end - now).total_seconds() / 3600

        # Extract short name (e.g., "Jan 27 - Feb 3" from full title)
        short_name = _extract_short_name(event_title)

        # Parse markets into bins
        sorted_markets = sorted(
            event_data["markets"],
            key=lambda x: x.get("groupItemThreshold", 0)
        )

        bins = []
        for m in sorted_markets:
            outcome = m.get("groupItemTitle", "")
            clob_ids = m.get("clobTokenIds", [])

            if isinstance(clob_ids, str):
                try:
                    clob_ids = json.loads(clob_ids)
                except:
                    clob_ids = []

            if not clob_ids:
                continue

            yes_token = clob_ids[0]

            # Parse upper bound
            upper_bound = float('inf')
            range_match = re.search(r"(\d+)\s*[-–]\s*(\d+)", outcome)
            if range_match:
                upper_bound = int(range_match.group(2))
            elif "<" in outcome or "under" in outcome.lower():
                under_match = re.search(r"[<]?\s*(\d+)", outcome)
                if under_match:
                    upper_bound = int(under_match.group(1)) - 1
            elif "+" in outcome or "or more" in outcome.lower():
                upper_bound = float('inf')

            bins.append({
                "upper_bound": upper_bound,
                "token_id": yes_token,
                "outcome": outcome,
            })

        bins.sort(key=lambda x: x["upper_bound"])

        settlement_date = end.date() if end else None

        events.append(EventInfo(
            title=event_title,
            short_name=short_name,
            start=start,
            end=end,
            xtracker_count=event_data["xtracker_count"],
            bins=bins,
            settlement_date=settlement_date,
            status=status,
            hours_remaining=hours_remaining,
        ))

    # Sort by start date
    events.sort(key=lambda e: e.start or datetime.min.replace(tzinfo=timezone.utc))

    return events


def _extract_short_name(title: str) -> str:
    """Extract short date range from event title."""
    import re
    # Match patterns like "January 27 - February 3" or "Jan 27 - Feb 3"
    match = re.search(
        r"(\w+)\s+(\d+)\s*[-–]\s*(\w+)\s+(\d+)",
        title
    )
    if match:
        m1, d1, m2, d2 = match.groups()
        # Abbreviate month names
        m1 = m1[:3]
        m2 = m2[:3]
        return f"{m1} {d1} - {m2} {d2}"
    return title


def display_events(events: List[EventInfo]) -> None:
    """Display all available events in a table."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    print()
    print("=" * 90)
    print("Available 7-day Musk Tweet Events")
    print("=" * 90)
    print(f"{'#':<3} {'Event':<25} {'Status':<20} {'Count':>6} {'Bins':>5} {'Hours Left':>10}")
    print("-" * 90)

    for i, event in enumerate(events, 1):
        status_str = event.status
        if event.status == "ACTIVE":
            status_str = f"ACTIVE ({event.hours_remaining:.1f}h left)"
        elif event.status == "NOT_STARTED":
            time_until = (event.start - now).total_seconds() / 3600 if event.start else 0
            status_str = f"STARTS in {time_until:.1f}h"

        count_str = str(event.xtracker_count) if event.xtracker_count is not None else "N/A"
        hours_str = f"{event.hours_remaining:.1f}" if event.hours_remaining > 0 else "-"

        print(f"{i:<3} {event.short_name:<25} {status_str:<20} {count_str:>6} {len(event.bins):>5} {hours_str:>10}")

    print("=" * 90)
    print()


def filter_events(
    events: List[EventInfo],
    event_filter: Optional[str] = None,
    active_only: bool = True,
) -> List[EventInfo]:
    """
    Filter events based on criteria.

    Args:
        events: List of all events
        event_filter: Optional filter string (matches short_name or title)
        active_only: If True, only return ACTIVE events

    Returns:
        Filtered list of events
    """
    filtered = events

    # Filter by status
    if active_only:
        filtered = [e for e in filtered if e.status == "ACTIVE"]

    # Filter by name
    if event_filter:
        filter_lower = event_filter.lower()
        filtered = [
            e for e in filtered
            if filter_lower in e.short_name.lower() or filter_lower in e.title.lower()
        ]

    return filtered


def get_musk_tweet_bins_from_api() -> tuple[List[Dict], Optional[date]]:
    """
    Fetch bin definitions from Polymarket API for current 7-day Musk tweet market.

    Returns:
        Tuple of (bins_list, settlement_date) where bins_list contains
        {"upper_bound": int, "token_id": str} for each bin.

    Note: This is kept for backward compatibility. Prefer get_all_7day_events().
    """
    events = get_all_7day_events()
    active_events = filter_events(events, active_only=True)

    if not active_events:
        logger.warning("No active 7-day events found")
        return [], None

    # Select the one with most time remaining
    best = max(active_events, key=lambda e: e.hours_remaining)

    logger.info(f"Selected event: {best.title}")
    logger.info(f"  Counting: {best.start} to {best.end}")
    logger.info(f"  Current XTracker count: {best.xtracker_count}")
    logger.info(f"Loaded {len(best.bins)} bins from API")

    return best.bins, best.settlement_date


def get_musk_tweet_bins() -> List[Dict]:
    """
    Get bin definitions for Musk tweet count market.

    First tries to fetch from API, falls back to environment variable.

    Returns list of {"upper_bound": int, "token_id": str} for each bin.
    """
    # Try API first
    try:
        bins, _ = get_musk_tweet_bins_from_api()
        if bins:
            return bins
    except Exception as e:
        logger.warning(f"Failed to fetch bins from API: {e}")

    # Fall back to environment variable
    logger.info("Falling back to MUSK_TWEET_TOKEN_IDS environment variable")

    # Define bin bounds (matching forecaster config)
    bin_bounds = [
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
        (550, 10000),  # 550+
    ]

    token_ids = os.environ.get("MUSK_TWEET_TOKEN_IDS", "").split(",")

    if len(token_ids) != len(bin_bounds) or not token_ids[0]:
        logger.warning(
            "MUSK_TWEET_TOKEN_IDS not set or incomplete. "
            "Set this environment variable with comma-separated YES token IDs for each bin."
        )
        # Return placeholder bins for dry-run testing
        return [
            {"upper_bound": upper, "token_id": f"placeholder_token_{i}"}
            for i, (lower, upper) in enumerate(bin_bounds)
        ]

    return [
        {"upper_bound": upper, "token_id": token_id}
        for (lower, upper), token_id in zip(bin_bounds, token_ids)
    ]


def get_settlement_date() -> Optional[date]:
    """
    Get settlement date from environment or compute default.

    The Musk tweet count market typically settles on a weekly basis.
    """
    settlement_str = os.environ.get("MUSK_TWEET_SETTLEMENT_DATE")
    if settlement_str:
        return date.fromisoformat(settlement_str)

    # Default: assume next Sunday at noon ET
    # This is a placeholder - should be fetched from Polymarket API
    return None


async def create_bot_for_event(
    event: EventInfo,
    clob_client: ClobClient,
    kelly_config: KellyConfig,
    forecaster_config: ForecasterConfig,
    bot_config: TradingBotConfig,
    capital_manager: CapitalManager,
    shared_data: Optional[SharedMarketData] = None,
) -> GASKellyTradingBot:
    """Create and setup a trading bot for a specific event."""
    from dataclasses import replace

    # Get available capital for this event from the pool
    available_for_event = capital_manager.get_available_for_event(event.short_name)

    # Create a copy of bot_config with event-specific settings
    event_bot_config = replace(
        bot_config,
        settlement_date=event.settlement_date,
        # Set initial capital to the event's max limit (actual usage controlled by manager)
        initial_capital=capital_manager.c_event_max,
    )

    # If shared data provided, use its EventStore
    event_store = None
    if shared_data:
        event_store = shared_data.event_store

    bot = GASKellyTradingBot(
        clob_client=clob_client,
        kelly_config=kelly_config,
        forecaster_config=forecaster_config,
        bot_config=event_bot_config,
        event_store=event_store,
    )

    await bot.setup(event.bins)

    # Register event with capital manager
    capital_manager.register_event(event.short_name)

    # Register event with shared data for probability tracking
    if shared_data:
        market_bins = [(b[0], b[1]) for b in bot._market_bins]
        # Compute market start date (7 days before settlement)
        market_start = bot.market_start_date if bot.market_start_date else (
            event.settlement_date - timedelta(days=7) if event.settlement_date else None
        )
        shared_data.register_event(
            event_name=event.short_name,
            market_bins=market_bins,
            settlement_date=event.settlement_date,
            market_start_date=market_start,
        )

    return bot


async def run_bot_tick(
    bot: GASKellyTradingBot,
    event_name: str,
    capital_manager: CapitalManager,
    event_lock: Optional[asyncio.Lock] = None,
    dry_run: bool = False,
) -> Optional[float]:
    """
    Run a single tick for a bot with capital management.

    The capital manager tracks collateral across events. Each bot operates
    with its own c_event_max limit. The manager enforces the global pool limit
    by tracking total collateral across all events.

    Thread-safe: uses event_lock to prevent concurrent ticks on the same event.

    Args:
        bot: The trading bot for this event
        event_name: Event identifier
        capital_manager: Shared capital manager
        event_lock: Optional lock for this event (if None, no locking)
        dry_run: If True, ignore capital limits (simulate unlimited funds)

    Returns:
        Current collateral after tick, or None if tick failed/skipped
    """
    # If lock provided, try to acquire without blocking
    if event_lock is not None:
        if event_lock.locked():
            logger.debug(f"[{event_name}] Tick already in progress, skipping")
            return None

        async with event_lock:
            return await _run_bot_tick_impl(bot, event_name, capital_manager, ignore_capital_limit=dry_run)
    else:
        return await _run_bot_tick_impl(bot, event_name, capital_manager, ignore_capital_limit=dry_run)


async def _run_bot_tick_impl(
    bot: GASKellyTradingBot,
    event_name: str,
    capital_manager: CapitalManager,
    ignore_capital_limit: bool = False,
) -> Optional[float]:
    """Internal tick implementation (called with lock held if applicable)."""
    # Check available capital in pool before tick
    pool_available = capital_manager.get_available_for_event(event_name)

    # Get current event collateral
    current_collateral = 0.0
    if bot.kelly_bot and bot.kelly_bot.portfolio:
        current_collateral = bot.kelly_bot.portfolio.total_collateral_used

    # Skip if pool is exhausted for this event (unless ignoring limits)
    if not ignore_capital_limit and pool_available <= 0 and current_collateral == 0:
        logger.warning(f"[{event_name}] Pool exhausted, skipping tick")
        return current_collateral

    # Set external capital limit on portfolio (so Kelly knows the constraint)
    if bot.kelly_bot and bot.kelly_bot.portfolio:
        if ignore_capital_limit:
            # No limit in dry-run mode (or when explicitly ignored)
            bot.kelly_bot.portfolio.set_external_capital_limit(None)
        else:
            # Pass pool constraint to portfolio
            bot.kelly_bot.portfolio.set_external_capital_limit(pool_available)

    logger.info(
        f"[{event_name}] Running tick (event collateral: ${current_collateral:.2f}, "
        f"pool available: ${pool_available:.2f}{' [ignored]' if ignore_capital_limit else ''})..."
    )

    result = await bot.run_tick()

    # Sync collateral with capital manager after tick
    if bot.kelly_bot and bot.kelly_bot.portfolio:
        new_collateral = bot.kelly_bot.portfolio.total_collateral_used
        await capital_manager.sync_event_collateral(event_name, new_collateral)

        if result:
            logger.info(
                f"[{event_name}] Tick completed: executed={result.num_executed}, "
                f"collateral=${new_collateral:.2f}"
            )
        return new_collateral

    return None


async def run_all_bots_tick(
    bots: List[tuple[str, GASKellyTradingBot]],
    capital_manager: CapitalManager,
    dry_run: bool = False,
) -> None:
    """Run a single tick for all bots concurrently."""
    tasks = [run_bot_tick(bot, name, capital_manager, dry_run=dry_run) for name, bot in bots]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Log any exceptions
    for (name, _), result in zip(bots, results):
        if isinstance(result, Exception):
            logger.error(f"[{name}] Tick failed with error: {result}", exc_info=result)


@dataclass
class TradingLoopContext:
    """Context for the continuous trading loop."""
    bots: List[tuple[str, GASKellyTradingBot]]
    capital_manager: CapitalManager
    clob_client: ClobClient
    kelly_config: KellyConfig
    forecaster_config: ForecasterConfig
    bot_config: TradingBotConfig
    event_filter: Optional[str]
    shared_data: Optional[SharedMarketData] = None  # Shared market data
    event_refresh_interval: int = 6  # Check for new events every N ticks
    probability_update_interval: int = 150  # Seconds between probability updates (2.5 min)
    orderbook_poll_interval: int = 15  # Seconds between orderbook polls (fallback)
    dry_run: bool = False  # If True, ignore capital limits

    # Locks for thread-safety
    _event_locks: Dict[str, asyncio.Lock] = None  # Per-event trading locks
    _probability_lock: asyncio.Lock = None  # Global probability update lock
    _fast_loop_running: bool = False  # Prevent concurrent fast loop iterations

    def __post_init__(self):
        """Initialize locks after dataclass creation."""
        if self._event_locks is None:
            object.__setattr__(self, '_event_locks', {})
        if self._probability_lock is None:
            object.__setattr__(self, '_probability_lock', asyncio.Lock())

    def get_event_lock(self, event_name: str) -> asyncio.Lock:
        """Get or create a lock for the specified event."""
        if event_name not in self._event_locks:
            self._event_locks[event_name] = asyncio.Lock()
        return self._event_locks[event_name]


async def check_for_new_events(ctx: TradingLoopContext) -> List[EventInfo]:
    """
    Check for new events that have become active.

    Returns list of new events not currently being traded.
    """
    current_event_names = {name for name, _ in ctx.bots}

    # Fetch fresh event list
    all_events = get_all_7day_events()

    # Filter to active events matching our filter criteria
    if ctx.event_filter:
        active_events = filter_events(all_events, event_filter=ctx.event_filter, active_only=True)
    else:
        active_events = filter_events(all_events, active_only=True)

    # Find events we're not already trading
    new_events = [e for e in active_events if e.short_name not in current_event_names]

    return new_events


async def add_bot_for_event(ctx: TradingLoopContext, event: EventInfo) -> None:
    """Add a new bot for an event that just became active."""
    logger.info(f"[{event.short_name}] New event detected, creating bot...")

    bot = await create_bot_for_event(
        event=event,
        clob_client=ctx.clob_client,
        kelly_config=ctx.kelly_config,
        forecaster_config=ctx.forecaster_config,
        bot_config=ctx.bot_config,
        capital_manager=ctx.capital_manager,
        shared_data=ctx.shared_data,
    )

    ctx.bots.append((event.short_name, bot))
    logger.info(f"[{event.short_name}] Bot added, now trading {len(ctx.bots)} events")


async def handle_completed_event(
    ctx: TradingLoopContext,
    event_name: str,
    bot: GASKellyTradingBot,
) -> None:
    """Handle an event that has completed (past settlement time)."""
    logger.info(f"[{event_name}] Event completed, shutting down bot...")

    # Get final collateral (represents positions that will settle)
    final_collateral = 0.0
    if bot.kelly_bot and bot.kelly_bot.portfolio:
        final_collateral = bot.kelly_bot.portfolio.total_collateral_used

    # Shutdown the bot
    await bot.shutdown()

    # For now, assume we need to wait for actual settlement
    # In production, you'd track positions and record actual payouts
    # For simplicity, we just release the collateral (assuming break-even)
    if final_collateral > 0:
        logger.info(
            f"[{event_name}] Releasing ${final_collateral:.2f} collateral "
            "(actual settlement payout TBD)"
        )
        await ctx.capital_manager.release_collateral(event_name, final_collateral)

    ctx.capital_manager.unregister_event(event_name)


async def run_probability_update_loop(
    ctx: TradingLoopContext,
    stop_event: asyncio.Event,
) -> None:
    """
    Slow loop: Update posts and probabilities every 2-2.5 minutes.

    This loop:
    1. Fetches new posts from XTracker (ONCE for all events)
    2. Updates cumulative counts for each event
    3. Recalculates probability tables

    Thread-safe: uses probability lock to prevent concurrent updates.
    """
    if not ctx.shared_data:
        logger.warning("No shared data configured, probability update loop disabled")
        return

    update_interval = ctx.probability_update_interval
    logger.info(f"Starting probability update loop (interval: {update_interval}s)")

    while not stop_event.is_set():
        try:
            # Acquire probability lock to prevent concurrent updates
            async with ctx._probability_lock:
                # 1. Fetch new posts (shared across all events)
                await ctx.shared_data.update_posts()

                # 2. Update cumulative counts for each event
                for event_name, bot in ctx.bots:
                    if bot.kelly_bot and bot.kelly_bot.portfolio:
                        # Get cumulative count for this event's date range
                        count = bot._get_market_cumulative_count()
                        ctx.shared_data.set_cumulative_count(event_name, count)

                # 3. Recalculate probabilities
                ctx.shared_data.update_probabilities()

                logger.info(
                    f"[ProbabilityLoop] Updated {len(ctx.bots)} events, "
                    f"next update in {update_interval}s"
                )

        except Exception as e:
            logger.error(f"Error in probability update loop: {e}", exc_info=True)

        # Wait for next update
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=update_interval)
            break
        except asyncio.TimeoutError:
            pass

    logger.info("Probability update loop stopped")


def log_ws_status(ctx: TradingLoopContext) -> None:
    """Log WebSocket callback status for all bots."""
    from datetime import datetime, timezone

    # Check if any bot has WS enabled
    any_ws_enabled = any(
        bot.kelly_bot and bot.kelly_bot.orderbook_manager and
        bot.kelly_bot.orderbook_manager.config.enabled
        for _, bot in ctx.bots
    )

    if not any_ws_enabled:
        # Don't spam logs if WS is disabled (e.g., in dry-run mode)
        return

    now = datetime.now(timezone.utc)
    ws_statuses = []

    for event_name, bot in ctx.bots:
        if bot._last_ws_callback_time:
            # Convert to UTC properly (astimezone, not replace)
            last_callback_utc = bot._last_ws_callback_time.astimezone(timezone.utc)
            age = (now - last_callback_utc).total_seconds()
            ws_statuses.append(f"{event_name}: {age:.0f}s ago (#{bot._ws_callback_count})")
        else:
            ws_statuses.append(f"{event_name}: no callbacks")

    if ws_statuses:
        logger.info(f"[WS Status] {' | '.join(ws_statuses)}")


async def run_trading_tick_fast(
    ctx: TradingLoopContext,
) -> None:
    """
    Fast trading tick: Use cached probabilities to evaluate trades.

    This is called frequently (on orderbook updates or periodic polling).
    Does NOT recalculate probabilities - uses cached values from slow loop.

    Thread-safe: uses per-event locks to prevent concurrent ticks on the same event.
    """
    # Prevent concurrent fast loop iterations
    if ctx._fast_loop_running:
        logger.debug("Fast loop already running, skipping")
        return

    ctx._fast_loop_running = True
    try:
        for event_name, bot in ctx.bots:
            try:
                # Get cached probabilities from shared data
                if ctx.shared_data:
                    cached_probs = ctx.shared_data.get_probabilities(event_name)
                    if cached_probs:
                        # Update bot's probability cache
                        bot._cached_probabilities = cached_probs

                # Get event lock and run tick
                event_lock = ctx.get_event_lock(event_name)
                await run_bot_tick(bot, event_name, ctx.capital_manager, event_lock, dry_run=ctx.dry_run)

            except Exception as e:
                logger.error(f"[{event_name}] Fast tick error: {e}")
    finally:
        ctx._fast_loop_running = False


async def run_all_bots_continuous(
    ctx: TradingLoopContext,
    tick_interval: int,
    stop_event: asyncio.Event,
) -> None:
    """
    Run trading with two-loop architecture:
    - Slow loop: Update posts and probabilities (every 2-2.5 min)
    - Fast loop: Check orderbooks and execute trades (every 15-30s)
    """
    logger.info(f"Starting two-loop trading architecture...")
    logger.info(f"  Probability updates: every {ctx.probability_update_interval}s")
    logger.info(f"  Orderbook checks: every {ctx.orderbook_poll_interval}s")

    # Start probability update loop as background task
    probability_task = asyncio.create_task(
        run_probability_update_loop(ctx, stop_event)
    )

    fast_tick_count = 0
    status_interval = max(1, 180 // ctx.orderbook_poll_interval)  # Show status every ~3 min

    # WS status logging interval (every ~10 seconds)
    ws_status_interval = max(1, 10 // ctx.orderbook_poll_interval)

    try:
        while not stop_event.is_set():
            try:
                fast_tick_count += 1

                # Display capital status periodically
                if fast_tick_count % status_interval == 1:
                    ctx.capital_manager.display_status()

                # Log WebSocket status periodically
                if fast_tick_count % ws_status_interval == 0:
                    log_ws_status(ctx)

                # Run fast trading tick for all bots
                await run_trading_tick_fast(ctx)

                # Check for completed events
                active_bots = []
                for name, bot in ctx.bots:
                    hours_elapsed, hours_remaining = bot._get_timing()
                    if hours_remaining <= 0:
                        await handle_completed_event(ctx, name, bot)
                    else:
                        active_bots.append((name, bot))

                ctx.bots = active_bots

                # Check for new events periodically (every ~30 fast ticks)
                if fast_tick_count % 30 == 0:
                    logger.info("Checking for new events...")
                    new_events = await check_for_new_events(ctx)
                    for event in new_events:
                        await add_bot_for_event(ctx, event)

                if not ctx.bots:
                    logger.info("All events completed, stopping trading loop")
                    break

                # Wait for next fast tick (orderbook poll interval)
                try:
                    await asyncio.wait_for(
                        stop_event.wait(),
                        timeout=ctx.orderbook_poll_interval
                    )
                    break  # Stop requested
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue

            except Exception as e:
                logger.error(f"Error in fast trading loop: {e}", exc_info=True)
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=30)
                    break
                except asyncio.TimeoutError:
                    pass

    finally:
        # Stop probability loop
        stop_event.set()
        await probability_task

    # Final status
    ctx.capital_manager.display_status()
    logger.info("Trading loop stopped")


async def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    logger.info("Starting GAS-Kelly trading bot...")
    logger.info(f"Dry run: {args.dry_run}")

    # Fetch all available events
    all_events = get_all_7day_events()

    if not all_events:
        logger.error("No 7-day events found!")
        return

    # Always display event list at startup
    display_events(all_events)

    # If --list-events, exit after displaying
    if args.list_events:
        return

    # Filter events
    if args.event:
        # User specified a specific event
        events_to_trade = filter_events(all_events, event_filter=args.event, active_only=False)
        if not events_to_trade:
            logger.error(f"No events matching '{args.event}' found!")
            logger.info("Use --list-events to see available events.")
            return
        # If matching event is not active, warn but allow
        for e in events_to_trade:
            if e.status != "ACTIVE":
                logger.warning(f"Event '{e.short_name}' is {e.status}, trading may not be effective.")
    else:
        # Trade all active events
        events_to_trade = filter_events(all_events, active_only=True)
        if not events_to_trade:
            logger.warning("No active events to trade!")
            logger.info("Use --event to specify a specific event, or wait for events to start.")
            return

    logger.info(f"Trading {len(events_to_trade)} event(s): {[e.short_name for e in events_to_trade]}")

    # Load config
    config_path = args.config
    if os.path.exists(config_path):
        yaml_config = load_config(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        yaml_config = {}
        logger.warning(f"Config file not found: {config_path}, using defaults")

    # Create CLOB client
    try:
        clob_client = create_clob_client()
        logger.info("Created authenticated CLOB client")
    except Exception as e:
        if args.dry_run:
            logger.warning(f"Could not create CLOB client ({e}), using mock for dry-run")
            clob_client = None
        else:
            raise

    # Create configs
    forecaster_config = ForecasterConfig.from_dict(
        yaml_config.get("forecaster", {}).copy()
    )

    kelly_config_dict = yaml_config.get("kelly", {}).copy()

    # Apply command-line overrides for per-event max
    if args.max_position_usd:
        kelly_config_dict.setdefault("collateral", {})["c_event_max"] = args.max_position_usd
        kelly_config_dict["collateral"]["c_bin_max"] = args.max_position_usd / 4

    kelly_config = KellyConfig.from_dict(kelly_config_dict)

    # Create shared capital manager
    c_event_max = kelly_config.collateral.c_event_max
    capital_manager = CapitalManager(
        total_capital=args.initial_capital,
        c_event_max=c_event_max,
    )

    logger.info(f"Capital pool: ${args.initial_capital:.2f} total, ${c_event_max:.2f} max per event")

    # Get timing configuration from config file
    trading_config = yaml_config.get("trading", {})
    slow_loop_interval = trading_config.get("slow_loop_interval_seconds", 150)
    fast_loop_interval = trading_config.get("fast_loop_interval_seconds", 15)
    forecast_cache_timeout = trading_config.get("forecast_cache_seconds", slow_loop_interval + 15)

    logger.info(
        f"Timing: slow_loop={slow_loop_interval}s, fast_loop={fast_loop_interval}s, "
        f"forecast_cache={forecast_cache_timeout}s"
    )

    # Bot config (settlement_date and initial_capital will be set per event)
    bot_config = TradingBotConfig(
        tick_interval_seconds=args.tick_interval,
        dry_run=args.dry_run,
        settlement_date=None,
        initial_capital=c_event_max,  # Each bot gets up to the event max
        training_days=args.training_days,
        use_gas=True,
        disable_websocket=args.no_ws,
        forecast_cache_timeout=forecast_cache_timeout,
    )

    # For dry-run without CLOB client, create a mock
    if clob_client is None:
        from unittest.mock import MagicMock
        logger.warning("Using mock CLOB client - no real orderbook data available")
        clob_client = MagicMock()
        mock_orderbook = MagicMock()
        mock_orderbook.bids = []
        mock_orderbook.asks = []
        clob_client.get_order_book.return_value = mock_orderbook

    # Create shared market data (optimization: shared EventStore across all events)
    logger.info("Initializing shared market data...")
    shared_data = SharedMarketData(
        forecaster_config=forecaster_config,
        training_days=args.training_days,
    )
    await shared_data.initialize(n_days=args.training_days)

    # Create bots for each event (using shared EventStore)
    bots: List[tuple[str, GASKellyTradingBot]] = []

    for event in events_to_trade:
        logger.info(f"Setting up bot for: {event.short_name}")
        logger.info(f"  XTracker count: {event.xtracker_count}, Bins: {len(event.bins)}")

        bot = await create_bot_for_event(
            event=event,
            clob_client=clob_client,
            kelly_config=kelly_config,
            forecaster_config=forecaster_config,
            bot_config=bot_config,
            capital_manager=capital_manager,
            shared_data=shared_data,
        )
        bots.append((event.short_name, bot))

    if not bots:
        logger.error("Failed to create any bots!")
        return

    # Create trading loop context
    ctx = TradingLoopContext(
        bots=bots,
        capital_manager=capital_manager,
        clob_client=clob_client,
        kelly_config=kelly_config,
        forecaster_config=forecaster_config,
        bot_config=bot_config,
        event_filter=args.event,
        shared_data=shared_data,
        probability_update_interval=slow_loop_interval,
        orderbook_poll_interval=fast_loop_interval,
        dry_run=args.dry_run,
    )

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal")
        stop_event.set()
        for name, bot in ctx.bots:
            bot.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        # Display initial capital status
        capital_manager.display_status()

        # Print initial state for each bot
        for name, bot in bots:
            state = bot.get_state_summary()
            logger.info(
                f"[{name}] Initial state: hours_elapsed={state['hours_elapsed']:.1f}, "
                f"hours_remaining={state['hours_remaining']:.1f}"
            )

        if args.single_tick:
            # Run single tick for all bots
            logger.info("Running single tick for all events...")
            await run_all_bots_tick(bots, capital_manager, dry_run=args.dry_run)
            # Display final capital status
            capital_manager.display_status()
        else:
            # Run continuous loop
            await run_all_bots_continuous(ctx, args.tick_interval, stop_event)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Shutdown all bots
        for name, bot in ctx.bots:
            logger.info(f"[{name}] Shutting down...")
            await bot.shutdown()

        # Final capital summary
        logger.info("=== Final Capital Summary ===")
        summary = capital_manager.get_summary()
        logger.info(f"Total Capital: ${summary['total_capital']:.2f}")
        logger.info(f"Available: ${summary['available']:.2f}")
        logger.info(f"In Positions: ${summary['total_collateral']:.2f}")
        logger.info(f"Realized P&L: ${summary['realized_pnl']:+.2f}")

        logger.info("All bots shutdown complete")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GAS-Kelly trading bot for Musk tweet count market"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run in dry-run mode (no real orders). Default: True",
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live mode (real orders). Overrides --dry-run.",
    )

    parser.add_argument(
        "--event",
        type=str,
        help="Trade specific event only (e.g., 'Jan 27 - Feb 3'). If not specified, trades all active 7-day events.",
    )

    parser.add_argument(
        "--list-events",
        action="store_true",
        help="List all available 7-day events and exit (no trading).",
    )

    parser.add_argument(
        "--max-position-usd",
        type=float,
        help="Maximum position size in USD per event",
    )

    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1000.0,
        help="Initial capital in USD (shared across all events). Default: 1000",
    )

    parser.add_argument(
        "--tick-interval",
        type=int,
        default=300,
        help="Seconds between optimization ticks. Default: 300 (5 min)",
    )

    parser.add_argument(
        "--training-days",
        type=int,
        default=45,
        help="Days of history for model fitting. Default: 45",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/musk_tweet_count.yaml",
        help="Path to config file",
    )

    parser.add_argument(
        "--single-tick",
        action="store_true",
        help="Run a single tick and exit (for testing)",
    )

    parser.add_argument(
        "--no-ws",
        action="store_true",
        help="Disable WebSocket and use REST polling only",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Handle --live flag
    if args.live:
        args.dry_run = False

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    asyncio.run(main(args))
