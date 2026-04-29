"""
Polymarket Musk Tweet Count Trading Bot

A comprehensive trading bot for Polymarket's Elon Musk tweet count markets that:
1. Fetches active markets and orderbooks via Gamma API and CLOB API
2. Retrieves xtracker metrics (current tweet count)
3. Calculates fair values (placeholder for user's logic)
4. Executes trades based on configurable thresholds
5. Manages positions and runs periodically
"""

import json
import os
import re
import time
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple
from enum import Enum

import requests
from ruamel.yaml import YAML
from dotenv import load_dotenv

from py_clob_client_v2.client import ClobClient
from py_clob_client_v2.clob_types import OrderArgs, OrderType
from py_clob_client_v2.constants import POLYGON

from src.utils.app_utils import get_logger
from src.utils.crypto_utils import load_private_key

# Load environment variables
load_dotenv()

logger = get_logger(name='musk_tweet_bot', enable_file_handler=True)


# =============================================================================
# Constants
# =============================================================================

GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"
DATA_API_URL = "https://data-api.polymarket.com"
XTRACKER_API_URL = "https://xtracker.polymarket.com/api"

# Chain ID for Polygon mainnet
CHAIN_ID = POLYGON

# Tag ID for Musk tweet count markets
MUSK_TWEET_TAG_ID = 972

# Default xtracker user handle for Musk
MUSK_XTRACKER_HANDLE = "elonmusk"

# Default config file path
DEFAULT_CONFIG_PATH = "./config/musk_tweet_count.yaml"


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Dict containing configuration
    """
    yaml = YAML()
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f)
        logger.info(f"Loaded config from {config_path}")
        return config or {}
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TradingConfig:
    """All configurable parameters for the trading bot."""

    # Edge thresholds
    min_edge_to_buy: float = 0.05        # 5% min edge to enter position
    min_edge_to_sell: float = 0.03       # 3% edge to exit position

    # Position limits
    max_position_size_usd: float = 100.0  # Max USD per position
    max_total_exposure_usd: float = 500.0 # Max total exposure

    # Safety margins
    error_margin: float = 0.02           # 2% safety margin

    # Timing
    check_interval_seconds: int = 300    # 5 min between checks

    # Order type
    use_limit_orders: bool = True        # Limit vs market orders
    limit_order_offset: float = 0.01     # How far from mid to place limits

    # Risk management
    stop_loss_pct: float = 0.50          # Cut at 50% loss
    take_profit_pct: float = 0.30        # Take at 30% profit

    # Liquidity requirements
    min_liquidity_usd: float = 100.0     # Min orderbook depth
    max_spread_pct: float = 0.10         # Max bid-ask spread (10%)

    # Dry run mode
    dry_run: bool = False                # If True, don't execute trades

    @classmethod
    def from_config(cls, config: Dict) -> 'TradingConfig':
        """
        Create TradingConfig from config dict (loaded from YAML).

        Args:
            config: Full config dict with 'trading' section

        Returns:
            TradingConfig instance
        """
        trading_config = config.get('trading', {})
        return cls(
            min_edge_to_buy=trading_config.get('min_edge_to_buy', 0.05),
            min_edge_to_sell=trading_config.get('min_edge_to_sell', 0.03),
            max_position_size_usd=trading_config.get('max_position_size_usd', 100.0),
            max_total_exposure_usd=trading_config.get('max_total_exposure_usd', 500.0),
            error_margin=trading_config.get('error_margin', 0.02),
            check_interval_seconds=trading_config.get('check_interval_seconds', 300),
            use_limit_orders=trading_config.get('use_limit_orders', True),
            limit_order_offset=trading_config.get('limit_order_offset', 0.01),
            stop_loss_pct=trading_config.get('stop_loss_pct', 0.50),
            take_profit_pct=trading_config.get('take_profit_pct', 0.30),
            min_liquidity_usd=trading_config.get('min_liquidity_usd', 100.0),
            max_spread_pct=trading_config.get('max_spread_pct', 0.10),
            dry_run=trading_config.get('dry_run', False),
        )


@dataclass
class OrderBookLevel:
    """Single level in the orderbook."""
    price: float
    size: float


@dataclass
class OrderBook:
    """Order book for a token."""
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_pct(self) -> Optional[float]:
        if self.mid_price and self.spread:
            return self.spread / self.mid_price
        return None

    def bid_depth_usd(self, levels: int = 5) -> float:
        """Total USD value in top N bid levels."""
        return sum(b.price * b.size for b in self.bids[:levels])

    def ask_depth_usd(self, levels: int = 5) -> float:
        """Total USD value in top N ask levels."""
        return sum(a.price * a.size for a in self.asks[:levels])


@dataclass
class MarketOption:
    """Single outcome option with orderbook data."""
    token_id: str
    outcome: str                         # e.g., "0-74", "75-99", "100+"
    outcome_index: int                   # 0, 1, 2, etc.
    orderbook: Optional[OrderBook] = None

    @property
    def best_bid(self) -> Optional[float]:
        return self.orderbook.best_bid if self.orderbook else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.orderbook.best_ask if self.orderbook else None

    @property
    def mid_price(self) -> Optional[float]:
        return self.orderbook.mid_price if self.orderbook else None


@dataclass
class Market:
    """Full market with options and xtracker metadata."""
    condition_id: str
    question: str
    description: str
    end_date: datetime
    options: List[MarketOption]
    start_date: Optional[datetime] = None  # Counting period start
    xtracker_count: Optional[int] = None   # Current tweet count from xtracker
    xtracker_updated_at: Optional[datetime] = None
    market_slug: Optional[str] = None
    active: bool = True

    @property
    def time_remaining_hours(self) -> float:
        """Hours remaining until market resolution (end of counting period)."""
        now = datetime.now(timezone.utc)
        delta = self.end_date - now
        return max(0, delta.total_seconds() / 3600)

    @property
    def time_elapsed_hours(self) -> float:
        """Hours elapsed since counting period started."""
        if not self.start_date:
            return 0.0
        now = datetime.now(timezone.utc)
        delta = now - self.start_date
        return max(0, delta.total_seconds() / 3600)

    @property
    def total_period_hours(self) -> float:
        """Total hours in the counting period."""
        if not self.start_date:
            return 24.0  # Default assumption
        delta = self.end_date - self.start_date
        return max(1, delta.total_seconds() / 3600)

    def get_option_by_outcome(self, outcome: str) -> Optional[MarketOption]:
        """Find option by outcome string."""
        for opt in self.options:
            if opt.outcome == outcome:
                return opt
        return None


@dataclass
class Position:
    """User's position in a market option."""
    token_id: str
    outcome: str
    size: float                          # Number of shares
    avg_price: float                     # Average entry price
    current_value: float                 # Current market value
    realized_pnl: float = 0.0

    @property
    def cost_basis(self) -> float:
        """Total cost of position."""
        return self.size * self.avg_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return self.current_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        if self.cost_basis > 0:
            return self.unrealized_pnl / self.cost_basis
        return 0.0


# =============================================================================
# API Clients
# =============================================================================

class GammaAPIClient:
    """Client for Gamma API - Market discovery and metadata."""

    def __init__(self, base_url: str = GAMMA_API_URL):
        self.base_url = base_url
        self.session = requests.Session()

    def get_events(
        self,
        tag_id: Optional[int] = None,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """
        Fetch events from Gamma API.

        Args:
            tag_id: Filter by tag ID (e.g., 972 for Musk tweets)
            active: Filter for active events
            closed: Filter for closed events
            limit: Number of results per page
            offset: Pagination offset

        Returns:
            List of event dictionaries (each contains nested markets)
        """
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
        }
        if tag_id is not None:
            params["tag_id"] = tag_id

        try:
            response = self.session.get(
                f"{self.base_url}/events",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch events: {e}")
            return []

    def get_musk_tweet_events(self) -> List[Dict]:
        """
        Fetch Musk tweet count events using tag_id=972.

        Returns:
            List of event dictionaries containing Musk tweet markets
        """
        events = self.get_events(
            tag_id=MUSK_TWEET_TAG_ID,
            active=True,
            closed=False
        )
        logger.info(f"Found {len(events)} Musk tweet events")
        return events

    @staticmethod
    def parse_counting_dates_from_title(
        title: str,
        require_7_days: bool = True
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Parse tweet counting dates from event title.

        Only matches regular weekly events like:
        - "Elon Musk # tweets January 13 - January 20, 2026?" -> (Jan 13, Jan 20)

        Skips irregular events like:
        - "Elon Musk # tweets in January 2026?" (monthly aggregate)
        - Events that don't span exactly 7 days

        Args:
            title: Event title containing date range
            require_7_days: If True, only accept events that span exactly 7 days

        Returns:
            Tuple of (start_date, end_date) as datetime objects, or (None, None) if not a regular event
        """
        # Pattern: "Month Day - Month Day, Year" or "Month Day-Month Day, Year"
        # This pattern will NOT match "in January 2026" (monthly aggregates)
        pattern = r"(\w+)\s+(\d{1,2})\s*[-–]\s*(\w+)\s+(\d{1,2}),?\s*(\d{4})"
        match = re.search(pattern, title)

        if not match:
            # This is likely a non-regular event (e.g., monthly aggregate)
            logger.info(f"Skipping non-regular event (no date range): {title}")
            return None, None

        try:
            start_month_str = match.group(1)
            start_day = int(match.group(2))
            end_month_str = match.group(3)
            end_day = int(match.group(4))
            year = int(match.group(5))

            # Parse month names
            month_map = {
                "january": 1, "february": 2, "march": 3, "april": 4,
                "may": 5, "june": 6, "july": 7, "august": 8,
                "september": 9, "october": 10, "november": 11, "december": 12,
                "jan": 1, "feb": 2, "mar": 3, "apr": 4,
                "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
            }

            start_month = month_map.get(start_month_str.lower())
            end_month = month_map.get(end_month_str.lower())

            if not start_month or not end_month:
                logger.info(f"Skipping event with unknown month: {title}")
                return None, None

            # Create datetime objects (counting is 12:00 PM ET to 11:59:59 AM ET)
            # Use America/New_York so DST (EDT=UTC-4 vs EST=UTC-5) is handled automatically.
            et = ZoneInfo("America/New_York")
            start_date = datetime(year, start_month, start_day, 12, 0, 0, tzinfo=et).astimezone(timezone.utc)
            end_date = datetime(year, end_month, end_day, 11, 59, 59, tzinfo=et).astimezone(timezone.utc)

            # Check if duration is exactly 7 days
            if require_7_days:
                duration_days = (end_date - start_date + timedelta(seconds=1)).days
                if duration_days != 7:
                    logger.info(f"Skipping non-7-day event ({duration_days} days): {title}")
                    return None, None

            logger.debug(f"Parsed counting dates: {start_date.date()} to {end_date.date()}")
            return start_date, end_date

        except (ValueError, AttributeError) as e:
            logger.info(f"Skipping event with invalid dates '{title}': {e}")
            return None, None

    def get_musk_tweet_markets(self, xtracker_client: Optional['XTrackerClient'] = None) -> List[Dict]:
        """
        Get all individual markets from Musk tweet events.

        Fetches xtracker data at event level (all markets under same event share tracking).

        Args:
            xtracker_client: Optional XTrackerClient for fetching tweet counts

        Returns:
            List of market dictionaries from all Musk tweet events
        """
        events = self.get_musk_tweet_events()
        markets = []

        for event in events:
            event_markets = event.get("markets", [])
            event_title = event.get("title", "")
            event_description = event.get("description", "")

            # Parse actual counting dates from event title (not tradable dates)
            # This will return None for non-regular events (monthly, non-7-day, etc.)
            counting_start, counting_end = self.parse_counting_dates_from_title(event_title)

            # Skip non-regular events (couldn't parse counting dates)
            if not counting_start or not counting_end:
                logger.debug(f"Skipping {len(event_markets)} markets from non-regular event")
                continue

            # Fetch xtracker data once per event (all markets share same tracking)
            xtracker_count = None
            xtracker_updated = None

            if xtracker_client:
                now = datetime.now(timezone.utc)

                # Boundary condition: if counting hasn't started yet, count is 0
                if now < counting_start:
                    logger.info(f"Counting not started yet for '{event_title}' (starts {counting_start.date()})")
                    xtracker_count = 0
                    xtracker_updated = now
                else:
                    # Counting has started, fetch from xtracker
                    xtracker_count, xtracker_updated = xtracker_client.get_current_count(
                        start_date=counting_start,
                        end_date=counting_end,
                    )

            # Add metadata to each market under this event
            for market in event_markets:
                if not market.get("eventTitle"):
                    market["eventTitle"] = event_title
                if not market.get("eventDescription"):
                    market["eventDescription"] = event_description

                # Store counting dates (from title, not tradable dates)
                market["countingStartDate"] = counting_start
                market["countingEndDate"] = counting_end

                # Store xtracker data (shared across all markets in event)
                market["xtrackerCount"] = xtracker_count
                market["xtrackerUpdatedAt"] = xtracker_updated

                markets.append(market)

        logger.info(f"Found {len(markets)} individual Musk tweet markets")
        return markets

    def get_market_by_id(self, condition_id: str) -> Optional[Dict]:
        """Fetch a single market by condition ID."""
        try:
            response = self.session.get(
                f"{self.base_url}/markets/{condition_id}",
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch market {condition_id}: {e}")
            return None

class XTrackerClient:
    """Client for XTracker API - Tweet count tracking data."""

    def __init__(self, base_url: str = XTRACKER_API_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self._trackings_cache: Dict[str, List[Dict]] = {}  # user_handle -> trackings

    def get_user_trackings(self, handle: str = MUSK_XTRACKER_HANDLE) -> List[Dict]:
        """
        Fetch all trackings for a user.

        Args:
            handle: Twitter handle (e.g., 'elonmusk')

        Returns:
            List of tracking dictionaries with id, startDate, endDate, etc.
        """
        if handle in self._trackings_cache:
            return self._trackings_cache[handle]

        try:
            response = self.session.get(
                f"{self.base_url}/users/{handle}",
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            # Extract trackings from response
            if data.get("success") and "data" in data:
                trackings = data["data"].get("trackings", [])
            else:
                trackings = data.get("trackings", [])

            self._trackings_cache[handle] = trackings
            logger.info(f"Fetched {len(trackings)} trackings for @{handle}")
            return trackings

        except requests.RequestException as e:
            logger.error(f"Failed to fetch trackings for @{handle}: {e}")
            return []

    def get_tracking_stats(self, tracking_id: str) -> Optional[Dict]:
        """
        Fetch detailed stats for a specific tracking.

        Args:
            tracking_id: UUID of the tracking

        Returns:
            Tracking data with stats including current tweet count
        """
        try:
            response = self.session.get(
                f"{self.base_url}/trackings/{tracking_id}",
                params={"includeStats": "true"},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            # Handle wrapped response
            if data.get("success") and "data" in data:
                return data["data"]
            return data

        except requests.RequestException as e:
            logger.error(f"Failed to fetch tracking stats for {tracking_id}: {e}")
            return None

    def find_tracking_for_period(
        self,
        start_date: datetime,
        end_date: datetime,
        handle: str = MUSK_XTRACKER_HANDLE,
    ) -> Optional[Dict]:
        """
        Find a tracking that matches the given time period by date (ignoring time).

        Tracking periods are typically from start_date 12:00 PM to end_date 11:59 AM.
        We match by calendar date only.

        Args:
            start_date: Period start date
            end_date: Period end date
            handle: Twitter handle

        Returns:
            Matching tracking dict or None
        """
        trackings = self.get_user_trackings(handle)

        # Extract just the date parts for comparison
        target_start_date = start_date.date()
        target_end_date = end_date.date()

        for tracking in trackings:
            try:
                t_start_str = tracking.get("startDate", "")
                t_end_str = tracking.get("endDate", "")

                if not t_start_str or not t_end_str:
                    continue

                t_start = datetime.fromisoformat(t_start_str.replace("Z", "+00:00"))
                t_end = datetime.fromisoformat(t_end_str.replace("Z", "+00:00"))

                # Compare by date only (year, month, day)
                if t_start.date() == target_start_date and t_end.date() == target_end_date:
                    logger.debug(f"Found matching tracking: {tracking.get('id')} for {target_start_date} - {target_end_date}")
                    return tracking

            except (ValueError, TypeError) as e:
                logger.debug(f"Error parsing tracking dates: {e}")
                continue

        # NOTE: Previously had a fallback that matched "containing" tracking periods
        # (e.g., Feb 13-20 would match a query for Feb 16-18). This was WRONG because
        # it returns the count for the full larger period, not the sub-period.
        # If no exact match, return None and let the system use the computed count.
        logger.warning(f"No tracking found for period {target_start_date} - {target_end_date}")
        return None

    def get_last_sync(self, handle: str = MUSK_XTRACKER_HANDLE) -> Optional[datetime]:
        """Fetch the lastSync timestamp for a user (lightweight poll).

        This does NOT use the trackings cache - it's a fast, uncached poll
        to detect when XTracker has synced new data from X/Twitter.
        """
        try:
            response = self.session.get(
                f"{self.base_url}/users/{handle}",
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("success") and "data" in data:
                sync_str = data["data"].get("lastSync")
                if sync_str:
                    return datetime.fromisoformat(sync_str.replace("Z", "+00:00"))
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch lastSync for @{handle}: {e}")
        return None

    def get_current_count(
        self,
        start_date: datetime,
        end_date: datetime,
        handle: str = MUSK_XTRACKER_HANDLE,
    ) -> Tuple[Optional[int], Optional[datetime]]:
        """
        Get current tweet count for a time period.

        Args:
            start_date: Period start date
            end_date: Period end date
            handle: Twitter handle

        Returns:
            Tuple of (count, updated_at) or (None, None) if not found
        """
        # Find matching tracking
        tracking = self.find_tracking_for_period(start_date, end_date, handle)
        if not tracking:
            logger.warning(f"No tracking found for period {start_date} - {end_date}")
            return None, None

        tracking_id = tracking.get("id")
        if not tracking_id:
            return None, None

        # Fetch detailed stats
        stats = self.get_tracking_stats(tracking_id)
        if not stats:
            return None, None

        # Extract count from stats
        # The count is in stats.total or directly in the response
        count = None
        if "stats" in stats:
            count = stats["stats"].get("total")
        if count is None:
            count = stats.get("total")
        if count is None:
            # Try to sum up daily counts
            daily = stats.get("daily", [])
            if daily:
                count = sum(d.get("count", 0) for d in daily)

        if count is not None:
            updated_str = stats.get("updatedAt", stats.get("updated_at"))
            updated_at = None
            if updated_str:
                try:
                    updated_at = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    updated_at = datetime.now(timezone.utc)
            else:
                updated_at = datetime.now(timezone.utc)

            logger.info(f"Current tweet count: {count} (tracking: {tracking_id})")
            return int(count), updated_at

        return None, None


class DataAPIClient:
    """Client for Data API - Portfolio and position data."""

    def __init__(self, base_url: str = DATA_API_URL):
        self.base_url = base_url
        self.session = requests.Session()

    def get_positions(self, address: str) -> List[Dict]:
        """
        Fetch user positions from Data API.

        Args:
            address: User's wallet address

        Returns:
            List of position dictionaries
        """
        try:
            response = self.session.get(
                f"{self.base_url}/positions",
                params={"user": address, "sizeThreshold": 0},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else data.get("positions", [])
        except requests.RequestException as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    def get_trades(
        self,
        address: str,
        limit: int = 100,
        market: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch trade history for user.

        Args:
            address: User's wallet address
            limit: Max trades to return
            market: Optional market condition ID filter

        Returns:
            List of trade dictionaries
        """
        params = {
            "user": address,
            "limit": limit,
        }
        if market:
            params["market"] = market

        try:
            response = self.session.get(
                f"{self.base_url}/trades",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else data.get("trades", [])
        except requests.RequestException as e:
            logger.error(f"Failed to fetch trades: {e}")
            return []


# =============================================================================
# Fair Value Calculator
# =============================================================================

class FairValueCalculator:
    """
    Calculate fair prices for market outcomes.

    This is a PLACEHOLDER implementation using a simple Gaussian model.
    Users should implement their own logic based on historical patterns,
    time-of-day effects, news sentiment, etc.
    """

    def __init__(
        self,
        historical_mean: float = 350.0,  # Average tweets per 7-day period
        historical_std: float = 100.0,   # Standard deviation
        time_decay_factor: float = 0.8,  # How uncertainty decreases over time
    ):
        """
        Initialize calculator with historical parameters.

        Args:
            historical_mean: Historical average tweet count
            historical_std: Historical standard deviation
            time_decay_factor: Factor for time-based uncertainty decay
        """
        self.historical_mean = historical_mean
        self.historical_std = historical_std
        self.time_decay_factor = time_decay_factor

    @classmethod
    def from_config(cls, config: Dict) -> 'FairValueCalculator':
        """
        Create FairValueCalculator from config dict (loaded from YAML).

        Args:
            config: Full config dict with 'fair_value' section

        Returns:
            FairValueCalculator instance
        """
        fv_config = config.get('fair_value', {})
        return cls(
            historical_mean=fv_config.get('historical_mean', 350.0),
            historical_std=fv_config.get('historical_std', 100.0),
            time_decay_factor=fv_config.get('time_decay_factor', 0.8),
        )

    def calculate_fair_prices(
        self,
        market: Market,
        current_time: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Calculate fair probabilities for each outcome.

        This placeholder uses a simple Gaussian model:
        - Projects final count based on current count and time remaining
        - Applies normal distribution to estimate probability of each range

        Args:
            market: Market with options and xtracker data
            current_time: Current time (defaults to now)

        Returns:
            Dict mapping outcome string to fair probability (0-1)
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        fair_prices = {}

        # If we have xtracker count, use it for projection
        if market.xtracker_count is not None:
            current_count = market.xtracker_count
            hours_remaining = market.time_remaining_hours
            hours_elapsed = market.time_elapsed_hours
            total_hours = market.total_period_hours

            # Ensure we have valid time data
            if hours_elapsed <= 0:
                hours_elapsed = max(1, total_hours - hours_remaining)

            # Simple linear projection with uncertainty
            # Estimate remaining tweets based on current rate
            current_rate = current_count / hours_elapsed if hours_elapsed > 0 else self.historical_mean / total_hours

            # Project final count
            projected_additional = current_rate * hours_remaining
            projected_final = current_count + projected_additional

            # Uncertainty decreases as we get closer to end
            remaining_fraction = hours_remaining / total_hours if total_hours > 0 else 0.5
            uncertainty = self.historical_std * remaining_fraction * self.time_decay_factor

            logger.debug(
                f"Projection: current={current_count}, elapsed={hours_elapsed:.1f}h, "
                f"remaining={hours_remaining:.1f}h, rate={current_rate:.2f}/hr, "
                f"projected_final={projected_final:.1f}, uncertainty={uncertainty:.1f}"
            )
        else:
            # No current count, use historical prior
            projected_final = self.historical_mean
            uncertainty = self.historical_std
            logger.warning("No xtracker count available, using historical prior")

        # Calculate probability for each outcome range
        for option in market.options:
            prob = self._calculate_range_probability(
                option.outcome,
                projected_final,
                uncertainty
            )
            fair_prices[option.outcome] = prob

        # Normalize probabilities to sum to 1
        total = sum(fair_prices.values())
        if total > 0:
            fair_prices = {k: v / total for k, v in fair_prices.items()}

        return fair_prices

    def _calculate_range_probability(
        self,
        outcome: str,
        mean: float,
        std: float
    ) -> float:
        """
        Calculate probability that final count falls in outcome range.

        Parses outcome strings like "0-74", "75-99", "100+", "Under 50", etc.

        Args:
            outcome: Outcome string describing a range
            mean: Projected mean final count
            std: Standard deviation of projection

        Returns:
            Probability (0-1)
        """
        lower, upper = self._parse_outcome_range(outcome)

        # Use normal CDF to calculate probability
        if std <= 0:
            # No uncertainty - deterministic
            if lower <= mean <= upper:
                return 1.0
            return 0.0

        # P(lower <= X <= upper) = CDF(upper) - CDF(lower)
        from math import erf, sqrt

        def normal_cdf(x, mu, sigma):
            return 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))

        p_upper = normal_cdf(upper, mean, std) if upper < float('inf') else 1.0
        p_lower = normal_cdf(lower, mean, std) if lower > float('-inf') else 0.0

        return max(0, p_upper - p_lower)

    def _parse_outcome_range(self, outcome: str) -> Tuple[float, float]:
        """
        Parse outcome string into (lower, upper) bounds.

        Handles formats like:
        - "0-19" -> (0, 19)
        - "20-39" -> (20, 39)
        - "540+" or "540 or more" -> (540, inf)
        - "Under 50" or "<50" -> (-inf, 49)

        Args:
            outcome: Outcome string

        Returns:
            Tuple of (lower, upper) bounds
        """
        outcome_str = outcome.strip()

        # Range format: "X-Y" (e.g., "0-19", "20-39", "540-559")
        # Use search instead of match to be more flexible
        range_match = re.search(r"(\d+)\s*[-–]\s*(\d+)", outcome_str)
        if range_match:
            return float(range_match.group(1)), float(range_match.group(2))

        # "X+" format (e.g., "540+")
        plus_match = re.search(r"(\d+)\s*\+", outcome_str)
        if plus_match:
            return float(plus_match.group(1)), float('inf')

        # "X or more" format
        or_more_match = re.search(r"(\d+)\s+or\s+more", outcome_str.lower())
        if or_more_match:
            return float(or_more_match.group(1)), float('inf')

        # ">X" or ">=X" format
        gt_match = re.search(r">=?\s*(\d+)", outcome_str)
        if gt_match:
            return float(gt_match.group(1)), float('inf')

        # "Under X" or "<X" or "less than X"
        under_match = re.search(r"under\s+(\d+)|<\s*(\d+)|less\s+than\s+(\d+)", outcome_str.lower())
        if under_match:
            value = under_match.group(1) or under_match.group(2) or under_match.group(3)
            return float('-inf'), float(value) - 1

        # Single number
        single_match = re.match(r"^(\d+)$", outcome_str)
        if single_match:
            val = float(single_match.group(1))
            return val, val

        # Default: return wide range for unknown formats
        logger.warning(f"Could not parse outcome range: {outcome}")
        return 0, float('inf')


# =============================================================================
# Trading Engine
# =============================================================================

class TradingEngine:
    """Evaluates trading opportunities based on fair values and positions."""

    def __init__(self, config: TradingConfig):
        self.config = config

    def evaluate_buy_opportunity(
        self,
        option: MarketOption,
        fair_price: float,
        current_position: Optional[Position],
        total_exposure: float,
    ) -> Optional[Dict]:
        """
        Evaluate if we should buy this option.

        Args:
            option: Market option to evaluate
            fair_price: Calculated fair probability
            current_position: Existing position if any
            total_exposure: Current total portfolio exposure

        Returns:
            Order dict with {side, price, size, reason} or None
        """
        # Check liquidity
        if not self._check_liquidity(option):
            return None

        best_ask = option.best_ask
        if best_ask is None:
            return None

        # Calculate edge
        edge = fair_price - best_ask

        if edge < self.config.min_edge_to_buy:
            return None

        # Check position limits
        current_size_usd = 0
        if current_position:
            current_size_usd = current_position.current_value

        remaining_position_capacity = self.config.max_position_size_usd - current_size_usd
        remaining_total_capacity = self.config.max_total_exposure_usd - total_exposure

        max_buy_usd = min(remaining_position_capacity, remaining_total_capacity)

        if max_buy_usd <= 0:
            logger.debug(f"Position limits reached for {option.outcome}")
            return None

        # Calculate order size (number of shares)
        # Size in shares = USD amount / price
        order_size = max_buy_usd / best_ask

        # Determine price
        if self.config.use_limit_orders:
            # Place limit slightly below ask
            order_price = best_ask - self.config.limit_order_offset
            order_price = max(0.01, min(0.99, order_price))  # Clamp to valid range
        else:
            order_price = best_ask

        return {
            "side": Side.BUY,
            "price": round(order_price, 2),
            "size": round(order_size, 2),
            "reason": f"Edge: {edge:.1%} (fair={fair_price:.2f}, ask={best_ask:.2f})",
            "edge": edge,
        }

    def evaluate_sell_opportunity(
        self,
        option: MarketOption,
        fair_price: float,
        position: Position,
    ) -> Optional[Dict]:
        """
        Evaluate if we should sell (close/reduce) position.

        Checks for:
        1. Overpriced (fair value below market)
        2. Stop-loss trigger
        3. Take-profit trigger

        Args:
            option: Market option to evaluate
            fair_price: Calculated fair probability
            position: Current position

        Returns:
            Order dict with {side, price, size, reason} or None
        """
        if position.size <= 0:
            return None

        best_bid = option.best_bid
        if best_bid is None:
            return None

        # Check stop-loss
        if position.unrealized_pnl_pct <= -self.config.stop_loss_pct:
            return {
                "side": Side.SELL,
                "price": round(best_bid, 2),
                "size": position.size,
                "reason": f"STOP-LOSS: {position.unrealized_pnl_pct:.1%} loss",
                "edge": 0,
            }

        # Check take-profit
        if position.unrealized_pnl_pct >= self.config.take_profit_pct:
            return {
                "side": Side.SELL,
                "price": round(best_bid, 2),
                "size": position.size,
                "reason": f"TAKE-PROFIT: {position.unrealized_pnl_pct:.1%} gain",
                "edge": 0,
            }

        # Check if overpriced (edge to sell)
        edge = best_bid - fair_price

        if edge >= self.config.min_edge_to_sell:
            # Sell some or all based on edge size
            sell_fraction = min(1.0, edge / 0.10)  # Scale up to full exit at 10% edge
            sell_size = position.size * sell_fraction

            if self.config.use_limit_orders:
                order_price = best_bid + self.config.limit_order_offset
                order_price = max(0.01, min(0.99, order_price))
            else:
                order_price = best_bid

            return {
                "side": Side.SELL,
                "price": round(order_price, 2),
                "size": round(sell_size, 2),
                "reason": f"Overpriced: {edge:.1%} edge (fair={fair_price:.2f}, bid={best_bid:.2f})",
                "edge": edge,
            }

        return None

    def _check_liquidity(self, option: MarketOption) -> bool:
        """Check if option has sufficient liquidity."""
        if option.orderbook is None:
            return False

        ob = option.orderbook

        # Check spread
        if ob.spread_pct is not None and ob.spread_pct > self.config.max_spread_pct:
            logger.debug(f"Spread too wide for {option.outcome}: {ob.spread_pct:.1%}")
            return False

        # Check depth
        ask_depth = ob.ask_depth_usd()
        if ask_depth < self.config.min_liquidity_usd:
            logger.debug(f"Insufficient ask liquidity for {option.outcome}: ${ask_depth:.2f}")
            return False

        return True


# =============================================================================
# Main Trading Bot
# =============================================================================

class PolymarketTradingBot:
    """Main trading bot that orchestrates all components."""

    def __init__(
        self,
        private_key: str,
        config: Optional[TradingConfig] = None,
        fair_value_calculator: Optional[FairValueCalculator] = None,
    ):
        """
        Initialize the trading bot.

        Args:
            private_key: Ethereum private key for signing
            config: Trading configuration
            fair_value_calculator: Custom fair value calculator
        """
        self.config = config or TradingConfig()
        self.fair_value_calculator = fair_value_calculator or FairValueCalculator()
        self.trading_engine = TradingEngine(self.config)

        # Initialize API clients
        self.gamma_client = GammaAPIClient()
        self.data_client = DataAPIClient()
        self.xtracker_client = XTrackerClient()

        # Initialize CLOB client with L2 authentication
        self.clob_client = ClobClient(
            host=CLOB_API_URL,
            key=private_key,
            chain_id=CHAIN_ID,
        )

        # Derive API credentials
        self._setup_api_credentials()

        # Cache for positions
        self._positions: Dict[str, Position] = {}
        self._total_exposure: float = 0.0

        logger.info("PolymarketTradingBot initialized")

    def _setup_api_credentials(self):
        """Setup or derive API credentials for L2 authentication."""
        try:
            # Try to derive existing API key
            api_creds = self.clob_client.derive_api_key()
            self.clob_client.set_api_creds(api_creds)
            logger.info("Derived existing API credentials")
        except Exception:
            try:
                # Create new API key if derivation fails
                api_creds = self.clob_client.create_api_key()
                self.clob_client.set_api_creds(api_creds)
                logger.info("Created new API credentials")
            except Exception as e:
                logger.error(f"Failed to setup API credentials: {e}")
                raise

    @property
    def address(self) -> str:
        """Get wallet address from CLOB client."""
        return self.clob_client.get_address()

    def fetch_positions(self) -> Dict[str, Position]:
        """
        Fetch current positions from Data API.

        Returns:
            Dict mapping token_id to Position
        """
        positions_data = self.data_client.get_positions(self.address)
        positions = {}
        total_exposure = 0.0

        for pos_data in positions_data:
            token_id = pos_data.get("asset", pos_data.get("token_id", ""))
            size = float(pos_data.get("size", pos_data.get("amount", 0)))

            if size <= 0:
                continue

            avg_price = float(pos_data.get("avgPrice", pos_data.get("average_price", 0)))
            current_price = float(pos_data.get("currentPrice", pos_data.get("price", avg_price)))

            position = Position(
                token_id=token_id,
                outcome=pos_data.get("outcome", ""),
                size=size,
                avg_price=avg_price,
                current_value=size * current_price,
                realized_pnl=float(pos_data.get("realizedPnl", 0)),
            )
            positions[token_id] = position
            total_exposure += position.current_value

        self._positions = positions
        self._total_exposure = total_exposure

        logger.info(f"Fetched {len(positions)} positions, total exposure: ${total_exposure:.2f}")
        return positions

    def fetch_open_orders(self) -> List[Dict]:
        """
        Fetch open orders from CLOB.

        Returns:
            List of open order dictionaries
        """
        try:
            orders = self.clob_client.get_orders()
            open_orders = [o for o in orders if o.get("status") == "LIVE"]
            logger.info(f"Fetched {len(open_orders)} open orders")
            return open_orders
        except Exception as e:
            logger.error(f"Failed to fetch open orders: {e}")
            return []

    def fetch_orderbook(self, token_id: str) -> Optional[OrderBook]:
        """
        Fetch orderbook for a token from CLOB.

        Args:
            token_id: Token ID to fetch orderbook for

        Returns:
            OrderBook or None if fetch fails or doesn't exist
        """
        try:
            ob_data = self.clob_client.get_order_book(token_id)

            bids = [
                OrderBookLevel(price=float(b.price), size=float(b.size))
                for b in ob_data.bids
            ]
            asks = [
                OrderBookLevel(price=float(a.price), size=float(a.size))
                for a in ob_data.asks
            ]

            # Sort bids descending, asks ascending
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)

            return OrderBook(bids=bids, asks=asks)
        except Exception as e:
            error_str = str(e)
            # 404 means no orderbook exists yet - this is normal for some markets
            if "404" in error_str or "No orderbook exists" in error_str:
                logger.debug(f"No orderbook for token {token_id[:16]}...")
            else:
                logger.warning(f"Failed to fetch orderbook for {token_id[:16]}...: {e}")
            return None

    def parse_market(self, market_data: Dict) -> Optional[Market]:
        """
        Parse raw market data into Market object with orderbooks.

        Args:
            market_data: Raw market data from Gamma API (enriched with xtracker data)

        Returns:
            Market object or None if parsing fails
        """
        try:
            condition_id = market_data.get("conditionId", market_data.get("condition_id", ""))

            # Use counting dates parsed from event title (not tradable dates)
            # These are already datetime objects set by get_musk_tweet_markets
            start_date = market_data.get("countingStartDate")
            end_date = market_data.get("countingEndDate")

            # Fall back to parsing from market data if not pre-set
            if not end_date:
                end_date_str = market_data.get("endDate", market_data.get("end_date"))
                if end_date_str:
                    if isinstance(end_date_str, str):
                        end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                    else:
                        end_date = datetime.fromtimestamp(end_date_str, tz=timezone.utc)
                else:
                    end_date = datetime.now(timezone.utc)

            # Get the outcome range from groupItemTitle (e.g., "0-19", "20-39")
            # The 'outcomes' field is just ["Yes", "No"], not the actual range
            outcome_range = market_data.get("groupItemTitle", "")
            if not outcome_range:
                # Try to extract from question as fallback
                question = market_data.get("question", "")
                range_match = re.search(r"(\d+[-–]\d+|\d+\+)", question)
                if range_match:
                    outcome_range = range_match.group(1)
                else:
                    outcome_range = "Unknown"

            # Get token IDs from clobTokenIds array
            # Index 0 is typically YES, Index 1 is typically NO
            # Note: clobTokenIds may be a JSON string or a list
            clob_token_ids = market_data.get("clobTokenIds", [])
            if isinstance(clob_token_ids, str):
                try:
                    clob_token_ids = json.loads(clob_token_ids)
                except (json.JSONDecodeError, TypeError):
                    clob_token_ids = []

            # Create a single MarketOption for the YES outcome of this range
            # (We trade the YES token for the range we think will win)
            options = []
            if clob_token_ids and len(clob_token_ids) > 0:
                yes_token_id = clob_token_ids[0]

                # Validate token ID is a proper string (not empty, not a list)
                if yes_token_id and isinstance(yes_token_id, str) and not yes_token_id.startswith("["):
                    option = MarketOption(
                        token_id=yes_token_id,
                        outcome=outcome_range,
                        outcome_index=market_data.get("groupItemThreshold", 0),
                    )

                    # Fetch orderbook for the YES token
                    option.orderbook = self.fetch_orderbook(yes_token_id)
                    options.append(option)
                else:
                    logger.warning(f"Invalid token ID for {outcome_range}: {yes_token_id}")

            # Use pre-fetched xtracker data (fetched at event level by get_musk_tweet_markets)
            xtracker_count = market_data.get("xtrackerCount")
            xtracker_updated = market_data.get("xtrackerUpdatedAt")

            market = Market(
                condition_id=condition_id,
                question=market_data.get("question", ""),
                description=market_data.get("description", market_data.get("eventDescription", "")),
                end_date=end_date,
                options=options,
                start_date=start_date,
                xtracker_count=xtracker_count,
                xtracker_updated_at=xtracker_updated,
                market_slug=market_data.get("slug", ""),
                active=market_data.get("active", True),
            )

            logger.info(
                f"Parsed market: {outcome_range} "
                f"(token={clob_token_ids[0][:8] if clob_token_ids else 'N/A'}..., xtracker={xtracker_count})"
            )
            return market

        except Exception as e:
            logger.error(f"Failed to parse market: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def place_order(
        self,
        token_id: str,
        side: Side,
        price: float,
        size: float,
    ) -> Optional[Dict]:
        """
        Place a limit order via CLOB client.

        Args:
            token_id: Token to trade
            side: BUY or SELL
            price: Limit price (0.01 to 0.99)
            size: Number of shares

        Returns:
            Order response or None if failed
        """
        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would place {side.value} order: {size} @ {price} for {token_id[:16]}...")
            return {
                "dry_run": True,
                "side": side.value,
                "price": price,
                "size": size,
                "token_id": token_id,
            }

        try:
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=side.value,
            )

            # Create and sign order
            signed_order = self.clob_client.create_order(order_args)

            # Post order
            response = self.clob_client.post_order(signed_order, OrderType.GTC)

            logger.info(f"Placed {side.value} order: {size} @ {price} for {token_id[:16]}...")
            logger.debug(f"Order response: {response}")

            # Add token_id to response for tracking
            if response:
                response["token_id"] = token_id
            return response

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    def place_market_order(
        self,
        token_id: str,
        side: Side,
        amount: float,
    ) -> Optional[Dict]:
        """
        Place a market order via CLOB client.

        Args:
            token_id: Token to trade
            side: BUY or SELL
            amount: USD amount for BUY, shares for SELL

        Returns:
            Order response or None if failed
        """
        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would place market {side.value}: {amount} for {token_id}")
            return {"dry_run": True, "side": side.value, "amount": amount}

        try:
            response = self.clob_client.create_market_order(
                token_id=token_id,
                amount=amount,
                side=side.value,
            )

            logger.info(f"Placed market {side.value}: {amount} for {token_id}")
            return response

        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            return None

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        if self.config.dry_run:
            logger.info("[DRY RUN] Would cancel all orders")
            return True

        try:
            self.clob_client.cancel_all()
            logger.info("Cancelled all open orders")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            return False

    def process_market(self, market: Market) -> List[Dict]:
        """
        Process a single market - evaluate opportunities and place orders.

        Args:
            market: Market to process

        Returns:
            List of placed orders
        """
        placed_orders = []

        # Calculate fair prices
        fair_prices = self.fair_value_calculator.calculate_fair_prices(market)

        logger.info(f"Fair prices for {market.question}")
        for outcome, price in fair_prices.items():
            logger.info(f"  {outcome}: {price: .2%}")

        # Evaluate each option
        for option in market.options:
            fair_price = fair_prices.get(option.outcome, 0.5)
            position = self._positions.get(option.token_id)

            # Check for sell opportunity first (if we have position)
            if position and position.size > 0:
                sell_signal = self.trading_engine.evaluate_sell_opportunity(
                    option, fair_price, position
                )
                if sell_signal:
                    logger.info(
                        f"SELL signal for {option.outcome}: {sell_signal['reason']}"
                    )
                    if self.config.use_limit_orders:
                        order = self.place_order(
                            option.token_id,
                            sell_signal["side"],
                            sell_signal["price"],
                            sell_signal["size"],
                        )
                    else:
                        order = self.place_market_order(
                            option.token_id,
                            sell_signal["side"],
                            sell_signal["size"],
                        )
                    if order:
                        order["outcome"] = option.outcome
                        placed_orders.append(order)
                    continue  # Don't buy if we just sold

            # Check for buy opportunity
            buy_signal = self.trading_engine.evaluate_buy_opportunity(
                option, fair_price, position, self._total_exposure
            )
            if buy_signal:
                logger.info(
                    f"BUY signal for {option.outcome}: {buy_signal['reason']}"
                )
                if self.config.use_limit_orders:
                    order = self.place_order(
                        option.token_id,
                        buy_signal["side"],
                        buy_signal["price"],
                        buy_signal["size"],
                    )
                else:
                    order = self.place_market_order(
                        option.token_id,
                        buy_signal["side"],
                        buy_signal["size"] * buy_signal["price"],  # Convert to USD
                    )
                if order:
                    order["outcome"] = option.outcome
                    placed_orders.append(order)

        return placed_orders

    def check_balance(self) -> float:
        """Check USDC balance."""
        try:
            from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType

            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            balance_info = self.clob_client.get_balance_allowance(params)
            # USDC has 6 decimals, so divide by 1e6
            balance = float(balance_info.get("balance", 0)) / 1e6
            logger.info(f"USDC balance: ${balance:.2f}")
            return balance
        except Exception as e:
            logger.error(f"Failed to check balance: {e}")
            return 0.0

    def cancel_stale_orders(
        self,
        open_orders: List[Dict],
        active_token_ids: set,
        new_order_token_ids: set,
    ) -> List[Dict]:
        """
        Cancel orders that are no longer relevant.

        Cancels orders for tokens that:
        - Are not in active markets anymore
        - Had new orders placed (to replace old unfilled orders)

        Args:
            open_orders: List of current open orders
            active_token_ids: Set of token IDs from current active markets
            new_order_token_ids: Set of token IDs we just placed orders for

        Returns:
            List of cancelled orders
        """
        cancelled = []

        for order in open_orders:
            token_id = order.get("asset_id", order.get("token_id", ""))
            order_id = order.get("id", order.get("order_id", ""))

            should_cancel = False
            reason = ""

            # Cancel if token is no longer in active markets
            if token_id and token_id not in active_token_ids:
                should_cancel = True
                reason = "token no longer active"

            # Cancel if we placed a new order for this token (replace old order)
            elif token_id in new_order_token_ids:
                should_cancel = True
                reason = "replaced with new order"

            if should_cancel and order_id:
                try:
                    if self.config.dry_run:
                        logger.info(f"[DRY RUN] Would cancel order {order_id[:8]}... ({reason})")
                        cancelled.append({"order_id": order_id, "reason": reason, "dry_run": True})
                    else:
                        self.clob_client.cancel(order_id)
                        logger.info(f"Cancelled order {order_id[:8]}... ({reason})")
                        cancelled.append({"order_id": order_id, "reason": reason})
                except Exception as e:
                    logger.warning(f"Failed to cancel order {order_id[:8]}...: {e}")

        return cancelled

    def run_once(self) -> Dict:
        """
        Run a single cycle of the bot.

        Returns:
            Summary dict with detailed information about the cycle.
        """
        summary = {
            "balance": 0.0,
            "events_found": 0,
            "events": [],  # List of {title, markets_count, xtracker_count}
            "markets_found": 0,
            "markets_processed": 0,
            "orders_placed": [],
            "orders_cancelled": [],
            "positions": [],
            "open_orders": [],
            "errors": [],
        }

        try:
            # Check balance
            balance = self.check_balance()
            summary["balance"] = balance
            if balance < 0:
                logger.warning("Low balance, skipping trading")
                summary["errors"].append("Low balance")
                return summary

            # Fetch current open orders (before processing)
            open_orders = self.fetch_open_orders()
            summary["open_orders"] = [
                {
                    "order_id": o.get("id", "")[:8] + "...",
                    "token_id": o.get("asset_id", o.get("token_id", ""))[:16] + "...",
                    "side": o.get("side", ""),
                    "price": o.get("price", ""),
                    "size": o.get("original_size", o.get("size", "")),
                }
                for o in open_orders
            ]

            # Fetch current positions
            self.fetch_positions()
            summary["positions"] = [
                {
                    "outcome": p.outcome,
                    "size": p.size,
                    "avg_price": p.avg_price,
                    "current_value": p.current_value,
                    "pnl_pct": f"{p.unrealized_pnl_pct:.1%}",
                }
                for p in self._positions.values()
            ]

            # Fetch Musk tweet markets using tag_id=972
            musk_markets = self.gamma_client.get_musk_tweet_markets(
                xtracker_client=self.xtracker_client
            )
            summary["markets_found"] = len(musk_markets)

            if not musk_markets:
                logger.info("No active Musk tweet markets found")
                self._print_summary(summary)
                return summary

            # Group markets by event for logging
            events_map: Dict[str, Dict] = {}
            for market_data in musk_markets:
                event_title = market_data.get("eventTitle", "Unknown Event")
                if event_title not in events_map:
                    events_map[event_title] = {
                        "title": event_title,
                        "markets_count": 0,
                        "xtracker_count": market_data.get("xtrackerCount"),
                        "counting_start": market_data.get("countingStartDate"),
                        "counting_end": market_data.get("countingEndDate"),
                    }
                events_map[event_title]["markets_count"] += 1

            summary["events_found"] = len(events_map)
            summary["events"] = list(events_map.values())

            # Track active token IDs and new order token IDs
            active_token_ids: set = set()
            new_order_token_ids: set = set()
            all_orders = []

            # Process each market
            for market_data in musk_markets:
                try:
                    market = self.parse_market(market_data)
                    if market and market.active:
                        # Track active token IDs
                        for option in market.options:
                            if option.token_id:
                                active_token_ids.add(option.token_id)

                        # Process market and place orders
                        orders = self.process_market(market)
                        for order in orders:
                            token_id = order.get("token_id", "")
                            if token_id:
                                new_order_token_ids.add(token_id)
                            all_orders.append(order)

                        summary["markets_processed"] += 1
                except Exception as e:
                    error_msg = f"Error processing market: {e}"
                    logger.error(error_msg)
                    summary["errors"].append(error_msg)

            # Record placed orders
            summary["orders_placed"] = [
                {
                    "side": o.get("side", ""),
                    "price": o.get("price", ""),
                    "size": o.get("size", ""),
                    "outcome": o.get("outcome", ""),
                    "dry_run": o.get("dry_run", False),
                }
                for o in all_orders
            ]

            # Cancel stale orders
            cancelled = self.cancel_stale_orders(
                open_orders=open_orders,
                active_token_ids=active_token_ids,
                new_order_token_ids=new_order_token_ids,
            )
            summary["orders_cancelled"] = cancelled

        except Exception as e:
            error_msg = f"Fatal error in run_once: {e}"
            logger.error(error_msg)
            import traceback
            logger.debug(traceback.format_exc())
            summary["errors"].append(error_msg)

        # Print detailed summary
        self._print_summary(summary)

        return summary

    def _print_summary(self, summary: Dict):
        """Print a detailed summary of the trading cycle."""
        print("\n" + "=" * 60)
        print("TRADING CYCLE SUMMARY")
        print("=" * 60)

        # Balance
        print(f"\nBalance: ${summary['balance']:.2f} USDC")

        # Events and markets
        print(f"\nEvents Found: {summary['events_found']}")
        for event in summary.get("events", []):
            start_date = event.get("counting_start")
            end_date = event.get("counting_end")
            date_range = ""
            if start_date and end_date:
                date_range = f" ({start_date.strftime('%b %d')} - {end_date.strftime('%b %d')})"
            xtracker = event.get("xtracker_count")
            xtracker_str = f", tweets: {xtracker}" if xtracker is not None else ""
            print(f"  - {event['title'][:50]}{date_range}")
            print(f"    Markets: {event['markets_count']}{xtracker_str}")

        print(f"\nTotal Markets: {summary['markets_found']} found, {summary['markets_processed']} processed")

        # Current positions
        positions = summary.get("positions", [])
        if positions:
            print(f"\nCurrent Positions ({len(positions)}):")
            for pos in positions:
                print(f"  - {pos['outcome']}: {pos['size']:.2f} @ ${pos['avg_price']:.2f} "
                      f"(value: ${pos['current_value']:.2f}, PnL: {pos['pnl_pct']})")
        else:
            print("\nCurrent Positions: None")

        # Open orders (before this cycle)
        open_orders = summary.get("open_orders", [])
        if open_orders:
            print(f"\nOpen Orders (at start): {len(open_orders)}")
            for order in open_orders[:5]:  # Show first 5
                print(f"  - {order['side']} {order['size']} @ {order['price']}")
            if len(open_orders) > 5:
                print(f"  ... and {len(open_orders) - 5} more")
        else:
            print("\nOpen Orders (at start): None")

        # Orders placed
        orders_placed = summary.get("orders_placed", [])
        if orders_placed:
            dry_run_label = " [DRY RUN]" if any(o.get("dry_run") for o in orders_placed) else ""
            print(f"\nOrders Placed{dry_run_label}: {len(orders_placed)}")
            for order in orders_placed:
                print(f"  - {order['side']} {order.get('size', 'N/A')} @ {order.get('price', 'N/A')} "
                      f"for {order.get('outcome', 'N/A')}")
        else:
            print("\nOrders Placed: None")

        # Orders cancelled
        orders_cancelled = summary.get("orders_cancelled", [])
        if orders_cancelled:
            dry_run_label = " [DRY RUN]" if any(o.get("dry_run") for o in orders_cancelled) else ""
            print(f"\nOrders Cancelled{dry_run_label}: {len(orders_cancelled)}")
            for order in orders_cancelled:
                print(f"  - {order['order_id']} ({order['reason']})")
        else:
            print("\nOrders Cancelled: None")

        # Errors
        errors = summary.get("errors", [])
        if errors:
            print(f"\nErrors: {len(errors)}")
            for error in errors:
                print(f"  - {error}")

        print("\n" + "=" * 60 + "\n")

    def run(self):
        """
        Run the bot continuously.

        Loops forever, running cycles at the configured interval.
        """
        logger.info(
            f"Starting continuous run with {self.config.check_interval_seconds}s interval"
        )

        cycle_count = 0
        while True:
            cycle_count += 1
            logger.info(f"=== Cycle {cycle_count} ===")

            try:
                summary = self.run_once()
                logger.info(
                    f"Cycle complete: {summary['markets_processed']} markets, "
                    f"{summary['orders_placed']} orders"
                )
                if summary["errors"]:
                    logger.warning(f"Errors: {summary['errors']}")

            except KeyboardInterrupt:
                logger.info("Interrupted by user, stopping...")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")

            # Sleep until next cycle
            logger.info(f"Sleeping for {self.config.check_interval_seconds}s...")
            time.sleep(self.config.check_interval_seconds)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Polymarket Musk Tweet Count Trading Bot"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to config file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run single cycle and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't execute trades, just log what would happen",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Seconds between cycles (overrides config)",
    )
    parser.add_argument(
        "--max-position",
        type=float,
        default=None,
        help="Max USD per position (overrides config)",
    )
    parser.add_argument(
        "--max-exposure",
        type=float,
        default=None,
        help="Max total exposure USD (overrides config)",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=None,
        help="Minimum edge to buy (overrides config)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    # Load private key (supports plain or encrypted)
    try:
        private_key = load_private_key()
    except ValueError as e:
        logger.error(str(e))
        print(f"Error: {e}")
        print("\nTo use an encrypted key:")
        print("  1. Encrypt: python -m src.utils.crypto_utils encrypt")
        print("  2. Set: export ENCRYPTED_POLYMARKET_PRIVATE_KEY='<output>'")
        print("  3. Run the bot (you'll be prompted for password)")
        return 1

    # Load config from YAML file
    yaml_config = load_config(args.config)

    # Create TradingConfig from YAML, with CLI overrides
    trading_config = TradingConfig.from_config(yaml_config)

    # Apply CLI overrides
    if args.dry_run:
        trading_config.dry_run = True
    if args.interval is not None:
        trading_config.check_interval_seconds = args.interval
    if args.max_position is not None:
        trading_config.max_position_size_usd = args.max_position
    if args.max_exposure is not None:
        trading_config.max_total_exposure_usd = args.max_exposure
    if args.min_edge is not None:
        trading_config.min_edge_to_buy = args.min_edge

    # Create FairValueCalculator from config
    fair_value_calculator = FairValueCalculator.from_config(yaml_config)

    # Create and run bot
    bot = PolymarketTradingBot(
        private_key=private_key,
        config=trading_config,
        fair_value_calculator=fair_value_calculator,
    )

    if args.once:
        summary = bot.run_once()
        if summary['errors']:
            print(f"  Errors: {summary['errors']}")
        return 0
    else:
        bot.run()
        return 0


if __name__ == "__main__":
    exit(main())
