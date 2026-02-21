"""
Data provider for backtesting.

Loads historical price data and caches XTracker posts to avoid repeated API calls.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PricePoint:
    """A single price observation."""
    timestamp: int
    datetime: datetime
    price: float


@dataclass
class BinPriceHistory:
    """Price history for a single bin."""
    bin_index: int
    outcome: str
    lower_bound: int
    upper_bound: int
    token_id: str
    is_winner: bool
    prices: List[PricePoint] = field(default_factory=list)

    def get_price_at(self, ts: int) -> Optional[float]:
        """Get price at or before the given timestamp."""
        # Binary search for efficiency
        if not self.prices:
            return None

        # Find the latest price at or before ts
        result = None
        for p in self.prices:
            if p.timestamp <= ts:
                result = p.price
            else:
                break
        return result


def parse_counting_dates(short_name: str, end_date: date) -> Tuple[date, date]:
    """
    Parse tweet counting window from short_name.

    Examples:
        "Nov 18 - Nov 25" -> (2025-11-18, 2025-11-25)
        "Jan 27 - Feb 3" -> (2026-01-27, 2026-02-03)
        "Elon Musk musk # tweets in Jan" -> (2026-01-01, 2026-02-01)
        "Elon Musk musk # tweets in Oct" -> (2025-10-01, 2025-11-01)

    Args:
        short_name: Event short name like "Nov 18 - Nov 25"
        end_date: Settlement date to derive year

    Returns:
        Tuple of (counting_start_date, counting_end_date)
    """
    import re

    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }

    # Pattern for monthly events: "tweets in {Month}" (e.g., "tweets in Jan", "tweets in October")
    # Counting window = 1st of that month to 1st of next month (= end_date)
    monthly_pattern = r"tweets\s+in\s+(\w+)"
    monthly_match = re.search(monthly_pattern, short_name, re.IGNORECASE)
    if monthly_match:
        month_str = monthly_match.group(1)
        month_num = month_map.get(month_str) or month_map.get(month_str.lower())
        if month_num:
            # The counting window is the full month.
            # end_date (settlement) is the 1st of the next month.
            # counting_start = 1st of that month
            # counting_end = end_date (1st of next month)
            start_year = end_date.year
            # If the month is December and end_date is in January, start is previous year
            if month_num > end_date.month:
                start_year -= 1
            try:
                counting_start = date(start_year, month_num, 1)
                return counting_start, end_date
            except ValueError:
                pass

    # Pattern for "Month DD - Month DD" format
    date_range_pattern = r"(\w{3})\s+(\d{1,2})\s*-\s*(\w{3})\s+(\d{1,2})"
    match = re.search(date_range_pattern, short_name)

    if match:
        start_month_str, start_day_str, end_month_str, end_day_str = match.groups()
        start_day = int(start_day_str)
        end_day = int(end_day_str)

        start_month = month_map.get(start_month_str, end_date.month)
        end_month = month_map.get(end_month_str, end_date.month)

        # Determine year - usually same as end_date, but handle year boundary
        start_year = end_date.year
        end_year = end_date.year

        # If start month > end month, start is in previous year
        if start_month > end_month:
            start_year -= 1

        try:
            counting_start = date(start_year, start_month, start_day)
            counting_end = date(end_year, end_month, end_day)
            return counting_start, counting_end
        except ValueError:
            pass

    # Fallback: log warning and assume 7-day window ending at end_date
    import logging
    logging.getLogger(__name__).warning(
        f"Could not parse counting dates from '{short_name}', falling back to 7-day window"
    )
    return end_date - timedelta(days=6), end_date


@dataclass
class EventPriceData:
    """All price data for an event."""
    event_id: str
    title: str
    short_name: str
    start_date: date  # When trading opens
    end_date: date    # Settlement date
    bins: List[BinPriceHistory] = field(default_factory=list)

    # Computed fields
    all_timestamps: List[int] = field(default_factory=list)
    winner_bin_index: Optional[int] = None
    counting_start_date: Optional[date] = None  # When tweet counting starts
    counting_end_date: Optional[date] = None    # When tweet counting ends

    def __post_init__(self):
        """Compute derived fields."""
        # Collect all unique timestamps across all bins
        ts_set = set()
        for bin_data in self.bins:
            for p in bin_data.prices:
                ts_set.add(p.timestamp)
            if bin_data.is_winner:
                self.winner_bin_index = bin_data.bin_index

        self.all_timestamps = sorted(ts_set)

        # Parse counting dates from short_name if not set
        if self.counting_start_date is None or self.counting_end_date is None:
            self.counting_start_date, self.counting_end_date = parse_counting_dates(
                self.short_name, self.end_date
            )

    def get_prices_at(self, ts: int) -> Dict[int, float]:
        """Get all bin prices at a given timestamp."""
        return {
            bin_data.bin_index: bin_data.get_price_at(ts)
            for bin_data in self.bins
            if bin_data.get_price_at(ts) is not None
        }


@dataclass
class SimulatedOrderbook:
    """Simulated orderbook from mid-price + spread."""
    bin_index: int
    mid_price: float
    spread: float

    @property
    def yes_bid(self) -> float:
        """Price to sell YES (bid)."""
        return max(0.001, self.mid_price - self.spread / 2)

    @property
    def yes_ask(self) -> float:
        """Price to buy YES (ask)."""
        return min(0.999, self.mid_price + self.spread / 2)

    @property
    def no_bid(self) -> float:
        """Price to sell NO (bid)."""
        return max(0.001, (1 - self.mid_price) - self.spread / 2)

    @property
    def no_ask(self) -> float:
        """Price to buy NO (ask)."""
        return min(0.999, (1 - self.mid_price) + self.spread / 2)


class HistoricalDataProvider:
    """
    Provides historical market data for backtesting.

    Loads price data from scraped files and caches XTracker posts.
    """

    def __init__(
        self,
        price_data_dir: Path,
        spread: float = 0.02,
        slippage: float = 0.005,
    ):
        """
        Initialize the data provider.

        Args:
            price_data_dir: Directory containing scraped price history
            spread: Bid-ask spread to simulate (default 2%)
            slippage: Additional execution cost (default 0.5%)
        """
        self.price_data_dir = Path(price_data_dir)
        self.spread = spread
        self.slippage = slippage

        # Cached data
        self._events: Dict[str, EventPriceData] = {}
        self._posts_cache: Optional[Dict] = None
        self._posts_cache_path: Optional[Path] = None

    def load_event(self, event_dir: str) -> Optional[EventPriceData]:
        """
        Load price data for a specific event.

        Args:
            event_dir: Directory name (e.g., "2024-11-01_Oct_25_-_Nov_1")

        Returns:
            EventPriceData or None if not found
        """
        if event_dir in self._events:
            return self._events[event_dir]

        event_path = self.price_data_dir / event_dir

        if not event_path.exists():
            logger.warning(f"Event directory not found: {event_path}")
            return None

        # Load event info
        event_info_path = event_path / "event_info.json"
        if not event_info_path.exists():
            logger.warning(f"Event info not found: {event_info_path}")
            return None

        with open(event_info_path) as f:
            event_info = json.load(f)

        # Parse dates
        try:
            start_date = date.fromisoformat(event_info["start_date"])
            end_date = date.fromisoformat(event_info["end_date"])
        except (KeyError, ValueError) as e:
            logger.warning(f"Invalid dates in event info: {e}")
            return None

        # Load bins info
        bins_data = []
        for bin_info in event_info.get("bins", []):
            bins_data.append(BinPriceHistory(
                bin_index=bin_info["bin_index"],
                outcome=bin_info["outcome"],
                lower_bound=bin_info["lower_bound"],
                upper_bound=bin_info["upper_bound"],
                token_id=bin_info["token_id"],
                is_winner=bin_info["is_winner"],
                prices=[],
            ))

        # Load price data
        prices_path = event_path / "prices.csv"
        if prices_path.exists():
            self._load_prices_csv(prices_path, bins_data)

        event_data = EventPriceData(
            event_id=event_info.get("event_id", ""),
            title=event_info.get("title", ""),
            short_name=event_info.get("short_name", ""),
            start_date=start_date,
            end_date=end_date,
            bins=bins_data,
        )

        self._events[event_dir] = event_data
        logger.info(
            f"Loaded event {event_data.short_name}: "
            f"{len(event_data.bins)} bins, "
            f"{len(event_data.all_timestamps)} timestamps"
        )

        return event_data

    def _load_prices_csv(
        self,
        prices_path: Path,
        bins_data: List[BinPriceHistory],
    ) -> None:
        """Load price data from CSV into bins."""
        bin_map = {b.bin_index: b for b in bins_data}

        with open(prices_path) as f:
            header = f.readline()  # Skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 5:
                    continue

                try:
                    bin_index = int(parts[0])
                    timestamp = int(parts[2])
                    dt = datetime.fromisoformat(parts[3])
                    price = float(parts[4])

                    if bin_index in bin_map:
                        bin_map[bin_index].prices.append(PricePoint(
                            timestamp=timestamp,
                            datetime=dt,
                            price=price,
                        ))
                except (ValueError, IndexError):
                    continue

        # Sort prices by timestamp for each bin
        for bin_data in bins_data:
            bin_data.prices.sort(key=lambda p: p.timestamp)

    def get_simulated_orderbook(
        self,
        event: EventPriceData,
        bin_index: int,
        timestamp: int,
    ) -> Optional[SimulatedOrderbook]:
        """
        Get simulated orderbook for a bin at a given time.

        Args:
            event: The event data
            bin_index: Which bin
            timestamp: Unix timestamp

        Returns:
            SimulatedOrderbook or None if no price data
        """
        if bin_index >= len(event.bins):
            return None

        bin_data = event.bins[bin_index]
        mid_price = bin_data.get_price_at(timestamp)

        if mid_price is None:
            return None

        return SimulatedOrderbook(
            bin_index=bin_index,
            mid_price=mid_price,
            spread=self.spread,
        )

    def get_all_orderbooks(
        self,
        event: EventPriceData,
        timestamp: int,
    ) -> Dict[int, SimulatedOrderbook]:
        """Get simulated orderbooks for all bins at a given time."""
        result = {}
        for bin_data in event.bins:
            ob = self.get_simulated_orderbook(event, bin_data.bin_index, timestamp)
            if ob:
                result[bin_data.bin_index] = ob
        return result

    def list_available_events(self) -> List[str]:
        """List all available event directories."""
        if not self.price_data_dir.exists():
            return []

        events = []
        for item in self.price_data_dir.iterdir():
            if item.is_dir() and (item / "event_info.json").exists():
                events.append(item.name)

        return sorted(events)


class CachedPostsProvider:
    """
    Caches XTracker posts data to avoid repeated API calls during backtest.

    Posts are cached per backtest run and saved to disk for reuse.
    """

    def __init__(
        self,
        cache_dir: Path,
        xtracker_handle: str = "elonmusk",
    ):
        """
        Initialize posts cache.

        Args:
            cache_dir: Directory to store cached posts
            xtracker_handle: XTracker handle to fetch posts for
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.xtracker_handle = xtracker_handle

        self._posts: List[Dict] = []
        self._loaded = False
        self._loaded_start: Optional[date] = None
        self._loaded_end: Optional[date] = None
        self._cache_file = self.cache_dir / f"posts_{xtracker_handle}.json"

    def load_or_fetch(self, start_date: date, end_date: date) -> List[Dict]:
        """
        Load posts from cache or fetch from API.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of post dictionaries
        """
        # Check if already loaded data covers the requested range
        if self._loaded:
            if (self._loaded_start and self._loaded_end and
                self._loaded_start <= start_date and self._loaded_end >= end_date):
                return self._posts
            else:
                # Need to reload for new range
                logger.info(f"Extending cache to cover {start_date} - {end_date}")
                self._loaded = False

        # Try to load from cache
        if self._cache_file.exists():
            try:
                with open(self._cache_file) as f:
                    cache_data = json.load(f)

                cached_start = date.fromisoformat(cache_data.get("start_date", ""))
                cached_end = date.fromisoformat(cache_data.get("end_date", ""))

                # Check if cache covers our date range
                if cached_start <= start_date and cached_end >= end_date:
                    self._posts = cache_data.get("posts", [])
                    self._loaded = True
                    self._loaded_start = cached_start
                    self._loaded_end = cached_end
                    logger.info(
                        f"Loaded {len(self._posts)} posts from cache "
                        f"({cached_start} to {cached_end})"
                    )
                    return self._posts

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to load posts cache: {e}")

        # Fetch from API
        self._posts = self._fetch_from_api(start_date, end_date)
        self._loaded = True
        self._loaded_start = start_date
        self._loaded_end = end_date

        # Save to cache
        self._save_cache(start_date, end_date)

        return self._posts

    def _fetch_from_api(self, start_date: date, end_date: date) -> List[Dict]:
        """Fetch posts from XTracker API."""
        from ..forecaster.data import XTrackerClient

        logger.info(f"Fetching posts from XTracker API ({start_date} to {end_date})...")

        client = XTrackerClient()

        # Fetch posts for the date range
        # Add buffer days to ensure we have enough data
        fetch_start = start_date - timedelta(days=7)
        fetch_end = end_date + timedelta(days=1)

        posts = client.get_posts(
            start_date=fetch_start.isoformat(),
            end_date=fetch_end.isoformat(),
            handle=self.xtracker_handle,
        )

        logger.info(f"Fetched {len(posts)} posts from XTracker API")

        return posts

    def _save_cache(self, start_date: date, end_date: date) -> None:
        """Save posts to cache file."""
        cache_data = {
            "handle": self.xtracker_handle,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "fetched_at": datetime.now().isoformat(),
            "num_posts": len(self._posts),
            "posts": self._posts,
        }

        with open(self._cache_file, "w") as f:
            json.dump(cache_data, f)

        logger.info(f"Saved {len(self._posts)} posts to cache: {self._cache_file}")

    def _get_post_timestamp(self, post: Dict) -> Optional[int]:
        """Extract Unix timestamp from a post."""
        ts = post.get("createdAt") or post.get("timestamp")
        if ts is None:
            return None

        if isinstance(ts, (int, float)):
            return int(ts)
        elif isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return int(dt.timestamp())
            except ValueError:
                return None
        return None

    def get_posts_before(self, timestamp: int) -> List[Dict]:
        """Get all posts before a given timestamp."""
        result = []
        for p in self._posts:
            ts = self._get_post_timestamp(p)
            if ts is not None and ts <= timestamp:
                result.append(p)
        return result

    def get_posts_in_range(
        self,
        start_ts: int,
        end_ts: int,
    ) -> List[Dict]:
        """Get posts within a timestamp range."""
        result = []
        for p in self._posts:
            ts = self._get_post_timestamp(p)
            if ts is not None and start_ts <= ts <= end_ts:
                result.append(p)
        return result

    def count_posts_in_range(
        self,
        start_ts: int,
        end_ts: int,
    ) -> int:
        """Count posts within a timestamp range."""
        return len(self.get_posts_in_range(start_ts, end_ts))
