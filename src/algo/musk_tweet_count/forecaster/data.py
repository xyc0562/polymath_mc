"""
Data layer for the forecasting model.

Handles:
- XTracker API client for fetching official post data
- Event storage and retrieval
- Contract-day calculations with proper timezone handling
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger(__name__)


@dataclass
class TweetEvent:
    """A single tweet/post event."""

    timestamp: datetime
    event_type: str = "tweet"  # "tweet" or "retweet"
    event_id: Optional[str] = None

    def __post_init__(self):
        # Ensure timestamp is timezone-aware
        if self.timestamp.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")


class ContractDayUtils:
    """Utilities for contract-day calculations."""

    def __init__(self, timezone: str = "America/New_York", boundary_hour: int = 12):
        """
        Initialize contract-day utilities.

        Args:
            timezone: Timezone for calculations (default: US Eastern)
            boundary_hour: Hour of day boundary (default: 12 = noon)
        """
        self.tz = ZoneInfo(timezone)
        self.boundary_hour = boundary_hour

    def get_contract_date(self, timestamp: datetime) -> date:
        """
        Get the contract-day for a timestamp.

        Contract-day boundary is at noon ET.
        [noon_d, noon_{d+1}) belongs to contract-day d.

        Args:
            timestamp: Timezone-aware datetime

        Returns:
            Contract date
        """
        ts_local = timestamp.astimezone(self.tz)
        # Subtract boundary hours, then take the date
        adjusted = ts_local - timedelta(hours=self.boundary_hour)
        return adjusted.date()

    def get_tau(self, timestamp: datetime, contract_date: date) -> int:
        """
        Get minutes since contract-day start (noon).

        Args:
            timestamp: Timezone-aware datetime
            contract_date: The contract date

        Returns:
            Minutes since noon (τ ∈ [0, 1440))
        """
        ts_local = timestamp.astimezone(self.tz)
        contract_start = datetime.combine(
            contract_date,
            time(self.boundary_hour, 0),
            tzinfo=self.tz
        )
        delta = ts_local - contract_start
        return int(delta.total_seconds() / 60)

    def get_contract_day_bounds(self, contract_date: date) -> Tuple[datetime, datetime]:
        """
        Get start and end datetimes for a contract-day.

        Args:
            contract_date: The contract date

        Returns:
            Tuple of (start_dt, end_dt) where end is exclusive
        """
        start = datetime.combine(
            contract_date,
            time(self.boundary_hour, 0),
            tzinfo=self.tz
        )
        end = datetime.combine(
            contract_date + timedelta(days=1),
            time(self.boundary_hour, 0),
            tzinfo=self.tz
        )
        return start, end

    def get_current_contract_date(self) -> date:
        """Get the current contract-day."""
        now = datetime.now(self.tz)
        return self.get_contract_date(now)

    def get_current_tau(self) -> int:
        """Get current τ (minutes since noon)."""
        now = datetime.now(self.tz)
        contract_date = self.get_contract_date(now)
        return self.get_tau(now, contract_date)

    def is_weekend(self, contract_date: date) -> bool:
        """Check if contract-day starts on weekend (Sat/Sun)."""
        return contract_date.weekday() in (5, 6)


class XTrackerClient:
    """Client for fetching official post data from XTracker API."""

    BASE_URL = "https://xtracker.polymarket.com/api"

    # XTracker data availability start date
    DATA_START_DATE = date(2025, 11, 1)

    def __init__(self, timeout: int = 60):
        """
        Initialize XTracker client.

        Args:
            timeout: Request timeout in seconds
        """
        self.session = requests.Session()
        self.timeout = timeout

    def fetch_all_posts(
        self,
        start_date: date,
        end_date: date,
        handle: str = "elonmusk",
    ) -> List[Dict]:
        """
        Fetch all posts in a date range with a single API call.

        The XTracker API returns all posts without pagination.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            handle: Twitter handle

        Returns:
            List of raw post dictionaries from API
        """
        url = f"{self.BASE_URL}/users/{handle}/posts"

        # Format dates as UTC ISO strings
        start_str = f"{start_date.isoformat()}T00:00:00.000Z"
        end_str = f"{end_date.isoformat()}T23:59:59.999Z"

        params = {
            "startDate": start_str,
            "endDate": end_str,
        }

        logger.info(f"Fetching posts from {start_date} to {end_date}...")

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            # API returns {"success": true, "data": [...]}
            if isinstance(data, dict):
                if not data.get("success", False):
                    logger.error(f"API returned success=false")
                    return []
                posts = data.get("data", [])
            elif isinstance(data, list):
                posts = data
            else:
                posts = []

            logger.info(f"Fetched {len(posts)} total posts")
            return posts

        except requests.RequestException as e:
            logger.error(f"Failed to fetch posts: {e}")
            return []

    def _parse_post(self, post: Dict) -> Optional[TweetEvent]:
        """
        Parse a single post from API response into TweetEvent.

        API response format:
        {
            "id": "...",
            "platformId": "...",
            "content": "RT @user: ..." or "regular tweet",
            "createdAt": "2026-01-25T13:06:41.000Z",
            "importedAt": "...",
            "metrics": null
        }
        """
        try:
            # Parse timestamp from createdAt field
            ts_str = post.get("createdAt")
            if not ts_str:
                return None

            # Handle Z suffix -> +00:00 for fromisoformat
            ts_str = ts_str.replace("Z", "+00:00")
            timestamp = datetime.fromisoformat(ts_str)

            # Determine event type from content
            content = post.get("content", "")
            event_type = "retweet" if content.startswith("RT @") else "tweet"

            return TweetEvent(
                timestamp=timestamp,
                event_type=event_type,
                event_id=post.get("id"),
            )

        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse post: {e}")
            return None

    def fetch_all_events(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        handle: str = "elonmusk",
    ) -> List[TweetEvent]:
        """
        Fetch all events in a date range.

        Args:
            start_date: Start date (default: XTracker start date)
            end_date: End date (default: today)
            handle: Twitter handle

        Returns:
            List of TweetEvent objects sorted by timestamp
        """
        if start_date is None:
            start_date = self.DATA_START_DATE

        if end_date is None:
            end_date = date.today()

        # Ensure we don't go before data availability
        if start_date < self.DATA_START_DATE:
            start_date = self.DATA_START_DATE

        posts = self.fetch_all_posts(start_date, end_date, handle)

        events = []
        for post in posts:
            event = self._parse_post(post)
            if event:
                events.append(event)

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)

        logger.info(f"Parsed {len(events)} events")
        return events

    def fetch_events_by_contract_day(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        contract_utils: Optional[ContractDayUtils] = None,
        handle: str = "elonmusk",
    ) -> Dict[date, List[TweetEvent]]:
        """
        Fetch all events and group by contract-day.

        This fetches all data in a single API call, then groups locally.

        Args:
            start_date: Start date (default: XTracker start date)
            end_date: End date (default: today)
            contract_utils: Contract-day utilities (creates new if None)
            handle: Twitter handle

        Returns:
            Dict mapping contract_date -> List[TweetEvent]
        """
        if contract_utils is None:
            contract_utils = ContractDayUtils()

        # Fetch all events
        events = self.fetch_all_events(start_date, end_date, handle)

        # Group by contract-day
        events_by_date: Dict[date, List[TweetEvent]] = {}

        for event in events:
            contract_date = contract_utils.get_contract_date(event.timestamp)

            if contract_date not in events_by_date:
                events_by_date[contract_date] = []

            events_by_date[contract_date].append(event)

        # Sort events within each day
        for contract_date in events_by_date:
            events_by_date[contract_date].sort(key=lambda e: e.timestamp)

        logger.info(f"Grouped into {len(events_by_date)} contract-days")

        return events_by_date

    # Legacy methods for backward compatibility

    def get_posts(
        self,
        start_date: str,
        end_date: str,
        handle: str = "elonmusk",
    ) -> List[Dict]:
        """Legacy method: Fetch posts from XTracker API."""
        # Parse ISO date strings to date objects
        start = datetime.fromisoformat(start_date.replace("Z", "+00:00")).date()
        end = datetime.fromisoformat(end_date.replace("Z", "+00:00")).date()
        return self.fetch_all_posts(start, end, handle)

    def get_contract_day_posts(
        self,
        contract_date: date,
        contract_utils: ContractDayUtils,
        handle: str = "elonmusk",
    ) -> List[TweetEvent]:
        """Legacy method: Fetch all posts for a specific contract-day."""
        # For single day, still use the bulk fetch but filter
        start_dt, end_dt = contract_utils.get_contract_day_bounds(contract_date)

        # Fetch with buffer to ensure we get all posts
        events = self.fetch_all_events(
            start_date=contract_date,
            end_date=contract_date + timedelta(days=1),
            handle=handle,
        )

        # Filter to just this contract-day
        result = [
            e for e in events
            if contract_utils.get_contract_date(e.timestamp) == contract_date
        ]

        return result

    def fetch_historical(
        self,
        n_days: int,
        contract_utils: ContractDayUtils,
        handle: str = "elonmusk",
    ) -> Dict[date, List[TweetEvent]]:
        """
        Legacy method: Fetch historical posts for training.

        Now fetches all data in a single API call.
        """
        today = contract_utils.get_current_contract_date()
        start_date = today - timedelta(days=n_days)

        # Ensure we don't go before data availability
        if start_date < self.DATA_START_DATE:
            start_date = self.DATA_START_DATE

        return self.fetch_events_by_contract_day(
            start_date=start_date,
            end_date=today,
            contract_utils=contract_utils,
            handle=handle,
        )


class EventStore:
    """Storage for tweet events with contract-day aggregation."""

    def __init__(
        self,
        contract_utils: ContractDayUtils,
        xtracker_client: Optional[XTrackerClient] = None,
    ):
        """
        Initialize event store.

        Args:
            contract_utils: Contract-day utilities
            xtracker_client: Optional XTracker client for data fetching
        """
        self.contract_utils = contract_utils
        self.xtracker_client = xtracker_client or XTrackerClient()

        # Storage: contract_date -> List[TweetEvent]
        self._events: Dict[date, List[TweetEvent]] = {}

        # Cache for contract-day counts
        self._counts_cache: Dict[date, int] = {}

    def add_event(self, event: TweetEvent) -> None:
        """Add a single event."""
        contract_date = self.contract_utils.get_contract_date(event.timestamp)

        if contract_date not in self._events:
            self._events[contract_date] = []

        self._events[contract_date].append(event)
        self._events[contract_date].sort(key=lambda e: e.timestamp)

        # Invalidate cache
        self._counts_cache.pop(contract_date, None)

    def add_events(self, events: List[TweetEvent]) -> None:
        """Add multiple events."""
        for event in events:
            self.add_event(event)

    def get_events(
        self,
        start: datetime,
        end: datetime,
    ) -> List[TweetEvent]:
        """
        Get events in a time range.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (exclusive)

        Returns:
            List of events in range
        """
        result = []

        # Determine contract dates to check
        start_date = self.contract_utils.get_contract_date(start)
        end_date = self.contract_utils.get_contract_date(end)

        current = start_date
        while current <= end_date:
            if current in self._events:
                for event in self._events[current]:
                    if start <= event.timestamp < end:
                        result.append(event)
            current += timedelta(days=1)

        return sorted(result, key=lambda e: e.timestamp)

    def get_contract_day_events(self, contract_date: date) -> List[TweetEvent]:
        """Get all events for a contract-day."""
        return self._events.get(contract_date, [])

    def get_contract_day_count(self, contract_date: date) -> int:
        """Get tweet count for a contract-day."""
        if contract_date in self._counts_cache:
            return self._counts_cache[contract_date]

        count = len(self._events.get(contract_date, []))
        self._counts_cache[contract_date] = count
        return count

    def get_contract_day_counts(self, n_days: int) -> Dict[date, int]:
        """
        Get counts for the last n completed contract-days.

        Args:
            n_days: Number of days

        Returns:
            Dict mapping contract_date -> count
        """
        today = self.contract_utils.get_current_contract_date()
        counts = {}

        for days_ago in range(1, n_days + 1):
            contract_date = today - timedelta(days=days_ago)
            counts[contract_date] = self.get_contract_day_count(contract_date)

        return counts

    def refresh_from_api(self, n_days: int = 90) -> None:
        """
        Refresh data from XTracker API.

        Args:
            n_days: Number of days to fetch
        """
        logger.info(f"Refreshing {n_days} days of data from XTracker API")

        history = self.xtracker_client.fetch_historical(
            n_days,
            self.contract_utils,
        )

        for contract_date, events in history.items():
            self._events[contract_date] = events
            self._counts_cache[contract_date] = len(events)

        logger.info(f"Loaded {sum(len(e) for e in self._events.values())} total events")

    def get_events_since_noon(self, contract_date: Optional[date] = None) -> List[TweetEvent]:
        """
        Get events since noon of the given contract-day.

        Args:
            contract_date: Contract date (default: today)

        Returns:
            List of events since noon
        """
        if contract_date is None:
            contract_date = self.contract_utils.get_current_contract_date()

        return self.get_contract_day_events(contract_date)

    def get_historical_timestamps(
        self,
        n_days: int,
    ) -> Dict[date, List[datetime]]:
        """
        Get timestamps for historical contract-days.

        Used for fitting intraday curves.

        Args:
            n_days: Number of days

        Returns:
            Dict mapping contract_date -> List[timestamp]
        """
        today = self.contract_utils.get_current_contract_date()
        result = {}

        for days_ago in range(1, n_days + 1):
            contract_date = today - timedelta(days=days_ago)
            events = self.get_contract_day_events(contract_date)
            result[contract_date] = [e.timestamp for e in events]

        return result

    def has_data_for_date(self, contract_date: date) -> bool:
        """Check if we have data for a contract-day."""
        return contract_date in self._events

    def get_date_range(self) -> Tuple[Optional[date], Optional[date]]:
        """Get the range of dates with data."""
        if not self._events:
            return None, None
        dates = sorted(self._events.keys())
        return dates[0], dates[-1]
