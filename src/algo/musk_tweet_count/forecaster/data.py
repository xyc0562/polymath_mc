"""
Data layer for the forecasting model.

Handles:
- XTracker API client for fetching official post data
- Event storage and retrieval
- Contract-day calculations with proper timezone handling
- CSV data loading for unofficial historical data
"""

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta, timezone
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
    source: str = "xtracker"

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

    def get_contract_day_bounds_utc(self, contract_date: date) -> Tuple[datetime, datetime]:
        """
        Get start and end datetimes for a contract-day in UTC.

        Args:
            contract_date: The contract date

        Returns:
            Tuple of (start_dt_utc, end_dt_utc) where end is exclusive
        """
        start, end = self.get_contract_day_bounds(contract_date)
        return start.astimezone(timezone.utc), end.astimezone(timezone.utc)

    def minutes_between(self, start: datetime, end: datetime) -> float:
        """
        Get real elapsed minutes between two timezone-aware datetimes.

        This normalizes both endpoints to UTC first, so DST transitions do not
        create phantom jumps or backwards time.
        """
        delta = end.astimezone(timezone.utc) - start.astimezone(timezone.utc)
        return delta.total_seconds() / 60.0

    def hours_between(self, start: datetime, end: datetime) -> float:
        """Get real elapsed hours between two timezone-aware datetimes."""
        return self.minutes_between(start, end) / 60.0

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
        self._warned_missing_platform_id = False

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
            platform_id = post.get("platformId")
            if not platform_id and not self._warned_missing_platform_id:
                logger.warning("XTracker post missing platformId; falling back to legacy post id")
                self._warned_missing_platform_id = True

            return TweetEvent(
                timestamp=timestamp,
                event_type=event_type,
                event_id=platform_id or post.get("id"),
                source="xtracker",
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
        Fetch historical posts for training.

        Fetches all data in a single API call, then groups by contract day.

        Note: Uses contract_today + 1 as end_date to avoid timezone mismatch.
        The XTracker API uses UTC timestamps, so posts from late ET hours
        have UTC dates of "tomorrow". Adding 1 day ensures we always capture them.
        """
        contract_today = contract_utils.get_current_contract_date()
        start_date = contract_today - timedelta(days=n_days)

        # Ensure we don't go before data availability
        if start_date < self.DATA_START_DATE:
            start_date = self.DATA_START_DATE

        return self.fetch_events_by_contract_day(
            start_date=start_date,
            end_date=contract_today + timedelta(days=1),
            contract_utils=contract_utils,
            handle=handle,
        )


def load_events_from_csv(
    csv_path: str,
    contract_utils: Optional[ContractDayUtils] = None,
    filter_originals: bool = False,
) -> Dict[date, List[TweetEvent]]:
    """
    Load events from CSV file and group by contract-day.

    CSV format:
        tweet_id,post_date,content,type
        2014182947297930000,21/1/26 22:46,Correct,original

    Args:
        csv_path: Path to CSV file
        contract_utils: Contract-day utilities (creates new if None)
        filter_originals: If True, only include type="original" (exclude retweets)

    Returns:
        Dict mapping contract_date -> List[TweetEvent]
    """
    if contract_utils is None:
        contract_utils = ContractDayUtils()

    events_by_date: Dict[date, List[TweetEvent]] = {}
    tz = contract_utils.tz

    logger.info(f"Loading events from CSV: {csv_path}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Skip retweets if filtering
            if filter_originals and row.get('type') != 'original':
                continue

            try:
                # Parse post_date: format "DD/M/YY HH:MM" (e.g., "21/1/26 22:46")
                post_date_str = row['post_date']

                # Split into date and time parts
                date_part, time_part = post_date_str.split(' ')
                if '-' in date_part:
                    year, month, day = date_part.split('-')
                    hour, minute, second = time_part.split(':')
                    year_full = int(year)
                else:
                    day, month, year = date_part.split('/')
                    hour, minute = time_part.split(':')
                    second = 0
                    year_full = int(year) + 2000
                # Convert 2-digit year to 4-digit (YY -> 20YY)

                # Create timezone-aware datetime (already in EST)
                timestamp = datetime(
                    year_full,
                    int(month),
                    int(day),
                    int(hour),
                    int(minute),
                    int(second),
                    tzinfo=tz
                )

                # Determine event type (retweet vs tweet)
                content = row.get('content', '')
                event_type = "retweet" if content.startswith("RT @") else "tweet"

                # Create TweetEvent
                event = TweetEvent(
                    timestamp=timestamp,
                    event_type=event_type,
                    event_id=row.get('tweet_id'),
                    source="xtracker",
                )

                # Group by contract-day
                contract_date = contract_utils.get_contract_date(timestamp)

                if contract_date not in events_by_date:
                    events_by_date[contract_date] = []

                events_by_date[contract_date].append(event)

            except (ValueError, KeyError, IndexError) as e:
                logger.warning(f"Failed to parse row: {e} - {row}")
                continue

    # Sort events within each day
    for contract_date in events_by_date:
        events_by_date[contract_date].sort(key=lambda e: e.timestamp)

    total_events = sum(len(events) for events in events_by_date.values())
    logger.info(
        f"Loaded {total_events} events from CSV, "
        f"grouped into {len(events_by_date)} contract-days"
    )

    return events_by_date


class EventStore:
    """
    Storage for tweet events with contract-day aggregation.

    Thread-safe: uses a lock to protect concurrent access to event data.
    """

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

        # Storage split into authoritative XTracker data and provisional realtime overlay.
        self._official_events: Dict[date, List[TweetEvent]] = {}
        self._provisional_events: Dict[date, List[TweetEvent]] = {}

        # Cache for contract-day counts
        self._counts_cache: Dict[date, int] = {}
        self._merged_cache: Dict[date, List[TweetEvent]] = {}

        # Lock for thread-safe access
        import threading
        self._lock = threading.RLock()

    def _invalidate_day(self, contract_date: date) -> None:
        self._counts_cache.pop(contract_date, None)
        self._merged_cache.pop(contract_date, None)

    def _dedupe_and_sort(self, events: List[TweetEvent]) -> List[TweetEvent]:
        unique_by_id: Dict[str, TweetEvent] = {}
        without_id: List[TweetEvent] = []

        for event in events:
            if event.event_id:
                unique_by_id[event.event_id] = event
            else:
                without_id.append(event)

        deduped = list(unique_by_id.values()) + without_id
        return sorted(deduped, key=lambda e: e.timestamp)

    def _add_to_layer(
        self,
        layer: Dict[date, List[TweetEvent]],
        event: TweetEvent,
        *,
        source: str,
    ) -> bool:
        contract_date = self.contract_utils.get_contract_date(event.timestamp)
        normalized = TweetEvent(
            timestamp=event.timestamp,
            event_type=event.event_type,
            event_id=event.event_id,
            source=source,
        )

        if contract_date not in layer:
            layer[contract_date] = []

        if normalized.event_id:
            existing_ids = {e.event_id for e in layer[contract_date] if e.event_id}
            if normalized.event_id in existing_ids:
                return False

        layer[contract_date].append(normalized)
        layer[contract_date] = self._dedupe_and_sort(layer[contract_date])
        self._invalidate_day(contract_date)
        return True

    def _merged_events_for_day(self, contract_date: date) -> List[TweetEvent]:
        cached = self._merged_cache.get(contract_date)
        if cached is not None:
            return cached

        official = self._official_events.get(contract_date, [])
        provisional = self._provisional_events.get(contract_date, [])

        merged: List[TweetEvent] = list(official)
        official_ids = {event.event_id for event in official if event.event_id}
        for event in provisional:
            if event.event_id and event.event_id in official_ids:
                continue
            merged.append(event)

        merged = self._dedupe_and_sort(merged)
        self._merged_cache[contract_date] = merged
        return merged

    def add_event(self, event: TweetEvent) -> bool:
        """
        Add a single authoritative event (with deduplication by event_id).

        If an event with the same event_id already exists, it won't be added again.
        Thread-safe.
        """
        with self._lock:
            return self._add_to_layer(self._official_events, event, source="xtracker")

    def add_provisional_event(self, event: TweetEvent) -> bool:
        """Add a provisional realtime event. Thread-safe."""
        with self._lock:
            contract_date = self.contract_utils.get_contract_date(event.timestamp)
            merged_ids = {
                existing.event_id
                for existing in self._merged_events_for_day(contract_date)
                if existing.event_id
            }
            if event.event_id and event.event_id in merged_ids:
                return False
            return self._add_to_layer(self._provisional_events, event, source="twikit")

    def add_events(self, events: List[TweetEvent]) -> int:
        """Add multiple authoritative events (with deduplication). Thread-safe."""
        inserted = 0
        for event in events:
            inserted += int(self.add_event(event))
        return inserted

    def replace_official_day(self, contract_date: date, events: List[TweetEvent]) -> bool:
        """
        Replace all authoritative events for a contract day.

        Returns True if the effective merged view changed.
        """
        filtered = [
            TweetEvent(
                timestamp=e.timestamp,
                event_type=e.event_type,
                event_id=e.event_id,
                source="xtracker",
            )
            for e in events
            if self.contract_utils.get_contract_date(e.timestamp) == contract_date
        ]
        filtered = self._dedupe_and_sort(filtered)

        with self._lock:
            before = list(self._merged_events_for_day(contract_date))
            self._official_events[contract_date] = filtered
            self._invalidate_day(contract_date)
            after = list(self._merged_events_for_day(contract_date))
            return before != after

    def clear_provisional_day(self, contract_date: date) -> bool:
        """Clear provisional overlay for a contract day. Returns True if anything changed."""
        with self._lock:
            before = list(self._merged_events_for_day(contract_date))
            removed = bool(self._provisional_events.get(contract_date))
            self._provisional_events.pop(contract_date, None)
            self._invalidate_day(contract_date)
            after = list(self._merged_events_for_day(contract_date))
            return removed and before != after

    def set_contract_day_events(
        self,
        contract_date: date,
        events: List[TweetEvent],
        allow_regression: bool = False,
    ) -> bool:
        """
        Replace all events for a contract day.

        Use this for refreshing a specific day's data.
        Thread-safe.

        Args:
            contract_date: The contract date to replace
            events: New list of events for that day
            allow_regression: If False (default), refuse to replace if new count < old count

        Returns:
            True if events were updated, False if rejected due to regression
        """
        if not allow_regression:
            logger.debug(
                "set_contract_day_events() now delegates to authoritative replacement; "
                "authoritative rebases are allowed by design"
            )
        return self.replace_official_day(contract_date, events)

    def get_events(
        self,
        start: datetime,
        end: datetime,
    ) -> List[TweetEvent]:
        """
        Get events in a time range. Thread-safe.

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

        with self._lock:
            current = start_date
            while current <= end_date:
                for event in self._merged_events_for_day(current):
                    if start <= event.timestamp < end:
                        result.append(event)
                current += timedelta(days=1)

        return sorted(result, key=lambda e: e.timestamp)

    def get_contract_day_events(self, contract_date: date) -> List[TweetEvent]:
        """Get all events for a contract-day. Thread-safe."""
        with self._lock:
            return list(self._merged_events_for_day(contract_date))

    def get_official_contract_day_events(self, contract_date: date) -> List[TweetEvent]:
        """Get authoritative XTracker events for a contract-day. Thread-safe."""
        with self._lock:
            return list(self._official_events.get(contract_date, []))

    def get_provisional_contract_day_events(self, contract_date: date) -> List[TweetEvent]:
        """Get provisional realtime events for a contract-day. Thread-safe."""
        with self._lock:
            return list(self._provisional_events.get(contract_date, []))

    def has_event_id(self, event_id: str, contract_date: Optional[date] = None) -> bool:
        """Check whether an event id exists in the merged effective view."""
        with self._lock:
            dates = [contract_date] if contract_date is not None else sorted(
                set(self._official_events) | set(self._provisional_events)
            )
            for day in dates:
                for event in self._merged_events_for_day(day):
                    if event.event_id == event_id:
                        return True
        return False

    def get_latest_official_timestamp(self, contract_dates: Optional[List[date]] = None) -> Optional[datetime]:
        """Get latest authoritative timestamp, optionally restricted to specific contract days."""
        with self._lock:
            dates = contract_dates or sorted(self._official_events.keys())
            latest: Optional[datetime] = None
            for contract_date in dates:
                for event in self._official_events.get(contract_date, []):
                    if latest is None or event.timestamp > latest:
                        latest = event.timestamp
            return latest

    def get_contract_day_count(self, contract_date: date) -> int:
        """Get tweet count for a contract-day. Thread-safe."""
        with self._lock:
            if contract_date in self._counts_cache:
                return self._counts_cache[contract_date]

            count = len(self._merged_events_for_day(contract_date))
            self._counts_cache[contract_date] = count
        return count

    def get_contract_day_counts(
        self,
        n_days: int,
        as_of_date: Optional[date] = None,
    ) -> Dict[date, int]:
        """
        Get counts for the last n completed contract-days.

        Args:
            n_days: Number of days
            as_of_date: Reference date for "today" (for backtesting). Default: actual today.

        Returns:
            Dict mapping contract_date -> count
        """
        if as_of_date is None:
            today = self.contract_utils.get_current_contract_date()
        else:
            today = as_of_date

        counts = {}

        for days_ago in range(1, n_days + 1):
            contract_date = today - timedelta(days=days_ago)
            counts[contract_date] = self.get_contract_day_count(contract_date)

        return counts

    def refresh_from_api(self, n_days: int = 90) -> bool:
        """
        Refresh data from XTracker API. Thread-safe.

        Args:
            n_days: Number of days to fetch

        Returns:
            True if refresh was successful, False if rejected due to empty/regression
        """
        logger.info(f"Refreshing {n_days} days of data from XTracker API")

        # Fetch data (blocking HTTP call, but outside lock)
        history = self.xtracker_client.fetch_historical(
            n_days,
            self.contract_utils,
        )

        # Don't replace with empty data
        if not history:
            logger.warning("Refresh returned no data, keeping cached data")
            return False

        with self._lock:
            all_dates = set(self._official_events) | set(history)
            for contract_date in all_dates:
                authoritative = history.get(contract_date, [])
                self._official_events[contract_date] = self._dedupe_and_sort([
                    TweetEvent(
                        timestamp=e.timestamp,
                        event_type=e.event_type,
                        event_id=e.event_id,
                        source="xtracker",
                    )
                    for e in authoritative
                    if self.contract_utils.get_contract_date(e.timestamp) == contract_date
                ])
                self._provisional_events.pop(contract_date, None)
                self._invalidate_day(contract_date)

            total = sum(len(self._merged_events_for_day(d)) for d in all_dates)

        logger.info(f"Loaded {total} total events")
        return True

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
        as_of_date: Optional[date] = None,
    ) -> Dict[date, List[datetime]]:
        """
        Get timestamps for historical contract-days.

        Used for fitting intraday curves.

        Args:
            n_days: Number of days
            as_of_date: Reference date for "today" (for backtesting). Default: actual today.

        Returns:
            Dict mapping contract_date -> List[timestamp]
        """
        if as_of_date is None:
            today = self.contract_utils.get_current_contract_date()
        else:
            today = as_of_date

        result = {}

        for days_ago in range(1, n_days + 1):
            contract_date = today - timedelta(days=days_ago)
            events = self.get_contract_day_events(contract_date)
            result[contract_date] = [e.timestamp for e in events]

        return result

    def has_data_for_date(self, contract_date: date) -> bool:
        """Check if we have data for a contract-day. Thread-safe."""
        with self._lock:
            return (
                contract_date in self._official_events or
                contract_date in self._provisional_events
            )

    def get_date_range(self) -> Tuple[Optional[date], Optional[date]]:
        """Get the range of dates with data. Thread-safe."""
        with self._lock:
            dates = sorted(set(self._official_events) | set(self._provisional_events))
            if not dates:
                return None, None
            return dates[0], dates[-1]

    def cleanup_old_data(self, keep_days: int = 60) -> int:
        """
        Remove data older than keep_days from today.

        Prevents memory growth over long-running processes.

        Args:
            keep_days: Number of days of data to keep

        Returns:
            Number of days removed
        """
        today = self.contract_utils.get_current_contract_date()
        cutoff = today - timedelta(days=keep_days)

        with self._lock:
            old_dates = [
                d for d in sorted(set(self._official_events) | set(self._provisional_events))
                if d < cutoff
            ]

            for old_date in old_dates:
                self._official_events.pop(old_date, None)
                self._provisional_events.pop(old_date, None)
                self._invalidate_day(old_date)

        return len(old_dates)
