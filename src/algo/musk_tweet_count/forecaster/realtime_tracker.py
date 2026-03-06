"""
Realtime twikit-based tracker for provisional tweet detection.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Deque, List, Optional, Set

from .data import ContractDayUtils, EventStore, TweetEvent

logger = logging.getLogger(__name__)

MUSK_USER_ID = "44196397"
DEFAULT_RECENT_ID_LIMIT = 500
DEFAULT_FAILURE_BACKOFF_SECONDS = 60.0


@dataclass
class RealtimePollResult:
    """Outcome from a realtime twikit poll."""

    events: List[TweetEvent]
    gap_detected: bool = False
    newest_timestamp: Optional[datetime] = None


class RealtimeTweetTracker:
    """Poll Musk's timeline via twikit for faster-than-XTracker detection."""

    def __init__(
        self,
        event_store: EventStore,
        contract_utils: ContractDayUtils,
        cookies_path: Path,
        poll_interval: float = 10.0,
        fetch_count: int = 40,
        late_tweet_grace_seconds: float = 120.0,
        recent_id_limit: int = DEFAULT_RECENT_ID_LIMIT,
        failure_backoff_seconds: float = DEFAULT_FAILURE_BACKOFF_SECONDS,
    ):
        self.event_store = event_store
        self.contract_utils = contract_utils
        self.cookies_path = Path(cookies_path)
        self.poll_interval = poll_interval
        self.fetch_count = fetch_count
        self.late_tweet_grace_seconds = late_tweet_grace_seconds
        self.failure_backoff_seconds = failure_backoff_seconds
        self._client = None
        self._initialized = False
        self._recent_ids: Set[str] = set()
        self._recent_id_order: Deque[str] = deque()
        self._recent_id_limit = recent_id_limit
        self._watermark: Optional[datetime] = None
        self._last_poll_time: Optional[datetime] = None
        self._consecutive_errors = 0
        self._backoff_until: Optional[datetime] = None

    @property
    def last_poll_time(self) -> Optional[datetime]:
        return self._last_poll_time

    @property
    def watermark(self) -> Optional[datetime]:
        return self._watermark

    @property
    def consecutive_errors(self) -> int:
        return self._consecutive_errors

    @property
    def backoff_until(self) -> Optional[datetime]:
        return self._backoff_until

    def _remember_id(self, tweet_id: str) -> None:
        if tweet_id in self._recent_ids:
            return
        self._recent_ids.add(tweet_id)
        self._recent_id_order.append(tweet_id)
        while len(self._recent_id_order) > self._recent_id_limit:
            old_id = self._recent_id_order.popleft()
            self._recent_ids.discard(old_id)

    def seed_from_official_store(self, last_xtracker_refresh_time: Optional[datetime]) -> None:
        """Seed the recent-ID cache and watermark from recent authoritative data."""
        today = self.contract_utils.get_current_contract_date()
        dates = [today - timedelta(days=1), today]
        latest = last_xtracker_refresh_time

        for contract_date in dates:
            for event in self.event_store.get_official_contract_day_events(contract_date):
                if event.event_id:
                    self._remember_id(event.event_id)
                if latest is None or event.timestamp > latest:
                    latest = event.timestamp

        self._watermark = latest
        logger.info(
            "Realtime tracker seeded: %d recent ids, watermark=%s",
            len(self._recent_ids),
            self._watermark.isoformat() if self._watermark else "none",
        )

    def should_poll(self, now: datetime) -> bool:
        if self._backoff_until and now < self._backoff_until:
            return False
        if self._last_poll_time is None:
            return True
        return (now - self._last_poll_time).total_seconds() >= self.poll_interval

    async def _ensure_initialized(self) -> bool:
        if self._initialized:
            return True

        try:
            from twikit import Client

            self._client = Client(language="en-US")
            self._client.load_cookies(str(self.cookies_path))
            self._initialized = True
            logger.info("Realtime tracker initialized from %s", self.cookies_path)
            return True
        except Exception as exc:
            logger.warning("Realtime tracker initialization failed: %s", exc)
            return False

    @staticmethod
    def _normalize_timestamp(timestamp: datetime) -> datetime:
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp

    def _tweet_to_event(self, tweet) -> Optional[TweetEvent]:
        created_at = getattr(tweet, "created_at_datetime", None)
        if created_at is None:
            return None

        event_type = "retweet" if getattr(tweet, "retweeted_tweet", None) is not None else "tweet"
        return TweetEvent(
            timestamp=self._normalize_timestamp(created_at),
            event_type=event_type,
            event_id=str(tweet.id),
            source="twikit",
        )

    async def poll_once(self) -> RealtimePollResult:
        """
        Poll Musk's timeline once.

        Returns provisional events sorted oldest-to-newest. If the fetched window
        is obviously truncated, returns `gap_detected=True` and no events so the
        caller can fall back to XTracker authority immediately.
        """
        now = datetime.now(self.contract_utils.tz)
        self._last_poll_time = now

        if not await self._ensure_initialized():
            return RealtimePollResult(events=[])

        try:
            tweets = list(await self._client.get_user_tweets(MUSK_USER_ID, "Tweets", count=self.fetch_count))
            tweets.sort(key=lambda tweet: getattr(tweet, "created_at_datetime", datetime.min.replace(tzinfo=self.contract_utils.tz)))

            if not tweets:
                self._consecutive_errors = 0
                self._backoff_until = None
                return RealtimePollResult(events=[])

            oldest_timestamp = self._normalize_timestamp(tweets[0].created_at_datetime)
            newest_timestamp = self._normalize_timestamp(tweets[-1].created_at_datetime)
            if (
                self._watermark is not None and
                len(tweets) >= self.fetch_count and
                oldest_timestamp > self._watermark
            ):
                logger.warning(
                    "Realtime tracker detected a timeline gap: oldest fetched tweet %s is newer than watermark %s",
                    oldest_timestamp.isoformat(),
                    self._watermark.isoformat(),
                )
                self._consecutive_errors = 0
                self._backoff_until = None
                return RealtimePollResult(events=[], gap_detected=True, newest_timestamp=newest_timestamp)

            cutoff = None
            if self._watermark is not None:
                cutoff = self._watermark - timedelta(seconds=self.late_tweet_grace_seconds)

            events: List[TweetEvent] = []
            for tweet in tweets:
                event = self._tweet_to_event(tweet)
                if event is None:
                    continue
                if event.event_id and event.event_id in self._recent_ids:
                    continue
                if cutoff is not None and event.timestamp <= cutoff:
                    continue

                if event.event_id:
                    self._remember_id(event.event_id)
                events.append(event)

            if self._watermark is None or newest_timestamp > self._watermark:
                self._watermark = newest_timestamp

            self._consecutive_errors = 0
            self._backoff_until = None
            return RealtimePollResult(events=events, newest_timestamp=newest_timestamp)

        except Exception as exc:
            self._consecutive_errors += 1
            if self._consecutive_errors >= 5:
                self._backoff_until = now + timedelta(seconds=self.failure_backoff_seconds)
                logger.error(
                    "Realtime tracker poll failed %d times; backing off until %s: %s",
                    self._consecutive_errors,
                    self._backoff_until.isoformat(),
                    exc,
                )
            else:
                logger.warning(
                    "Realtime tracker poll failed (%d): %s",
                    self._consecutive_errors,
                    exc,
                )
            return RealtimePollResult(events=[])
