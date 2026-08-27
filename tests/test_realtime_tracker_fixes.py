import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from src.algo.musk_tweet_count.forecaster.data import ContractDayUtils
from src.algo.musk_tweet_count.forecaster.realtime_tracker import RealtimeTweetTracker

T0 = datetime(2026, 7, 8, 12, 0, 0, tzinfo=timezone.utc)


MUSK_ID = "44196397"


def _tweet(offset_seconds: int, tweet_id: str, reply_to_user: str = None,
           reply_to_status: str = None) -> SimpleNamespace:
    legacy = {}
    if reply_to_user is not None:
        legacy["in_reply_to_user_id_str"] = reply_to_user
    if reply_to_status is not None:
        legacy["in_reply_to_status_id_str"] = reply_to_status
    return SimpleNamespace(
        created_at_datetime=T0 + timedelta(seconds=offset_seconds),
        id=tweet_id,
        retweeted_tweet=None,
        _legacy=legacy,
    )


class FakeTwikitClient:
    """Returns queued batches (or raises queued exceptions) per poll."""

    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = []
        self.set_cookie_calls = 0

    async def get_user_tweets(self, user_id, tweet_type, count):
        self.calls.append((user_id, tweet_type, count))
        item = self.batches.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def set_cookies(self, cookies, clear_cookies=False):
        self.set_cookie_calls += 1


def _make_tracker(client, fetch_count=5, watermark=None) -> RealtimeTweetTracker:
    tracker = RealtimeTweetTracker(
        event_store=None,
        contract_utils=ContractDayUtils(),
        cookies_path=Path("/nonexistent/cookies.json"),
        fetch_count=fetch_count,
    )
    tracker._initialized = True
    tracker._client = client
    tracker._cookie_list = [{"twid": "cookie-a"}, {"twid": "cookie-b"}]
    tracker._cookie_index = 0
    tracker._watermark = watermark
    return tracker


def test_poll_uses_replies_timeline():
    client = FakeTwikitClient([[_tweet(10, "1")]])
    tracker = _make_tracker(client)

    asyncio.run(tracker.poll_once())

    assert len(client.calls) == 1
    _, tweet_type, _ = client.calls[0]
    assert tweet_type == "Replies"


def test_gap_advances_watermark_and_next_poll_resumes():
    full_window = [_tweet(1000 + i, f"g{i}") for i in range(5)]
    client = FakeTwikitClient([full_window, list(full_window)])
    tracker = _make_tracker(client, fetch_count=5, watermark=T0)

    first = asyncio.run(tracker.poll_once())

    assert first.gap_detected is True
    assert first.events == []
    # Watermark must advance past the fetched window (pre-fix it stayed at
    # T0, so every later poll re-detected the same gap until restart)
    assert tracker.watermark == T0 + timedelta(seconds=1004)

    second = asyncio.run(tracker.poll_once())

    assert second.gap_detected is False
    assert second.events == []  # all ids remembered during the gap poll


def test_new_tweet_after_gap_is_detected_incrementally():
    full_window = [_tweet(1000 + i, f"g{i}") for i in range(5)]
    next_window = [_tweet(1001 + i, f"g{i + 1}") for i in range(4)] + [_tweet(1100, "new")]
    client = FakeTwikitClient([full_window, next_window])
    tracker = _make_tracker(client, fetch_count=5, watermark=T0)

    asyncio.run(tracker.poll_once())  # gap poll
    result = asyncio.run(tracker.poll_once())

    assert result.gap_detected is False
    assert [e.event_id for e in result.events] == ["new"]
    assert tracker.watermark == T0 + timedelta(seconds=1100)


def test_reply_to_other_account_is_activity_not_count():
    """Resolution rule: replies only count when Musk replies to his own
    thread. Replies to others must not enter the count — but surface as
    activity events (session-state signal)."""
    batch = [
        _tweet(10, "top-level"),
        _tweet(20, "reply-other", reply_to_user="12345", reply_to_status="999"),
        _tweet(30, "self-thread", reply_to_user=MUSK_ID, reply_to_status="998"),
    ]
    client = FakeTwikitClient([batch])
    tracker = _make_tracker(client)

    result = asyncio.run(tracker.poll_once())

    assert [e.event_id for e in result.events] == ["top-level", "self-thread"]
    assert [e.event_id for e in result.activity_events] == ["reply-other"]


def test_reply_with_unknown_target_is_activity():
    # in_reply_to_status set but user id missing -> conservative: not counted
    batch = [_tweet(10, "mystery-reply", reply_to_status="997")]
    client = FakeTwikitClient([batch])
    tracker = _make_tracker(client)

    result = asyncio.run(tracker.poll_once())

    assert result.events == []
    assert [e.event_id for e in result.activity_events] == ["mystery-reply"]


def test_activity_events_are_deduped_across_polls():
    batch = [_tweet(10, "reply-other", reply_to_user="12345", reply_to_status="999")]
    client = FakeTwikitClient([batch, list(batch)])
    tracker = _make_tracker(client)

    first = asyncio.run(tracker.poll_once())
    second = asyncio.run(tracker.poll_once())

    assert [e.event_id for e in first.activity_events] == ["reply-other"]
    assert second.activity_events == []


def test_tweet_without_legacy_payload_is_kept():
    # Objects without _legacy (e.g. future twikit changes) count as
    # top-level posts rather than being silently dropped
    tweet = SimpleNamespace(
        created_at_datetime=T0 + timedelta(seconds=10),
        id="no-legacy",
        retweeted_tweet=None,
    )
    client = FakeTwikitClient([[tweet]])
    tracker = _make_tracker(client)

    result = asyncio.run(tracker.poll_once())

    assert [e.event_id for e in result.events] == ["no-legacy"]


def test_network_error_does_not_rotate_cookie():
    client = FakeTwikitClient([TimeoutError("connect timeout")])
    tracker = _make_tracker(client)

    result = asyncio.run(tracker.poll_once())

    assert result.events == []
    assert tracker.consecutive_errors == 1
    assert tracker._cookie_index == 0  # no rotation on cookie-agnostic error
    assert client.set_cookie_calls == 0


def test_cookie_error_rotates_cookie():
    from twikit.errors import Unauthorized

    client = FakeTwikitClient([Unauthorized("bad cookie")])
    tracker = _make_tracker(client)

    result = asyncio.run(tracker.poll_once())

    assert result.events == []
    assert tracker.last_cookie_error == "unauthorized"
    assert tracker._cookie_index == 1
    assert client.set_cookie_calls == 1
