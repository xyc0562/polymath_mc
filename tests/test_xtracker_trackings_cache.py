from datetime import datetime, timezone

import requests

from src.algo.musk_tweet_count import musk_tweet_count as mtc
from src.algo.musk_tweet_count.musk_tweet_count import XTrackerClient


class _Resp:
    def __init__(self, trackings):
        self._trackings = trackings

    def raise_for_status(self):
        pass

    def json(self):
        return {"success": True, "data": {"trackings": self._trackings}}


class _Session:
    def __init__(self):
        self.responses = []
        self.calls = 0

    def get(self, url, timeout=None):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return _Resp(r)


def _tracking(tid, start, end):
    return {"id": tid, "startDate": start, "endDate": end}


OLD = _tracking("old", "2026-09-01T16:00:00.000Z", "2026-09-08T15:59:59.000Z")
NEW = _tracking("new", "2026-09-08T16:00:00.000Z", "2026-09-15T15:59:59.000Z")


def _client(monkeypatch, clock):
    monkeypatch.setattr(mtc.time, "monotonic", lambda: clock[0])
    client = XTrackerClient()
    client.session = _Session()
    return client


def test_tracking_created_after_first_fetch_is_found_after_ttl(monkeypatch):
    clock = [1000.0]
    client = _client(monkeypatch, clock)
    client.session.responses = [[OLD], [OLD, NEW]]
    start = datetime(2026, 9, 8, 12, tzinfo=timezone.utc)
    end = datetime(2026, 9, 15, 12, tzinfo=timezone.utc)

    assert client.find_tracking_for_period(start, end) is None

    # Within TTL: served from cache, no refetch.
    clock[0] += XTrackerClient.TRACKINGS_CACHE_TTL_SECONDS - 1
    assert client.find_tracking_for_period(start, end) is None
    assert client.session.calls == 1

    # After TTL: refetched, the newly listed tracking is found.
    clock[0] += 2
    assert client.find_tracking_for_period(start, end)["id"] == "new"
    assert client.session.calls == 2


def test_refresh_failure_keeps_stale_trackings(monkeypatch):
    clock = [1000.0]
    client = _client(monkeypatch, clock)
    client.session.responses = [[OLD], requests.ConnectionError("down")]

    assert client.get_user_trackings() == [OLD]
    clock[0] += XTrackerClient.TRACKINGS_CACHE_TTL_SECONDS + 1
    assert client.get_user_trackings() == [OLD]
    assert client.session.calls == 2
