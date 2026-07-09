import asyncio
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import requests

from src.algo.musk_tweet_count.forecaster.data import XTrackerClient
from src.algo.musk_tweet_count.forecaster.multi_event_manager import (
    EventInfo,
    MultiEventManager,
    RefreshOutcome,
)


ET = ZoneInfo("America/New_York")


def _make_event_info(settlement_date: date) -> EventInfo:
    return EventInfo(
        event_id="123",
        title="Elon Musk # of tweets test",
        short_name="Test Event",
        settlement_date=settlement_date,
        market_start_date=settlement_date - timedelta(days=7),
        bins=[],
    )


# ---------- XTrackerClient.last_fetch_failed ----------


def test_fetch_all_posts_network_error_sets_last_fetch_failed(monkeypatch):
    client = XTrackerClient()

    def raise_timeout(*args, **kwargs):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(client.session, "get", raise_timeout)

    posts = client.fetch_all_posts(date(2026, 7, 1), date(2026, 7, 2))

    assert posts == []
    assert client.last_fetch_failed is True


def test_fetch_all_posts_success_false_sets_last_fetch_failed(monkeypatch):
    client = XTrackerClient()
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"success": False},
    )
    monkeypatch.setattr(client.session, "get", lambda *a, **k: response)

    posts = client.fetch_all_posts(date(2026, 7, 1), date(2026, 7, 2))

    assert posts == []
    assert client.last_fetch_failed is True


def test_fetch_all_posts_success_clears_last_fetch_failed(monkeypatch):
    client = XTrackerClient()
    client.last_fetch_failed = True
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"success": True, "data": []},
    )
    monkeypatch.setattr(client.session, "get", lambda *a, **k: response)

    posts = client.fetch_all_posts(date(2026, 7, 1), date(2026, 7, 2))

    assert posts == []
    assert client.last_fetch_failed is False


def test_refresh_outcome_defaults_fetch_failed_false():
    assert RefreshOutcome().fetch_failed is False


# ---------- health degraded includes orderbook WS ----------


def _healthy_report(orderbook_connected: bool) -> dict:
    return {
        "status": "healthy",
        "frozen_events": 0,
        "errors_count": 0,
        "user_stream": {"enabled": True, "connected": True},
        "orderbook_ws": {"enabled": True, "connected": orderbook_connected},
        "realtime_tracker": {"enabled": False},
    }


def _health_stub() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(dry_run=False),
        is_data_fresh=lambda: (True, ""),
    )


def test_health_degraded_when_orderbook_ws_disconnected():
    stub = _health_stub()

    degraded = MultiEventManager._is_health_degraded(stub, _healthy_report(False))

    assert degraded is True


def test_health_not_degraded_when_orderbook_ws_connected():
    stub = _health_stub()

    degraded = MultiEventManager._is_health_degraded(stub, _healthy_report(True))

    assert degraded is False


# ---------- fetch_all_positions valuation + pagination ----------


def _positions_page(items):
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: items)


def test_fetch_all_positions_values_resolved_losers_at_zero(monkeypatch):
    from src.algo.musk_tweet_count.forecaster import multi_event_manager as mem

    pages = [
        _positions_page(
            [
                # Live position: currentValue trusted
                {"asset": "tok_live", "size": 100.0, "avgPrice": 0.40,
                 "initialValue": 40.0, "currentValue": 50.0},
                # Resolved loser: currentValue == 0 must be trusted, not cost
                {"asset": "tok_dead", "size": 8650.0, "avgPrice": 0.0458,
                 "initialValue": 396.98, "currentValue": 0},
                # Missing currentValue: fall back to cost
                {"asset": "tok_nocur", "size": 10.0, "avgPrice": 0.25,
                 "initialValue": 2.5},
            ]
        )
    ]
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params)
        return pages.pop(0)

    monkeypatch.setattr(mem.requests, "get", fake_get)
    stub = SimpleNamespace(wallet_address="0xABCDEF1234567890")

    positions = asyncio.run(MultiEventManager.fetch_all_positions(stub))

    assert positions["tok_live"]["current_value"] == 50.0
    assert positions["tok_live"]["cost_basis"] == 40.0
    assert positions["tok_dead"]["current_value"] == 0.0
    assert positions["tok_dead"]["cost_basis"] == 396.98
    assert positions["tok_nocur"]["current_value"] == 2.5
    assert len(calls) == 1


def test_fetch_all_positions_paginates(monkeypatch):
    from src.algo.musk_tweet_count.forecaster import multi_event_manager as mem

    full_page = [
        {"asset": f"tok_{i}", "size": 1.0, "avgPrice": 0.5,
         "initialValue": 0.5, "currentValue": 0.5}
        for i in range(500)
    ]
    second_page = [
        {"asset": "tok_last", "size": 1.0, "avgPrice": 0.5,
         "initialValue": 0.5, "currentValue": 0.5}
    ]
    pages = [_positions_page(full_page), _positions_page(second_page)]
    offsets = []

    def fake_get(url, params=None, timeout=None):
        offsets.append(params["offset"])
        return pages.pop(0)

    monkeypatch.setattr(mem.requests, "get", fake_get)
    stub = SimpleNamespace(wallet_address="0xABCDEF1234567890")

    positions = asyncio.run(MultiEventManager.fetch_all_positions(stub))

    assert offsets == [0, 500]
    assert len(positions) == 501


# ---------- settlement P&L: settled-guard + posts fallback + return value ----------


def _settlement_stub(now_offset_days: int, trackings_count, posts_count=None):
    """Build a stub manager whose 'now' is settlement_date + offset."""
    calls = {"trackings": 0, "posts": 0, "breakdown": 0}

    async def get_authoritative_count(event_info):
        calls["trackings"] += 1
        return trackings_count, None

    def compute_count_from_posts(event_info):
        calls["posts"] += 1
        return posts_count

    def _compute_settlement_pnl_breakdown(event_info, bot, actual_count):
        calls["breakdown"] += 1
        return -123.45, []

    stub = SimpleNamespace(
        _tz=ET,
        get_authoritative_count=get_authoritative_count,
        compute_count_from_posts=compute_count_from_posts,
        _compute_settlement_pnl_breakdown=_compute_settlement_pnl_breakdown,
        _fmt_usd=lambda v: f"${v:.2f}",
        _calls=calls,
    )
    return stub


def test_settlement_pnl_skipped_before_settlement():
    # Settlement is tomorrow: cleanup (e.g. shutdown) must not log settlement P&L
    tomorrow = datetime.now(ET).date() + timedelta(days=2)
    stub = _settlement_stub(0, trackings_count=100)
    event_info = _make_event_info(tomorrow)

    result = asyncio.run(
        MultiEventManager._log_settlement_pnl(stub, "123", event_info, None, 3000.0)
    )

    assert result is None
    assert stub._calls["trackings"] == 0


def test_settlement_pnl_uses_trackings_count_when_available():
    yesterday = datetime.now(ET).date() - timedelta(days=1)
    stub = _settlement_stub(0, trackings_count=222)
    event_info = _make_event_info(yesterday)

    result = asyncio.run(
        MultiEventManager._log_settlement_pnl(stub, "123", event_info, None, 3000.0)
    )

    assert result == -123.45
    assert stub._calls["posts"] == 0


def test_settlement_pnl_falls_back_to_posts_store():
    yesterday = datetime.now(ET).date() - timedelta(days=1)
    stub = _settlement_stub(0, trackings_count=None, posts_count=222)
    event_info = _make_event_info(yesterday)

    result = asyncio.run(
        MultiEventManager._log_settlement_pnl(stub, "123", event_info, None, 3000.0)
    )

    assert result == -123.45
    assert stub._calls["posts"] == 1


def test_settlement_pnl_skipped_when_no_usable_count():
    yesterday = datetime.now(ET).date() - timedelta(days=1)
    stub = _settlement_stub(0, trackings_count=None, posts_count=0)
    event_info = _make_event_info(yesterday)

    result = asyncio.run(
        MultiEventManager._log_settlement_pnl(stub, "123", event_info, None, 3000.0)
    )

    assert result is None
    assert stub._calls["breakdown"] == 0
