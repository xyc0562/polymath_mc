from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from src.algo.musk_tweet_count.forecaster.multi_event_manager import (
    EventInfo,
    FillNotification,
    MultiEventManager,
)
from src.algo.musk_tweet_count.kelly.executor import BalanceAllowanceErrorContext


def _base_health(*, errors_count: int = 0) -> dict:
    return {
        "status": "healthy",
        "uptime_hours": 12.0,
        "active_events": 2,
        "pending_events": 1,
        "available_capital": 250.0,
        "allocated_capital": 750.0,
        "total_pnl": 42.5,
        "errors_count": errors_count,
        "last_data_refresh_source": "xtracker",
        "shared_data_version": 7,
        "user_stream": {
            "connected": True,
            "pending_orders": 1,
            "fill_count": 5,
        },
        "realtime_tracker": {
            "enabled": True,
            "consecutive_errors": 0,
            "backoff_until": None,
            "last_poll_time": "2026-03-06T12:00:00+00:00",
        },
        "active_event_names": ["Event A", "Event B"],
    }


def _make_manager() -> MultiEventManager:
    manager = object.__new__(MultiEventManager)
    manager.slack_notifier = SimpleNamespace(
        health_interval_seconds=0,
        health_on_change_only=True,
        fill_summary_interval_seconds=3600,
        fill_summary_max_examples=2,
        balance_allowance_cooldown_seconds=1800,
    )
    manager._tz = ZoneInfo("UTC")
    manager._last_slack_health_notification = datetime.now(manager._tz)
    manager._last_slack_health_signature = MultiEventManager._health_digest_signature(
        _base_health()
    )
    manager._pending_fill_notifications = []
    manager._first_pending_fill_at = None
    return manager


def test_health_digest_change_only_still_sends_when_interval_zero():
    manager = _make_manager()
    sent = []
    states = iter([
        _base_health(),
        _base_health(errors_count=1),
    ])

    manager.get_health = lambda: next(states)
    manager._notify_slack = lambda level, title, lines=None, **kwargs: sent.append(
        (level, title, lines or [], kwargs)
    )

    manager._maybe_notify_health()
    manager._maybe_notify_health()

    assert len(sent) == 1
    level, title, lines, _ = sent[0]
    assert level == "info"
    assert title == "Health digest"
    assert any("errors=1" in line for line in lines)


def test_fill_summary_batches_multiple_confirmed_fills():
    manager = _make_manager()
    now = datetime.now(manager._tz)
    manager._first_pending_fill_at = now - timedelta(hours=2)
    manager._pending_fill_notifications = [
        FillNotification(
            event_short_name="Event A",
            bin_index=1,
            bin_range="10-19",
            side="BUY",
            status="CONFIRMED",
            size=5.0,
            price=0.30,
            notional=1.50,
            timestamp=now - timedelta(minutes=90),
        ),
        FillNotification(
            event_short_name="Event A",
            bin_index=2,
            bin_range="20-29",
            side="SELL",
            status="CONFIRMED",
            size=4.0,
            price=0.60,
            notional=2.40,
            timestamp=now - timedelta(minutes=60),
        ),
        FillNotification(
            event_short_name="Event B",
            bin_index=0,
            bin_range="0-9",
            side="BUY",
            status="CONFIRMED",
            size=2.0,
            price=0.80,
            notional=1.60,
            timestamp=now - timedelta(minutes=30),
        ),
    ]
    sent = []
    manager._notify_slack = lambda level, title, lines=None, **kwargs: sent.append(
        (level, title, lines or [], kwargs)
    )

    manager._maybe_flush_fill_summaries()

    assert len(sent) == 1
    level, title, lines, _ = sent[0]
    assert level == "info"
    assert title == "Fill summary"
    assert "fills=3" in lines[0]
    assert "total_notional=$5.50" in lines[0]
    assert "Event A: fills=2 notional=$3.90" in lines
    assert "Event B: fills=1 notional=$1.60" in lines
    assert "samples:" in lines
    assert manager._pending_fill_notifications == []
    assert manager._first_pending_fill_at is None


def test_balance_allowance_alert_uses_configured_cooldown():
    manager = _make_manager()
    sent = []
    manager._notify_slack = lambda level, title, lines=None, **kwargs: sent.append(
        (level, title, lines or [], kwargs)
    )

    event_info = EventInfo(
        event_id="event-123",
        title="Event Title",
        short_name="Event A",
        settlement_date=datetime(2026, 3, 10).date(),
        market_start_date=datetime(2026, 3, 3).date(),
        bins=[],
    )
    context = BalanceAllowanceErrorContext(
        event_name="Event A",
        action="SELL_NO",
        side="SELL",
        token_type="NO",
        bin_index=2,
        bin_range="20-29",
        token_id="token-2",
        requested_size=25.0,
        requested_price=0.44,
        requested_limit_price=0.441,
        requested_notional=11.0,
        reservation_price=0.39,
        edge=0.1282,
        utility_gain=0.0123,
        error="not enough balance / allowance",
    )

    manager._notify_balance_allowance_error(event_info, context)

    assert len(sent) == 1
    level, title, _, kwargs = sent[0]
    assert level == "warning"
    assert title == "Balance / allowance rejection: Event A"
    assert kwargs["dedupe_key"] == "balance_allowance:event-123:2:token-2"
    assert kwargs["cooldown_seconds"] == 1800
