from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from src.algo.musk_tweet_count.forecaster.multi_event_manager import (
    ActiveEvent,
    EventInfo,
    FillNotification,
    MultiEventManager,
)
from src.algo.musk_tweet_count.kelly.config import KellyConfig
from src.algo.musk_tweet_count.kelly.executor import BalanceAllowanceErrorContext
from src.algo.musk_tweet_count.kelly.orderbook import OrderbookLevel, UnifiedOrderbook
from src.algo.musk_tweet_count.kelly.portfolio import Portfolio
from src.algo.musk_tweet_count.kelly.user_stream import FillEvent, OrderStatus


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
        fill_summary_quiet_seconds=300,
        fill_summary_max_examples=2,
        balance_allowance_cooldown_seconds=1800,
    )
    manager.kelly_config = KellyConfig()
    manager._tz = ZoneInfo("UTC")
    manager._slack_tz = ZoneInfo("Asia/Singapore")
    manager._last_slack_health_notification = datetime.now(manager._tz)
    manager._last_slack_health_signature = MultiEventManager._health_digest_signature(
        _base_health()
    )
    manager._pending_fill_notifications = []
    manager._first_pending_fill_at = None
    manager._last_pending_fill_at = None
    manager._seen_fill_notification_keys = {}
    manager._active_events = {}
    return manager


def _make_active_event(event_id: str, short_name: str) -> ActiveEvent:
    portfolio = Portfolio(
        initial_capital=100.0,
        capital=82.0,
        num_bins=2,
        bin_upper_bounds=[9, 19],
        probabilities=[0.35, 0.65],
    )
    pos = portfolio.ensure_position(1, "token-1")
    pos.yes_shares = 12.0
    pos.yes_avg_cost = 0.41
    pos.collateral_used = pos.yes_shares * pos.yes_avg_cost

    orderbooks = {
        "token-0": UnifiedOrderbook(
            bin_index=0,
            yes_token_id="token-0",
            yes_bids=[OrderbookLevel(price=0.18, size=50.0)],
            yes_asks=[OrderbookLevel(price=0.22, size=50.0)],
        ),
        "token-1": UnifiedOrderbook(
            bin_index=1,
            yes_token_id="token-1",
            yes_bids=[OrderbookLevel(price=0.42, size=40.0)],
            yes_asks=[OrderbookLevel(price=0.45, size=40.0)],
        ),
    }
    orderbook_manager = SimpleNamespace(
        get_orderbook=lambda token_id: orderbooks.get(token_id),
    )
    kelly_bot = SimpleNamespace(
        portfolio=portfolio,
        orderbook_manager=orderbook_manager,
        bin_token_ids={0: "token-0", 1: "token-1"},
    )
    bot = SimpleNamespace(
        kelly_bot=kelly_bot,
        _market_bins=[(0, 9), (10, 19)],
        _cached_probabilities=[0.35, 0.65],
        _cached_forecast_mean=16.0,
        _cached_forecast_std=3.0,
        _cached_forecast_breakdown={"past_count": 12, "past_days": 6, "remaining_days": 1},
        _cached_forecast_time=datetime(2026, 3, 7, 0, 0, tzinfo=timezone.utc),
        _get_market_cumulative_count=lambda: 12,
        _last_known_count=12,
        get_effective_count_for_dead_bins=lambda count: count,
        _get_timing=lambda: (120.0, 24.0),
    )
    info = EventInfo(
        event_id=event_id,
        title=f"{short_name} title",
        short_name=short_name,
        settlement_date=date(2026, 3, 10),
        market_start_date=date(2026, 3, 3),
        bins=[],
    )
    return ActiveEvent(
        info=info,
        bot=bot,
        task=None,
        started_at=datetime(2026, 3, 7, 0, 0, tzinfo=timezone.utc),
        allocated_capital=100.0,
    )


def test_health_digest_ignores_error_counter_increments_after_first_error():
    manager = _make_manager()
    sent = []
    manager._last_slack_health_signature = MultiEventManager._health_digest_signature(
        _base_health(errors_count=1)
    )

    manager.get_health = lambda: _base_health(errors_count=2)
    manager._notify_slack = lambda level, title, lines=None, **kwargs: sent.append(
        (level, title, lines or [], kwargs)
    )

    manager._maybe_notify_health()

    assert sent == []


def test_fmt_slack_timestamp_uses_sgt():
    manager = _make_manager()

    assert (
        manager._fmt_slack_timestamp("2026-03-07T00:00:00+00:00")
        == "2026-03-07 08:00:00 SGT"
    )


def test_fill_summary_batches_multiple_confirmed_fills():
    manager = _make_manager()
    now = datetime.now(manager._tz)
    manager._first_pending_fill_at = now - timedelta(minutes=25)
    manager._last_pending_fill_at = now - timedelta(minutes=15)
    manager._active_events = {
        "event-a": _make_active_event("event-a", "Event A"),
        "event-b": _make_active_event("event-b", "Event B"),
    }
    manager._pending_fill_notifications = [
        FillNotification(
            event_id="event-a",
            event_short_name="Event A",
            order_id="order-a1",
            match_id="match-a1",
            bin_index=1,
            bin_range="10-19",
            side="BUY",
            status="CONFIRMED",
            size=5.0,
            price=0.30,
            notional=1.50,
            timestamp=now - timedelta(minutes=20),
        ),
        FillNotification(
            event_id="event-a",
            event_short_name="Event A",
            order_id="order-a2",
            match_id="match-a2",
            bin_index=2,
            bin_range="20-29",
            side="SELL",
            status="CONFIRMED",
            size=4.0,
            price=0.60,
            notional=2.40,
            timestamp=now - timedelta(minutes=18),
        ),
        FillNotification(
            event_id="event-b",
            event_short_name="Event B",
            order_id="order-b1",
            match_id="match-b1",
            bin_index=0,
            bin_range="0-9",
            side="BUY",
            status="CONFIRMED",
            size=2.0,
            price=0.80,
            notional=1.60,
            timestamp=now - timedelta(minutes=16),
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
    assert "SGT" in lines[0]
    assert "total_notional=$5.50" in lines[0]
    assert "Event A: fills=2 notional=$3.90" in lines
    assert "Event B: fills=1 notional=$1.60" in lines
    assert "samples:" in lines
    assert any("holdings: capital=$82.00" in line for line in lines)
    assert any("Range     Holdings" in line for line in lines)
    assert any("forecast_as_of=2026-03-07 08:00:00 SGT" in line for line in lines)
    assert any("Bin  Range" in line for line in lines)
    assert manager._pending_fill_notifications == []
    assert manager._first_pending_fill_at is None
    assert manager._last_pending_fill_at is None


def test_notify_fill_dedupes_repeated_confirmed_fill():
    manager = _make_manager()
    event_info = EventInfo(
        event_id="event-a",
        title="Event Title",
        short_name="Event A",
        settlement_date=date(2026, 3, 10),
        market_start_date=date(2026, 3, 3),
        bins=[],
    )
    fill = FillEvent(
        order_id="order-1",
        token_id="token-1",
        side="BUY",
        price=0.42,
        size=10.0,
        status=OrderStatus.CONFIRMED,
        timestamp=datetime(2026, 3, 7, 0, 0, tzinfo=timezone.utc),
        match_id="match-1",
    )

    manager._notify_fill(event_info, 1, fill, bin_range="10-19")
    manager._notify_fill(event_info, 1, fill, bin_range="10-19")

    assert len(manager._pending_fill_notifications) == 1


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
