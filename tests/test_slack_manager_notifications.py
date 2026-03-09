import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from src.algo.musk_tweet_count.forecaster.multi_event_manager import (
    ActiveEvent,
    EventInfo,
    FillNotification,
    MultiEventManager,
)
from src.algo.musk_tweet_count.kelly.capital_pool import CapitalPool, CapitalPoolConfig
from src.algo.musk_tweet_count.kelly.config import KellyConfig
from src.algo.musk_tweet_count.kelly.executor import BalanceAllowanceErrorContext
from src.algo.musk_tweet_count.kelly.orderbook import OrderbookLevel, UnifiedOrderbook
from src.algo.musk_tweet_count.kelly.portfolio import Portfolio
from src.algo.musk_tweet_count.kelly.user_stream import FillEvent, OrderStatus


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
    manager.config = SimpleNamespace(
        dry_run=False,
        max_data_age_seconds=300,
    )
    manager._tz = ZoneInfo("UTC")
    manager._slack_tz = ZoneInfo("Asia/Singapore")
    now = datetime.now(manager._tz)
    manager._last_data_refresh_time = now - timedelta(minutes=1)
    manager._last_data_refresh_source = "xtracker"
    manager._last_xtracker_refresh_time = now - timedelta(minutes=1)
    manager._last_slack_health_notification = now
    manager._last_slack_health_signature = None
    manager._last_slack_health_degraded = None
    manager._pending_fill_notifications = []
    manager._first_pending_fill_at = None
    manager._last_pending_fill_at = None
    manager._seen_fill_notification_keys = {}
    manager._last_integrity_alert_state = {}
    manager._active_events = {}
    manager._pending_events = {}
    manager._completed_events = []
    manager._running = True
    manager._start_time = now - timedelta(hours=12)
    manager._last_cleanup_time = now - timedelta(hours=1)
    manager._errors_count = 0
    manager.shared_data_version = 7
    manager.capital_pool = SimpleNamespace(
        get_summary=lambda: {
            "total_capital": 250.0,
            "available_capital": 50.0,
            "allocated_capital": 200.0,
            "total_value": 250.0,
            "num_active_events": len(manager._active_events),
            "num_settled_events": 0,
            "active_events": {},
            "config": {},
        },
        get_performance_summary=lambda: {
            "num_events": 0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "win_rate": 0.0,
        },
    )
    return manager


def _make_active_event(
    event_id: str,
    short_name: str,
    *,
    allocated_capital: float = 100.0,
    event_budget: float = 100.0,
    frozen: bool = False,
    missing_bid: bool = False,
) -> ActiveEvent:
    portfolio = Portfolio(
        initial_capital=100.0,
        capital=95.08,
        num_bins=2,
        bin_upper_bounds=[9, 19],
        probabilities=[0.35, 0.65],
    )
    pos = portfolio.ensure_position(1, "token-1")
    pos.yes_shares = 12.0
    pos.yes_avg_cost = 0.41
    pos.collateral_used = pos.yes_shares * pos.yes_avg_cost

    yes_bids = [] if missing_bid else [OrderbookLevel(price=0.42, size=40.0)]
    yes_asks = [OrderbookLevel(price=0.45, size=40.0)]
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
            yes_bids=yes_bids,
            yes_asks=yes_asks,
        ),
    }
    orderbook_manager = SimpleNamespace(
        get_orderbook=lambda token_id: orderbooks.get(token_id),
    )
    integrity = {
        "frozen": frozen,
        "reason": "overlay_conflict" if frozen else None,
        "overlay_entries": 1 if frozen else 0,
        "deadline_at": "2026-03-07T01:00:00+00:00" if frozen else None,
        "last_forced_api_recovery_at": None,
        "unmatched_api_delta_count": 0,
    }
    kelly_bot = SimpleNamespace(
        config=SimpleNamespace(
            collateral=SimpleNamespace(c_event_max=event_budget),
        ),
        portfolio=portfolio,
        orderbook_manager=orderbook_manager,
        bin_token_ids={0: "token-0", 1: "token-1"},
    )
    bot = SimpleNamespace(
        kelly_bot=kelly_bot,
        get_state_summary=lambda: {"integrity": integrity},
    )
    info = EventInfo(
        event_id=event_id,
        title=f"{short_name} title",
        short_name=short_name,
        settlement_date=date(2026, 3, 10),
        market_start_date=date(2026, 3, 3),
        bins=[
            {"lower_bound": 0, "upper_bound": 9, "token_id": "token-0", "no_token_id": "no-token-0"},
            {"lower_bound": 10, "upper_bound": 19, "token_id": "token-1", "no_token_id": "no-token-1"},
        ],
    )
    return ActiveEvent(
        info=info,
        bot=bot,
        task=None,
        started_at=datetime(2026, 3, 7, 0, 0, tzinfo=timezone.utc),
        allocated_capital=allocated_capital,
    )


def _base_health(
    manager: MultiEventManager,
    *,
    errors_count: int = 0,
    realtime_errors: int = 0,
    realtime_backoff: str | None = None,
    user_stream_connected: bool = True,
) -> dict:
    capital_snapshot = manager._build_capital_snapshot()
    frozen_events = [event.short_name for event in capital_snapshot.events if event.frozen]
    return {
        "status": "healthy",
        "timestamp": datetime.now(manager._tz).isoformat(),
        "uptime_hours": 12.0,
        "uptime_days": 0.5,
        "active_events": len(manager._active_events),
        "pending_events": 1,
        "completed_events_in_history": 0,
        "total_events_started": len(manager._active_events),
        "total_events_completed": 0,
        "available_capital": 50.0,
        "allocated_capital": 200.0,
        "total_capital": 250.0,
        "total_pnl": 0.0,
        "errors_count": errors_count,
        "last_data_refresh_source": "xtracker",
        "last_data_refresh_time": manager._last_data_refresh_time.isoformat(),
        "last_xtracker_refresh_time": manager._last_xtracker_refresh_time.isoformat(),
        "shared_data_version": 7,
        "capital_snapshot": capital_snapshot,
        "user_stream": {
            "enabled": True,
            "connected": user_stream_connected,
            "pending_orders": 1,
            "fill_count": 5,
            "message_count": 8,
            "last_message_age_seconds": 4.0,
        },
        "realtime_tracker": {
            "enabled": True,
            "consecutive_errors": realtime_errors,
            "backoff_until": realtime_backoff,
            "last_poll_time": "2026-03-06T12:00:00+00:00",
        },
        "active_event_names": [event.short_name for event in capital_snapshot.events],
        "frozen_events": len(frozen_events),
        "frozen_event_names": frozen_events,
    }


def test_fmt_slack_timestamp_uses_sgt():
    manager = _make_manager()

    assert (
        manager._fmt_slack_timestamp("2026-03-07T00:00:00+00:00")
        == "2026-03-07 08:00:00 SGT"
    )


def test_capital_snapshot_uses_best_bid_liquidation():
    manager = _make_manager()
    manager._active_events = {"event-a": _make_active_event("event-a", "Event A")}

    snapshot = manager._build_capital_snapshot()

    assert snapshot.baseline_total == 250.0
    assert snapshot.unallocated_idle == 150.0
    assert snapshot.alloc_budget_total == 100.0
    assert snapshot.event_idle_cash_total == 95.08
    assert round(snapshot.open_cost_total, 2) == 4.92
    assert round(snapshot.open_liq_total, 2) == 5.04
    assert round(snapshot.open_pnl_total, 2) == 0.12
    assert round(snapshot.asset_now_total, 2) == 250.12
    assert snapshot.events[0].short_name == "Event A"


def test_capital_snapshot_uses_effective_event_budget_not_restored_basis():
    manager = _make_manager()
    manager._active_events = {
        "event-a": _make_active_event(
            "event-a",
            "Event A",
            allocated_capital=32.0,
            event_budget=100.0,
        )
    }

    snapshot = manager._build_capital_snapshot()

    assert snapshot.alloc_budget_total == 100.0
    assert snapshot.unallocated_idle == 150.0
    assert round(snapshot.event_idle_cash_total, 2) == 95.08
    assert round(snapshot.asset_now_total, 2) == 250.12
    assert snapshot.events[0].alloc_budget == 100.0
    assert round(snapshot.events[0].idle_cash, 2) == 95.08


def test_startup_digest_uses_capital_rollup():
    manager = _make_manager()
    manager._active_events = {"event-a": _make_active_event("event-a", "Event A")}
    sent = []
    manager.get_health = lambda: _base_health(manager)
    manager._notify_slack = lambda level, title, lines=None, **kwargs: sent.append(
        (level, title, lines or [], kwargs)
    )

    manager._notify_startup()

    assert len(sent) == 1
    level, title, lines, _ = sent[0]
    assert level == "info"
    assert title == "Startup digest"
    assert any("subsystems:" in line for line in lines)
    assert any("capital: baseline=$250.00 alloc=$100.00" in line for line in lines)
    assert any("asset_now=$250.12" in line for line in lines)


def test_event_capital_line_stays_self_consistent_when_restored_basis_is_smaller():
    manager = _make_manager()
    manager._active_events = {
        "event-a": _make_active_event(
            "event-a",
            "Event A",
            allocated_capital=32.0,
            event_budget=100.0,
        )
    }

    lines = manager._format_event_capital_lines(manager._build_capital_snapshot().events)

    assert any(
        "Event A | alloc=$100.00 idle=$95.08 cost=$4.92 liq=$5.04 pnl=$0.12 asset=$100.12"
        in line
        for line in lines
    )


def test_health_digest_is_clamped_when_healthy():
    manager = _make_manager()
    manager._active_events = {"event-a": _make_active_event("event-a", "Event A")}
    sent = []
    healthy = _base_health(manager)
    manager.get_health = lambda: healthy
    manager._notify_slack = lambda level, title, lines=None, **kwargs: sent.append(
        (level, title, lines or [], kwargs)
    )
    manager._last_slack_health_signature = MultiEventManager._health_digest_signature(healthy)
    manager._last_slack_health_notification = datetime.now(manager._tz) - timedelta(hours=1)
    manager._last_slack_health_degraded = False

    manager._maybe_notify_health()

    assert sent == []

    manager._last_slack_health_notification = datetime.now(manager._tz) - timedelta(hours=7)
    manager._maybe_notify_health()

    assert len(sent) == 1
    assert sent[0][1] == "Health digest"
    assert any("capital: baseline=$250.00 alloc=$100.00" in line for line in sent[0][2])


def test_health_digest_sends_immediately_on_degraded_change():
    manager = _make_manager()
    manager._active_events = {"event-a": _make_active_event("event-a", "Event A")}
    sent = []
    healthy = _base_health(manager)
    degraded = _base_health(manager, realtime_errors=3)
    manager._notify_slack = lambda level, title, lines=None, **kwargs: sent.append(
        (level, title, lines or [], kwargs)
    )
    manager._last_slack_health_signature = MultiEventManager._health_digest_signature(healthy)
    manager._last_slack_health_notification = datetime.now(manager._tz)
    manager._last_slack_health_degraded = False
    manager.get_health = lambda: degraded

    manager._maybe_notify_health()

    assert len(sent) == 1
    assert any("realtime=degraded" in line for line in sent[0][2])


def test_fill_digest_batches_by_event_without_samples():
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
            token_type="YES",
        ),
        FillNotification(
            event_id="event-a",
            event_short_name="Event A",
            order_id="order-a2",
            match_id="match-a2",
            bin_index=1,
            bin_range="10-19",
            side="SELL",
            status="CONFIRMED",
            size=4.0,
            price=0.60,
            notional=2.40,
            timestamp=now - timedelta(minutes=18),
            token_type="YES",
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
            token_type="NO",
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
    assert title == "Fill digest"
    assert "gross_buy=$3.10" in lines[0]
    assert "gross_sell=$2.40" in lines[0]
    assert "net_signed=$0.70" in lines[0]
    assert any("Event A | fills=2 buy=$1.50 sell=$2.40 net=$-0.90" in line for line in lines)
    assert any("Event B | fills=1 buy=$1.60 sell=$0.00 net=$1.60" in line for line in lines)
    assert "samples:" not in lines
    assert not any("holdings:" in line for line in lines)
    assert not any("forecast_as_of" in line for line in lines)
    assert manager._pending_fill_notifications == []


def test_notify_fill_dedupes_repeated_confirmed_fill():
    manager = _make_manager()
    active = _make_active_event("event-a", "Event A")
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

    manager._notify_fill(active.info, 1, fill, bin_range="10-19")
    manager._notify_fill(active.info, 1, fill, bin_range="10-19")

    assert len(manager._pending_fill_notifications) == 1
    assert manager._pending_fill_notifications[0].token_type == "YES"


def test_format_exception_details_include_type_and_location():
    manager = _make_manager()

    try:
        raise KeyError("range")
    except KeyError as exc:
        lines = manager._format_exception_details(exc)

    assert lines[0] == "KeyError: 'range'"
    assert any("in test_format_exception_details_include_type_and_location" in line for line in lines[1:])


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
    level, title, lines, kwargs = sent[0]
    assert level == "warning"
    assert title == "Balance / allowance rejection: Event A"
    assert kwargs["dedupe_key"] == "balance_allowance:event-123:2:token-2"
    assert kwargs["cooldown_seconds"] == 1800
    assert any("impact=order not sent or not fully executable" in line for line in lines)
    assert any("action=inspect balance, allowances, and recent fills before retrying" in line for line in lines)


def test_capital_pool_logs_use_explicit_terms(caplog):
    async def scenario():
        pool = CapitalPool(CapitalPoolConfig(total_capital=500.0), max_per_event=200.0)
        await pool.request_capital("event-a", 150.0)
        await pool.return_capital("event-a", 175.0)
        await pool.restore_allocation("event-b", 80.0)
        await pool.set_total_from_api(520.0)

    with caplog.at_level(logging.INFO):
        asyncio.run(scenario())

    messages = [record.message for record in caplog.records]
    assert any("[CAPITAL][ALLOCATE] event=event-a" in message for message in messages)
    assert any("[CAPITAL][RELEASE] event=event-a" in message for message in messages)
    assert any("[CAPITAL][RESTORE] event=event-b" in message for message in messages)
    assert any("[CAPITAL][SYNC_API] baseline_total=$520.00" in message for message in messages)
