import asyncio
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Optional
from zoneinfo import ZoneInfo

from src.algo.musk_tweet_count.forecaster.config import ForecasterConfig
from src.algo.musk_tweet_count.forecaster.data import ContractDayUtils, EventStore, TweetEvent
from src.algo.musk_tweet_count.forecaster.multi_event_manager import (
    ActiveEvent,
    EventInfo,
    MultiEventManager,
)
from src.algo.musk_tweet_count.forecaster.trading_bot import (
    GASKellyTradingBot,
    TickRequest,
    TradingBotConfig,
)
from src.algo.musk_tweet_count.kelly.config import KellyConfig
from src.algo.musk_tweet_count.kelly.executor import TickResult


class DummyXTrackerClient:
    def fetch_historical(self, n_days, contract_utils, handle="elonmusk"):
        return {}


class DummyKellyBot:
    def __init__(self):
        self.orderbook_manager = None
        self.portfolio = None

    async def run_tick(self, **kwargs):
        return TickResult(
            num_candidates=0,
            num_executed=0,
            total_utility_gain=0.0,
            executions=[],
            elapsed_seconds=0.0,
        )


def _make_store() -> tuple[ContractDayUtils, EventStore]:
    contract_utils = ContractDayUtils()
    event_store = EventStore(contract_utils, DummyXTrackerClient())
    return contract_utils, event_store


def _make_event(
    contract_utils: ContractDayUtils,
    contract_day: date,
    hours_after_noon: int,
    event_id: str,
    *,
    source: str,
) -> TweetEvent:
    start_dt, _ = contract_utils.get_contract_day_bounds(contract_day)
    return TweetEvent(
        timestamp=start_dt + timedelta(hours=hours_after_noon),
        event_type="tweet",
        event_id=event_id,
        source=source,
    )


def _make_bot() -> GASKellyTradingBot:
    _, event_store = _make_store()
    return GASKellyTradingBot(
        clob_client=SimpleNamespace(),
        kelly_config=KellyConfig(),
        forecaster_config=ForecasterConfig(),
        bot_config=TradingBotConfig(sync_driven=True, dry_run=True),
        event_store=event_store,
    )


def test_event_store_merged_view_prefers_official_ids():
    contract_utils, event_store = _make_store()
    contract_day = date(2026, 1, 10)

    official = _make_event(contract_utils, contract_day, 1, "tweet-1", source="xtracker")
    provisional_dup = _make_event(contract_utils, contract_day, 2, "tweet-1", source="twikit")
    provisional_new = _make_event(contract_utils, contract_day, 3, "tweet-2", source="twikit")

    assert event_store.add_event(official) is True
    assert event_store.add_provisional_event(provisional_dup) is False
    assert event_store.add_provisional_event(provisional_new) is True

    merged = event_store.get_contract_day_events(contract_day)
    assert [event.event_id for event in merged] == ["tweet-1", "tweet-2"]
    assert [event.source for event in merged] == ["xtracker", "twikit"]
    assert event_store.get_contract_day_count(contract_day) == 2


def test_authoritative_replace_plus_overlay_clear_rebases_effective_count():
    contract_utils, event_store = _make_store()
    contract_day = date(2026, 1, 11)

    official = _make_event(contract_utils, contract_day, 1, "official-1", source="xtracker")
    provisional = _make_event(contract_utils, contract_day, 2, "prov-2", source="twikit")
    replacement = _make_event(contract_utils, contract_day, 1, "official-1", source="xtracker")

    assert event_store.add_event(official) is True
    assert event_store.add_provisional_event(provisional) is True
    assert event_store.get_contract_day_count(contract_day) == 2

    assert event_store.replace_official_day(contract_day, [replacement]) is False
    assert event_store.clear_provisional_day(contract_day) is True
    assert event_store.get_contract_day_count(contract_day) == 1
    assert [event.event_id for event in event_store.get_contract_day_events(contract_day)] == [
        "official-1"
    ]


def test_sync_tick_requests_coalesce_to_latest_version_and_highest_priority(monkeypatch):
    bot = _make_bot()
    calls = []

    async def fake_execute_tick(*, log_header, recompute, allow_authoritative_rebase=False):
        calls.append(
            {
                "log_header": log_header,
                "recompute": recompute,
                "allow_authoritative_rebase": allow_authoritative_rebase,
            }
        )
        return TickResult(
            num_candidates=0,
            num_executed=0,
            total_utility_gain=0.0,
            executions=[],
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(bot, "_execute_tick", fake_execute_tick)

    bot.request_sync_tick("realtime", 10)
    bot.request_sync_tick("realtime", 11)
    bot.request_sync_tick("xtracker", 10)

    request = bot._pop_pending_sync_tick()
    assert request is not None
    assert request.data_version == 11
    assert request.source == "xtracker"

    asyncio.run(bot._execute_sync_tick_request(request))

    assert calls == [
        {
            "log_header": True,
            "recompute": True,
            "allow_authoritative_rebase": False,
        }
    ]
    assert bot._last_processed_data_version == 11
    assert bot._last_processed_tick_source == "xtracker"


def test_xtracker_noop_cycles_reuse_data_version_but_still_run_once_per_sequence(monkeypatch):
    bot = _make_bot()
    calls = []

    async def fake_execute_tick(*, log_header, recompute, allow_authoritative_rebase=False):
        calls.append((log_header, recompute, allow_authoritative_rebase))
        return TickResult(
            num_candidates=0,
            num_executed=0,
            total_utility_gain=0.0,
            executions=[],
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(bot, "_execute_tick", fake_execute_tick)

    asyncio.run(bot._execute_sync_tick_request(TickRequest("xtracker", 31, 100)))
    asyncio.run(bot._execute_sync_tick_request(TickRequest("xtracker", 31, 100)))
    asyncio.run(bot._execute_sync_tick_request(TickRequest("xtracker", 31, 101)))

    assert calls == [
        (True, True, False),
        (True, True, False),
    ]
    assert bot._last_processed_data_version == 31
    assert bot._last_processed_sync_tick_request_sequence == 101
    assert bot._last_processed_tick_source == "xtracker"


def test_manager_can_queue_xtracker_ticks_for_all_active_events_on_noop_refresh():
    manager = object.__new__(MultiEventManager)
    manager.shared_data_version = 31
    manager._sync_tick_request_sequence = 40
    manager._events_lock = asyncio.Lock()

    async def fake_wait_for_stabilization(event_id, active):
        return True

    manager._maybe_wait_for_event_stabilization = fake_wait_for_stabilization

    class FakeBot:
        def __init__(self):
            self.requests = []

        def request_sync_tick(self, **kwargs):
            self.requests.append(kwargs)

    start_day = date(2026, 3, 1)
    settlement_day = date(2026, 3, 8)
    info_a = EventInfo(
        event_id="event-a",
        title="Event A",
        short_name="Mar 01 - Mar 08",
        settlement_date=settlement_day,
        market_start_date=start_day,
        bins=[],
    )
    info_b = EventInfo(
        event_id="event-b",
        title="Event B",
        short_name="Mar 02 - Mar 09",
        settlement_date=date(2026, 3, 9),
        market_start_date=date(2026, 3, 2),
        bins=[],
    )
    bot_a = FakeBot()
    bot_b = FakeBot()
    manager._active_events = {
        "event-a": ActiveEvent(
            info=info_a,
            bot=bot_a,
            task=SimpleNamespace(),
            started_at=datetime.now(UTC),
            allocated_capital=100.0,
        ),
        "event-b": ActiveEvent(
            info=info_b,
            bot=bot_b,
            task=SimpleNamespace(),
            started_at=datetime.now(UTC),
            allocated_capital=100.0,
        ),
    }

    asyncio.run(
        manager._queue_sync_tick_requests(
            "xtracker",
            set(),
            queue_all_active=True,
        )
    )

    expected = {
        "source": "xtracker",
        "data_version": 31,
        "request_sequence": 41,
        "allow_authoritative_rebase": False,
    }
    assert bot_a.requests == [expected]
    assert bot_b.requests == [expected]


def test_authoritative_rebase_resets_regression_baseline():
    bot = _make_bot()
    bot.kelly_bot = DummyKellyBot()
    bot._cached_forecast_mean = 12.0
    bot._cached_forecast_std = 3.0
    bot._data_freshness_checker = lambda: (True, 0.0)
    bot._last_known_count = 9
    bot._get_timing = lambda: (24.0, 12.0)
    bot._get_market_cumulative_count = lambda: 7

    result = asyncio.run(bot._run_tick_impl(log_header=False, allow_authoritative_rebase=True))

    assert result is not None
    assert bot._last_known_count == 7


# ---------------------------------------------------------------------------
# Discovery reconciliation / phantom-event eviction
# ---------------------------------------------------------------------------


def _make_phantom_manager() -> MultiEventManager:
    async def _no_allocation(event_id):
        return None

    manager = object.__new__(MultiEventManager)
    manager._events_lock = asyncio.Lock()
    manager._active_events = {}
    manager._pending_events = {}
    manager._delisted_miss_count = {}
    manager._evicted_event_ids = set()
    manager.capital_pool = SimpleNamespace(get_allocation=_no_allocation)
    return manager


def _make_active_event(
    event_id: str,
    short_name: str = "Mar 01 - Mar 08",
    *,
    yes_shares: float = 0.0,
    no_shares: float = 0.0,
    pending_count: int = 0,
    cancel_recorder: Optional[list] = None,
) -> ActiveEvent:
    from src.algo.musk_tweet_count.kelly.portfolio import BinPosition

    info = EventInfo(
        event_id=event_id,
        title=f"Event {event_id}",
        short_name=short_name,
        settlement_date=date(2026, 3, 8),
        market_start_date=date(2026, 3, 1),
        bins=[],
    )

    portfolio = SimpleNamespace(positions={})
    if yes_shares > 0 or no_shares > 0:
        portfolio.positions[0] = BinPosition(
            bin_index=0,
            yes_token_id="tok",
            yes_shares=yes_shares,
            no_shares=no_shares,
        )

    executor = SimpleNamespace(get_pending_count=lambda count=pending_count: count)
    kelly_bot = SimpleNamespace(portfolio=portfolio, kelly_executor=executor)
    bot = SimpleNamespace(kelly_bot=kelly_bot)

    class FakeTask:
        def __init__(self, recorder):
            self._recorder = recorder

        def cancel(self):
            if self._recorder is not None:
                self._recorder.append("cancelled")

    return ActiveEvent(
        info=info,
        bot=bot,
        task=FakeTask(cancel_recorder),
        started_at=datetime.now(UTC),
        allocated_capital=1000.0,
    )


def test_phantom_event_zero_exposure_evicted_after_two_misses():
    manager = _make_phantom_manager()
    cancels: list = []
    manager._active_events = {
        "phantom": _make_active_event("phantom", cancel_recorder=cancels),
        "live": _make_active_event("live"),
    }

    # First discovery: phantom missing — first miss, no eviction yet.
    asyncio.run(manager._reconcile_against_discovery({"live"}))
    assert cancels == []
    assert manager._delisted_miss_count["phantom"] == 1

    # Second discovery: phantom still missing — eviction fires.
    asyncio.run(manager._reconcile_against_discovery({"live"}))
    assert cancels == ["cancelled"]
    # Miss counter is cleared on eviction so a future re-list starts fresh.
    assert "phantom" not in manager._delisted_miss_count


def test_phantom_miss_counter_resets_when_event_reappears():
    manager = _make_phantom_manager()
    cancels: list = []
    manager._active_events = {
        "intermittent": _make_active_event("intermittent", cancel_recorder=cancels),
    }

    # Miss 1
    asyncio.run(manager._reconcile_against_discovery({"other"}))
    assert manager._delisted_miss_count["intermittent"] == 1

    # Event reappears in next discovery — counter must reset, no eviction.
    asyncio.run(manager._reconcile_against_discovery({"intermittent"}))
    assert "intermittent" not in manager._delisted_miss_count
    assert cancels == []

    # New first miss; still no eviction (only 1 miss).
    asyncio.run(manager._reconcile_against_discovery({"other"}))
    assert manager._delisted_miss_count["intermittent"] == 1
    assert cancels == []


def test_phantom_with_positions_is_never_evicted():
    manager = _make_phantom_manager()
    cancels: list = []
    manager._active_events = {
        "held": _make_active_event(
            "held",
            yes_shares=100.0,
            cancel_recorder=cancels,
        ),
    }

    # Many consecutive misses; positions block eviction every time.
    for _ in range(5):
        asyncio.run(manager._reconcile_against_discovery({"other"}))

    assert cancels == []
    assert manager._delisted_miss_count["held"] == 5


def test_phantom_with_pending_orders_is_never_evicted():
    manager = _make_phantom_manager()
    cancels: list = []
    manager._active_events = {
        "live_orders": _make_active_event(
            "live_orders",
            pending_count=3,
            cancel_recorder=cancels,
        ),
    }

    for _ in range(3):
        asyncio.run(manager._reconcile_against_discovery({"other"}))

    assert cancels == []


def test_pending_event_missing_from_discovery_dropped_after_two_misses():
    manager = _make_phantom_manager()
    pending_info = EventInfo(
        event_id="pending-1",
        title="Pending",
        short_name="Mar 02 - Mar 09",
        settlement_date=date(2026, 3, 9),
        market_start_date=date(2026, 3, 2),
        bins=[],
    )
    manager._pending_events = {"pending-1": pending_info}

    # First miss: deferred (same threshold as active events — a single
    # transient Gamma blip must not drop a pending event).
    asyncio.run(manager._reconcile_against_discovery({"some-other-event"}))
    assert "pending-1" in manager._pending_events

    # Second consecutive miss: dropped (no capital allocation held).
    asyncio.run(manager._reconcile_against_discovery({"some-other-event"}))
    assert "pending-1" not in manager._pending_events


def test_reconciliation_no_op_when_all_events_present():
    manager = _make_phantom_manager()
    cancels: list = []
    manager._active_events = {
        "a": _make_active_event("a", cancel_recorder=cancels),
        "b": _make_active_event("b", cancel_recorder=cancels),
    }

    asyncio.run(manager._reconcile_against_discovery({"a", "b"}))

    assert cancels == []
    assert manager._delisted_miss_count == {}


def test_no_exposure_helper_treats_missing_portfolio_as_evictable():
    info = EventInfo(
        event_id="nascent",
        title="Nascent",
        short_name="Mar 03 - Mar 10",
        settlement_date=date(2026, 3, 10),
        market_start_date=date(2026, 3, 3),
        bins=[],
    )
    bot = SimpleNamespace(kelly_bot=None)
    active = ActiveEvent(
        info=info,
        bot=bot,
        task=SimpleNamespace(),
        started_at=datetime.now(UTC),
        allocated_capital=0.0,
    )

    assert MultiEventManager._event_has_no_exposure(active) is True


# ---------------------------------------------------------------------------
# Settlement P&L computation at event cleanup
# ---------------------------------------------------------------------------


def _make_event_info_with_bins() -> EventInfo:
    return EventInfo(
        event_id="ev1",
        title="Musk count Mar 01 - Mar 08",
        short_name="Mar 01 - Mar 08",
        settlement_date=date(2026, 3, 8),
        market_start_date=date(2026, 3, 1),
        bins=[
            {"lower_bound": 100, "upper_bound": 119, "token_id": "y0", "no_token_id": "n0"},
            {"lower_bound": 120, "upper_bound": 139, "token_id": "y1", "no_token_id": "n1"},
            {"lower_bound": 140, "upper_bound": 159, "token_id": "y2", "no_token_id": "n2"},
            {"lower_bound": 500, "upper_bound": float("inf"), "token_id": "y3", "no_token_id": "n3"},
        ],
    )


def _make_bot_with_positions(positions: dict) -> SimpleNamespace:
    from src.algo.musk_tweet_count.kelly.portfolio import BinPosition

    portfolio = SimpleNamespace(positions={})
    for bin_idx, spec in positions.items():
        portfolio.positions[bin_idx] = BinPosition(
            bin_index=bin_idx,
            yes_token_id=f"y{bin_idx}",
            yes_shares=spec.get("yes_shares", 0.0),
            no_shares=spec.get("no_shares", 0.0),
            yes_avg_cost=spec.get("yes_avg_cost", 0.0),
            no_avg_cost=spec.get("no_avg_cost", 0.0),
        )
    kelly_bot = SimpleNamespace(portfolio=portfolio)
    return SimpleNamespace(kelly_bot=kelly_bot)


def test_settlement_pnl_winning_yes_pays_one_minus_cost():
    info = _make_event_info_with_bins()
    bot = _make_bot_with_positions({
        1: {"yes_shares": 100.0, "yes_avg_cost": 0.20},  # 120-139 wins
    })

    pnl, breakdown = MultiEventManager._compute_settlement_pnl_breakdown(
        info, bot, actual_count=125,
    )

    assert pnl == 80.0  # 100 * (1.0 - 0.20)
    assert len(breakdown) == 1
    assert breakdown[0]["is_winning_bin"] is True
    assert breakdown[0]["pnl"] == 80.0


def test_settlement_pnl_losing_yes_loses_cost():
    info = _make_event_info_with_bins()
    bot = _make_bot_with_positions({
        0: {"yes_shares": 50.0, "yes_avg_cost": 0.30},  # 100-119 doesn't contain 125
    })

    pnl, breakdown = MultiEventManager._compute_settlement_pnl_breakdown(
        info, bot, actual_count=125,
    )

    assert pnl == -15.0  # 50 * (0.0 - 0.30)
    assert breakdown[0]["is_winning_bin"] is False


def test_settlement_pnl_winning_no_loses_one_minus_cost_inverted():
    """NO shares on the winning bin pay $0; loss = no_shares * no_avg_cost."""
    info = _make_event_info_with_bins()
    bot = _make_bot_with_positions({
        1: {"no_shares": 200.0, "no_avg_cost": 0.85},  # 120-139 wins
    })

    pnl, breakdown = MultiEventManager._compute_settlement_pnl_breakdown(
        info, bot, actual_count=125,
    )

    assert pnl == -170.0  # 200 * (0.0 - 0.85)


def test_settlement_pnl_losing_no_pays_one_minus_cost():
    """NO shares on a losing bin pay $1; gain = no_shares * (1 - no_avg_cost)."""
    info = _make_event_info_with_bins()
    bot = _make_bot_with_positions({
        2: {"no_shares": 100.0, "no_avg_cost": 0.90},  # 140-159 doesn't contain 125
    })

    pnl, breakdown = MultiEventManager._compute_settlement_pnl_breakdown(
        info, bot, actual_count=125,
    )

    assert abs(pnl - 10.0) < 1e-9  # 100 * (1.0 - 0.90)


def test_settlement_pnl_mixed_positions_sum_correctly():
    info = _make_event_info_with_bins()
    bot = _make_bot_with_positions({
        0: {"yes_shares": 50.0, "yes_avg_cost": 0.10},     # loses: -5
        1: {                                                # bin 120-139 wins
            "yes_shares": 100.0, "yes_avg_cost": 0.30,    # +70
            "no_shares": 20.0, "no_avg_cost": 0.80,       # -16
        },
        2: {"no_shares": 80.0, "no_avg_cost": 0.95},      # bin doesn't win, NO pays 1: +4
    })

    pnl, breakdown = MultiEventManager._compute_settlement_pnl_breakdown(
        info, bot, actual_count=125,
    )

    # -5 + 70 - 16 + 4 = 53
    assert abs(pnl - 53.0) < 1e-9
    assert len(breakdown) == 3


def test_settlement_pnl_top_bin_open_ended_wins_with_high_count():
    info = _make_event_info_with_bins()
    bot = _make_bot_with_positions({
        3: {"yes_shares": 25.0, "yes_avg_cost": 0.05},  # 500+ contains 1000
    })

    pnl, _ = MultiEventManager._compute_settlement_pnl_breakdown(
        info, bot, actual_count=1000,
    )

    assert abs(pnl - 23.75) < 1e-9  # 25 * (1.0 - 0.05)


def test_settlement_pnl_skips_dust_positions():
    info = _make_event_info_with_bins()
    bot = _make_bot_with_positions({
        0: {"yes_shares": 0.005, "yes_avg_cost": 0.10},  # below dust threshold
    })

    pnl, breakdown = MultiEventManager._compute_settlement_pnl_breakdown(
        info, bot, actual_count=110,
    )

    assert pnl == 0.0
    assert breakdown == []


def test_settlement_pnl_bin_boundary_inclusive():
    """A count equal to the bin's upper or lower bound is inside the bin."""
    info = _make_event_info_with_bins()
    bot = _make_bot_with_positions({
        0: {"yes_shares": 10.0, "yes_avg_cost": 0.50},  # 100-119
        2: {"yes_shares": 10.0, "yes_avg_cost": 0.50},  # 140-159
    })

    # count=119 → bin 0 wins, bin 2 loses
    pnl, breakdown = MultiEventManager._compute_settlement_pnl_breakdown(
        info, bot, actual_count=119,
    )
    by_bin = {entry["bin_index"]: entry for entry in breakdown}
    assert by_bin[0]["is_winning_bin"] is True
    assert by_bin[2]["is_winning_bin"] is False

    # count=140 → bin 0 loses, bin 2 wins
    _, breakdown = MultiEventManager._compute_settlement_pnl_breakdown(
        info, bot, actual_count=140,
    )
    by_bin = {entry["bin_index"]: entry for entry in breakdown}
    assert by_bin[0]["is_winning_bin"] is False
    assert by_bin[2]["is_winning_bin"] is True


def test_settlement_pnl_no_portfolio_returns_zero():
    info = _make_event_info_with_bins()
    bot = SimpleNamespace(kelly_bot=None)

    pnl, breakdown = MultiEventManager._compute_settlement_pnl_breakdown(
        info, bot, actual_count=125,
    )

    assert pnl == 0.0
    assert breakdown == []


def test_settlement_pnl_skips_bin_out_of_range():
    """Positions on bin indices not in event_info.bins are silently skipped
    rather than crashing — defends against state drift."""
    info = _make_event_info_with_bins()
    bot = _make_bot_with_positions({
        99: {"yes_shares": 100.0, "yes_avg_cost": 0.50},  # bin 99 doesn't exist
        1: {"yes_shares": 10.0, "yes_avg_cost": 0.10},
    })

    pnl, breakdown = MultiEventManager._compute_settlement_pnl_breakdown(
        info, bot, actual_count=125,
    )

    assert len(breakdown) == 1
    assert breakdown[0]["bin_index"] == 1
    assert abs(pnl - 9.0) < 1e-9  # 10 * (1.0 - 0.10)


def test_log_settlement_pnl_skips_when_authoritative_count_unavailable(caplog):
    import logging
    manager = object.__new__(MultiEventManager)
    manager._fmt_usd = MultiEventManager._fmt_usd
    manager._tz = ZoneInfo("America/New_York")

    async def fake_get_count(event_info):
        return None, None

    manager.get_authoritative_count = fake_get_count
    # Posts-store fallback also has nothing usable
    manager.compute_count_from_posts = lambda event_info: 0

    info = _make_event_info_with_bins()
    bot = _make_bot_with_positions({1: {"yes_shares": 100.0, "yes_avg_cost": 0.20}})

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(manager._log_settlement_pnl("ev1", info, bot, 1000.0))

    assert result is None
    assert any("no usable count" in r.message for r in caplog.records)


def test_log_settlement_pnl_emits_pnl_log_on_success(caplog):
    import logging
    manager = object.__new__(MultiEventManager)
    manager._fmt_usd = MultiEventManager._fmt_usd
    manager._tz = ZoneInfo("America/New_York")

    async def fake_get_count(event_info):
        return 125, datetime.now(UTC)

    manager.get_authoritative_count = fake_get_count

    info = _make_event_info_with_bins()
    bot = _make_bot_with_positions({
        1: {"yes_shares": 100.0, "yes_avg_cost": 0.20},  # bin 120-139 wins → +80
    })

    with caplog.at_level(logging.INFO):
        result = asyncio.run(manager._log_settlement_pnl("ev1", info, bot, 1000.0))

    assert result is not None
    assert abs(result - 80.0) < 1e-9
    messages = [r.message for r in caplog.records]
    assert any(
        "[EVENT][SETTLEMENT_PNL]" in m
        and "actual_count=125" in m
        and "settlement_pnl=$80.00" in m
        for m in messages
    )
    # per-bin breakdown line should appear
    assert any("WIN" in m and "120-139" in m for m in messages)


def test_no_exposure_helper_treats_pending_count_error_as_blocking():
    """If we can't determine pending order count we MUST refuse to evict —
    a noisy bot is way better than silently dropping a real event."""
    info = EventInfo(
        event_id="grumpy",
        title="Grumpy",
        short_name="Mar 03 - Mar 10",
        settlement_date=date(2026, 3, 10),
        market_start_date=date(2026, 3, 3),
        bins=[],
    )

    def broken_pending_count():
        raise RuntimeError("executor exploded")

    executor = SimpleNamespace(get_pending_count=broken_pending_count)
    kelly_bot = SimpleNamespace(
        portfolio=SimpleNamespace(positions={}),
        kelly_executor=executor,
    )
    bot = SimpleNamespace(kelly_bot=kelly_bot)
    active = ActiveEvent(
        info=info,
        bot=bot,
        task=SimpleNamespace(),
        started_at=datetime.now(UTC),
        allocated_capital=1000.0,
    )

    assert MultiEventManager._event_has_no_exposure(active) is False
