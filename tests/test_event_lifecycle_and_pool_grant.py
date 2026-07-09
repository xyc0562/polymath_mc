import asyncio
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

from src.algo.musk_tweet_count.forecaster.data import ContractDayUtils
from src.algo.musk_tweet_count.forecaster.multi_event_manager import (
    EventInfo,
    MultiEventManager,
)
from src.algo.musk_tweet_count.kelly.capital_pool import CapitalPool, CapitalPoolConfig
from src.algo.musk_tweet_count.kelly.config import CollateralConfig, KellyConfig
from src.algo.musk_tweet_count.kelly.integration import KellyTradingBot
from src.algo.musk_tweet_count.kelly.portfolio import Portfolio


def _make_pool(total=10_000.0, max_per_event=3_000.0) -> CapitalPool:
    pool = CapitalPool(
        CapitalPoolConfig(total_capital=total),
        max_per_event=max_per_event,
    )
    pool._initialized_from_api = True
    return pool


def _event_info(event_id="e1", days_out=5) -> EventInfo:
    settlement = date.today() + timedelta(days=days_out)
    return EventInfo(
        event_id=event_id,
        title="Musk tweets test",
        short_name="Test Event",
        settlement_date=settlement,
        market_start_date=settlement - timedelta(days=7),
        bins=[],
    )


def _reconcile_stub(pool, pending=None):
    return SimpleNamespace(
        _events_lock=asyncio.Lock(),
        _active_events={},
        _pending_events=dict(pending or {}),
        _delisted_miss_count={},
        _evicted_event_ids=set(),
        capital_pool=pool,
        DELISTED_EVICTION_THRESHOLD=MultiEventManager.DELISTED_EVICTION_THRESHOLD,
        _event_has_no_exposure=MultiEventManager._event_has_no_exposure,
    )


def _reconcile(stub, discovered_ids):
    asyncio.run(MultiEventManager._reconcile_against_discovery(stub, discovered_ids))


# ---------- item 7: pending events ----------


def test_pending_event_with_allocation_never_dropped():
    pool = _make_pool()
    asyncio.run(pool.restore_allocation("e1", 800.0))
    stub = _reconcile_stub(pool, pending={"e1": _event_info("e1")})

    for _ in range(4):  # far past the threshold
        _reconcile(stub, discovered_ids=set())

    assert "e1" in stub._pending_events


def test_pending_event_without_allocation_dropped_after_threshold():
    pool = _make_pool()
    stub = _reconcile_stub(pool, pending={"e1": _event_info("e1")})

    _reconcile(stub, discovered_ids=set())
    assert "e1" in stub._pending_events  # first miss: deferred

    _reconcile(stub, discovered_ids=set())
    assert "e1" not in stub._pending_events  # second miss: dropped


def test_pending_miss_counter_resets_on_rediscovery():
    pool = _make_pool()
    stub = _reconcile_stub(pool, pending={"e1": _event_info("e1")})

    _reconcile(stub, discovered_ids=set())
    _reconcile(stub, discovered_ids={"e1"})  # reappears
    _reconcile(stub, discovered_ids=set())

    assert "e1" in stub._pending_events  # back to miss 1/2


# ---------- item 7: eviction reversibility ----------


def _add_event_stub():
    async def _noop_sync():
        return None

    return SimpleNamespace(
        _events_lock=asyncio.Lock(),
        _active_events={},
        _pending_events={},
        _completed_events=["e1"],
        _evicted_event_ids=set(),
        contract_utils=ContractDayUtils(),
        config=SimpleNamespace(event_trading_rules=None),
        kelly_config=SimpleNamespace(t_stop_hours=3.0),
        _sync_user_stream_markets=_noop_sync,
        _running=False,
    )


def test_evicted_event_can_be_readded():
    stub = _add_event_stub()
    stub._evicted_event_ids = {"e1"}

    added = asyncio.run(MultiEventManager.add_event(stub, _event_info("e1")))

    assert added is True
    assert "e1" in stub._pending_events
    assert "e1" not in stub._completed_events
    assert "e1" not in stub._evicted_event_ids


def test_settled_completed_event_not_readded():
    stub = _add_event_stub()  # completed but NOT evicted

    added = asyncio.run(MultiEventManager.add_event(stub, _event_info("e1")))

    assert added is False
    assert "e1" not in stub._pending_events
    assert "e1" in stub._completed_events


# ---------- item 10: pool grant top-up ----------


def test_request_capital_tops_up_restored_allocation():
    pool = _make_pool(total=10_000.0, max_per_event=3_000.0)
    asyncio.run(pool.restore_allocation("e1", 800.0))
    assert pool._available == pytest.approx(9_200.0)

    granted = asyncio.run(pool.request_capital("e1"))

    assert granted == pytest.approx(3_000.0)
    assert pool._available == pytest.approx(7_000.0)
    allocation = asyncio.run(pool.get_allocation("e1"))
    assert allocation.initial_allocation == pytest.approx(3_000.0)
    assert allocation.current_value == pytest.approx(3_000.0)


def test_top_up_bounded_by_available_capital():
    pool = _make_pool(total=1_000.0, max_per_event=3_000.0)
    asyncio.run(pool.restore_allocation("e1", 800.0))
    assert pool._available == pytest.approx(200.0)

    granted = asyncio.run(pool.request_capital("e1"))

    assert granted == pytest.approx(1_000.0)  # 800 restored + all 200 available
    assert pool._available == pytest.approx(0.0)


def test_no_clawback_when_restored_exceeds_max_per_event():
    pool = _make_pool(total=10_000.0, max_per_event=3_000.0)
    asyncio.run(pool.restore_allocation("e1", 5_000.0))
    available_before = pool._available

    granted = asyncio.run(pool.request_capital("e1"))

    assert granted == pytest.approx(5_000.0)
    assert pool._available == pytest.approx(available_before)


# ---------- item 10: event budget respects the grant ----------


def _budget_bot(c_event_max=3_000.0, external_limit=None) -> KellyTradingBot:
    bot = object.__new__(KellyTradingBot)
    bot.config = KellyConfig(collateral=CollateralConfig(c_event_max=c_event_max))
    bot.portfolio = Portfolio(
        initial_capital=100.0,
        capital=100.0,
        num_bins=1,
        probabilities=[1.0],
    )
    bot.portfolio.set_external_capital_limit(external_limit)
    return bot


def test_event_budget_defaults_to_c_event_max():
    bot = _budget_bot(external_limit=None)

    assert bot._effective_event_budget() == pytest.approx(3_000.0)


def test_event_budget_capped_by_short_pool_grant():
    bot = _budget_bot(external_limit=800.0)

    assert bot._effective_event_budget() == pytest.approx(800.0)


def test_event_budget_never_exceeds_c_event_max():
    bot = _budget_bot(external_limit=5_000.0)

    assert bot._effective_event_budget() == pytest.approx(3_000.0)
