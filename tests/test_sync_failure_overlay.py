import asyncio
import time

from src.algo.musk_tweet_count.kelly.config import KellyConfig, RateLimitConfig
from src.algo.musk_tweet_count.kelly.executor import (
    SYNC_STALENESS_HALT_SECONDS,
    KellyExecutor,
    OverlayFillFragment,
)
from src.algo.musk_tweet_count.kelly.candidates import TradeAction
from src.algo.musk_tweet_count.kelly.portfolio import Portfolio


def _make_executor(sync_portfolio) -> KellyExecutor:
    portfolio = Portfolio(
        initial_capital=1000.0,
        capital=1000.0,
        num_bins=3,
        probabilities=[0.2, 0.3, 0.5],
    )
    return KellyExecutor(
        config=KellyConfig(rate_limit=RateLimitConfig()),
        portfolio=portfolio,
        token_ids={i: f"yes-{i}" for i in range(3)},
        no_token_ids={i: f"no-{i}" for i in range(3)},
        sync_portfolio=sync_portfolio,
        event_name="test-event",
    )


def _buy_fragment(bin_index=0, size=100.0, price=0.10) -> OverlayFillFragment:
    return OverlayFillFragment(
        fill_key="fill-1",
        order_id="order-1",
        token_id=f"yes-{bin_index}",
        action=TradeAction.BUY_YES,
        bin_index=bin_index,
        price=price,
        original_size=size,
        remaining_size=size,
        confirmed_at=time.time(),
    )


def _run_tick_capturing_portfolio(executor):
    """Run one tick; return list of portfolios handed to the planner."""
    captured = []

    def fake_compute(portfolio, orderbooks, hours_to_settlement, verbose=False, **kwargs):
        captured.append(portfolio)
        return []

    executor._compute_optimal_trades = fake_compute
    executor._get_orderbooks = lambda: {}
    asyncio.run(executor.run_tick(hours_to_settlement=12.0))
    return captured


async def _failing_sync():
    raise RuntimeError("positions API down")


def test_sync_failure_within_window_plans_on_base_plus_overlay():
    executor = _make_executor(_failing_sync)
    executor._last_successful_sync_time = time.time() - 30
    executor._overlay_ledger.append(_buy_fragment(bin_index=0, size=100.0, price=0.10))

    captured = _run_tick_capturing_portfolio(executor)

    assert len(captured) == 1
    planned = captured[0]
    position = planned.get_position(0)
    # The confirmed-but-unreconciled fill MUST be visible to the planner
    # (the pre-fix code handed it the bare API base and re-bought the bin)
    assert position is not None
    assert position.yes_shares == 100.0
    assert planned.capital == 1000.0 - 100.0 * 0.10


def test_sync_failure_beyond_window_halts_tick():
    executor = _make_executor(_failing_sync)
    executor._last_successful_sync_time = (
        time.time() - SYNC_STALENESS_HALT_SECONDS - 60
    )

    captured = _run_tick_capturing_portfolio(executor)

    assert captured == []  # tick halted before planning


def test_sync_failure_with_no_prior_success_halts_tick():
    executor = _make_executor(_failing_sync)
    assert executor._last_successful_sync_time is None

    captured = _run_tick_capturing_portfolio(executor)

    assert captured == []


def test_successful_sync_records_timestamp():
    async def ok_sync():
        return None

    executor = _make_executor(ok_sync)
    assert executor._last_successful_sync_time is None

    captured = _run_tick_capturing_portfolio(executor)

    assert len(captured) == 1
    assert executor._last_successful_sync_time is not None


def test_recovered_sync_after_stale_halt_resumes_planning():
    calls = {"n": 0}

    async def flaky_sync():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("positions API down")
        return None

    executor = _make_executor(flaky_sync)
    # First tick: no prior success + failure -> halt
    captured = _run_tick_capturing_portfolio(executor)
    assert captured == []

    # Second tick: sync recovers -> planning resumes
    captured = _run_tick_capturing_portfolio(executor)
    assert len(captured) == 1
    assert executor._last_successful_sync_time is not None
