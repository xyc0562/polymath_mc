import asyncio
import logging
from datetime import datetime, timezone

import pytest

from src.algo.musk_tweet_count.kelly import candidates as candidates_module
from src.algo.musk_tweet_count.kelly import executor as executor_module
from src.algo.musk_tweet_count.kelly.candidates import TradeAction, TradeCandidate, generate_candidates
from src.algo.musk_tweet_count.kelly.config import KellyConfig, RateLimitConfig
from src.algo.musk_tweet_count.kelly.executor import KellyExecutor
from src.algo.musk_tweet_count.kelly.integration import KellyTradingBot
from src.algo.musk_tweet_count.kelly.orderbook import OrderbookLevel, UnifiedOrderbook
from src.algo.musk_tweet_count.kelly.portfolio import Portfolio
from src.algo.musk_tweet_count.kelly.user_stream import FillEvent, OrderStatus


def _make_candidate(
    *,
    action: TradeAction = TradeAction.BUY_YES,
    size: float = 10.0,
    price: float = 0.2,
    bin_index: int = 0,
    utility_gain: float = 0.01,
    reservation_price: float = 0.25,
    threshold_price: float = 0.0,
    execution_bound_price: float = 0.0,
) -> TradeCandidate:
    return TradeCandidate(
        bin_index=bin_index,
        action=action,
        size=size,
        price=price,
        utility_gain=utility_gain,
        reservation_price=reservation_price,
        edge=0.05,
        limit_price=price,
        threshold_price=threshold_price or reservation_price,
        execution_bound_price=execution_bound_price or threshold_price or reservation_price,
    )


def _make_fill(
    *,
    order_id: str = "order-1",
    token_id: str = "yes-0",
    side: str = "BUY",
    price: float = 0.2,
    size: float = 10.0,
    status: OrderStatus = OrderStatus.CONFIRMED,
    match_id: str = "match-1",
) -> FillEvent:
    return FillEvent(
        order_id=order_id,
        token_id=token_id,
        side=side,
        price=price,
        size=size,
        status=status,
        timestamp=datetime.now(timezone.utc),
        match_id=match_id,
    )


def _make_executor(
    *,
    capital: float = 100.0,
    rate_limit: RateLimitConfig | None = None,
    probabilities: list[float] | None = None,
) -> KellyExecutor:
    probabilities = probabilities or [1.0]
    config = KellyConfig(rate_limit=rate_limit or RateLimitConfig())
    portfolio = Portfolio(
        initial_capital=capital,
        capital=capital,
        num_bins=len(probabilities),
        probabilities=probabilities,
    )
    executor = KellyExecutor(
        config=config,
        portfolio=portfolio,
        token_ids={i: f"yes-{i}" for i in range(len(probabilities))},
        no_token_ids={i: f"no-{i}" for i in range(len(probabilities))},
        event_name="test-event",
    )
    return executor


def _make_orderbook(
    *,
    bin_index: int = 0,
    yes_bids: list[tuple[float, float]] | None = None,
    yes_asks: list[tuple[float, float]] | None = None,
) -> UnifiedOrderbook:
    return UnifiedOrderbook(
        bin_index=bin_index,
        yes_token_id=f"yes-{bin_index}",
        yes_bids=[OrderbookLevel(price=price, size=size) for price, size in (yes_bids or [])],
        yes_asks=[OrderbookLevel(price=price, size=size) for price, size in (yes_asks or [])],
    )


def _make_sync_bot(
    *,
    num_bins: int = 1,
    probabilities: list[float] | None = None,
    initial_capital: float = 1_000.0,
) -> KellyTradingBot:
    probabilities = probabilities or [1.0 / max(1, num_bins)] * num_bins
    bot = KellyTradingBot(
        clob_client=object(),
        config=KellyConfig(),
        probability_model=lambda *_args: probabilities,
        dry_run=False,
        wallet_address="0xabc",
        event_name="sync-test",
    )
    bot._setup_complete = True
    bot.bin_token_ids = {i: f"yes-{i}" for i in range(num_bins)}
    bot.bin_no_token_ids = {i: f"no-{i}" for i in range(num_bins)}
    bot.portfolio = Portfolio(
        initial_capital=initial_capital,
        capital=initial_capital,
        num_bins=num_bins,
        probabilities=probabilities,
    )
    return bot


def test_resolve_live_fak_submission_buy_yes_reprices_up_within_band():
    executor = _make_executor()
    orderbook = _make_orderbook(
        yes_asks=[(0.40, 20.0), (0.41, 10.0)],
    )
    candidate = _make_candidate(
        action=TradeAction.BUY_YES,
        size=25.0,
        price=0.40,
        threshold_price=0.41,
        execution_bound_price=0.41,
    )

    resolution = executor._resolve_live_fak_submission(
        candidate,
        token_id="yes-0",
        orderbook=orderbook,
        requested_size=25.0,
        requested_limit_price=0.40,
        execution_bound_price=0.41,
    )

    assert resolution is not None
    assert resolution.submitted_size == 25
    assert resolution.submitted_limit_price == pytest.approx(0.41)
    assert resolution.submitted_vwap == pytest.approx((20 * 0.40 + 5 * 0.41) / 25)
    assert resolution.price_moved is True
    assert resolution.size_reduced is False


def test_resolve_live_fak_submission_buy_no_reprices_up_within_band():
    executor = _make_executor()
    orderbook = _make_orderbook(
        yes_bids=[(0.60, 20.0), (0.59, 10.0)],
    )
    candidate = _make_candidate(
        action=TradeAction.BUY_NO,
        size=25.0,
        price=0.40,
        threshold_price=0.41,
        execution_bound_price=0.41,
    )

    resolution = executor._resolve_live_fak_submission(
        candidate,
        token_id="no-0",
        orderbook=orderbook,
        requested_size=25.0,
        requested_limit_price=0.40,
        execution_bound_price=0.41,
    )

    assert resolution is not None
    assert resolution.submitted_size == 25
    assert resolution.submitted_limit_price == pytest.approx(0.41)
    assert resolution.submitted_vwap == pytest.approx((20 * 0.40 + 5 * 0.41) / 25)
    assert resolution.price_moved is True
    assert resolution.size_reduced is False


def test_resolve_live_fak_submission_sell_yes_reprices_down_within_band():
    executor = _make_executor()
    orderbook = _make_orderbook(
        yes_bids=[(0.60, 20.0), (0.59, 10.0)],
    )
    candidate = _make_candidate(
        action=TradeAction.SELL_YES,
        size=25.0,
        price=0.60,
        threshold_price=0.59,
        execution_bound_price=0.59,
    )

    resolution = executor._resolve_live_fak_submission(
        candidate,
        token_id="yes-0",
        orderbook=orderbook,
        requested_size=25.0,
        requested_limit_price=0.60,
        execution_bound_price=0.59,
    )

    assert resolution is not None
    assert resolution.submitted_size == 25
    assert resolution.submitted_limit_price == pytest.approx(0.59)
    assert resolution.submitted_vwap == pytest.approx((20 * 0.60 + 5 * 0.59) / 25)
    assert resolution.price_moved is True
    assert resolution.size_reduced is False


def test_resolve_live_fak_submission_sell_no_reprices_down_within_band():
    executor = _make_executor()
    orderbook = _make_orderbook(
        yes_asks=[(0.40, 20.0), (0.41, 10.0)],
    )
    candidate = _make_candidate(
        action=TradeAction.SELL_NO,
        size=25.0,
        price=0.60,
        threshold_price=0.59,
        execution_bound_price=0.59,
    )

    resolution = executor._resolve_live_fak_submission(
        candidate,
        token_id="no-0",
        orderbook=orderbook,
        requested_size=25.0,
        requested_limit_price=0.60,
        execution_bound_price=0.59,
    )

    assert resolution is not None
    assert resolution.submitted_size == 25
    assert resolution.submitted_limit_price == pytest.approx(0.59)
    assert resolution.submitted_vwap == pytest.approx((20 * 0.60 + 5 * 0.59) / 25)
    assert resolution.price_moved is True
    assert resolution.size_reduced is False


def test_resolve_live_fak_submission_respects_tighter_execution_bound():
    executor = _make_executor()
    buy_orderbook = _make_orderbook(
        yes_asks=[(0.40, 20.0), (0.41, 10.0), (0.42, 20.0)],
    )
    buy_candidate = _make_candidate(
        action=TradeAction.BUY_YES,
        size=35.0,
        price=0.40,
        threshold_price=0.45,
        execution_bound_price=0.41,
    )

    buy_resolution = executor._resolve_live_fak_submission(
        buy_candidate,
        token_id="yes-0",
        orderbook=buy_orderbook,
        requested_size=35.0,
        requested_limit_price=0.40,
        execution_bound_price=0.41,
    )

    assert buy_resolution is not None
    assert buy_resolution.submitted_limit_price == pytest.approx(0.41)
    assert buy_resolution.submitted_size == 30
    assert buy_resolution.size_reduced is True

    sell_orderbook = _make_orderbook(
        yes_bids=[(0.60, 20.0), (0.59, 10.0), (0.58, 20.0)],
    )
    sell_candidate = _make_candidate(
        action=TradeAction.SELL_YES,
        size=35.0,
        price=0.60,
        threshold_price=0.55,
        execution_bound_price=0.59,
    )

    sell_resolution = executor._resolve_live_fak_submission(
        sell_candidate,
        token_id="yes-0",
        orderbook=sell_orderbook,
        requested_size=35.0,
        requested_limit_price=0.60,
        execution_bound_price=0.59,
    )

    assert sell_resolution is not None
    assert sell_resolution.submitted_limit_price == pytest.approx(0.59)
    assert sell_resolution.submitted_size == 30
    assert sell_resolution.size_reduced is True


def test_resolve_live_fak_submission_reduces_size_only_after_band_exhausted():
    executor = _make_executor()
    orderbook = _make_orderbook(
        yes_asks=[(0.40, 5.0), (0.41, 7.0)],
    )
    candidate = _make_candidate(
        action=TradeAction.BUY_YES,
        size=20.0,
        price=0.40,
        threshold_price=0.41,
        execution_bound_price=0.41,
    )

    resolution = executor._resolve_live_fak_submission(
        candidate,
        token_id="yes-0",
        orderbook=orderbook,
        requested_size=20.0,
        requested_limit_price=0.40,
        execution_bound_price=0.41,
    )

    assert resolution is not None
    assert resolution.submitted_limit_price == pytest.approx(0.41)
    assert resolution.submitted_size == 12
    assert resolution.size_reduced is True


def test_resolve_live_fak_submission_handles_buy_2dp_inside_search():
    executor = _make_executor()

    class TickSizeExecutor:
        def get_tick_size(self, token_id):
            del token_id
            return "0.001"

    executor.order_executor = TickSizeExecutor()
    orderbook = _make_orderbook(
        yes_asks=[(0.203, 5.0), (0.204, 50.0)],
    )
    candidate = _make_candidate(
        action=TradeAction.BUY_YES,
        size=5.0,
        price=0.203,
        threshold_price=0.204,
        execution_bound_price=0.204,
    )

    resolution = executor._resolve_live_fak_submission(
        candidate,
        token_id="yes-0",
        orderbook=orderbook,
        requested_size=5.0,
        requested_limit_price=0.203,
        execution_bound_price=0.204,
    )

    assert resolution is not None
    assert resolution.submitted_size == 5
    assert resolution.submitted_limit_price == pytest.approx(0.204)
    assert resolution.submitted_vwap == pytest.approx(0.203)
    assert resolution.price_moved is True
    assert resolution.size_reduced is False


def test_confirmed_buy_overlay_applied_once_and_reconciled():
    executor = _make_executor()
    candidate = _make_candidate(size=12.0, price=0.25)
    fill = _make_fill(size=12.0, price=0.25)

    recorded = executor._record_confirmed_fill(
        fill_key=executor._confirmed_fill_key(fill, "yes-0"),
        order_id=fill.order_id,
        candidate=candidate,
        token_id="yes-0",
        fill_event=fill,
        now=100.0,
    )

    assert recorded is True

    effective = executor._build_effective_portfolio()
    position = effective.get_position(0)
    assert position is not None
    assert position.yes_shares == 12.0
    assert effective.capital == 97.0

    previous_api = executor.api_base_portfolio._copy()
    current_api = executor.api_base_portfolio._copy()
    current_api.execute_buy_yes(0, 12.0, 0.25, "yes-0")
    executor.api_base_portfolio = current_api
    executor.portfolio = current_api

    executor._reconcile_overlay_against_api(previous_api, current_api)

    assert executor.get_integrity_summary()["overlay_entries"] == 0

    reconciled = executor._build_effective_portfolio()
    reconciled_position = reconciled.get_position(0)
    assert reconciled_position is not None
    assert reconciled_position.yes_shares == 12.0
    assert reconciled.capital == 97.0


def test_unmatched_api_delta_is_accepted_without_freeze():
    executor = _make_executor()
    previous_api = executor.api_base_portfolio._copy()
    current_api = executor.api_base_portfolio._copy()
    current_api.execute_buy_yes(0, 5.0, 0.4, "yes-0")

    executor._reconcile_overlay_against_api(previous_api, current_api)

    integrity = executor.get_integrity_summary()
    assert integrity["frozen"] is False
    assert integrity["unmatched_api_delta_count"] == 1


def test_overlay_residual_within_api_tolerance_is_cleared(caplog):
    executor = _make_executor(
        capital=200.0,
        rate_limit=RateLimitConfig(
            overlay_reconciliation_api_tolerance_fraction=0.05,
        ),
    )
    candidate = _make_candidate(size=20.0, price=0.25)
    fill = _make_fill(size=20.0, price=0.25, match_id="match-tol")

    executor._record_confirmed_fill(
        fill_key="match-tol",
        order_id=fill.order_id,
        candidate=candidate,
        token_id="yes-0",
        fill_event=fill,
        now=10.0,
    )

    previous_api = executor.api_base_portfolio._copy()
    current_api = executor.api_base_portfolio._copy()
    current_api.execute_buy_yes(0, 19.3, 0.25, "yes-0")

    with caplog.at_level(logging.INFO):
        executor._reconcile_overlay_against_api(previous_api, current_api)

    integrity = executor.get_integrity_summary()
    assert integrity["overlay_entries"] == 0
    assert integrity["frozen"] is False
    assert "Cleared residual overlay within API tolerance" in caplog.text


def test_overlay_residual_above_api_tolerance_is_retained():
    executor = _make_executor(
        capital=200.0,
        rate_limit=RateLimitConfig(
            overlay_reconciliation_api_tolerance_fraction=0.05,
        ),
    )
    candidate = _make_candidate(size=20.0, price=0.25)
    fill = _make_fill(size=20.0, price=0.25, match_id="match-notol")

    executor._record_confirmed_fill(
        fill_key="match-notol",
        order_id=fill.order_id,
        candidate=candidate,
        token_id="yes-0",
        fill_event=fill,
        now=10.0,
    )

    previous_api = executor.api_base_portfolio._copy()
    current_api = executor.api_base_portfolio._copy()
    current_api.execute_buy_yes(0, 18.0, 0.25, "yes-0")

    executor._reconcile_overlay_against_api(previous_api, current_api)

    integrity = executor.get_integrity_summary()
    assert integrity["overlay_entries"] == 1
    assert integrity["frozen"] is False


def test_overlay_api_tolerance_is_capped_for_large_positions():
    executor = _make_executor(
        capital=1_000.0,
        rate_limit=RateLimitConfig(
            overlay_reconciliation_api_tolerance_fraction=0.05,
            overlay_reconciliation_api_tolerance_max_shares=1.0,
        ),
    )
    candidate = _make_candidate(size=20.0, price=0.25)
    fill = _make_fill(size=20.0, price=0.25, match_id="match-cap")

    executor._record_confirmed_fill(
        fill_key="match-cap",
        order_id=fill.order_id,
        candidate=candidate,
        token_id="yes-0",
        fill_event=fill,
        now=10.0,
    )

    executor.api_base_portfolio.execute_buy_yes(0, 480.0, 0.25, "yes-0")
    executor.portfolio = executor.api_base_portfolio
    previous_api = executor.api_base_portfolio._copy()
    current_api = executor.api_base_portfolio._copy()
    current_api.execute_buy_yes(0, 18.5, 0.25, "yes-0")
    executor.api_base_portfolio = current_api
    executor.portfolio = current_api

    assert executor._overlay_api_tolerance_shares(498.5) == 1.0

    executor._reconcile_overlay_against_api(previous_api, current_api)

    integrity = executor.get_integrity_summary()
    assert integrity["overlay_entries"] == 1
    assert integrity["frozen"] is False


def test_duplicate_confirmed_fill_with_conflicting_economics_freezes():
    executor = _make_executor()
    candidate = _make_candidate(size=10.0, price=0.2)
    fill = _make_fill(size=10.0, price=0.2, match_id="match-dup")

    first = executor._record_confirmed_fill(
        fill_key="match-dup",
        order_id=fill.order_id,
        candidate=candidate,
        token_id="yes-0",
        fill_event=fill,
        now=10.0,
    )
    second = executor._record_confirmed_fill(
        fill_key="match-dup",
        order_id=fill.order_id,
        candidate=candidate,
        token_id="yes-0",
        fill_event=_make_fill(
            order_id=fill.order_id,
            size=12.0,
            price=0.2,
            match_id="match-dup",
        ),
        now=11.0,
    )

    assert first is True
    assert second is False
    assert executor.get_integrity_summary()["frozen"] is True


def test_hard_deadline_drops_overlay_and_clears_freeze():
    executor = _make_executor()
    candidate = _make_candidate(size=8.0, price=0.3)
    fill = _make_fill(size=8.0, price=0.3, match_id="match-recover")

    executor._record_confirmed_fill(
        fill_key="match-recover",
        order_id=fill.order_id,
        candidate=candidate,
        token_id="yes-0",
        fill_event=fill,
        now=0.0,
    )
    executor._freeze_integrity("test freeze", now=100.0)
    executor._maybe_force_api_recovery(now=401.0)

    integrity = executor.get_integrity_summary()
    assert integrity["frozen"] is False
    assert integrity["overlay_entries"] == 0
    assert integrity["last_forced_api_recovery_at"] is not None


def test_enforce_integrity_deadline_syncs_and_clears_without_tick():
    executor = _make_executor()
    candidate = _make_candidate(size=6.0, price=0.25)
    fill = _make_fill(size=6.0, price=0.25, match_id="match-deadline")
    sync_calls = []
    executor._last_api_snapshot = executor.api_base_portfolio._copy()

    executor._record_confirmed_fill(
        fill_key="match-deadline",
        order_id=fill.order_id,
        candidate=candidate,
        token_id="yes-0",
        fill_event=fill,
        now=0.0,
    )
    executor._freeze_integrity("deadline test", now=10.0)

    async def sync():
        sync_calls.append(True)
        updated = executor.api_base_portfolio._copy()
        updated.execute_buy_yes(0, 6.0, 0.25, "yes-0")
        executor.api_base_portfolio = updated
        executor.portfolio = updated

    executor.sync_portfolio = sync

    recovered = asyncio.run(executor.enforce_integrity_deadline(now=400.0))

    integrity = executor.get_integrity_summary()
    assert recovered is True
    assert len(sync_calls) == 1
    assert integrity["frozen"] is False
    assert integrity["overlay_entries"] == 0
    assert integrity["last_forced_api_recovery_at"] is None


def test_kelly_bot_status_exposes_integrity_fields():
    bot = KellyTradingBot(
        clob_client=object(),
        config=KellyConfig(),
        probability_model=lambda *_args: [1.0],
        dry_run=True,
        event_name="status-event",
    )

    class FakeExecutor:
        def get_pending_count(self):
            return 2

        def get_pending_collateral(self):
            return 12.5

        def get_integrity_summary(self):
            return {
                "frozen": True,
                "reason": "test",
                "frozen_at": "2026-03-06T00:00:00+00:00",
                "deadline_at": "2026-03-06T00:05:00+00:00",
                "overlay_entries": 3,
                "oldest_overlay_age_seconds": 42.0,
                "last_forced_api_recovery_at": None,
                "unmatched_api_delta_count": 1,
            }

    bot.kelly_executor = FakeExecutor()
    status = bot.get_status()

    assert status["integrity"]["frozen"] is True
    assert status["integrity"]["overlay_entries"] == 3
    assert status["pending_orders"]["count"] == 2
    assert status["pending_orders"]["collateral"] == 12.5


def test_sync_positions_api_failure_preserves_portfolio_state(monkeypatch):
    bot = KellyTradingBot(
        clob_client=object(),
        config=KellyConfig(),
        probability_model=lambda *_args: [1.0],
        dry_run=False,
        wallet_address="0xabc",
        event_name="sync-failure",
    )
    bot._setup_complete = True
    bot.bin_token_ids = {0: "yes-0"}
    bot.bin_no_token_ids = {0: "no-0"}
    bot.portfolio = Portfolio(
        initial_capital=100.0,
        capital=80.0,
        num_bins=1,
        probabilities=[1.0],
    )
    bot.portfolio.execute_buy_yes(0, 10.0, 0.2, "yes-0")

    before = bot.portfolio._copy()

    async def fail_fetch_positions(_wallet_address):
        raise RuntimeError("408 Request Timeout")

    async def fake_balance():
        return 999.0

    monkeypatch.setattr(bot, "fetch_positions_from_api", fail_fetch_positions)
    monkeypatch.setattr(bot, "fetch_usdc_balance", fake_balance)

    with pytest.raises(RuntimeError, match="408"):
        asyncio.run(bot.sync_positions_from_api("0xabc"))

    position = bot.portfolio.get_position(0)
    assert position is not None
    assert position.yes_shares == before.get_position(0).yes_shares
    assert position.yes_avg_cost == before.get_position(0).yes_avg_cost
    assert bot.portfolio.capital == before.capital
    assert bot.portfolio.total_collateral_used == before.total_collateral_used


def test_sync_positions_uses_initial_value_for_missing_avg_price_and_blocks_cap_overrun(monkeypatch):
    bot = _make_sync_bot(probabilities=[0.1])
    bot.portfolio.execute_buy_no(0, 243.0, 0.919, "yes-0")

    async def fake_fetch_positions(_wallet_address):
        return {
            "no-0": {
                "shares": 243.2,
                "avg_price": 0.0,
                "value": 243.2 * 0.919,
            }
        }

    async def fake_balance():
        return 0.0

    monkeypatch.setattr(bot, "fetch_positions_from_api", fake_fetch_positions)
    monkeypatch.setattr(bot, "fetch_usdc_balance", fake_balance)

    asyncio.run(bot.sync_positions_from_api("0xabc"))

    position = bot.portfolio.get_position(0)
    assert position is not None
    assert position.no_shares == pytest.approx(243.2)
    assert position.no_avg_cost == pytest.approx(0.919)
    assert position.collateral_used == pytest.approx(243.2 * 0.919)
    assert position.has_no_unpriced_increment is False

    executor = KellyExecutor(
        config=KellyConfig(),
        portfolio=bot.portfolio,
        token_ids={0: "yes-0"},
        no_token_ids={0: "no-0"},
        event_name="sync-test",
    )
    candidate = _make_candidate(
        action=TradeAction.BUY_NO,
        size=112.0,
        price=0.914,
        bin_index=0,
        utility_gain=0.01,
        reservation_price=0.96,
    )
    orderbooks = {0: _make_orderbook(bin_index=0, yes_bids=[(0.086, 500.0)], yes_asks=[(0.087, 500.0)])}

    optimal_size = executor._find_optimal_size_on(
        bot.portfolio,
        candidate,
        orderbooks,
        24.0,
        KellyConfig(),
    )

    assert optimal_size < 2.0


@pytest.mark.parametrize(
    ("is_no", "token_id", "shares", "price"),
    [
        (False, "yes-0", 40.0, 0.27),
        (True, "no-0", 55.0, 0.83),
    ],
)
def test_sync_positions_missing_avg_price_uses_initial_value_without_uncertainty(
    monkeypatch,
    is_no,
    token_id,
    shares,
    price,
):
    bot = _make_sync_bot(probabilities=[0.4])

    async def fake_fetch_positions(_wallet_address):
        return {
            token_id: {
                "shares": shares,
                "avg_price": 0.0,
                "value": shares * price,
            }
        }

    async def fake_balance():
        return 0.0

    monkeypatch.setattr(bot, "fetch_positions_from_api", fake_fetch_positions)
    monkeypatch.setattr(bot, "fetch_usdc_balance", fake_balance)

    asyncio.run(bot.sync_positions_from_api("0xabc"))

    position = bot.portfolio.get_position(0)
    assert position is not None
    if is_no:
        assert position.no_shares == pytest.approx(shares)
        assert position.no_avg_cost == pytest.approx(price)
        assert position.has_no_unpriced_increment is False
    else:
        assert position.yes_shares == pytest.approx(shares)
        assert position.yes_avg_cost == pytest.approx(price)
        assert position.has_yes_unpriced_increment is False
    assert position.collateral_used == pytest.approx(shares * price)


def test_sync_positions_missing_price_can_use_overlay_hint(monkeypatch):
    bot = _make_sync_bot(probabilities=[0.4])
    bot.portfolio.execute_buy_yes(0, 20.0, 0.25, "yes-0")

    class FakeExecutor:
        def get_overlay_price_hint_for_api_increase(self, *, bin_index, is_no, share_increase):
            assert bin_index == 0
            assert is_no is False
            assert share_increase == pytest.approx(5.0)
            return 5.0, 0.41

    bot.kelly_executor = FakeExecutor()

    async def fake_fetch_positions(_wallet_address):
        return {
            "yes-0": {
                "shares": 25.0,
                "avg_price": 0.0,
                "value": 0.0,
            }
        }

    async def fake_balance():
        return 0.0

    monkeypatch.setattr(bot, "fetch_positions_from_api", fake_fetch_positions)
    monkeypatch.setattr(bot, "fetch_usdc_balance", fake_balance)

    asyncio.run(bot.sync_positions_from_api("0xabc"))

    position = bot.portfolio.get_position(0)
    assert position is not None
    assert position.yes_shares == pytest.approx(25.0)
    assert position.yes_avg_cost == pytest.approx((20.0 * 0.25 + 5.0 * 0.41) / 25.0)
    assert position.has_yes_unpriced_increment is False


def test_truly_unpriceable_increment_blocks_same_side_add_but_not_other_bins(monkeypatch):
    bot = _make_sync_bot(num_bins=2, probabilities=[0.2, 0.8])
    bot.portfolio.execute_buy_no(0, 30.0, 0.7, "yes-0")

    class FakeExecutor:
        def get_overlay_price_hint_for_api_increase(self, *, bin_index, is_no, share_increase):
            assert bin_index == 0
            assert is_no is True
            assert share_increase == pytest.approx(5.0)
            return 0.0, 0.0

    bot.kelly_executor = FakeExecutor()

    async def fake_fetch_positions(_wallet_address):
        return {
            "no-0": {
                "shares": 35.0,
                "avg_price": 0.0,
                "value": 0.0,
            }
        }

    async def fake_balance():
        return 0.0

    monkeypatch.setattr(bot, "fetch_positions_from_api", fake_fetch_positions)
    monkeypatch.setattr(bot, "fetch_usdc_balance", fake_balance)

    asyncio.run(bot.sync_positions_from_api("0xabc"))

    position = bot.portfolio.get_position(0)
    assert position is not None
    assert position.has_no_unpriced_increment is True
    assert position.no_unpriced_shares == pytest.approx(5.0)
    assert position.no_unpriced_reserve == pytest.approx(5.0)

    orderbooks = {
        0: _make_orderbook(bin_index=0, yes_bids=[(0.12, 200.0)], yes_asks=[(0.13, 200.0)]),
        1: _make_orderbook(bin_index=1, yes_bids=[(0.30, 200.0)], yes_asks=[(0.31, 200.0)]),
    }

    candidates = generate_candidates(
        portfolio=bot.portfolio,
        orderbooks=orderbooks,
        config=KellyConfig(),
        hours_to_settlement=24.0,
        verbose=True,
    )

    assert any(c.bin_index == 1 and c.action == TradeAction.BUY_YES for c in candidates)
    assert all(not (c.bin_index == 0 and c.action == TradeAction.BUY_NO) for c in candidates)

    sold = bot.portfolio.simulate_sell_no(0, 3.0, 0.75)
    sold_pos = sold.get_position(0)
    assert sold_pos is not None
    assert sold_pos.no_unpriced_shares == pytest.approx(2.0)


def test_uncertainty_auto_clears_when_later_priced_sync_arrives(monkeypatch):
    bot = _make_sync_bot(probabilities=[0.4])

    async def fake_balance():
        return 0.0

    states = [
        {
            "yes-0": {
                "shares": 10.0,
                "avg_price": 0.0,
                "value": 0.0,
            }
        },
        {
            "yes-0": {
                "shares": 10.0,
                "avg_price": 0.33,
                "value": 3.3,
            }
        },
    ]

    async def fake_fetch_positions(_wallet_address):
        return states.pop(0)

    monkeypatch.setattr(bot, "fetch_positions_from_api", fake_fetch_positions)
    monkeypatch.setattr(bot, "fetch_usdc_balance", fake_balance)

    asyncio.run(bot.sync_positions_from_api("0xabc"))
    position = bot.portfolio.get_position(0)
    assert position is not None
    assert position.has_yes_unpriced_increment is True

    asyncio.run(bot.sync_positions_from_api("0xabc"))
    position = bot.portfolio.get_position(0)
    assert position is not None
    assert position.has_yes_unpriced_increment is False
    assert position.yes_avg_cost == pytest.approx(0.33)


def test_reduced_shares_preserve_local_cost_and_shrink_unpriced_first(monkeypatch):
    bot = _make_sync_bot(probabilities=[0.4])
    bot.portfolio.execute_buy_yes(0, 10.0, 0.25, "yes-0")
    position = bot.portfolio.get_position(0)
    assert position is not None
    position.yes_unpriced_shares = 4.0
    position.yes_unpriced_reserve = 4.0
    position.recompute_collateral_used()

    async def fake_fetch_positions(_wallet_address):
        return {
            "yes-0": {
                "shares": 7.0,
                "avg_price": 0.0,
                "value": 0.0,
            }
        }

    async def fake_balance():
        return 0.0

    monkeypatch.setattr(bot, "fetch_positions_from_api", fake_fetch_positions)
    monkeypatch.setattr(bot, "fetch_usdc_balance", fake_balance)

    asyncio.run(bot.sync_positions_from_api("0xabc"))

    position = bot.portfolio.get_position(0)
    assert position is not None
    assert position.yes_shares == pytest.approx(7.0)
    assert position.yes_avg_cost == pytest.approx(0.25)
    assert position.yes_unpriced_shares == pytest.approx(1.0)
    assert position.yes_unpriced_reserve == pytest.approx(1.0)
    assert position.collateral_used == pytest.approx((6.0 * 0.25) + 1.0)
    assert bot.portfolio.capital == pytest.approx(bot.config.collateral.c_event_max - position.collateral_used)


def test_sync_logging_reports_resolution_sources_and_uncertainty(monkeypatch, caplog):
    bot = _make_sync_bot(probabilities=[0.5])
    bot.portfolio.execute_buy_no(0, 10.0, 0.8, "yes-0")

    class FakeExecutor:
        def get_overlay_price_hint_for_api_increase(self, *, bin_index, is_no, share_increase):
            return 0.0, 0.0

    bot.kelly_executor = FakeExecutor()

    states = [
        {
            "no-0": {
                "shares": 12.0,
                "avg_price": 0.0,
                "value": 0.0,
            }
        },
        {
            "no-0": {
                "shares": 12.0,
                "avg_price": 0.0,
                "value": 9.96,
            }
        },
    ]

    async def fake_fetch_positions(_wallet_address):
        return states.pop(0)

    async def fake_balance():
        return 0.0

    monkeypatch.setattr(bot, "fetch_positions_from_api", fake_fetch_positions)
    monkeypatch.setattr(bot, "fetch_usdc_balance", fake_balance)

    with caplog.at_level(logging.INFO):
        asyncio.run(bot.sync_positions_from_api("0xabc"))
        asyncio.run(bot.sync_positions_from_api("0xabc"))

    assert "source=unpriced_reserve" in caplog.text
    assert "cost basis unresolved; blocking same-side adds" in caplog.text
    assert "source=initialValue" in caplog.text
    assert "cost basis uncertainty cleared" in caplog.text


def test_missing_api_side_with_positive_conditional_balance_is_retained_and_sellable(monkeypatch):
    bot = _make_sync_bot(probabilities=[0.2])
    bot.portfolio.execute_buy_no(0, 251.1, 0.86523, "yes-0")
    warnings = []
    bot.on_position_sync_warning = warnings.append

    async def fake_fetch_positions(_wallet_address):
        return {}

    async def fake_balance():
        return 0.0

    monkeypatch.setattr(bot, "fetch_positions_from_api", fake_fetch_positions)
    monkeypatch.setattr(bot, "fetch_usdc_balance", fake_balance)
    monkeypatch.setattr(
        bot,
        "_fetch_conditional_balance_shares",
        lambda token_id, refresh=True: 201.8 if token_id == "no-0" else None,
    )

    asyncio.run(bot.sync_positions_from_api("0xabc"))

    position = bot.portfolio.get_position(0)
    assert position is not None
    assert position.no_shares == pytest.approx(201.8)
    assert position.no_avg_cost == pytest.approx(0.86523)
    assert position.has_no_api_missing_unverified is True
    assert len(warnings) == 1
    assert warnings[0].side == "NO"
    assert warnings[0].verified_balance_shares == pytest.approx(201.8)

    orderbooks = {
        0: _make_orderbook(
            bin_index=0,
            yes_bids=[(0.11, 400.0)],
            yes_asks=[(0.13, 400.0)],
        )
    }
    candidates = generate_candidates(
        portfolio=bot.portfolio,
        orderbooks=orderbooks,
        config=KellyConfig(),
        hours_to_settlement=6.0,
        verbose=True,
    )

    assert any(c.action == TradeAction.SELL_NO for c in candidates)
    assert all(c.action != TradeAction.BUY_YES for c in candidates)
    assert bot.get_status()["position_sync"]["warning_count"] == 1


def test_missing_api_side_verified_zero_clears_position(monkeypatch):
    bot = _make_sync_bot(probabilities=[0.4])
    bot.portfolio.execute_buy_no(0, 75.0, 0.81, "yes-0")

    async def fake_fetch_positions(_wallet_address):
        return {}

    async def fake_balance():
        return 0.0

    monkeypatch.setattr(bot, "fetch_positions_from_api", fake_fetch_positions)
    monkeypatch.setattr(bot, "fetch_usdc_balance", fake_balance)
    monkeypatch.setattr(bot, "_fetch_conditional_balance_shares", lambda *_args, **_kwargs: 0.0)

    asyncio.run(bot.sync_positions_from_api("0xabc"))

    position = bot.portfolio.get_position(0)
    assert position is not None
    assert position.no_shares == 0.0
    assert position.has_no_api_missing_unverified is False
    assert bot.get_status()["position_sync"]["warning_count"] == 0


def test_generate_candidates_logs_sell_candidates_in_priority_order(monkeypatch, caplog):
    portfolio = Portfolio(
        initial_capital=100.0,
        capital=90.0,
        num_bins=1,
        probabilities=[0.5],
    )
    portfolio.execute_buy_yes(0, 20.0, 0.5, "yes-0")
    orderbooks = {
        0: _make_orderbook(
            bin_index=0,
            yes_bids=[(0.45, 100.0)],
            yes_asks=[(0.46, 100.0)],
        )
    }
    sell_candidate = _make_candidate(
        action=TradeAction.SELL_YES,
        size=20.0,
        price=0.45,
        bin_index=0,
        utility_gain=0.012,
        reservation_price=0.40,
        threshold_price=0.42,
    )

    monkeypatch.setattr(candidates_module, "_generate_buy_yes_candidate", lambda **_kwargs: None)
    monkeypatch.setattr(candidates_module, "_generate_sell_yes_candidate", lambda **_kwargs: sell_candidate)
    monkeypatch.setattr(candidates_module, "_generate_buy_no_candidate", lambda **_kwargs: None)
    monkeypatch.setattr(candidates_module, "_generate_sell_no_candidate", lambda **_kwargs: None)

    with caplog.at_level(logging.INFO):
        candidates = generate_candidates(
            portfolio=portfolio,
            orderbooks=orderbooks,
            config=KellyConfig(),
            hours_to_settlement=12.0,
            verbose=True,
        )

    assert len(candidates) == 1
    assert candidates[0].action == TradeAction.SELL_YES
    assert "All SELL candidates (priority order):" in caplog.text
    assert "Bin 0 SELL_YES" in caplog.text


def test_compute_optimal_trades_skips_blocking_sell_and_keeps_sell_priority(monkeypatch, caplog):
    executor = _make_executor(probabilities=[0.2, 0.3, 0.5])
    candidates = [
        _make_candidate(
            action=TradeAction.SELL_YES,
            bin_index=0,
            size=50.0,
            price=0.72,
            utility_gain=0.010,
            reservation_price=0.60,
            threshold_price=0.60,
        ),
        _make_candidate(
            action=TradeAction.SELL_YES,
            bin_index=1,
            size=75.0,
            price=0.68,
            utility_gain=0.011,
            reservation_price=0.55,
            threshold_price=0.55,
        ),
        _make_candidate(
            action=TradeAction.BUY_YES,
            bin_index=2,
            size=80.0,
            price=0.10,
            utility_gain=0.040,
            reservation_price=0.20,
            threshold_price=0.18,
        ),
    ]
    call_count = {"n": 0}

    def fake_generate(*, portfolio, orderbooks, config, hours_to_settlement, verbose=False):
        del portfolio, orderbooks, config, hours_to_settlement, verbose
        call_count["n"] += 1
        if call_count["n"] == 1:
            return list(candidates)
        return []

    monkeypatch.setattr(executor_module, "generate_candidates", fake_generate)
    executor._find_optimal_size_on = (
        lambda portfolio, candidate, orderbooks, hours, tick_config=None: candidate.size
    )

    def fake_simulate_trade(portfolio, candidate):
        after = portfolio._copy()
        after._mock_candidate_bin = candidate.bin_index
        return after

    monkeypatch.setattr(executor, "_simulate_trade", fake_simulate_trade)
    monkeypatch.setattr(
        candidates_module,
        "_compute_portfolio_utility_gain",
        lambda before, after, config: {0: 0.005, 1: 0.020, 2: 0.030}[after._mock_candidate_bin],
    )

    with caplog.at_level(logging.INFO):
        planned = executor._compute_optimal_trades(
            executor.portfolio,
            {},
            hours_to_settlement=12.0,
            verbose=True,
        )

    assert len(planned) == 1
    assert planned[0].action == TradeAction.SELL_YES
    assert planned[0].bin_index == 1
    assert "Reject SELL_YES bin=0: sized_utility_below_min" in caplog.text
    assert "[test-event][SIM iter=0] SELL_YES bin=1" in caplog.text


def test_compute_optimal_trades_skips_sell_with_size_below_one(monkeypatch):
    executor = _make_executor(probabilities=[0.4, 0.6])
    candidates = [
        _make_candidate(
            action=TradeAction.SELL_YES,
            bin_index=0,
            size=10.0,
            price=0.50,
            utility_gain=0.010,
        ),
        _make_candidate(
            action=TradeAction.SELL_YES,
            bin_index=1,
            size=25.0,
            price=0.55,
            utility_gain=0.011,
        ),
    ]
    call_count = {"n": 0}

    def fake_generate(*, portfolio, orderbooks, config, hours_to_settlement, verbose=False):
        del portfolio, orderbooks, config, hours_to_settlement, verbose
        call_count["n"] += 1
        if call_count["n"] == 1:
            return list(candidates)
        return []

    monkeypatch.setattr(executor_module, "generate_candidates", fake_generate)
    executor._find_optimal_size_on = (
        lambda portfolio, candidate, orderbooks, hours, tick_config=None:
        0.0 if candidate.bin_index == 0 else candidate.size
    )

    def fake_simulate_trade(portfolio, candidate):
        after = portfolio._copy()
        after._mock_candidate_bin = candidate.bin_index
        return after

    monkeypatch.setattr(executor, "_simulate_trade", fake_simulate_trade)
    monkeypatch.setattr(
        candidates_module,
        "_compute_portfolio_utility_gain",
        lambda before, after, config: 0.020 if after._mock_candidate_bin == 1 else 0.0,
    )

    planned = executor._compute_optimal_trades(
        executor.portfolio,
        {},
        hours_to_settlement=12.0,
        verbose=False,
    )

    assert len(planned) == 1
    assert planned[0].action == TradeAction.SELL_YES
    assert planned[0].bin_index == 1


def test_compute_optimal_trades_skips_fak_cooldown_and_uses_next_candidate(monkeypatch):
    executor = _make_executor(probabilities=[0.4, 0.6])
    executor._record_fak_failure(0)
    candidates = [
        _make_candidate(
            action=TradeAction.SELL_YES,
            bin_index=0,
            size=30.0,
            price=0.40,
            utility_gain=0.010,
        ),
        _make_candidate(
            action=TradeAction.SELL_YES,
            bin_index=1,
            size=35.0,
            price=0.41,
            utility_gain=0.011,
        ),
    ]
    call_count = {"n": 0}

    def fake_generate(*, portfolio, orderbooks, config, hours_to_settlement, verbose=False):
        del portfolio, orderbooks, config, hours_to_settlement, verbose
        call_count["n"] += 1
        if call_count["n"] == 1:
            return list(candidates)
        return []

    monkeypatch.setattr(executor_module, "generate_candidates", fake_generate)
    executor._find_optimal_size_on = (
        lambda portfolio, candidate, orderbooks, hours, tick_config=None: candidate.size
    )

    def fake_simulate_trade(portfolio, candidate):
        after = portfolio._copy()
        after._mock_candidate_bin = candidate.bin_index
        return after

    monkeypatch.setattr(executor, "_simulate_trade", fake_simulate_trade)
    monkeypatch.setattr(
        candidates_module,
        "_compute_portfolio_utility_gain",
        lambda before, after, config: 0.020 if after._mock_candidate_bin == 1 else 0.0,
    )

    planned = executor._compute_optimal_trades(
        executor.portfolio,
        {},
        hours_to_settlement=12.0,
        verbose=False,
    )

    assert len(planned) == 1
    assert planned[0].action == TradeAction.SELL_YES
    assert planned[0].bin_index == 1


def test_run_tick_does_not_rebuy_when_api_is_stale_after_confirmed_fill():
    class FakeLiveOrderExecutor:
        def __init__(self):
            self.dry_run = False
            self.calls = []
            self.executor = None
            self.bin_ranges = {}

        def place_batch_orders(self, orders):
            responses = []
            loop = asyncio.get_running_loop()
            for order in orders:
                order_id = f"order-{len(self.calls) + 1}"
                self.calls.append(dict(order))
                fill = _make_fill(
                    order_id=order_id,
                    token_id=order["token_id"],
                    side=order["side"],
                    price=order["price"],
                    size=order["size"],
                    match_id=f"match-{order_id}",
                )
                loop.call_soon(self.executor.handle_fill, fill)
                responses.append({"orderID": order_id})
            return responses

        def cancel_order(self, _order_id):
            return True

        def _fetch_conditional_balance_allowance(self, token_id, refresh=True):
            return {"token_id": token_id, "refresh": refresh}

    async def run_case():
        order_executor = FakeLiveOrderExecutor()
        executor = _make_executor(
            rate_limit=RateLimitConfig(
                max_orders_per_tick=3,
                tick_timeout_seconds=5.0,
                overlay_reconciliation_grace_seconds=90.0,
                integrity_freeze_max_seconds=180.0,
            )
        )
        executor.order_executor = order_executor
        order_executor.executor = executor
        executor._post_confirm_delay = 0.0

        async def stale_sync():
            # Intentionally leave the API base unchanged across iterations.
            return None

        executor.sync_portfolio = stale_sync
        executor._get_orderbooks = lambda: {
            0: _make_orderbook(
                yes_asks=[(0.20, 100.0)],
            )
        }

        def fake_compute(portfolio, orderbooks, hours_to_settlement, verbose=False):
            del orderbooks, hours_to_settlement, verbose
            position = portfolio.get_position(0)
            yes_shares = position.yes_shares if position else 0.0
            if yes_shares < 1.0:
                return [_make_candidate(size=20.0, price=0.2)]
            return []

        executor._compute_optimal_trades = fake_compute

        result = await executor.run_tick(hours_to_settlement=2.0)
        integrity = executor.get_integrity_summary()

        assert result.num_executed == 1
        assert len(order_executor.calls) == 1
        assert integrity["frozen"] is False
        assert integrity["overlay_entries"] == 1

    asyncio.run(run_case())


def test_run_tick_does_not_batch_merge_orders_with_different_execution_bounds():
    class FakeLiveOrderExecutor:
        def __init__(self):
            self.dry_run = False
            self.calls = []
            self.executor = None
            self.bin_ranges = {}

        def get_tick_size(self, token_id):
            del token_id
            return "0.01"

        def place_batch_orders(self, orders):
            responses = []
            loop = asyncio.get_running_loop()
            for order in orders:
                order_id = f"order-{len(self.calls) + 1}"
                self.calls.append(dict(order))
                fill = _make_fill(
                    order_id=order_id,
                    token_id=order["token_id"],
                    side=order["side"],
                    price=order["price"],
                    size=order["size"],
                    match_id=f"match-{order_id}",
                )
                loop.call_soon(self.executor.handle_fill, fill)
                responses.append({"orderID": order_id})
            return responses

        def cancel_order(self, _order_id):
            return True

        def _fetch_conditional_balance_allowance(self, token_id, refresh=True):
            return {"token_id": token_id, "refresh": refresh}

    async def run_case():
        order_executor = FakeLiveOrderExecutor()
        executor = _make_executor(
            rate_limit=RateLimitConfig(
                max_orders_per_tick=2,
                tick_timeout_seconds=5.0,
                overlay_reconciliation_grace_seconds=90.0,
                integrity_freeze_max_seconds=180.0,
            )
        )
        executor.order_executor = order_executor
        order_executor.executor = executor
        executor._post_confirm_delay = 0.0

        async def stale_sync():
            return None

        executor.sync_portfolio = stale_sync
        orderbooks = {
            0: _make_orderbook(
                yes_asks=[(0.20, 100.0)],
            )
        }
        executor._get_orderbooks = lambda: orderbooks

        calls = {"n": 0}

        def fake_compute(portfolio, orderbooks, hours_to_settlement, verbose=False):
            del portfolio, orderbooks, hours_to_settlement, verbose
            calls["n"] += 1
            if calls["n"] > 1:
                return []
            return [
                _make_candidate(
                    action=TradeAction.BUY_YES,
                    size=20.0,
                    price=0.20,
                    threshold_price=0.25,
                    execution_bound_price=0.25,
                ),
                _make_candidate(
                    action=TradeAction.BUY_YES,
                    size=20.0,
                    price=0.20,
                    threshold_price=0.23,
                    execution_bound_price=0.23,
                ),
            ]

        executor._compute_optimal_trades = fake_compute

        result = await executor.run_tick(hours_to_settlement=2.0)

        assert result.num_executed == 2
        assert len(order_executor.calls) == 2

    asyncio.run(run_case())


def test_run_tick_bookkeeping_uses_submitted_values_after_repricing_and_size_reduction():
    class FakeLiveOrderExecutor:
        def __init__(self):
            self.dry_run = False
            self.calls = []
            self.executor = None
            self.bin_ranges = {}

        def get_tick_size(self, token_id):
            del token_id
            return "0.01"

        def place_batch_orders(self, orders):
            responses = []
            loop = asyncio.get_running_loop()
            for order in orders:
                order_id = f"order-{len(self.calls) + 1}"
                self.calls.append(dict(order))
                fill = _make_fill(
                    order_id=order_id,
                    token_id=order["token_id"],
                    side=order["side"],
                    price=order["price"],
                    size=order["size"],
                    match_id=f"match-{order_id}",
                )
                loop.call_soon(self.executor.handle_fill, fill)
                responses.append({"orderID": order_id})
            return responses

        def cancel_order(self, _order_id):
            return True

        def _fetch_conditional_balance_allowance(self, token_id, refresh=True):
            return {"token_id": token_id, "refresh": refresh}

    async def run_case():
        order_executor = FakeLiveOrderExecutor()
        executor = _make_executor(
            capital=1_000.0,
            rate_limit=RateLimitConfig(
                max_orders_per_tick=1,
                tick_timeout_seconds=5.0,
                overlay_reconciliation_grace_seconds=90.0,
                integrity_freeze_max_seconds=180.0,
            ),
        )
        executor.order_executor = order_executor
        order_executor.executor = executor
        executor._post_confirm_delay = 0.0

        async def stale_sync():
            return None

        executor.sync_portfolio = stale_sync
        orderbooks = {
            0: _make_orderbook(
                yes_asks=[(0.40, 20.0), (0.41, 10.0)],
            )
        }
        executor._get_orderbooks = lambda: orderbooks

        calls = {"n": 0}

        def fake_compute(portfolio, orderbooks, hours_to_settlement, verbose=False):
            del portfolio, orderbooks, hours_to_settlement, verbose
            calls["n"] += 1
            if calls["n"] > 1:
                return []
            return [
                _make_candidate(
                    action=TradeAction.BUY_YES,
                    size=35.0,
                    price=0.40,
                    utility_gain=0.14,
                    threshold_price=0.45,
                    execution_bound_price=0.41,
                )
            ]

        executor._compute_optimal_trades = fake_compute

        result = await executor.run_tick(hours_to_settlement=2.0)

        assert result.num_executed == 1
        assert len(order_executor.calls) == 1
        assert order_executor.calls[0]["size"] == 30
        assert order_executor.calls[0]["price"] == pytest.approx(0.41)

        executed_candidate = result.executions[0].candidate
        assert executed_candidate.size == pytest.approx(30.0)
        assert executed_candidate.limit_price == pytest.approx(0.41)
        assert executed_candidate.price == pytest.approx((20 * 0.40 + 10 * 0.41) / 30)
        assert executed_candidate.utility_gain == pytest.approx(0.12)
        assert result.total_utility_gain == pytest.approx(0.12)

    asyncio.run(run_case())


def test_run_tick_logs_submission_resolution_for_buy_and_sell(caplog):
    class FakeLiveOrderExecutor:
        def __init__(self):
            self.dry_run = False
            self.calls = []
            self.executor = None
            self.bin_ranges = {}

        def get_tick_size(self, token_id):
            del token_id
            return "0.01"

        def place_batch_orders(self, orders):
            responses = []
            loop = asyncio.get_running_loop()
            for order in orders:
                order_id = f"order-{len(self.calls) + 1}"
                self.calls.append(dict(order))
                fill = _make_fill(
                    order_id=order_id,
                    token_id=order["token_id"],
                    side=order["side"],
                    price=order["price"],
                    size=order["size"],
                    match_id=f"match-{order_id}",
                )
                loop.call_soon(self.executor.handle_fill, fill)
                responses.append({"orderID": order_id})
            return responses

        def cancel_order(self, _order_id):
            return True

        def _fetch_conditional_balance_allowance(self, token_id, refresh=True):
            return {"token_id": token_id, "refresh": refresh}

    async def run_case(
        *,
        event_name,
        portfolio,
        orderbook,
        candidate,
        probabilities,
    ):
        order_executor = FakeLiveOrderExecutor()
        executor = _make_executor(
            capital=portfolio.initial_capital,
            rate_limit=RateLimitConfig(
                max_orders_per_tick=1,
                tick_timeout_seconds=5.0,
                overlay_reconciliation_grace_seconds=90.0,
                integrity_freeze_max_seconds=180.0,
            ),
            probabilities=probabilities,
        )
        executor.portfolio = portfolio
        executor.api_base_portfolio = portfolio
        executor.order_executor = order_executor
        executor.event_name = event_name
        order_executor.executor = executor
        executor._post_confirm_delay = 0.0

        async def stale_sync():
            return None

        executor.sync_portfolio = stale_sync
        executor._get_orderbooks = lambda: {0: orderbook}
        executor.set_log_context(
            probabilities=probabilities,
            orderbooks={0: orderbook},
            bin_ranges={0: "demo-bin"},
            hours_to_settlement=2.0,
        )

        calls = {"n": 0}

        def fake_compute(portfolio, orderbooks, hours_to_settlement, verbose=False):
            del portfolio, orderbooks, hours_to_settlement, verbose
            calls["n"] += 1
            if calls["n"] > 1:
                return []
            return [candidate]

        executor._compute_optimal_trades = fake_compute

        with caplog.at_level(logging.INFO):
            result = await executor.run_tick(hours_to_settlement=2.0)

        return order_executor.calls, result

    async def main():
        buy_portfolio = Portfolio(
            initial_capital=1_000.0,
            capital=1_000.0,
            num_bins=1,
            probabilities=[1.0],
        )
        buy_orderbook = _make_orderbook(
            yes_asks=[(0.40, 20.0), (0.41, 10.0)],
        )
        buy_candidate = _make_candidate(
            action=TradeAction.BUY_YES,
            size=35.0,
            price=0.40,
            utility_gain=0.14,
            threshold_price=0.45,
            execution_bound_price=0.41,
        )

        buy_calls, buy_result = await run_case(
            event_name="TEST_BUY_RESOLUTION",
            portfolio=buy_portfolio,
            orderbook=buy_orderbook,
            candidate=buy_candidate,
            probabilities=[1.0],
        )

        assert buy_result.num_executed == 1
        assert buy_calls == [
            {
                "token_id": "yes-0",
                "side": "BUY",
                "price": 0.41,
                "size": 30,
                "execution_bound_price": 0.41,
                "fak_resolved": True,
            }
        ]

        sell_portfolio = Portfolio(
            initial_capital=1_000.0,
            capital=700.0,
            num_bins=1,
            probabilities=[1.0],
        )
        sell_portfolio.execute_buy_no(0, 40.0, 0.55, "yes-0")
        sell_orderbook = _make_orderbook(
            yes_asks=[(0.40, 20.0), (0.41, 10.0)],
        )
        sell_candidate = _make_candidate(
            action=TradeAction.SELL_NO,
            size=25.0,
            price=0.60,
            utility_gain=0.09,
            threshold_price=0.55,
            execution_bound_price=0.59,
        )
        sell_candidate.limit_price = 0.60

        sell_calls, sell_result = await run_case(
            event_name="TEST_SELL_RESOLUTION",
            portfolio=sell_portfolio,
            orderbook=sell_orderbook,
            candidate=sell_candidate,
            probabilities=[1.0],
        )

        assert sell_result.num_executed == 1
        assert sell_calls == [
            {
                "token_id": "no-0",
                "side": "SELL",
                "price": 0.59,
                "size": 25,
                "execution_bound_price": 0.59,
                "fak_resolved": True,
            }
        ]

    asyncio.run(main())

    assert (
        "Submission resolve BUY_YES bin=0: requested=35 @ limit 0.4000 "
        "bound=0.4100 -> submitted=30 @ limit 0.4100"
    ) in caplog.text
    assert (
        "Submission resolve SELL_NO bin=0: requested=25 @ limit 0.6000 "
        "bound=0.5900 -> submitted=25 @ limit 0.5900"
    ) in caplog.text
