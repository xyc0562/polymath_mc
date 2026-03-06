import asyncio
from datetime import datetime, timezone

from src.algo.musk_tweet_count.kelly.candidates import TradeAction, TradeCandidate
from src.algo.musk_tweet_count.kelly.config import KellyConfig, RateLimitConfig
from src.algo.musk_tweet_count.kelly.executor import KellyExecutor
from src.algo.musk_tweet_count.kelly.integration import KellyTradingBot
from src.algo.musk_tweet_count.kelly.portfolio import Portfolio
from src.algo.musk_tweet_count.kelly.user_stream import FillEvent, OrderStatus


def _make_candidate(
    *,
    action: TradeAction = TradeAction.BUY_YES,
    size: float = 10.0,
    price: float = 0.2,
    bin_index: int = 0,
) -> TradeCandidate:
    return TradeCandidate(
        bin_index=bin_index,
        action=action,
        size=size,
        price=price,
        utility_gain=0.01,
        reservation_price=0.25,
        edge=0.05,
        limit_price=price,
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
) -> KellyExecutor:
    config = KellyConfig(rate_limit=rate_limit or RateLimitConfig())
    portfolio = Portfolio(
        initial_capital=capital,
        capital=capital,
        num_bins=1,
        probabilities=[1.0],
    )
    executor = KellyExecutor(
        config=config,
        portfolio=portfolio,
        token_ids={0: "yes-0"},
        no_token_ids={0: "no-0"},
        event_name="test-event",
    )
    return executor


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
                integrity_freeze_max_seconds=300.0,
            )
        )
        executor.order_executor = order_executor
        order_executor.executor = executor
        executor._post_confirm_delay = 0.0

        async def stale_sync():
            # Intentionally leave the API base unchanged across iterations.
            return None

        executor.sync_portfolio = stale_sync
        executor._get_orderbooks = lambda: {}

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
