"""Tests for maker mode: ActivityStateTracker, GTD executor wrappers,
QuoteManager gates / desired quotes / reconcile diff / kill switches /
shadow mode, and the capital + fill integration hooks.

Each test maps to an entry in the maker boundary-condition inventory.
"""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from src.algo.musk_tweet_count.kelly.activity_state import ActivityStateTracker
from src.algo.musk_tweet_count.kelly.candidates import TradeAction
from src.algo.musk_tweet_count.kelly.config import (
    KellyConfig,
    MakerConfig,
    RateLimitConfig,
)
from src.algo.musk_tweet_count.kelly.maker_event_log import MakerEventLog
from src.algo.musk_tweet_count.kelly.executor import (
    ORDER_MANAGER_BREAKER,
    KellyExecutor,
    OrderExecutor,
    SYNC_STALENESS_HALT_SECONDS,
)
from src.algo.musk_tweet_count.kelly.orderbook import OrderbookLevel, UnifiedOrderbook
from src.algo.musk_tweet_count.kelly.portfolio import Portfolio
from src.algo.musk_tweet_count.kelly.quote_manager import QuoteManager, RestingOrder
from src.algo.musk_tweet_count.kelly.user_stream import PendingOrder, UserStreamClient


@pytest.fixture(autouse=True)
def _closed_breaker():
    ORDER_MANAGER_BREAKER.reset()
    yield
    ORDER_MANAGER_BREAKER.reset()


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


class FakeOrderExecutor:
    def __init__(self):
        self.dry_run = False
        self.placed = []
        self.cancelled = []
        self.open_orders_response = []
        self.open_orders_calls = 0
        self.cancel_result = True
        self.fail_place = False
        self._seq = 0

    def get_tick_size(self, token_id):
        return "0.01"

    def place_gtd_order(self, token_id, side, price, size, ttl_seconds):
        self.placed.append(
            {
                "token_id": token_id,
                "side": side,
                "price": price,
                "size": size,
                "ttl_seconds": ttl_seconds,
            }
        )
        if self.fail_place:
            return None
        self._seq += 1
        return {"orderID": f"ord-{self._seq}"}

    def cancel_orders(self, order_ids):
        self.cancelled.append(list(order_ids))
        return self.cancel_result

    def get_open_orders(self):
        self.open_orders_calls += 1
        return self.open_orders_response


class FakeKellyExecutor:
    def __init__(self, portfolio):
        self._integrity_state = SimpleNamespace(frozen=False)
        self._last_successful_sync_time = time.time()
        self._pending_orders = {}
        self.remembered = []
        self._base = portfolio
        self.maker_collateral_provider = None
        self.maker_pre_trade_hook = None

    def _build_effective_portfolio(self):
        return self._base._copy()

    def _remember_order_context(self, order_id, candidate, token_id):
        self.remembered.append((order_id, candidate, token_id))


def _book(bin_index, bid, ask, token="yes-0", ts=None):
    return UnifiedOrderbook(
        bin_index=bin_index,
        yes_token_id=token,
        yes_bids=[OrderbookLevel(price=bid, size=500.0)],
        yes_asks=[OrderbookLevel(price=ask, size=500.0)],
        last_updated=ts if ts is not None else time.time(),
    )


def _quiet_tracker(maker: MakerConfig) -> ActivityStateTracker:
    tracker = ActivityStateTracker(
        quiet_window_seconds=maker.quiet_window_seconds,
        storm_window_seconds=maker.storm_window_seconds,
        storm_count=maker.storm_count,
    )
    # Rewind the conservative startup seed so the state is quiet now.
    tracker._last_event_ts = time.time() - maker.quiet_window_seconds - 10
    tracker.note_poll()
    return tracker


def make_qm(mode="live", maker_kwargs=None, kelly_kwargs=None, portfolio=None):
    maker = MakerConfig(mode=mode, **(maker_kwargs or {}))
    kelly_fields = dict(
        min_buy_utility=1e-9,
        rate_limit=RateLimitConfig(),
        maker=maker,
    )
    kelly_fields.update(kelly_kwargs or {})
    kelly = KellyConfig(**kelly_fields)
    if portfolio is None:
        portfolio = Portfolio(
            initial_capital=100.0,
            capital=100.0,
            num_bins=3,
            probabilities=[0.5, 0.3, 0.2],
            bin_upper_bounds=[10, 20, 30],
        )
    kelly_executor = FakeKellyExecutor(portfolio)
    order_executor = FakeOrderExecutor()
    qm = QuoteManager(
        maker_config=maker,
        kelly_config=kelly,
        order_executor=order_executor,
        kelly_executor=kelly_executor,
        portfolio=portfolio,
        token_ids={0: "yes-0", 1: "yes-1", 2: "yes-2"},
        no_token_ids={0: "no-0", 1: "no-1", 2: "no-2"},
        event_name="maker-test",
        activity_tracker=_quiet_tracker(maker),
    )
    return qm, order_executor, kelly_executor, portfolio


def _standard_books():
    # Bin 0: fair 0.5, bid 0.38 / ask 0.44 -> BUY_YES quote at 0.39
    return {0: _book(0, 0.38, 0.44, token="yes-0")}


# ----------------------------------------------------------------------
# ActivityStateTracker
# ----------------------------------------------------------------------


def test_tracker_seeds_not_quiet_then_quiet_after_window():
    tracker = ActivityStateTracker(quiet_window_seconds=1200.0)
    now = time.time()
    assert tracker.state(now) == "active"  # startup seed: fail closed
    assert tracker.state(now + 1201) == "quiet"


def test_tracker_event_makes_active_then_decays():
    tracker = ActivityStateTracker(quiet_window_seconds=1200.0)
    now = time.time()
    tracker.note_event(now + 2000)
    assert tracker.state(now + 2100) == "active"
    assert tracker.state(now + 2000 + 1201) == "quiet"


def test_tracker_storm_on_burst():
    tracker = ActivityStateTracker(
        quiet_window_seconds=1200.0, storm_window_seconds=1800.0, storm_count=5
    )
    now = time.time()
    for i in range(5):
        tracker.note_event(now + i * 60)
    assert tracker.state(now + 300) == "storm"
    # Burst ages out of the storm window -> quiet again
    assert tracker.state(now + 240 + 1801 + 1200) == "quiet"


def test_tracker_poll_staleness():
    tracker = ActivityStateTracker()
    assert tracker.seconds_since_poll() is None
    now = time.time()
    tracker.note_poll(now)
    assert tracker.seconds_since_poll(now + 30) == pytest.approx(30.0)


# ----------------------------------------------------------------------
# OrderExecutor GTD wrappers
# ----------------------------------------------------------------------


class FakeClobClient:
    def __init__(self):
        self.created = []
        self.posted = []
        self.cancelled_batches = []
        self.raise_on_cancel = False
        self.raise_on_open_orders = False

    def create_order(self, order_args):
        self.created.append(order_args)
        return {"signed": True}

    def post_order(self, signed_order, order_type):
        self.posted.append((signed_order, order_type))
        return {"orderID": "gtd-1", "success": True}

    def cancel_orders(self, order_ids):
        self.cancelled_batches.append(list(order_ids))
        if self.raise_on_cancel:
            raise RuntimeError("cancel failed")
        return {"canceled": order_ids}

    def get_open_orders(self):
        if self.raise_on_open_orders:
            raise RuntimeError("listing down")
        return [{"id": "abc", "asset_id": "tok"}]


def test_place_gtd_order_signs_expiration_and_gtd_type():
    from py_clob_client_v2.clob_types import OrderType

    client = FakeClobClient()
    executor = OrderExecutor(clob_client=client, dry_run=False, event_name="t")

    before = time.time()
    response = executor.place_gtd_order(
        token_id="tok", side="BUY", price=0.39, size=100.0, ttl_seconds=360.0
    )

    assert response["orderID"] == "gtd-1"
    order_args = client.created[0]
    assert order_args.expiration >= int(before + 360) - 1
    assert order_args.expiration <= int(time.time() + 360) + 1
    assert order_args.size == 100
    assert client.posted[0][1] == OrderType.GTD


def test_place_gtd_order_rejects_dust_buy():
    client = FakeClobClient()
    executor = OrderExecutor(clob_client=client, dry_run=False, event_name="t")

    # 5 shares at $0.10 = $0.50: below 15 shares AND below $1 value
    assert executor.place_gtd_order("tok", "BUY", 0.10, 5.0, 360.0) is None
    assert client.created == []


def test_cancel_orders_batch_and_failure():
    client = FakeClobClient()
    executor = OrderExecutor(clob_client=client, dry_run=False, event_name="t")

    assert executor.cancel_orders(["a", "b"]) is True
    assert client.cancelled_batches == [["a", "b"]]
    assert executor.cancel_orders([]) is True  # no-op, no API call
    assert len(client.cancelled_batches) == 1

    client.raise_on_cancel = True
    assert executor.cancel_orders(["c"]) is False


def test_get_open_orders_error_returns_none():
    client = FakeClobClient()
    executor = OrderExecutor(clob_client=client, dry_run=False, event_name="t")

    assert executor.get_open_orders() == [{"id": "abc", "asset_id": "tok"}]
    client.raise_on_open_orders = True
    assert executor.get_open_orders() is None


# ----------------------------------------------------------------------
# User-stream stale janitor override
# ----------------------------------------------------------------------


def test_pending_order_stale_after_override():
    client = UserStreamClient(api_key="k", api_secret="s", api_passphrase="p")
    stale_seen = []

    async def on_stale(pending):
        stale_seen.append(pending.order_id)

    client.on_stale_order = on_stale

    old = time.time() - 120  # past the 30s FAK default
    fak = PendingOrder(
        order_id="fak-1", token_id="t", side="BUY", price=0.5, size=10,
        bin_index=0, created_at=old,
    )
    maker = PendingOrder(
        order_id="maker-1", token_id="t", side="BUY", price=0.5, size=10,
        bin_index=0, created_at=old, stale_after=420.0,
    )
    client._pending_orders = {"fak-1": fak, "maker-1": maker}

    asyncio.run(client._check_stale_orders())

    assert stale_seen == ["fak-1"]
    assert "maker-1" in client._pending_orders

    # Past its own stale_after the maker order is janitored too
    maker.created_at = time.time() - 421
    asyncio.run(client._check_stale_orders())
    assert "maker-1" not in client._pending_orders


# ----------------------------------------------------------------------
# Maker-side fill attribution (user channel trade events)
# ----------------------------------------------------------------------


def _mk_pending(order_id="mk-1", size=100.0, resting=True):
    return PendingOrder(
        order_id=order_id,
        token_id="yes-0",
        side="BUY",
        price=0.39,
        size=size,
        bin_index=0,
        resting=resting,
        stale_after=420.0 if resting else None,
    )


def _trade_payload(trade_id="trade-1", taker_id="taker-1", status="MATCHED", makers=None):
    return {
        "event_type": "trade",
        "id": trade_id,
        "taker_order_id": taker_id,
        "asset_id": "yes-0",
        "side": "SELL",
        "price": "0.39",
        "size": "100",
        "status": status,
        "maker_orders": makers or [],
    }


def _fill_client():
    client = UserStreamClient(api_key="k", api_secret="s", api_passphrase="p")
    received = []
    client.on_fill = received.append
    return client, received


def test_maker_side_fill_attributed_from_maker_orders():
    client, received = _fill_client()
    pending = _mk_pending()
    client._pending_orders = {"mk-1": pending}

    payload = _trade_payload(makers=[
        {"order_id": "mk-1", "matched_amount": "40", "price": "0.39", "asset_id": "yes-0"},
        {"order_id": "counterparty-2", "matched_amount": "60", "price": "0.40"},
    ])
    asyncio.run(client._handle_trade_event(payload))

    ours = [f for f in received if f.order_id == "mk-1"]
    assert len(ours) == 1
    fill = ours[0]
    assert fill.size == pytest.approx(40.0)  # OUR matched portion, not the taker's 100
    assert fill.price == pytest.approx(0.39)
    assert fill.side == "BUY"  # our side from the pending order, not the taker's SELL
    assert fill.match_id == "trade-1:mk-1"

    # Taker fill still dispatched exactly as before (unknown downstream)
    assert any(f.order_id == "taker-1" and f.size == 100.0 for f in received)

    # Partial fill: order keeps resting, tracking accumulates
    assert pending.filled_size == pytest.approx(40.0)
    assert "mk-1" in client._pending_orders


def test_maker_partial_confirmed_keeps_pending_until_full():
    client, received = _fill_client()
    pending = _mk_pending(size=100.0)
    client._pending_orders = {"mk-1": pending}

    partial = _trade_payload(status="CONFIRMED", makers=[
        {"order_id": "mk-1", "matched_amount": "40", "price": "0.39"},
    ])
    asyncio.run(client._handle_trade_event(partial))
    # CONFIRMED partial on a RESTING order: remainder still live -> tracked
    assert "mk-1" in client._pending_orders
    assert pending.filled_size == pytest.approx(40.0)

    # Duplicate resend of the same trade dedups (no double count)
    asyncio.run(client._handle_trade_event(partial))
    assert pending.filled_size == pytest.approx(40.0)

    completing = _trade_payload(trade_id="trade-2", status="CONFIRMED", makers=[
        {"order_id": "mk-1", "matched_amount": "60", "price": "0.39"},
    ])
    asyncio.run(client._handle_trade_event(completing))
    assert pending.filled_size == pytest.approx(100.0)
    assert "mk-1" not in client._pending_orders


def test_fak_confirmed_removal_unchanged():
    # Legacy taker semantics: a FAK order is done once its trade confirms,
    # even partially filled (the remainder was killed by the exchange).
    client, _ = _fill_client()
    client._pending_orders = {"fak-1": _mk_pending("fak-1", resting=False)}

    asyncio.run(client._handle_trade_event(
        _trade_payload(taker_id="fak-1", status="CONFIRMED")
    ))
    assert "fak-1" not in client._pending_orders


def test_counterparty_maker_entries_ignored():
    client, received = _fill_client()
    client._pending_orders = {}

    asyncio.run(client._handle_trade_event(_trade_payload(makers=[
        {"order_id": "someone-else", "matched_amount": "100", "price": "0.39"},
    ])))

    # Only the taker fill is dispatched (and dropped downstream as unknown)
    assert [f.order_id for f in received] == ["taker-1"]


def test_two_own_maker_orders_one_trade_distinct_match_ids():
    client, received = _fill_client()
    client._pending_orders = {
        "mk-1": _mk_pending("mk-1"),
        "mk-2": _mk_pending("mk-2"),
    }

    asyncio.run(client._handle_trade_event(_trade_payload(makers=[
        {"order_id": "mk-1", "matched_amount": "30", "price": "0.39"},
        {"order_id": "mk-2", "matched_amount": "70", "price": "0.38"},
    ])))

    ours = {f.order_id: f for f in received if f.order_id.startswith("mk-")}
    assert ours["mk-1"].match_id == "trade-1:mk-1"
    assert ours["mk-2"].match_id == "trade-1:mk-2"
    assert client._pending_orders["mk-1"].filled_size == pytest.approx(30.0)
    assert client._pending_orders["mk-2"].filled_size == pytest.approx(70.0)


def test_place_quote_registers_resting_pending_order():
    class FakeUserStream:
        def __init__(self):
            self.added = []

        async def add_pending_order(self, pending):
            self.added.append(pending)

    qm, _, _, _ = make_qm()
    stream = FakeUserStream()
    qm.user_stream = stream

    async def main():
        await qm.reconcile(orderbooks=_standard_books(), hours_to_settlement=48.0)
        await asyncio.sleep(0)  # let the fire-and-forget registration run

    asyncio.run(main())

    assert len(stream.added) == 1
    pending = stream.added[0]
    assert pending.resting is True
    assert pending.stale_after == qm.config.ttl_seconds + 60.0
    assert pending.side == "BUY"


# ----------------------------------------------------------------------
# Gates
# ----------------------------------------------------------------------


def test_gate_final_window():
    qm, _, _, _ = make_qm()
    ok, reason = qm._check_gates(hours_to_settlement=11.0, now=time.time())
    assert not ok and reason.startswith("final_")
    ok, _ = qm._check_gates(hours_to_settlement=13.0, now=time.time())
    assert ok


def test_gate_breaker_open():
    qm, _, _, _ = make_qm()
    ORDER_MANAGER_BREAKER.record_failure()
    ok, reason = qm._check_gates(48.0, time.time())
    assert not ok and reason == "order_manager_breaker_open"


def test_gate_integrity_frozen():
    qm, _, ke, _ = make_qm()
    ke._integrity_state.frozen = True
    ok, reason = qm._check_gates(48.0, time.time())
    assert not ok and reason == "integrity_frozen"


def test_gate_sync_stale():
    qm, _, ke, _ = make_qm()
    ke._last_successful_sync_time = time.time() - SYNC_STALENESS_HALT_SECONDS - 1
    ok, reason = qm._check_gates(48.0, time.time())
    assert not ok and reason == "position_sync_stale"


def test_gate_activity_states():
    now = time.time()

    qm, _, _, _ = make_qm()
    qm.activity_tracker = None
    assert qm._check_gates(48.0, now) == (False, "no_activity_tracker")

    qm, _, _, _ = make_qm()
    qm.activity_tracker._last_poll_ts = now - 500  # stale poll
    ok, reason = qm._check_gates(48.0, now)
    assert not ok and reason == "activity_tracker_stale"

    qm, _, _, _ = make_qm()
    qm.activity_tracker.note_event(now - 60)  # in-session
    ok, reason = qm._check_gates(48.0, now)
    assert not ok and reason == "activity_active"

    qm, _, _, _ = make_qm()
    for i in range(5):
        # 5 posts within the 30-minute storm window
        qm.activity_tracker.note_event(now - 300 - i)
    ok, reason = qm._check_gates(48.0, now)
    assert not ok and reason == "activity_storm"


def test_gates_pass_when_all_healthy():
    qm, _, _, _ = make_qm()
    ok, reason = qm._check_gates(48.0, time.time())
    assert ok, reason


# ----------------------------------------------------------------------
# Desired quotes
# ----------------------------------------------------------------------


def test_quote_improves_touch_within_threshold():
    qm, _, _, _ = make_qm()
    desired = qm.compute_desired_quotes(_standard_books())

    assert len(desired) == 1
    q = desired[0]
    assert q.bin_index == 0
    assert q.action == TradeAction.BUY_YES
    assert q.token_id == "yes-0"
    assert q.price == pytest.approx(0.39)  # best bid 0.38 + one tick
    # dollars = min(max_quote 150, bin room 75, budget 75, capital 100) = 75
    assert q.size == int(75.0 / 0.39)


def test_quote_price_capped_by_edge_threshold():
    # Fair 0.5, book 0.48/0.55. The YES quote would sit exactly at the
    # friction-adjusted threshold (0.48); the NO side (0.46 vs NO fair
    # 0.50, 4c edge) is the better Kelly entry and wins the bin. Either
    # way no quote may exceed its side's threshold.
    qm, _, _, _ = make_qm()
    desired = qm.compute_desired_quotes({0: _book(0, 0.48, 0.55, token="yes-0")})
    assert len(desired) == 1
    q = desired[0]
    assert q.action == TradeAction.BUY_NO
    assert q.price == pytest.approx(0.46)

    # A side whose touch already outbids its threshold is never quoted:
    # YES touch 0.49 > 0.48 threshold -> only the NO side may appear.
    qm, _, _, _ = make_qm()
    desired = qm.compute_desired_quotes({0: _book(0, 0.49, 0.55, token="yes-0")})
    assert all(d.action == TradeAction.BUY_NO for d in desired)


def test_no_quote_below_min_spread():
    qm, _, _, _ = make_qm()
    desired = qm.compute_desired_quotes({0: _book(0, 0.42, 0.44, token="yes-0")})
    assert desired == []


def test_no_quote_dead_bin():
    qm, _, _, portfolio = make_qm()
    portfolio.dead_bins = [0]
    desired = qm.compute_desired_quotes(_standard_books())
    assert desired == []


def test_no_quote_stale_book():
    qm, _, _, _ = make_qm()
    stale = _book(0, 0.38, 0.44, token="yes-0", ts=time.time() - 120)
    desired = qm.compute_desired_quotes({0: stale})
    assert desired == []


def test_no_quote_outside_quote_zone():
    qm, _, _, _ = make_qm(maker_kwargs={"quote_zone_min": 0.50})
    desired = qm.compute_desired_quotes(_standard_books())
    assert desired == []


def test_min_buy_utility_blocks_quotes():
    qm, _, _, _ = make_qm(kelly_kwargs={"min_buy_utility": 1.0})
    desired = qm.compute_desired_quotes(_standard_books())
    assert desired == []


def test_buy_no_side_selected_when_fair_below_market():
    # Bin 2: fair YES 0.2 -> fair NO 0.8. YES book 0.24/0.30 gives
    # NO book bid 0.70 / ask 0.76 -> BUY_NO quote at 0.71 on the NO token.
    qm, _, _, _ = make_qm()
    desired = qm.compute_desired_quotes({2: _book(2, 0.24, 0.30, token="yes-2")})

    assert len(desired) == 1
    q = desired[0]
    assert q.action == TradeAction.BUY_NO
    assert q.token_id == "no-2"
    assert q.price == pytest.approx(0.71)


def test_quote_sizing_respects_budget_and_dust_floor():
    qm, _, _, portfolio = make_qm(maker_kwargs={"budget_fraction": 0.01})
    # budget = 0.01 * 500 = $5 < min_quote_usd 10 -> no quote
    desired = qm.compute_desired_quotes(_standard_books())
    assert desired == []


# ----------------------------------------------------------------------
# Reconcile (live)
# ----------------------------------------------------------------------


def _reconcile(qm, books, hours=48.0):
    asyncio.run(qm.reconcile(orderbooks=books, hours_to_settlement=hours))


def test_reconcile_places_and_registers():
    qm, oe, ke, _ = make_qm()
    _reconcile(qm, _standard_books())

    assert len(oe.placed) == 1
    assert oe.placed[0]["token_id"] == "yes-0"
    assert oe.placed[0]["ttl_seconds"] == qm.config.ttl_seconds
    assert len(qm._orders) == 1

    order = next(iter(qm._orders.values()))
    assert qm.open_collateral() == pytest.approx(order.price * order.size)

    # Fill machinery registration: candidate in pending + remembered context
    (candidate, token_id) = ke._pending_orders[order.order_id]
    assert candidate.kind == "maker"
    assert candidate.bin_index == 0
    assert token_id == "yes-0"
    assert ke.remembered[0][0] == order.order_id


def test_reconcile_keeps_stable_quote():
    qm, oe, _, _ = make_qm()
    books = _standard_books()
    _reconcile(qm, books)
    assert len(oe.placed) == 1

    oe.open_orders_response = [
        {"id": oid, "asset_id": o.token_id} for oid, o in qm._orders.items()
    ]
    _reconcile(qm, books)

    assert len(oe.placed) == 1  # unchanged quote was not reposted
    assert oe.cancelled == []
    assert len(qm._orders) == 1


def test_reconcile_cancels_unknown_orders_on_own_tokens():
    qm, oe, _, _ = make_qm()
    oe.open_orders_response = [
        {"id": "leftover-1", "asset_id": "yes-1"},   # ours: cancel
        {"id": "other-mkt", "asset_id": "not-ours"},  # ignore
    ]
    _reconcile(qm, _standard_books())

    assert ["leftover-1"] in oe.cancelled
    assert all("other-mkt" not in batch for batch in oe.cancelled)


def test_reconcile_drops_registry_entries_missing_from_exchange():
    qm, oe, ke, _ = make_qm()
    _reconcile(qm, _standard_books())
    assert len(qm._orders) == 1

    # Exchange no longer lists it (filled or expired) -> dropped from the
    # registry WITHOUT a cancel call; with no books nothing new is desired
    oe.open_orders_response = []
    oe.cancelled.clear()
    _reconcile(qm, {})

    assert qm._orders == {}
    assert qm.open_collateral() == 0.0
    assert oe.cancelled == []


def test_reconcile_gate_failure_cancels_all():
    qm, oe, ke, _ = make_qm()
    _reconcile(qm, _standard_books())
    order_ids = list(qm._orders.keys())
    assert order_ids

    oe.open_orders_response = [
        {"id": oid, "asset_id": o.token_id} for oid, o in qm._orders.items()
    ]
    ke._integrity_state.frozen = True
    _reconcile(qm, _standard_books())

    assert order_ids in oe.cancelled
    assert qm._orders == {}


def test_reconcile_reprices_on_price_move():
    qm, oe, _, _ = make_qm()
    _reconcile(qm, _standard_books())
    first_id = next(iter(qm._orders))

    oe.open_orders_response = [
        {"id": oid, "asset_id": o.token_id} for oid, o in qm._orders.items()
    ]
    # Touch moves 0.38 -> 0.40: desired 0.41, differs by 2 ticks
    _reconcile(qm, {0: _book(0, 0.40, 0.46, token="yes-0")})

    assert [first_id] in oe.cancelled
    assert len(oe.placed) == 2
    assert oe.placed[1]["price"] == pytest.approx(0.41)


def test_reconcile_reposts_expiring_order():
    qm, oe, _, _ = make_qm()
    books = _standard_books()
    _reconcile(qm, books)
    first_id = next(iter(qm._orders))

    # Age the order to within the repost window (ttl/3 remaining)
    qm._orders[first_id].expires_at = time.time() + qm.config.ttl_seconds / 4
    oe.open_orders_response = [
        {"id": oid, "asset_id": o.token_id} for oid, o in qm._orders.items()
    ]
    _reconcile(qm, books)

    assert [first_id] in oe.cancelled
    assert len(oe.placed) == 2


def test_reconcile_listing_failure_fails_closed():
    qm, oe, _, _ = make_qm()
    oe.open_orders_response = None
    _reconcile(qm, _standard_books())

    assert oe.placed == []
    assert oe.cancelled == []


def test_failed_reprice_cancel_does_not_double_post():
    qm, oe, _, _ = make_qm()
    _reconcile(qm, _standard_books())
    assert len(oe.placed) == 1

    oe.open_orders_response = [
        {"id": oid, "asset_id": o.token_id} for oid, o in qm._orders.items()
    ]
    oe.cancel_result = False  # reprice cancel will fail
    _reconcile(qm, {0: _book(0, 0.40, 0.46, token="yes-0")})

    # Old order still resting -> no replacement posted on the same key
    assert len(oe.placed) == 1
    assert len(qm._orders) == 1


def test_cancel_failure_keeps_collateral_tracked():
    qm, oe, _, _ = make_qm()
    _reconcile(qm, _standard_books())
    collateral = qm.open_collateral()
    assert collateral > 0

    oe.cancel_result = False
    asyncio.run(qm.cancel_all("test"))

    # Order stays tracked (collateral counted) until exchange truth
    # confirms it is gone; GTD expiry bounds the worst case.
    assert qm.open_collateral() == pytest.approx(collateral)

    oe.cancel_result = True
    asyncio.run(qm.cancel_all("test"))
    assert qm.open_collateral() == 0.0


# ----------------------------------------------------------------------
# Kill switches and self-trade guard
# ----------------------------------------------------------------------


def test_kill_cancels_and_arms_cooldown():
    qm, oe, _, _ = make_qm()
    books = _standard_books()
    _reconcile(qm, books)
    order_ids = list(qm._orders.keys())

    qm.kill("realtime_post")

    assert order_ids in oe.cancelled
    assert qm._orders == {}

    # During cooldown nothing is quoted even though gates would pass
    oe.open_orders_response = []
    _reconcile(qm, books)
    assert len(oe.placed) == 1  # no new placement

    # After cooldown quoting resumes
    qm._cooldown_until = time.time() - 1
    _reconcile(qm, books)
    assert len(oe.placed) == 2


def test_cancel_bin_only_affects_that_bin():
    # Budget high enough to fund quotes on both bins
    qm, oe, _, _ = make_qm(maker_kwargs={"budget_fraction": 0.5})
    books = {
        0: _book(0, 0.38, 0.44, token="yes-0"),
        2: _book(2, 0.24, 0.30, token="yes-2"),
    }
    _reconcile(qm, books)
    assert len(qm._orders) == 2

    bin0_ids = [oid for oid, o in qm._orders.items() if o.bin_index == 0]
    asyncio.run(qm.cancel_bin(0))

    assert bin0_ids in oe.cancelled
    assert len(qm._orders) == 1
    assert next(iter(qm._orders.values())).bin_index == 2


def test_attach_wires_executor_hooks():
    qm, _, ke, _ = make_qm()
    qm.attach()
    assert ke.maker_collateral_provider == qm.open_collateral
    assert ke.maker_pre_trade_hook == qm.cancel_bin


def test_manager_kill_routes_to_all_bots():
    from src.algo.musk_tweet_count.forecaster.multi_event_manager import (
        MultiEventManager,
    )

    killed = []

    class FakeQM:
        def kill(self, reason):
            killed.append(reason)

    def active(qm):
        return SimpleNamespace(
            bot=SimpleNamespace(kelly_bot=SimpleNamespace(quote_manager=qm)),
            info=SimpleNamespace(short_name="ev"),
        )

    stub = SimpleNamespace(
        _active_events={
            "a": active(FakeQM()),
            "b": active(None),  # maker disabled for this event: skipped
            "c": active(FakeQM()),
        }
    )

    MultiEventManager._kill_maker_quotes(stub, "realtime_post")
    assert killed == ["realtime_post", "realtime_post"]


# ----------------------------------------------------------------------
# Capital integration: effective portfolio subtracts resting collateral
# ----------------------------------------------------------------------


def test_effective_portfolio_subtracts_maker_collateral():
    portfolio = Portfolio(
        initial_capital=1000.0,
        capital=1000.0,
        num_bins=3,
        probabilities=[0.2, 0.3, 0.5],
    )
    executor = KellyExecutor(
        config=KellyConfig(rate_limit=RateLimitConfig()),
        portfolio=portfolio,
        token_ids={i: f"yes-{i}" for i in range(3)},
        no_token_ids={i: f"no-{i}" for i in range(3)},
        sync_portfolio=None,
        event_name="test-event",
    )

    assert executor._build_effective_portfolio().capital == 1000.0

    executor.maker_collateral_provider = lambda: 40.0
    assert executor._build_effective_portfolio().capital == 960.0

    # Clamped at zero, never negative
    executor.maker_collateral_provider = lambda: 5000.0
    assert executor._build_effective_portfolio().capital == 0.0

    # Provider failure must not break planning
    def boom():
        raise RuntimeError("registry unavailable")

    executor.maker_collateral_provider = boom
    assert executor._build_effective_portfolio().capital == 1000.0


# ----------------------------------------------------------------------
# Shadow mode
# ----------------------------------------------------------------------


def test_shadow_places_virtual_quotes_without_rest_calls():
    qm, oe, _, _ = make_qm(mode="shadow")
    _reconcile(qm, _standard_books())

    assert len(qm._orders) == 1
    assert oe.placed == []
    assert oe.cancelled == []
    assert oe.open_orders_calls == 0
    assert qm.open_collateral() == 0.0  # nothing actually rests


def test_shadow_would_fill_detection():
    qm, oe, _, _ = make_qm(mode="shadow")
    _reconcile(qm, _standard_books())
    assert len(qm._orders) == 1  # virtual bid at 0.39

    # Book trades down through the virtual bid (best bid now 0.35)
    _reconcile(qm, {0: _book(0, 0.35, 0.44, token="yes-0")})

    assert qm._shadow_would_fills == 1
    assert oe.placed == []


def test_shadow_gate_failure_drops_virtual_quotes():
    qm, _, ke, _ = make_qm(mode="shadow")
    _reconcile(qm, _standard_books())
    assert len(qm._orders) == 1

    ke._integrity_state.frozen = True
    _reconcile(qm, _standard_books())
    assert qm._orders == {}


def test_shadow_expires_virtual_quotes():
    qm, _, _, _ = make_qm(mode="shadow")
    _reconcile(qm, _standard_books())
    oid = next(iter(qm._orders))
    qm._orders[oid].expires_at = time.time() - 1

    # Empty books: nothing new desired, expiry still processed
    _reconcile(qm, {})
    assert oid not in qm._orders


# ----------------------------------------------------------------------
# Structured event log + forward markout
# ----------------------------------------------------------------------


def _read_events(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_event_log_writes_records_and_noop_when_disabled(tmp_path):
    log = MakerEventLog(str(tmp_path), "My Event / 2026", "shadow")
    assert log.enabled
    log.emit("quote_placed", now=100.0, bin=0, price=0.39)
    log.emit("would_fill", now=101.0, bin=0, price=0.39)
    log.close()

    records = _read_events(log.path)
    assert [r["type"] for r in records] == ["quote_placed", "would_fill"]
    assert records[0]["event"] == "My Event / 2026"
    assert records[0]["mode"] == "shadow"
    assert records[0]["ts"] == 100.0

    disabled = MakerEventLog("", "e", "shadow")
    assert not disabled.enabled
    assert disabled.path is None
    disabled.emit("quote_placed", now=1.0)  # no-op, must not raise


def test_event_log_disabled_by_default_hermetic():
    # Default MakerConfig has no event_log_dir, so a bare QuoteManager opens
    # no file — unit tests stay hermetic and runs opt in via the CLI.
    qm, _, _, _ = make_qm(mode="shadow")
    assert qm.event_log.path is None
    _reconcile(qm, _standard_books())


def test_shadow_reconcile_emits_structured_events(tmp_path):
    qm, _, _, _ = make_qm(mode="shadow", maker_kwargs={"event_log_dir": str(tmp_path)})
    _reconcile(qm, _standard_books())              # places a virtual quote
    _reconcile(qm, {0: _book(0, 0.35, 0.44)})      # book drops -> would-fill
    qm.event_log.close()

    records = _read_events(qm.event_log.path)
    types = [r["type"] for r in records]
    assert "quote_placed" in types
    assert "would_fill" in types

    placed = next(r for r in records if r["type"] == "quote_placed")
    assert placed["token_id"] == "yes-0"
    assert placed["side"] == TradeAction.BUY_YES.value
    assert "fair_value" in placed and "spread" in placed

    filled = next(r for r in records if r["type"] == "would_fill")
    assert filled["token_id"] == "yes-0"
    assert filled["fair_value"] == 0.5
    assert "fill_mid" in filled and "rested_seconds" in filled


def test_shadow_emits_no_quotes_diag_when_gate_open(tmp_path):
    # Gate open (healthy fixture) but every bin fails a quote filter: bin 0's
    # spread is too tight, bins 1-2 have no book. The reason histogram is
    # emitted so an empty desired-set is diagnosable — the signal that was
    # missing when the shadow ran 11 days and placed zero quotes.
    qm, _, _, _ = make_qm(mode="shadow", maker_kwargs={"event_log_dir": str(tmp_path)})
    tight = {0: _book(0, 0.42, 0.44, token="yes-0")}  # spread 0.02 < min_spread 0.03
    _reconcile(qm, tight)
    _reconcile(qm, tight)  # identical histogram -> deduped, no second emit
    qm.event_log.close()

    diags = [r for r in _read_events(qm.event_log.path) if r["type"] == "no_quotes"]
    assert len(diags) == 1
    reasons = diags[0]["reasons"]
    assert reasons.get("spread_below_min", 0) >= 1
    assert reasons.get("no_orderbook", 0) == 2
    assert "quoted" not in reasons


def test_no_quotes_diag_not_emitted_when_quote_placed(tmp_path):
    # When a quote IS produced, the diagnostic must stay silent.
    qm, _, _, _ = make_qm(mode="shadow", maker_kwargs={"event_log_dir": str(tmp_path)})
    _reconcile(qm, _standard_books())
    qm.event_log.close()
    diags = [r for r in _read_events(qm.event_log.path) if r["type"] == "no_quotes"]
    assert diags == []


def test_forward_markout_sampling_records_horizon_mids(tmp_path):
    qm, _, _, _ = make_qm(mode="shadow", maker_kwargs={"event_log_dir": str(tmp_path)})
    t0 = 1_000_000.0
    order = RestingOrder(
        order_id="mk-1",
        bin_index=0,
        action=TradeAction.BUY_YES,
        token_id="yes-0",
        price=0.39,
        size=100.0,
        placed_at=t0,
        expires_at=t0 + 360,
        fair_value=0.5,
    )
    qm._register_markout(order, fill_mid=0.41, now=t0)

    # A horizon is only sampled once its offset has elapsed; the mid rises.
    qm._sample_markouts({0: _book(0, 0.44, 0.46)}, now=t0 + 300)    # mid 0.45
    assert len(qm._pending_markouts) == 1                          # not yet complete
    qm._sample_markouts({0: _book(0, 0.49, 0.51)}, now=t0 + 900)    # mid 0.50
    qm._sample_markouts({0: _book(0, 0.54, 0.56)}, now=t0 + 1800)   # mid 0.55 -> done
    assert qm._pending_markouts == []
    qm.event_log.close()

    mk = next(r for r in _read_events(qm.event_log.path) if r["type"] == "markout")
    assert mk["side"] == TradeAction.BUY_YES.value
    assert mk["fill_price"] == 0.39
    assert mk["horizon_mids"]["300"] == pytest.approx(0.45)
    assert mk["horizon_mids"]["1800"] == pytest.approx(0.55)
    # Markout is the quoted token's mid drift from our buy price.
    assert mk["markout"]["300"] == pytest.approx(0.06)
    assert mk["markout"]["1800"] == pytest.approx(0.16)


def test_markout_finalizes_with_null_when_book_gone(tmp_path):
    qm, _, _, _ = make_qm(mode="shadow", maker_kwargs={"event_log_dir": str(tmp_path)})
    t0 = 2_000_000.0
    order = RestingOrder(
        order_id="mk-2",
        bin_index=0,
        action=TradeAction.BUY_YES,
        token_id="yes-0",
        price=0.40,
        size=50.0,
        placed_at=t0,
        expires_at=t0 + 360,
        fair_value=0.5,
    )
    qm._register_markout(order, fill_mid=0.42, now=t0)
    # No book for the bin at any horizon -> samples are null, still finalizes.
    qm._sample_markouts({}, now=t0 + 300)
    qm._sample_markouts({}, now=t0 + 900)
    qm._sample_markouts({}, now=t0 + 1800)
    assert qm._pending_markouts == []
    qm.event_log.close()

    mk = next(r for r in _read_events(qm.event_log.path) if r["type"] == "markout")
    assert mk["horizon_mids"] == {"300": None, "900": None, "1800": None}
    assert mk["markout"] == {"300": None, "900": None, "1800": None}
