"""Tests for SimulatedOrderGateway: taker fills, maker discard, fee accounting, ExecutionReport shape."""

from datetime import date, datetime, timezone

from src.algo.deribit_leadlag.backtest.data_provider import SimulatedOrderbook
from src.algo.deribit_leadlag.backtest.sim_gateway import SimFill, SimulatedOrderGateway
from src.algo.deribit_leadlag.config import (
    AllocationConfig,
    OrderConfig,
    SignalConfig,
)
from src.algo.deribit_leadlag.position_manager import (
    BinKey,
    DesiredAction,
    OpenOrder,
    OrderAction,
    PositionManager,
)
from src.algo.deribit_leadlag.polymarket_discovery import ThresholdMarket
from src.algo.deribit_leadlag.settlement import CompatibilityClass


def _make_market(condition_id: str = "cond-1") -> ThresholdMarket:
    return ThresholdMarket(
        condition_id=condition_id,
        question="",
        strike=78000.0,
        expiry_date=date(2026, 4, 22),
        resolution_time_utc=datetime(2026, 4, 22, 16, 0, tzinfo=timezone.utc),
        yes_token_id="yes-tok",
        no_token_id="no-tok",
        yes_price=0.55,
        no_price=0.45,
        event_id="",
        volume=0.0,
        description="",
        settlement=None,
    )


def _make_position_mgr() -> PositionManager:
    mgr = PositionManager(
        alloc_config=AllocationConfig(50.0, 150.0, 500.0, 30.0),
        order_config=OrderConfig(0.02, 0.15, 0.20, 0.02, 0.03, 0.072),
        signal_config=SignalConfig(),
    )
    market = _make_market()
    key = BinKey(market.expiry_date, market.strike)
    mgr.update_markets([market], {key: CompatibilityClass.TIME_ADJUSTED})
    return mgr


def test_taker_buy_yes_fills_at_recorded_ask_and_moves_inventory():
    mgr = _make_position_mgr()
    fills = []
    gw = SimulatedOrderGateway(
        position_mgr=mgr,
        fee_rate=0.072,
        max_order_size_usd=30.0,
        fill_callback=fills.append,
    )

    now = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)
    gw.set_clock(now)
    gw.set_orderbooks({"yes-tok": SimulatedOrderbook(best_yes_bid=0.50, best_yes_ask=0.55)})

    action = OrderAction(
        bin_key=BinKey(date(2026, 4, 22), 78000.0),
        action=DesiredAction.BUY_YES,
        token_id="yes-tok",
        price=0.55,
        size=50,
        is_maker=False,
        edge=0.10,
    )
    report = gw.execute_actions([action])

    assert len(report.posted_actions) == 1
    assert len(report.raw_results) == 1
    assert report.raw_results[0]["orderID"].startswith("sim-")
    # ExecutionReport shape compatibility check
    assert hasattr(report, "canceled_order_ids")
    assert hasattr(report, "posted_actions")
    assert hasattr(report, "raw_results")

    assert len(fills) == 1
    f = fills[0]
    assert f.direction == "BUY"
    assert f.side == "YES"
    # Filled at the snapshot's best ask, not the requested price.
    assert f.fill_price == 0.55
    # Size is capped to max_order_size_usd / price = 30 / 0.55 = 54 → min(50, 54) = 50.
    assert f.fill_size == 50
    # Fee = 0.072 * 0.55 * 0.45 * 50
    assert abs(f.fee - (0.072 * 0.55 * 0.45 * 50)) < 1e-6

    # Position moved through PositionManager.handle_fill
    bin_state = mgr.get_bins()[BinKey(date(2026, 4, 22), 78000.0)]
    assert bin_state.yes_position == 50.0


def test_taker_buy_no_crosses_mirrored_no_ask():
    mgr = _make_position_mgr()
    fills = []
    gw = SimulatedOrderGateway(mgr, 0.072, 30.0, fills.append)

    gw.set_clock(datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc))
    # YES bid 0.50 / ask 0.55  =>  NO ask = 1 − 0.50 = 0.50
    gw.set_orderbooks({"yes-tok": SimulatedOrderbook(best_yes_bid=0.50, best_yes_ask=0.55)})

    action = OrderAction(
        bin_key=BinKey(date(2026, 4, 22), 78000.0),
        action=DesiredAction.BUY_NO,
        token_id="no-tok",
        price=0.50,
        size=40,
        is_maker=False,
        edge=0.10,
    )
    gw.execute_actions([action])
    assert len(fills) == 1
    assert fills[0].fill_price == 0.50
    assert fills[0].side == "NO"

    bin_state = mgr.get_bins()[BinKey(date(2026, 4, 22), 78000.0)]
    assert bin_state.no_position == fills[0].fill_size


def test_maker_actions_discarded_under_discard_policy():
    mgr = _make_position_mgr()
    fills = []
    gw = SimulatedOrderGateway(mgr, 0.072, 30.0, fills.append, maker_policy="discard")
    gw.set_clock(datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc))
    gw.set_orderbooks({"yes-tok": SimulatedOrderbook(best_yes_bid=0.50, best_yes_ask=0.55)})

    action = OrderAction(
        bin_key=BinKey(date(2026, 4, 22), 78000.0),
        action=DesiredAction.BUY_YES,
        token_id="yes-tok",
        price=0.50,
        size=50,
        is_maker=True,
        edge=0.05,
    )
    report = gw.execute_actions([action])
    assert report.posted_actions == []
    assert report.raw_results == []
    assert fills == []

    bin_state = mgr.get_bins()[BinKey(date(2026, 4, 22), 78000.0)]
    assert bin_state.yes_position == 0.0


def test_maker_actions_filled_at_post_price_under_instant_optimistic():
    """Default policy fills makers at the algo's chosen post price with zero fee."""
    mgr = _make_position_mgr()
    fills = []
    gw = SimulatedOrderGateway(
        mgr, fee_rate=0.072, max_order_size_usd=30.0,
        fill_callback=fills.append, maker_policy="instant_optimistic",
    )
    gw.set_clock(datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc))
    gw.set_orderbooks({"yes-tok": SimulatedOrderbook(best_yes_bid=0.10, best_yes_ask=0.55)})

    # Maker post: BUY_YES at the resting bid 0.10. Optimistic mode fills at 0.10.
    action = OrderAction(
        bin_key=BinKey(date(2026, 4, 22), 78000.0),
        action=DesiredAction.BUY_YES,
        token_id="yes-tok",
        price=0.10,
        size=50,
        is_maker=True,
        edge=0.40,
    )
    report = gw.execute_actions([action])

    assert len(report.posted_actions) == 1
    assert report.raw_results[0]["type"] == "maker_optimistic"
    assert len(fills) == 1
    assert fills[0].fill_price == 0.10
    assert fills[0].fee == 0.0  # optimistic maker pays no fee


def test_invalid_maker_policy_raises():
    mgr = _make_position_mgr()
    try:
        SimulatedOrderGateway(mgr, 0.072, 30.0, None, maker_policy="badness")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown maker_policy")


def test_no_orderbook_or_no_quote_skips_fill():
    mgr = _make_position_mgr()
    fills = []
    gw = SimulatedOrderGateway(mgr, 0.072, 30.0, fills.append)
    gw.set_clock(datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc))
    # YES ask is None — BUY_YES has nothing to cross.
    gw.set_orderbooks({"yes-tok": SimulatedOrderbook(best_yes_bid=0.50, best_yes_ask=None)})

    action = OrderAction(
        bin_key=BinKey(date(2026, 4, 22), 78000.0),
        action=DesiredAction.BUY_YES,
        token_id="yes-tok",
        price=0.55,
        size=50,
        is_maker=False,
        edge=0.10,
    )
    report = gw.execute_actions([action])
    assert report.posted_actions == []
    assert fills == []


def test_apply_execution_report_consumes_sim_gateway_output():
    """Sim ExecutionReport is byte-compatible with PositionManager.apply_execution_report."""
    mgr = _make_position_mgr()
    gw = SimulatedOrderGateway(mgr, 0.072, 30.0, None)
    gw.set_clock(datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc))
    gw.set_orderbooks({"yes-tok": SimulatedOrderbook(best_yes_bid=0.50, best_yes_ask=0.55)})

    action = OrderAction(
        bin_key=BinKey(date(2026, 4, 22), 78000.0),
        action=DesiredAction.BUY_YES,
        token_id="yes-tok",
        price=0.55,
        size=50,
        is_maker=False,
        edge=0.10,
    )
    report = gw.execute_actions([action])

    # Must not raise — exercises the same code path as production.
    mgr.apply_execution_report(report)


def test_cancel_ids_echo_back_in_report():
    mgr = _make_position_mgr()
    gw = SimulatedOrderGateway(mgr, 0.072, 30.0, None)
    gw.set_clock(datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc))
    gw.set_orderbooks({"yes-tok": SimulatedOrderbook(best_yes_bid=0.50, best_yes_ask=0.55)})

    action = OrderAction(
        bin_key=BinKey(date(2026, 4, 22), 78000.0),
        action=DesiredAction.BUY_YES,
        token_id="yes-tok",
        price=0.55,
        size=50,
        is_maker=False,
        edge=0.10,
        cancel_order_ids=["stale-A", "stale-B", "stale-A"],  # dup intentional
    )
    report = gw.execute_actions([action])
    assert sorted(report.canceled_order_ids) == ["stale-A", "stale-B"]
