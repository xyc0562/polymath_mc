from datetime import date, datetime, timezone
from types import SimpleNamespace

from src.algo.deribit_leadlag.config import AllocationConfig, OrderConfig, SignalConfig
from src.algo.deribit_leadlag.implied_probs import ImpliedProb
from src.algo.deribit_leadlag.position_manager import (
    BinKey,
    DesiredAction,
    OpenOrder,
    PositionManager,
)
from src.algo.deribit_leadlag.polymarket_discovery import ThresholdMarket
from src.algo.deribit_leadlag.settlement import CompatibilityClass


def _make_market(
    *,
    yes_price: float = 0.42,
    no_price: float = 0.58,
    yes_token_id: str = "yes-token",
    no_token_id: str = "no-token",
) -> ThresholdMarket:
    return ThresholdMarket(
        condition_id="cond-1",
        question="Bitcoin above $60,000 on April 5?",
        strike=60000.0,
        expiry_date=date(2026, 4, 5),
        resolution_time_utc=datetime(2026, 4, 5, 16, 0, 0, tzinfo=timezone.utc),
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        yes_price=yes_price,
        no_price=no_price,
        event_id="evt-1",
        volume=1000.0,
        description="",
        settlement=None,
    )


def _make_manager() -> PositionManager:
    return PositionManager(
        alloc_config=AllocationConfig(per_bin_max_usd=50.0, per_date_max_usd=150.0, total_max_usd=500.0, max_order_size_usd=30.0),
        order_config=OrderConfig(maker_min_edge=0.02, taker_min_edge=0.15, emergency_exit_edge=0.20, reprice_edge_threshold=0.02, emergency_prob_shift=0.03, polymarket_crypto_fee_rate=0.072),
        signal_config=SignalConfig(
            min_call_spread_usd=15.0,
            max_bounds_width_maker=0.50,
            max_bounds_width_taker=0.30,
            min_prob=0.03,
            max_prob=0.97,
            min_time_to_expiry_hours=4.0,
            max_time_to_expiry_days=7.0,
            time_adjusted_basis_haircut=0.02,
            no_next_day_extra_haircut=0.03,
            max_bracket_dk=3000.0,
        ),
        wallet_address="0xabc",
    )


def test_handle_fill_sell_reduces_position_and_open_order_size():
    manager = _make_manager()
    market = _make_market()
    key = BinKey(market.expiry_date, market.strike)
    manager.update_markets([market], {key: CompatibilityClass.TIME_ADJUSTED})

    bin_state = manager.get_bins()[key]
    bin_state.yes_position = 10.0
    bin_state.open_orders.append(
        OpenOrder(
            order_id="sell-1",
            bin_key=key,
            side="SELL",
            token_id=market.yes_token_id,
            price=0.55,
            size=10.0,
            posted_at=0.0,
            edge_at_post=0.0,
            is_maker=True,
        )
    )

    manager.handle_fill(order_id="sell-1", token_id=market.yes_token_id, side="SELL", filled_size=4.0)
    assert bin_state.yes_position == 6.0
    assert bin_state.open_orders[0].size == 6.0

    manager.handle_fill(order_id="sell-1", token_id=market.yes_token_id, side="SELL", filled_size=6.0)
    assert bin_state.yes_position == 0.0
    assert bin_state.open_orders == []


def test_compute_deltas_uses_live_price_and_cancels_stale_order():
    manager = _make_manager()
    market = _make_market(yes_price=0.42, no_price=0.58)
    key = BinKey(market.expiry_date, market.strike)
    manager.update_markets([market], {key: CompatibilityClass.TIME_ADJUSTED})

    bin_state = manager.get_bins()[key]
    bin_state.open_orders.append(
        OpenOrder(
            order_id="stale-yes",
            bin_key=key,
            side="BUY",
            token_id=market.yes_token_id,
            price=0.42,
            size=40.0,
            posted_at=0.0,
            edge_at_post=0.0,
            is_maker=True,
        )
    )

    implied_prob = ImpliedProb(
        prob_conservative=0.35,
        prob_mid=0.36,
        prob_aggressive=0.37,
        method="call_spread",
        forward=0.0,
        T_years=0.1,
    )
    orderbook = SimpleNamespace(
        best_yes_bid=0.30,
        best_yes_ask=0.36,
        best_no_bid=0.64,
        best_no_ask=0.70,
    )

    manager.compute_targets({(market.expiry_date, market.strike): implied_prob}, {market.yes_token_id: orderbook})
    actions = manager.compute_deltas()

    assert len(actions) == 1
    action = actions[0]
    assert action.action == DesiredAction.BUY_YES
    assert action.is_maker is True
    assert action.price == 0.30
    assert action.cancel_order_ids == ["stale-yes"]

    report = SimpleNamespace(
        canceled_order_ids=["stale-yes"],
        posted_actions=[action],
        raw_results=[{"orderID": "new-yes"}],
    )
    manager.apply_execution_report(report)

    follow_up_actions = manager.compute_deltas()
    assert follow_up_actions == []


def test_large_edge_routes_to_taker_path():
    manager = _make_manager()
    market = _make_market(yes_price=0.20, no_price=0.80)
    key = BinKey(market.expiry_date, market.strike)
    manager.update_markets([market], {key: CompatibilityClass.TIME_ADJUSTED})

    implied_prob = ImpliedProb(
        prob_conservative=0.50,
        prob_mid=0.51,
        prob_aggressive=0.52,
        method="call_spread",
        forward=0.0,
        T_years=0.1,
    )
    orderbook = SimpleNamespace(
        best_yes_bid=0.19,
        best_yes_ask=0.20,
        best_no_bid=0.80,
        best_no_ask=0.81,
    )

    manager.compute_targets({(market.expiry_date, market.strike): implied_prob}, {market.yes_token_id: orderbook})
    actions = manager.compute_deltas()

    assert len(actions) == 1
    assert actions[0].action == DesiredAction.BUY_YES
    assert actions[0].is_maker is False
    assert actions[0].price == 0.20
