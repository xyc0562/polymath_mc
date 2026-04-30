"""Behavioral tests for LivePositionSource.

Focus: the two paths flagged in code review where preserving the legacy
`PositionManager.fetch_*` semantics matters:

  - empty wallet must NOT zero existing inventory
  - failed get_orders() must NOT leave stale orders in place
"""

from datetime import date, datetime, timezone

from src.algo.deribit_leadlag.config import AllocationConfig, OrderConfig, SignalConfig
from src.algo.deribit_leadlag.position_manager import (
    BinKey,
    OpenOrder,
    PositionManager,
)
from src.algo.deribit_leadlag.position_source import LivePositionSource
from src.algo.deribit_leadlag.polymarket_discovery import ThresholdMarket
from src.algo.deribit_leadlag.settlement import CompatibilityClass


def _make_market(
    *,
    yes_token_id: str = "yes-token",
    no_token_id: str = "no-token",
    strike: float = 60000.0,
    condition_id: str = "cond-1",
) -> ThresholdMarket:
    return ThresholdMarket(
        condition_id=condition_id,
        question="Bitcoin above $60,000 on April 5?",
        strike=strike,
        expiry_date=date(2026, 4, 5),
        resolution_time_utc=datetime(2026, 4, 5, 16, 0, 0, tzinfo=timezone.utc),
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        yes_price=0.42,
        no_price=0.58,
        event_id="evt-1",
        volume=1000.0,
        description="",
        settlement=None,
    )


def _make_manager() -> PositionManager:
    return PositionManager(
        alloc_config=AllocationConfig(
            per_bin_max_usd=50.0,
            per_date_max_usd=150.0,
            total_max_usd=500.0,
            max_order_size_usd=30.0,
        ),
        order_config=OrderConfig(
            maker_min_edge=0.02,
            taker_min_edge=0.15,
            emergency_exit_edge=0.20,
            reprice_edge_threshold=0.02,
            emergency_prob_shift=0.03,
            polymarket_crypto_fee_rate=0.072,
        ),
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
    )


def test_fetch_positions_empty_wallet_leaves_inventory_untouched():
    """Regression: an empty wallet must not zero out positions learned from fills."""
    manager = _make_manager()
    market = _make_market()
    key = BinKey(market.expiry_date, market.strike)
    manager.update_markets([market], {key: CompatibilityClass.TIME_ADJUSTED})

    bin_state = manager.get_bins()[key]
    bin_state.yes_position = 7.0
    bin_state.no_position = 3.0

    source = LivePositionSource(wallet_address="")
    source.fetch_positions(manager)

    # Holdings learned from fills must survive the no-op fetch.
    assert bin_state.yes_position == 7.0
    assert bin_state.no_position == 3.0


class _FailingClob:
    def get_orders(self):
        raise RuntimeError("clob unreachable")


def test_fetch_open_orders_clears_stale_orders_when_clob_fails():
    """Regression: a failed get_orders() must not preserve stale pre-cancel orders.

    The startup sequence cancels all inherited orders, then re-fetches to
    confirm the clean state. If the re-fetch fails, the manager must end up
    with no live orders — not the pre-cancel snapshot, which would cause
    compute_deltas to hold off cancel/repost work indefinitely.
    """
    manager = _make_manager()
    market = _make_market()
    key = BinKey(market.expiry_date, market.strike)
    manager.update_markets([market], {key: CompatibilityClass.TIME_ADJUSTED})

    bin_state = manager.get_bins()[key]
    bin_state.open_orders.append(
        OpenOrder(
            order_id="stale-1",
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

    source = LivePositionSource(wallet_address="0xabc")
    source.fetch_open_orders(manager, _FailingClob())

    assert bin_state.open_orders == []
