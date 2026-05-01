"""Tests for analyze.fifo_resolve, basis-error stats, and CSV/JSON output.

Focus on regressions called out by code review:
  - fifo_resolve must not mutate the input FillRecord list.
  - basis-error must use side-aware conservative probability.
"""

from datetime import date, datetime, timezone

from src.algo.deribit_leadlag.backtest.analyze import (
    ResolvedTrade,
    basis_errors_by_stratum,
    fifo_resolve,
    summarize,
)
from src.algo.deribit_leadlag.backtest.ledger import FillRecord, SettlementRecord
from src.algo.deribit_leadlag.position_manager import DesiredAction


def _make_buy(*, ts, condition_id="cond", side="YES", price=0.55, size=100, fee=0.0,
              prob_cons=0.60, prob_agg=0.65) -> FillRecord:
    return FillRecord(
        timestamp=ts,
        condition_id=condition_id,
        yes_token_id=f"{condition_id}-yes",
        no_token_id=f"{condition_id}-no",
        expiry_date=date(2026, 4, 22),
        strike=78000.0,
        side=side,
        direction="BUY",
        action=DesiredAction.BUY_YES if side == "YES" else DesiredAction.BUY_NO,
        fill_price=price,
        fill_size=size,
        fee=fee,
        edge_at_post=0.10,
        order_id="oid-buy",
        prob_conservative_at_entry=prob_cons,
        prob_mid_at_entry=(prob_cons + prob_agg) / 2,
        prob_aggressive_at_entry=prob_agg,
        bounds_width=prob_agg - prob_cons,
        has_next_day=True,
        hours_to_resolution=18.0,
    )


def _make_sell(*, ts, condition_id="cond", side="YES", price=0.65, size=100, fee=0.0) -> FillRecord:
    return FillRecord(
        timestamp=ts,
        condition_id=condition_id,
        yes_token_id=f"{condition_id}-yes",
        no_token_id=f"{condition_id}-no",
        expiry_date=date(2026, 4, 22),
        strike=78000.0,
        side=side,
        direction="SELL",
        action=DesiredAction.SELL_YES if side == "YES" else DesiredAction.SELL_NO,
        fill_price=price,
        fill_size=size,
        fee=fee,
        edge_at_post=0.0,
        order_id="oid-sell",
        prob_conservative_at_entry=None,
        prob_mid_at_entry=None,
        prob_aggressive_at_entry=None,
        bounds_width=None,
        has_next_day=None,
        hours_to_resolution=None,
    )


def test_fifo_resolve_does_not_mutate_input_fills():
    """Regression: matcher must not modify the input fills list.

    The CLI writes fills.csv from the same list AFTER calling
    fifo_resolve. A mutation would corrupt the export and make a
    second fifo_resolve pass produce different results.
    """
    buy = _make_buy(
        ts=datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc),
        size=100, price=0.55, fee=1.78,
    )
    sell = _make_sell(
        ts=datetime(2026, 4, 22, 12, 5, tzinfo=timezone.utc),
        size=40, price=0.60, fee=0.5,
    )
    fills = [buy, sell]

    # Snapshot pre-state
    pre_buy_size, pre_buy_fee = buy.fill_size, buy.fee
    pre_sell_size, pre_sell_fee = sell.fill_size, sell.fee

    trades = fifo_resolve(fills, [])

    # Original FillRecords must be byte-identical
    assert buy.fill_size == pre_buy_size
    assert buy.fee == pre_buy_fee
    assert sell.fill_size == pre_sell_size
    assert sell.fee == pre_sell_fee

    # ...and the matcher still emitted a partial close
    assert len(trades) == 1
    assert trades[0].size == 40
    assert trades[0].buy_price == 0.55
    assert trades[0].sell_price == 0.60

    # Re-running on the SAME list must give the SAME result (idempotent)
    trades2 = fifo_resolve(fills, [])
    assert len(trades2) == 1
    assert trades2[0].size == 40


def test_basis_error_for_no_side_uses_one_minus_prob_aggressive():
    """Regression: NO leg's basis-error must compare against 1 - prob_aggressive
    (the conservative NO probability), not the YES-side conservative.

    Reproduces the metric bug where NO trades reported basis_error ≈ +0.85
    instead of the true ~−0.04 because the analyzer used the wrong
    reference. The current side-aware implementation uses
    `model_prob_conservative_for_side`.
    """
    # Buy NO at 0.88; YES probabilities: conservative=0.04, aggressive=0.06.
    #   Conservative NO probability = 1 - 0.06 = 0.94.
    #   Sell NO at 0.90 → basis_error = 0.90 − 0.94 = −0.04
    buy = _make_buy(
        ts=datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc),
        side="NO", price=0.88, size=50, prob_cons=0.04, prob_agg=0.06,
    )
    sell = _make_sell(
        ts=datetime(2026, 4, 22, 12, 5, tzinfo=timezone.utc),
        side="NO", price=0.90, size=50,
    )
    trades = fifo_resolve([buy, sell], [])
    assert len(trades) == 1
    t = trades[0]

    side_prob = t.model_prob_conservative_for_side
    assert side_prob is not None
    assert abs(side_prob - 0.94) < 1e-9  # 1 - prob_aggressive_YES

    summary = summarize(trades)
    overall = summary["basis_error_overall"]
    assert overall["n"] == 1
    assert abs(overall["mean"] - (-0.04)) < 1e-9


def test_basis_error_for_yes_side_uses_prob_conservative_directly():
    """YES leg's basis-error compares against prob_conservative_at_entry."""
    buy = _make_buy(
        ts=datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc),
        side="YES", price=0.55, size=50, prob_cons=0.60, prob_agg=0.65,
    )
    sell = _make_sell(
        ts=datetime(2026, 4, 22, 12, 5, tzinfo=timezone.utc),
        side="YES", price=0.62, size=50,
    )
    trades = fifo_resolve([buy, sell], [])
    t = trades[0]

    assert t.model_prob_conservative_for_side == 0.60

    summary = summarize(trades)
    overall = summary["basis_error_overall"]
    assert abs(overall["mean"] - (0.62 - 0.60)) < 1e-9


def test_settlement_unmatched_when_no_settlement_record():
    """A buy with no settlement record stays unmatched (not silently lost)."""
    buy = _make_buy(
        ts=datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc),
        size=100, price=0.55, fee=0.0,
    )
    trades = fifo_resolve([buy], [])  # no settlement provided
    assert trades == []  # no resolved trade emitted, position implicitly stays open
