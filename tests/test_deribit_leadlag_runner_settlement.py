"""Tests for BacktestRunner settlement boundary semantics."""

from datetime import date, datetime, timezone

from src.algo.deribit_leadlag.backtest.ledger import FillLedger
from src.algo.deribit_leadlag.backtest.sim_gateway import SimFill
from src.algo.deribit_leadlag.position_manager import DesiredAction, BinKey


def _seed_buy_yes(ledger: FillLedger, *, condition_id: str, expiry: date, strike: float, size: int, price: float, ts: datetime, side: str = "YES"):
    direction = "BUY"
    yes_tok = f"{condition_id}-yes"
    no_tok = f"{condition_id}-no"
    ledger.record_fill(SimFill(
        timestamp=ts,
        bin_key=BinKey(expiry, strike),
        condition_id=condition_id,
        yes_token_id=yes_tok,
        no_token_id=no_tok,
        action=DesiredAction.BUY_YES if side == "YES" else DesiredAction.BUY_NO,
        side=side,
        direction=direction,
        token_id=yes_tok if side == "YES" else no_tok,
        fill_price=price,
        fill_size=size,
        fee=0.072 * price * (1 - price) * size,
        edge_at_post=0.10,
        order_id=f"sim-{condition_id}",
    ))


def test_settlement_pays_winning_yes_full_dollar_and_zeros_inventory():
    ledger = FillLedger()
    expiry = date(2026, 4, 22)
    settle_dt = datetime(2026, 4, 22, 16, 0, tzinfo=timezone.utc)

    _seed_buy_yes(
        ledger,
        condition_id="cond-A",
        expiry=expiry,
        strike=78000.0,
        size=100,
        price=0.55,
        ts=datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc),
    )

    yes_before, no_before = ledger.snapshot_by_token()
    assert yes_before["cond-A-yes"] == 100

    records = ledger.settle_expiry(
        timestamp=settle_dt,
        expiry_date=expiry,
        outcomes_by_condition={"cond-A": 1},  # YES won
    )
    assert len(records) == 1
    rec = records[0]
    assert rec.yes_resolved == 1
    assert rec.cash_payoff == 100.0  # full dollar payoff
    assert rec.yes_shares_settled == 100

    yes_after, no_after = ledger.snapshot_by_token()
    assert "cond-A-yes" not in yes_after
    assert "cond-A-no" not in no_after


def test_settlement_pays_zero_for_losing_yes():
    ledger = FillLedger()
    expiry = date(2026, 4, 22)
    _seed_buy_yes(
        ledger,
        condition_id="cond-B",
        expiry=expiry,
        strike=78000.0,
        size=100,
        price=0.55,
        ts=datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc),
    )
    records = ledger.settle_expiry(
        timestamp=datetime(2026, 4, 22, 16, 0, tzinfo=timezone.utc),
        expiry_date=expiry,
        outcomes_by_condition={"cond-B": -1},  # NO won — YES loses
    )
    assert len(records) == 1
    assert records[0].cash_payoff == 0.0


def test_settlement_pays_no_side_when_no_won():
    ledger = FillLedger()
    expiry = date(2026, 4, 22)
    _seed_buy_yes(
        ledger,
        condition_id="cond-C",
        expiry=expiry,
        strike=78000.0,
        size=80,
        price=0.45,
        ts=datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc),
        side="NO",
    )
    records = ledger.settle_expiry(
        timestamp=datetime(2026, 4, 22, 16, 0, tzinfo=timezone.utc),
        expiry_date=expiry,
        outcomes_by_condition={"cond-C": -1},
    )
    assert len(records) == 1
    assert records[0].cash_payoff == 80.0
    assert records[0].no_shares_settled == 80


def test_settlement_only_touches_target_expiry():
    """A second expiry's positions must be untouched."""
    ledger = FillLedger()
    e1 = date(2026, 4, 22)
    e2 = date(2026, 4, 23)
    _seed_buy_yes(ledger, condition_id="A", expiry=e1, strike=78000.0, size=10, price=0.55,
                  ts=datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc))
    _seed_buy_yes(ledger, condition_id="B", expiry=e2, strike=80000.0, size=20, price=0.45,
                  ts=datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc))

    records = ledger.settle_expiry(
        timestamp=datetime(2026, 4, 22, 16, 0, tzinfo=timezone.utc),
        expiry_date=e1,
        outcomes_by_condition={"A": 1, "B": 1},
    )
    assert len(records) == 1
    assert records[0].condition_id == "A"

    yes_after, _ = ledger.snapshot_by_token()
    assert "A-yes" not in yes_after
    assert yes_after["B-yes"] == 20  # B untouched


def test_runner_settle_crossings_runs_exactly_once_per_expiry():
    """Boundary-cross logic must not double-settle on multi-tick gaps."""
    from src.algo.deribit_leadlag.backtest.runner import BacktestRunner
    from src.algo.deribit_leadlag.backtest.sim_gateway import SimulatedOrderGateway
    from src.algo.deribit_leadlag.config import (
        AllocationConfig, OrderConfig, SignalConfig,
    )
    from src.algo.deribit_leadlag.position_manager import PositionManager

    ledger = FillLedger()
    expiry = date(2026, 4, 22)
    _seed_buy_yes(
        ledger,
        condition_id="cond-X",
        expiry=expiry,
        strike=78000.0,
        size=100,
        price=0.55,
        ts=datetime(2026, 4, 22, 11, 0, tzinfo=timezone.utc),
    )

    mgr = PositionManager(
        alloc_config=AllocationConfig(50.0, 150.0, 500.0, 30.0),
        order_config=OrderConfig(0.02, 0.15, 0.20, 0.02, 0.03, 0.072),
        signal_config=SignalConfig(),
    )
    gw = SimulatedOrderGateway(mgr, 0.072, 30.0, None)

    runner = BacktestRunner(
        provider=None,  # not used in this test
        position_mgr=mgr,
        gateway=gw,
        ledger=ledger,
        signal_config=SignalConfig(),
        outcomes_by_condition={"cond-X": 1},
    )
    runner._resolution_time_by_expiry[expiry] = datetime(2026, 4, 22, 16, 0, tzinfo=timezone.utc)

    # First crossing: settles.
    runner._settle_crossings(
        prev_ts=datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc),
        now_ts=datetime(2026, 4, 22, 16, 30, tzinfo=timezone.utc),
    )
    assert len(ledger.settlements()) == 1

    # Second crossing: no-op because already in _settled_expiries.
    runner._settle_crossings(
        prev_ts=datetime(2026, 4, 22, 16, 30, tzinfo=timezone.utc),
        now_ts=datetime(2026, 4, 22, 17, 0, tzinfo=timezone.utc),
    )
    assert len(ledger.settlements()) == 1
