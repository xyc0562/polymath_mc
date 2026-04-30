"""
Lead-lag backtest CLI.

    python -m scripts.run_leadlag_backtest \\
        --db data/leadlag_recorder.db \\
        --start 2026-04-11 --end 2026-04-28 \\
        --output-dir data/backtests/run-1

Loads the production YAML at config/deribit_leadlag.yaml, applies CLI
overrides, runs the backtest end-to-end, writes fills.csv / trades.csv /
summary.json under --output-dir, and prints stdout tables.

The backtest replays recorded snapshots through the production
`run_strategy_tick` decision pipeline and a `SimulatedOrderGateway`. v1 is
taker-only; maker actions are discarded so the resulting PnL is a lower
bound on what production could have captured via maker rebates.
"""

import argparse
import logging
import os
import sys
from datetime import date, datetime, time as dtime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algo.deribit_leadlag.backtest.analyze import (
    fifo_resolve,
    print_full_report,
    summarize,
    write_artifacts,
)
from src.algo.deribit_leadlag.backtest.data_provider import SqliteMarketDataProvider
from src.algo.deribit_leadlag.backtest.ledger import FillLedger
from src.algo.deribit_leadlag.backtest.runner import BacktestRunner, _load_outcomes
from src.algo.deribit_leadlag.backtest.sim_gateway import SimulatedOrderGateway
from src.algo.deribit_leadlag.config import LeadLagConfig
from src.algo.deribit_leadlag.position_manager import PositionManager


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lead-lag offline backtest")
    p.add_argument("--config", default="config/deribit_leadlag.yaml")
    p.add_argument("--db", default="data/leadlag_recorder.db")
    p.add_argument("--start", default="2026-04-11", help="YYYY-MM-DD inclusive")
    p.add_argument("--end", default="2026-04-28", help="YYYY-MM-DD inclusive")
    p.add_argument("--tick-stride", type=int, default=1,
                   help="Replay every Nth snapshot (1 = every recorded minute)")
    p.add_argument(
        "--maker-policy",
        default="instant_optimistic",
        choices=("instant_optimistic", "discard"),
        help=(
            "How simulator handles is_maker=True actions. "
            "'instant_optimistic' (default, upper-bound PnL) fills at the algo's "
            "post price immediately; 'discard' (lower-bound PnL) drops them — "
            "useful to validate the trading core without relying on the "
            "optimistic maker assumption."
        ),
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to write fills/trades/summary; defaults to data/backtests/<timestamp>",
    )
    p.add_argument("--verbose", "-v", action="store_true")

    # Signal sweepables (override YAML).
    p.add_argument("--basis-haircut", type=float, default=None)
    p.add_argument("--no-next-day-extra-haircut", type=float, default=None)
    p.add_argument("--min-call-spread-usd", type=float, default=None)
    p.add_argument("--max-bracket-dk", type=float, default=None)
    p.add_argument("--max-bounds-width-taker", type=float, default=None)
    p.add_argument("--max-bounds-width-maker", type=float, default=None)
    p.add_argument("--min-prob", type=float, default=None)
    p.add_argument("--max-prob", type=float, default=None)
    p.add_argument("--min-time-to-expiry-hours", type=float, default=None)

    # Order sweepables.
    p.add_argument("--taker-min-edge", type=float, default=None)
    p.add_argument("--maker-min-edge", type=float, default=None)
    p.add_argument("--emergency-exit-edge", type=float, default=None)
    p.add_argument("--fee-rate", type=float, default=None,
                   help="Polymarket taker fee rate (default: from YAML)")

    # Allocation sweepables.
    p.add_argument("--per-bin-max-usd", type=float, default=None)
    p.add_argument("--per-date-max-usd", type=float, default=None)
    p.add_argument("--total-max-usd", type=float, default=None)
    p.add_argument("--max-order-size-usd", type=float, default=None)
    return p


def _apply_cli_overrides(cfg: LeadLagConfig, args: argparse.Namespace) -> None:
    """Mutate the YAML-loaded config with any --flag the user passed."""
    if args.basis_haircut is not None:
        cfg.signal.time_adjusted_basis_haircut = args.basis_haircut
    if args.no_next_day_extra_haircut is not None:
        cfg.signal.no_next_day_extra_haircut = args.no_next_day_extra_haircut
    if args.min_call_spread_usd is not None:
        cfg.signal.min_call_spread_usd = args.min_call_spread_usd
    if args.max_bracket_dk is not None:
        cfg.signal.max_bracket_dk = args.max_bracket_dk
    if args.max_bounds_width_taker is not None:
        cfg.signal.max_bounds_width_taker = args.max_bounds_width_taker
    if args.max_bounds_width_maker is not None:
        cfg.signal.max_bounds_width_maker = args.max_bounds_width_maker
    if args.min_prob is not None:
        cfg.signal.min_prob = args.min_prob
    if args.max_prob is not None:
        cfg.signal.max_prob = args.max_prob
    if args.min_time_to_expiry_hours is not None:
        cfg.signal.min_time_to_expiry_hours = args.min_time_to_expiry_hours

    if args.taker_min_edge is not None:
        cfg.order.taker_min_edge = args.taker_min_edge
    if args.maker_min_edge is not None:
        cfg.order.maker_min_edge = args.maker_min_edge
    if args.emergency_exit_edge is not None:
        cfg.order.emergency_exit_edge = args.emergency_exit_edge
    if args.fee_rate is not None:
        cfg.order.polymarket_crypto_fee_rate = args.fee_rate

    if args.per_bin_max_usd is not None:
        cfg.allocation.per_bin_max_usd = args.per_bin_max_usd
    if args.per_date_max_usd is not None:
        cfg.allocation.per_date_max_usd = args.per_date_max_usd
    if args.total_max_usd is not None:
        cfg.allocation.total_max_usd = args.total_max_usd
    if args.max_order_size_usd is not None:
        cfg.allocation.max_order_size_usd = args.max_order_size_usd


def _default_output_dir() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("data", "backtests", f"run-{stamp}")


def main() -> int:
    args = _build_argparser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    cfg = LeadLagConfig.from_yaml(args.config)
    _apply_cli_overrides(cfg, args)

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    start_ts = datetime.combine(start_date, dtime.min, tzinfo=timezone.utc)
    # End is inclusive of the entire day (and its 16:00 UTC settlement crossing).
    end_ts = datetime.combine(end_date + timedelta(days=1), dtime.min, tzinfo=timezone.utc)

    output_dir = args.output_dir or _default_output_dir()
    logger.info("Output: %s", output_dir)
    logger.info(
        "Config overrides applied. taker_min_edge=%.4f basis_haircut=%.4f no_next_day_extra=%.4f fee_rate=%.4f",
        cfg.order.taker_min_edge,
        cfg.signal.time_adjusted_basis_haircut,
        cfg.signal.no_next_day_extra_haircut,
        cfg.order.polymarket_crypto_fee_rate,
    )

    outcomes = _load_outcomes(args.db)

    position_mgr = PositionManager(
        alloc_config=cfg.allocation,
        order_config=cfg.order,
        signal_config=cfg.signal,
    )
    ledger = FillLedger()
    gateway = SimulatedOrderGateway(
        position_mgr=position_mgr,
        fee_rate=cfg.order.polymarket_crypto_fee_rate,
        max_order_size_usd=cfg.allocation.max_order_size_usd,
        fill_callback=ledger.record_fill,
        maker_policy=args.maker_policy,
    )

    with SqliteMarketDataProvider(args.db, start_ts, end_ts, tick_stride=args.tick_stride) as provider:
        runner = BacktestRunner(
            provider=provider,
            position_mgr=position_mgr,
            gateway=gateway,
            ledger=ledger,
            signal_config=cfg.signal,
            outcomes_by_condition=outcomes,
        )
        artifacts = runner.run()

    trades = fifo_resolve(artifacts.fills, artifacts.settlements)
    summary = summarize(trades)

    print_full_report(summary)
    write_artifacts(output_dir, artifacts.fills, trades, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
