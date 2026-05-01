"""
Backtest runner: orchestrates per-tick replay + boundary-cross settlement.

Mirrors the live `_strategy_tick` path but additionally:
  - Rebuilds the position snapshot from `FillLedger` before each tick so
    `PositionManager` sees the correct inventory.
  - Captures the `adjusted_probs` map produced inside the tick so each
    fill can be annotated with its decision-time conservative probability
    (basis-error metric).
  - Detects when a tick crosses an expiry's resolution timestamp and
    emits synthetic settlement fills against the recorded outcomes.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple

from ..config import SignalConfig
from ..deribit_client import DeribitOption
from ..implied_probs import time_to_resolution_years
from ..polymarket_discovery import ThresholdMarket
from ..position_manager import BinKey, PositionManager
from ..settlement import CompatibilityClass
from ..signal_comparator import (
    AdjustedReferenceProb,
    build_adjusted_prob_map,
)
from .data_provider import SimulatedOrderbook, SqliteMarketDataProvider, TickSnapshot
from .ledger import FillLedger, SettlementRecord
from .sim_gateway import SimulatedOrderGateway

logger = logging.getLogger(__name__)


def _run_strategy_tick_with_probs(
    *,
    now: datetime,
    options: List[DeribitOption],
    markets: List[ThresholdMarket],
    compat_map: Dict[BinKey, CompatibilityClass],
    orderbooks_by_token: Dict[str, SimulatedOrderbook],
    signal_config: SignalConfig,
    position_mgr: PositionManager,
    order_gateway,
) -> Tuple[Optional[object], Dict[Tuple[date, float], AdjustedReferenceProb]]:
    """Like `strategy_tick.run_strategy_tick` but exposes the prob map.

    Identical decision path: same `build_adjusted_prob_map → compute_targets
    → compute_deltas → execute_actions → apply_execution_report` pipeline.
    Returns `(report, adjusted_probs)` so the runner can annotate fills.
    """
    if not options:
        return None, {}

    adjusted_probs = build_adjusted_prob_map(
        options=options,
        now=now,
        markets=markets,
        signal_config=signal_config,
        compatibility_map=compat_map,
    )
    if not adjusted_probs:
        return None, adjusted_probs

    position_mgr.compute_targets(adjusted_probs, orderbooks_by_token)
    actions = position_mgr.compute_deltas()
    if not actions:
        return None, adjusted_probs

    report = order_gateway.execute_actions(actions)
    position_mgr.apply_execution_report(report)
    return report, adjusted_probs


def _hours_to_resolution_map(
    now: datetime,
    markets: List[ThresholdMarket],
) -> Dict[Tuple[date, float], float]:
    out: Dict[Tuple[date, float], float] = {}
    for m in markets:
        T = time_to_resolution_years(now, m.resolution_time_utc)
        out[(m.expiry_date, m.strike)] = T * 365.25 * 24
    return out


def _load_outcomes(db_path: str) -> Dict[str, int]:
    """Read definitive settlement outcomes keyed by `condition_id`.

    Skips rows with `resolved = 0` (unresolved or indeterminate) so the
    ledger leaves those positions open rather than scoring them as losses.
    Backfill the settlements table via
    scripts/backfill_leadlag_settlements.py before relying on backtest PnL.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT condition_id, resolved FROM settlements WHERE resolved != 0"
    ).fetchall()
    conn.close()

    return {cid: int(resolved) for cid, resolved in rows}


@dataclass
class BacktestArtifacts:
    fills: List[object] = field(default_factory=list)  # FillRecord
    settlements: List[SettlementRecord] = field(default_factory=list)
    n_ticks: int = 0
    n_skipped_no_options: int = 0
    n_skipped_no_probs: int = 0


class BacktestRunner:
    def __init__(
        self,
        provider: SqliteMarketDataProvider,
        position_mgr: PositionManager,
        gateway: SimulatedOrderGateway,
        ledger: FillLedger,
        signal_config: SignalConfig,
        outcomes_by_condition: Dict[str, int],
    ):
        self._provider = provider
        self._position_mgr = position_mgr
        self._gateway = gateway
        self._ledger = ledger
        self._signal_config = signal_config
        self._outcomes = outcomes_by_condition
        self._settled_expiries: Set[date] = set()
        # Map (expiry_date, strike) -> resolution_time_utc, populated as we see markets.
        self._resolution_time_by_expiry: Dict[date, datetime] = {}

    def run(self) -> BacktestArtifacts:
        artifacts = BacktestArtifacts()
        prev_ts: Optional[datetime] = None

        for tick in self._provider.iter_snapshots():
            artifacts.n_ticks += 1

            # Track resolution times so settlement crossing logic has them.
            for m in tick.markets:
                self._resolution_time_by_expiry.setdefault(m.expiry_date, m.resolution_time_utc)

            # Settlement crossings happen BEFORE this tick's strategy run, so
            # any expiries whose resolution_time fell between (prev_ts, tick.timestamp]
            # are realized first.
            self._settle_crossings(prev_ts, tick.timestamp)

            # Rebuild position state from ledger BEFORE the strategy decision.
            yes_by_token, no_by_token = self._ledger.snapshot_by_token()
            self._position_mgr.update_markets(tick.markets, tick.compat_map)
            self._position_mgr.set_position_snapshot(yes_by_token, no_by_token)
            self._position_mgr.set_open_orders({})

            # Wire the gateway to this tick.
            self._gateway.set_clock(tick.timestamp)
            self._gateway.set_orderbooks(tick.orderbooks_by_token)

            report, adjusted_probs = _run_strategy_tick_with_probs(
                now=tick.timestamp,
                options=tick.options,
                markets=tick.markets,
                compat_map=tick.compat_map,
                orderbooks_by_token=tick.orderbooks_by_token,
                signal_config=self._signal_config,
                position_mgr=self._position_mgr,
                order_gateway=self._gateway,
            )

            if not tick.options:
                artifacts.n_skipped_no_options += 1
            elif not adjusted_probs:
                artifacts.n_skipped_no_probs += 1

            # Annotate any fills emitted this tick with their decision-time prob.
            if adjusted_probs:
                hours_map = _hours_to_resolution_map(tick.timestamp, tick.markets)
                self._ledger.annotate_last(adjusted_probs, hours_map)

            prev_ts = tick.timestamp

        # Final settlement sweep — anything resolved before our last tick that
        # we missed (e.g., recording window clipped past a 16:00 boundary).
        if prev_ts is not None:
            self._settle_crossings(None, prev_ts + timedelta(seconds=1))

        artifacts.fills = self._ledger.fills()
        artifacts.settlements = self._ledger.settlements()
        logger.info(
            "Backtest run finished: %d ticks (%d skipped no_options, %d skipped no_probs); "
            "%d fills, %d settlements",
            artifacts.n_ticks,
            artifacts.n_skipped_no_options,
            artifacts.n_skipped_no_probs,
            len(artifacts.fills),
            len(artifacts.settlements),
        )
        return artifacts

    def _settle_crossings(
        self,
        prev_ts: Optional[datetime],
        now_ts: datetime,
    ) -> None:
        """Settle any expiry whose resolution timestamp lies in (prev_ts, now_ts]."""
        for expiry, resolution_dt in list(self._resolution_time_by_expiry.items()):
            if expiry in self._settled_expiries:
                continue
            crossed = (prev_ts is None or resolution_dt > prev_ts) and resolution_dt <= now_ts
            if not crossed:
                continue
            records = self._ledger.settle_expiry(
                timestamp=resolution_dt,
                expiry_date=expiry,
                outcomes_by_condition=self._outcomes,
            )
            self._settled_expiries.add(expiry)
            if records:
                logger.info(
                    "Settled expiry %s: %d position rows realized",
                    expiry, len(records),
                )
