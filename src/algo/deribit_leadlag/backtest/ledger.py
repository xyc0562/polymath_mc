"""
Fill ledger: backtest's source of truth for inventory and PnL accounting.

The runner pushes a snapshot from this ledger into `PositionManager` at the
top of every tick (via `set_position_snapshot`), so the production
`compute_targets` / `compute_deltas` see the right inventory. Fees are
recorded here, NOT in `PositionManager` — that mirrors production where
fees come from exchange trade reports, not from the inventory layer.

Each fill is annotated with the `AdjustedReferenceProb` it was decided on
during the same tick. This is what makes the basis-error metric possible:
`basis_error = realized_payoff − prob_conservative_at_entry`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from ..position_manager import DesiredAction
from ..signal_comparator import AdjustedReferenceProb
from .sim_gateway import SimFill

logger = logging.getLogger(__name__)


@dataclass
class FillRecord:
    """A single recorded fill plus its decision-time annotation."""

    timestamp: datetime
    condition_id: str
    yes_token_id: str
    no_token_id: str
    expiry_date: date
    strike: float
    side: str  # "YES" or "NO"
    direction: str  # "BUY" or "SELL"
    action: DesiredAction
    fill_price: float
    fill_size: int
    fee: float
    edge_at_post: float
    order_id: str

    # Annotated post-hoc from the tick's adjusted_probs map.
    prob_conservative_at_entry: Optional[float] = None
    prob_mid_at_entry: Optional[float] = None
    prob_aggressive_at_entry: Optional[float] = None
    bounds_width: Optional[float] = None
    has_next_day: Optional[bool] = None
    hours_to_resolution: Optional[float] = None


@dataclass
class SettlementRecord:
    """One synthetic fill emitted when a Polymarket bin settles."""

    timestamp: datetime
    condition_id: str
    expiry_date: date
    strike: float
    yes_resolved: int  # +1 (YES won), -1 (NO won), 0 (unresolved)
    yes_shares_settled: float
    no_shares_settled: float
    cash_payoff: float


class FillLedger:
    """Per-fill log + token-keyed inventory snapshot for the runner."""

    def __init__(self) -> None:
        self._fills: List[FillRecord] = []
        self._settlements: List[SettlementRecord] = []
        self._yes_by_token: Dict[str, float] = {}
        self._no_by_token: Dict[str, float] = {}

    # --- mutation ---

    def record_fill(self, sim_fill: SimFill) -> FillRecord:
        rec = FillRecord(
            timestamp=sim_fill.timestamp,
            condition_id=sim_fill.condition_id,
            yes_token_id=sim_fill.yes_token_id,
            no_token_id=sim_fill.no_token_id,
            expiry_date=sim_fill.bin_key.expiry_date,
            strike=sim_fill.bin_key.strike,
            side=sim_fill.side,
            direction=sim_fill.direction,
            action=sim_fill.action,
            fill_price=sim_fill.fill_price,
            fill_size=sim_fill.fill_size,
            fee=sim_fill.fee,
            edge_at_post=sim_fill.edge_at_post,
            order_id=sim_fill.order_id,
        )

        # Update inventory snapshot. SELLs reduce, BUYs add. Floor at 0
        # because the live position API never goes negative.
        token_dict = self._yes_by_token if sim_fill.side == "YES" else self._no_by_token
        prior = token_dict.get(sim_fill.token_id, 0.0)
        if sim_fill.direction == "BUY":
            token_dict[sim_fill.token_id] = prior + sim_fill.fill_size
        else:
            new_val = max(0.0, prior - sim_fill.fill_size)
            if new_val > 0:
                token_dict[sim_fill.token_id] = new_val
            else:
                token_dict.pop(sim_fill.token_id, None)

        self._fills.append(rec)
        return rec

    def annotate_last(
        self,
        adjusted_probs: Dict[Tuple[date, float], AdjustedReferenceProb],
        hours_to_resolution_by_key: Dict[Tuple[date, float], float],
    ) -> None:
        """Backfill prob/bounds annotation onto fills recorded this tick.

        Called by the runner once per tick after `run_strategy_tick`
        returns. Walks fills in reverse and stops at the first one already
        annotated, so the cost is O(per-tick fills).
        """
        for rec in reversed(self._fills):
            if rec.prob_conservative_at_entry is not None:
                break
            key = (rec.expiry_date, rec.strike)
            adj = adjusted_probs.get(key)
            if adj is not None:
                rec.prob_conservative_at_entry = adj.prob_conservative
                rec.prob_mid_at_entry = adj.prob_mid
                rec.prob_aggressive_at_entry = adj.prob_aggressive
                rec.bounds_width = adj.bounds_width
                rec.has_next_day = adj.has_next_day
            hours = hours_to_resolution_by_key.get(key)
            if hours is not None:
                rec.hours_to_resolution = hours

    def settle_expiry(
        self,
        timestamp: datetime,
        expiry_date: date,
        outcomes_by_condition: Dict[str, int],
    ) -> List[SettlementRecord]:
        """Realize PnL for positions whose `expiry_date` matches and whose
        condition_id has a definitive outcome in `outcomes_by_condition`.

        `outcomes_by_condition[condition_id] -> +1` (YES won) or `-1` (NO
        won). Missing condition_ids and any 0 entries are treated as
        unresolved: the position is LEFT IN PLACE, no SettlementRecord is
        emitted, and a warning is logged. The analyzer's FIFO matcher will
        skip those (cid, side) keys at end-of-run, so unresolved markets
        don't leak into PnL as fictitious losses. Backfill the
        `settlements` table (gamma → Binance fallback) before relying on
        backtest PnL.
        """
        # Group ledger inventory by (condition_id, side).
        # Map token_id -> (condition_id, side, expiry_date) by replaying fill records.
        token_meta: Dict[str, Tuple[str, str, date]] = {}
        for rec in self._fills:
            token = rec.yes_token_id if rec.side == "YES" else rec.no_token_id
            token_meta[token] = (rec.condition_id, rec.side, rec.expiry_date)

        records: List[SettlementRecord] = []
        unresolved_cids: set[str] = set()

        # Settle YES side.
        for token, shares in list(self._yes_by_token.items()):
            meta = token_meta.get(token)
            if meta is None:
                continue
            cid, side, exp = meta
            if exp != expiry_date or side != "YES":
                continue
            yes_resolved = outcomes_by_condition.get(cid)
            if yes_resolved is None or yes_resolved == 0:
                unresolved_cids.add(cid)
                continue  # leave position open
            payoff = shares if yes_resolved == 1 else 0.0
            records.append(
                SettlementRecord(
                    timestamp=timestamp,
                    condition_id=cid,
                    expiry_date=exp,
                    strike=self._strike_for_token(token),
                    yes_resolved=yes_resolved,
                    yes_shares_settled=shares,
                    no_shares_settled=0.0,
                    cash_payoff=payoff,
                )
            )
            self._yes_by_token.pop(token, None)
        # Settle NO side.
        for token, shares in list(self._no_by_token.items()):
            meta = token_meta.get(token)
            if meta is None:
                continue
            cid, side, exp = meta
            if exp != expiry_date or side != "NO":
                continue
            yes_resolved = outcomes_by_condition.get(cid)
            if yes_resolved is None or yes_resolved == 0:
                unresolved_cids.add(cid)
                continue
            payoff = shares if yes_resolved == -1 else 0.0
            records.append(
                SettlementRecord(
                    timestamp=timestamp,
                    condition_id=cid,
                    expiry_date=exp,
                    strike=self._strike_for_token(token),
                    yes_resolved=yes_resolved,
                    yes_shares_settled=0.0,
                    no_shares_settled=shares,
                    cash_payoff=payoff,
                )
            )
            self._no_by_token.pop(token, None)

        if unresolved_cids:
            logger.warning(
                "settle_expiry %s: %d condition_id(s) lack a definitive outcome; "
                "leaving positions open. Backfill `settlements` table to score them. "
                "Unresolved: %s",
                expiry_date, len(unresolved_cids),
                ", ".join(sorted(unresolved_cids))[:200],
            )

        self._settlements.extend(records)
        return records

    # --- read-only views ---

    def snapshot_by_token(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Return (yes_by_token, no_by_token) for `set_position_snapshot`."""
        return dict(self._yes_by_token), dict(self._no_by_token)

    def fills(self) -> List[FillRecord]:
        return list(self._fills)

    def settlements(self) -> List[SettlementRecord]:
        return list(self._settlements)

    def _strike_for_token(self, token: str) -> float:
        for rec in self._fills:
            if rec.yes_token_id == token or rec.no_token_id == token:
                return rec.strike
        return 0.0
