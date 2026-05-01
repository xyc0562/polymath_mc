"""
Trade analysis for the lead-lag backtest.

Produces:
  - `fills.csv` — every taker fill, with decision-time annotation.
  - `trades.csv` — FIFO-resolved buy/sell or buy/settlement pairs with PnL.
  - `summary.json` — totals, return %, basis-error mean/stdev per stratum.
  - stdout tables — top-line PnL plus the basis-error report stratified by
    `has_next_day`, side, hours-to-resolution bucket, bounds-width bucket,
    and fill-price bucket.

The headline metric is *basis error*:

    basis_error = realized_payoff − prob_conservative_at_entry

If mean basis_error ≥ −time_adjusted_basis_haircut, the haircut is at least
adequate. Substantially below means the haircut is too thin.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from .ledger import FillRecord, SettlementRecord

logger = logging.getLogger(__name__)


# ----- FIFO matcher -----


@dataclass
class ResolvedTrade:
    """Buy matched against a sell or a settlement payoff."""

    timestamp_open: datetime
    timestamp_close: datetime
    condition_id: str
    expiry_date: date
    strike: float
    side: str  # "YES" or "NO"
    size: float
    buy_price: float
    sell_price: float  # 0.0 or 1.0 for settlement, sell_price for closed_early
    buy_fee: float
    sell_fee: float
    pnl: float  # (sell − buy) * size − fees
    resolution: str  # "closed_early" | "settlement_win" | "settlement_loss"
    # Raw YES-side probabilities at entry (always reported relative to the YES
    # outcome). Use `model_prob_conservative_for_side` for the side-correct
    # value to compare against `realized_payoff` in the basis-error metric.
    prob_conservative_at_entry: Optional[float]
    prob_mid_at_entry: Optional[float]
    prob_aggressive_at_entry: Optional[float]
    bounds_width: Optional[float]
    has_next_day: Optional[bool]
    hours_to_resolution_at_entry: Optional[float]

    @property
    def model_prob_conservative_for_side(self) -> Optional[float]:
        """Model's worst-case probability our side wins.

        For a YES leg this is just `prob_conservative_at_entry` (the
        YES-side conservative). For a NO leg, the conservative NO
        probability is `1 - prob_aggressive_at_entry` — the worst-case
        for our trade is when YES is most likely (aggressive).
        """
        if self.side == "YES":
            return self.prob_conservative_at_entry
        if self.prob_aggressive_at_entry is None:
            return None
        return 1.0 - self.prob_aggressive_at_entry


def _yes_payoff_for_side(side: str, yes_resolved: int) -> float:
    """$1 if our `side` won, else $0. yes_resolved: +1=YES won, -1=NO won."""
    if side == "YES":
        return 1.0 if yes_resolved == 1 else 0.0
    return 1.0 if yes_resolved == -1 else 0.0


def fifo_resolve(
    fills: Iterable[FillRecord],
    settlements: Iterable[SettlementRecord],
) -> List[ResolvedTrade]:
    """Walk fills + settlements chronologically and produce per-trade PnL.

    For each `(condition_id, side)`:
      1. BUY pushes onto a FIFO queue of open positions.
      2. SELL pops from the front, emits `closed_early` trades.
      3. After all fills, the settlement event matches every remaining open
         BUY against the YES/NO outcome.
    """
    settlements_by_key: Dict[Tuple[str, str], SettlementRecord] = {}
    for s in settlements:
        # Settlements are recorded once per condition × side; key by both.
        settlements_by_key[(s.condition_id, "YES")] = s
        settlements_by_key[(s.condition_id, "NO")] = s

    buys_open: Dict[Tuple[str, str], Deque[FillRecord]] = defaultdict(deque)
    resolved: List[ResolvedTrade] = []

    sorted_fills = sorted(fills, key=lambda f: f.timestamp)
    for fill in sorted_fills:
        key = (fill.condition_id, fill.side)
        if fill.direction == "BUY":
            buys_open[key].append(fill)
            continue

        # SELL: match FIFO.
        size_remaining = float(fill.fill_size)
        # Per-fill SELL fee is allocated proportionally if it splits across buys.
        while size_remaining > 0 and buys_open[key]:
            buy = buys_open[key][0]
            available = float(buy.fill_size)
            matched = min(size_remaining, available)
            frac_buy = matched / float(buy.fill_size) if buy.fill_size else 0.0
            frac_sell = matched / float(fill.fill_size) if fill.fill_size else 0.0

            buy_fee_alloc = buy.fee * frac_buy
            sell_fee_alloc = fill.fee * frac_sell
            pnl = (fill.fill_price - buy.fill_price) * matched - buy_fee_alloc - sell_fee_alloc

            resolved.append(
                ResolvedTrade(
                    timestamp_open=buy.timestamp,
                    timestamp_close=fill.timestamp,
                    condition_id=buy.condition_id,
                    expiry_date=buy.expiry_date,
                    strike=buy.strike,
                    side=buy.side,
                    size=matched,
                    buy_price=buy.fill_price,
                    sell_price=fill.fill_price,
                    buy_fee=buy_fee_alloc,
                    sell_fee=sell_fee_alloc,
                    pnl=pnl,
                    resolution="closed_early",
                    prob_conservative_at_entry=buy.prob_conservative_at_entry,
                    prob_mid_at_entry=buy.prob_mid_at_entry,
                    prob_aggressive_at_entry=buy.prob_aggressive_at_entry,
                    bounds_width=buy.bounds_width,
                    has_next_day=buy.has_next_day,
                    hours_to_resolution_at_entry=buy.hours_to_resolution,
                )
            )

            buy.fill_size = int(buy.fill_size - matched)
            buy.fee -= buy_fee_alloc
            if buy.fill_size <= 0:
                buys_open[key].popleft()
            size_remaining -= matched

    # Settle remaining open buys.
    for key, queue in buys_open.items():
        if not queue:
            continue
        cid, side = key
        settle = settlements_by_key.get((cid, side))
        if settle is None:
            # No settlement record (gamma still pending or label not backfilled).
            continue
        payoff = _yes_payoff_for_side(side, settle.yes_resolved)
        for buy in queue:
            size = float(buy.fill_size)
            pnl = (payoff - buy.fill_price) * size - buy.fee
            resolution = "settlement_win" if payoff > 0 else "settlement_loss"
            resolved.append(
                ResolvedTrade(
                    timestamp_open=buy.timestamp,
                    timestamp_close=settle.timestamp,
                    condition_id=cid,
                    expiry_date=buy.expiry_date,
                    strike=buy.strike,
                    side=side,
                    size=size,
                    buy_price=buy.fill_price,
                    sell_price=payoff,
                    buy_fee=buy.fee,
                    sell_fee=0.0,
                    pnl=pnl,
                    resolution=resolution,
                    prob_conservative_at_entry=buy.prob_conservative_at_entry,
                    prob_mid_at_entry=buy.prob_mid_at_entry,
                    prob_aggressive_at_entry=buy.prob_aggressive_at_entry,
                    bounds_width=buy.bounds_width,
                    has_next_day=buy.has_next_day,
                    hours_to_resolution_at_entry=buy.hours_to_resolution,
                )
            )

    return resolved


# ----- basis-error stats -----


@dataclass
class BasisErrorStats:
    n: int
    mean: float
    stdev: float


def _stats(values: List[float]) -> BasisErrorStats:
    n = len(values)
    if n == 0:
        return BasisErrorStats(0, 0.0, 0.0)
    mean = sum(values) / n
    if n < 2:
        return BasisErrorStats(n, mean, 0.0)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return BasisErrorStats(n, mean, math.sqrt(var))


def _hours_bucket(hours: Optional[float]) -> str:
    if hours is None:
        return "unknown"
    if hours < 12:
        return "<12h"
    if hours < 24:
        return "12-24h"
    if hours < 48:
        return "24-48h"
    if hours < 96:
        return "48-96h"
    return "≥96h"


def _bounds_bucket(width: Optional[float]) -> str:
    if width is None:
        return "unknown"
    if width < 0.05:
        return "<0.05"
    if width < 0.10:
        return "0.05-0.10"
    if width < 0.20:
        return "0.10-0.20"
    if width < 0.30:
        return "0.20-0.30"
    return "≥0.30"


def _price_bucket(price: float) -> str:
    if price < 0.05:
        return "<0.05"
    if price < 0.10:
        return "0.05-0.10"
    if price < 0.30:
        return "0.10-0.30"
    if price < 0.50:
        return "0.30-0.50"
    if price < 0.70:
        return "0.50-0.70"
    if price < 0.90:
        return "0.70-0.90"
    return "≥0.90"


def basis_errors_by_stratum(
    trades: List[ResolvedTrade],
) -> Dict[str, Dict[str, BasisErrorStats]]:
    """Compute basis_error stats grouped by each stratifier."""
    groups: Dict[str, Dict[str, List[float]]] = {
        "side": defaultdict(list),
        "has_next_day": defaultdict(list),
        "resolution": defaultdict(list),
        "hours_bucket": defaultdict(list),
        "bounds_bucket": defaultdict(list),
        "price_bucket": defaultdict(list),
    }

    for t in trades:
        side_prob = t.model_prob_conservative_for_side
        if side_prob is None:
            continue
        # Realized payoff for this leg: closed_early uses sell_price; settlement
        # uses 0/1. Compare against the conservative model prob FOR THE SIDE
        # we're holding (1 - prob_aggressive_YES for NO legs).
        realized = t.sell_price
        basis_error = realized - side_prob

        groups["side"][t.side].append(basis_error)
        groups["has_next_day"][str(t.has_next_day)].append(basis_error)
        groups["resolution"][t.resolution].append(basis_error)
        groups["hours_bucket"][_hours_bucket(t.hours_to_resolution_at_entry)].append(basis_error)
        groups["bounds_bucket"][_bounds_bucket(t.bounds_width)].append(basis_error)
        groups["price_bucket"][_price_bucket(t.buy_price)].append(basis_error)

    return {
        dim: {bucket: _stats(vals) for bucket, vals in buckets.items()}
        for dim, buckets in groups.items()
    }


# ----- output -----


def write_fills_csv(path: str, fills: List[FillRecord]) -> None:
    if not fills:
        logger.warning("No fills to write to %s", path)
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "condition_id", "expiry_date", "strike", "side", "direction",
            "action", "fill_price", "fill_size", "fee", "edge_at_post",
            "prob_conservative_at_entry", "prob_mid_at_entry", "prob_aggressive_at_entry",
            "bounds_width", "has_next_day", "hours_to_resolution", "order_id",
        ])
        for r in fills:
            writer.writerow([
                r.timestamp.isoformat(),
                r.condition_id,
                r.expiry_date.isoformat(),
                f"{r.strike:.0f}",
                r.side,
                r.direction,
                r.action.value if hasattr(r.action, "value") else str(r.action),
                f"{r.fill_price:.4f}",
                r.fill_size,
                f"{r.fee:.4f}",
                f"{r.edge_at_post:.4f}",
                f"{r.prob_conservative_at_entry:.4f}" if r.prob_conservative_at_entry is not None else "",
                f"{r.prob_mid_at_entry:.4f}" if r.prob_mid_at_entry is not None else "",
                f"{r.prob_aggressive_at_entry:.4f}" if r.prob_aggressive_at_entry is not None else "",
                f"{r.bounds_width:.4f}" if r.bounds_width is not None else "",
                r.has_next_day if r.has_next_day is not None else "",
                f"{r.hours_to_resolution:.2f}" if r.hours_to_resolution is not None else "",
                r.order_id,
            ])


def write_trades_csv(path: str, trades: List[ResolvedTrade]) -> None:
    if not trades:
        logger.warning("No resolved trades to write to %s", path)
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_open", "timestamp_close", "condition_id", "expiry_date",
            "strike", "side", "size", "buy_price", "sell_price", "buy_fee", "sell_fee",
            "pnl", "resolution", "prob_conservative_at_entry", "prob_mid_at_entry",
            "prob_aggressive_at_entry", "model_prob_conservative_for_side",
            "bounds_width", "has_next_day", "hours_to_resolution_at_entry",
        ])
        for t in trades:
            side_prob = t.model_prob_conservative_for_side
            writer.writerow([
                t.timestamp_open.isoformat(),
                t.timestamp_close.isoformat(),
                t.condition_id,
                t.expiry_date.isoformat(),
                f"{t.strike:.0f}",
                t.side,
                f"{t.size:.0f}",
                f"{t.buy_price:.4f}",
                f"{t.sell_price:.4f}",
                f"{t.buy_fee:.4f}",
                f"{t.sell_fee:.4f}",
                f"{t.pnl:.4f}",
                t.resolution,
                f"{t.prob_conservative_at_entry:.4f}" if t.prob_conservative_at_entry is not None else "",
                f"{t.prob_mid_at_entry:.4f}" if t.prob_mid_at_entry is not None else "",
                f"{t.prob_aggressive_at_entry:.4f}" if t.prob_aggressive_at_entry is not None else "",
                f"{side_prob:.4f}" if side_prob is not None else "",
                f"{t.bounds_width:.4f}" if t.bounds_width is not None else "",
                t.has_next_day if t.has_next_day is not None else "",
                f"{t.hours_to_resolution_at_entry:.2f}" if t.hours_to_resolution_at_entry is not None else "",
            ])


def summarize(trades: List[ResolvedTrade]) -> dict:
    if not trades:
        return {
            "n_trades": 0,
            "total_pnl": 0.0,
            "total_size": 0.0,
            "win_rate": 0.0,
            "avg_pnl_per_trade": 0.0,
            "basis_error_overall": asdict(BasisErrorStats(0, 0.0, 0.0)),
            "by_stratum": {},
        }

    total_pnl = sum(t.pnl for t in trades)
    total_size = sum(t.size for t in trades)
    n_win = sum(1 for t in trades if t.pnl > 0)
    n = len(trades)

    overall_basis = [
        t.sell_price - t.model_prob_conservative_for_side
        for t in trades
        if t.model_prob_conservative_for_side is not None
    ]

    by_stratum_raw = basis_errors_by_stratum(trades)
    by_stratum = {
        dim: {bucket: asdict(s) for bucket, s in buckets.items()}
        for dim, buckets in by_stratum_raw.items()
    }

    return {
        "n_trades": n,
        "total_pnl": total_pnl,
        "total_size": total_size,
        "win_rate": n_win / n,
        "avg_pnl_per_trade": total_pnl / n,
        "basis_error_overall": asdict(_stats(overall_basis)),
        "by_stratum": by_stratum,
    }


def write_summary_json(path: str, summary: dict) -> None:
    def _default(obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        raise TypeError(f"Unserializable: {type(obj).__name__}")

    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=_default)


def print_topline(summary: dict) -> None:
    print()
    print("=" * 72)
    print("LEAD-LAG BACKTEST — TOPLINE")
    print("=" * 72)
    print(f"  Trades resolved:   {summary['n_trades']}")
    if summary["n_trades"] == 0:
        print("  (no trades — nothing to report)")
        return
    print(f"  Total PnL:         ${summary['total_pnl']:+,.2f}")
    print(f"  Total size traded: {summary['total_size']:,.0f} shares")
    print(f"  Win rate:          {summary['win_rate']*100:.1f}%")
    print(f"  Avg PnL/trade:     ${summary['avg_pnl_per_trade']:+,.4f}")
    be = summary["basis_error_overall"]
    print(
        f"  Basis-error:       n={be['n']}  mean={be['mean']:+.4f}  stdev={be['stdev']:.4f}"
    )


def print_basis_table(
    summary: dict,
    dim: str,
    title: str,
) -> None:
    buckets = summary["by_stratum"].get(dim, {})
    if not buckets:
        return
    print()
    print(f"--- Basis-error by {title} ---")
    print(f"  {'bucket':<14} {'n':>6}  {'mean':>9}  {'stdev':>8}")
    for bucket in sorted(buckets.keys()):
        s = buckets[bucket]
        print(f"  {bucket:<14} {s['n']:>6}  {s['mean']:>+9.4f}  {s['stdev']:>8.4f}")


def print_full_report(summary: dict) -> None:
    print_topline(summary)
    if summary["n_trades"] == 0:
        return
    print_basis_table(summary, "side", "side")
    print_basis_table(summary, "has_next_day", "has_next_day")
    print_basis_table(summary, "resolution", "resolution")
    print_basis_table(summary, "hours_bucket", "hours-to-resolution at entry")
    print_basis_table(summary, "bounds_bucket", "bounds_width")
    print_basis_table(summary, "price_bucket", "fill price")
    print()


def write_artifacts(
    output_dir: str,
    fills: List[FillRecord],
    trades: List[ResolvedTrade],
    summary: dict,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    write_fills_csv(os.path.join(output_dir, "fills.csv"), fills)
    write_trades_csv(os.path.join(output_dir, "trades.csv"), trades)
    write_summary_json(os.path.join(output_dir, "summary.json"), summary)
    logger.info("Wrote %s/{fills.csv,trades.csv,summary.json}", output_dir)
