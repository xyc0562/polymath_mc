"""
SQLite-backed market-data provider for the lead-lag backtest.

Iterates `snapshots` rows in order and yields a `TickSnapshot` per recorded
tick, with reconstructed `DeribitOption` and `ThresholdMarket` lists plus the
top-of-book orderbook per Polymarket YES token. The `compat_map` carries the
recorder's parsed `CompatibilityClass` so the backtest can skip `REJECT` bins
without re-parsing market descriptions (which the recorder didn't store).

Design notes:
  - Rebuilds dataclasses from raw rows. The pre-baked `adjusted_probs` table
    is intentionally NOT consumed here — driving every tick through the
    production `signal_comparator.build_adjusted_prob_map` keeps the haircut
    and fee configuration fully sweepable.
  - `underlying_price` drifts up to ~$350 within a single snapshot in the
    recorder; we preserve the per-row value when reconstructing
    `DeribitOption.*_price_btc`.
  - `ThresholdMarket.resolution_time_utc` is synthesized as 12:00 ET → UTC
    via `poly_settlement_dt_utc` (DST-aware). Fields not stored by the
    recorder (`description`, `settlement`, `event_id`, `question`) are
    populated with empty defaults; downstream code never dereferences them.
  - NO orderbook is mirrored from YES top-of-book (recorder only stores the
    YES side). Asymmetric NO-side liquidity is unmodeled in v1.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, Iterator, List, Optional

from ..deribit_client import DeribitOption
from ..polymarket_discovery import ThresholdMarket
from ..position_manager import BinKey
from ..settlement import CompatibilityClass

# Reuse the DST-aware helper from the settlement-backfill script.
from scripts.backfill_leadlag_settlements import poly_settlement_dt_utc

logger = logging.getLogger(__name__)


@dataclass
class SimulatedOrderbook:
    """Top-of-book orderbook view consumed by `position_manager._get_execution_prices`.

    PositionManager reads four attributes via `getattr`: `best_yes_bid`,
    `best_yes_ask`, `best_no_bid`, `best_no_ask`. NO is the strict
    YES-mirror.
    """

    best_yes_bid: Optional[float]
    best_yes_ask: Optional[float]

    @property
    def best_no_bid(self) -> Optional[float]:
        return None if self.best_yes_ask is None else 1.0 - self.best_yes_ask

    @property
    def best_no_ask(self) -> Optional[float]:
        return None if self.best_yes_bid is None else 1.0 - self.best_yes_bid


@dataclass
class TickSnapshot:
    """One recorded snapshot, fully reconstructed for a strategy tick."""

    timestamp: datetime  # tz-aware UTC
    snapshot_id: int
    spot: Optional[float]
    options: List[DeribitOption]
    markets: List[ThresholdMarket]
    compat_map: Dict[BinKey, CompatibilityClass] = field(default_factory=dict)
    orderbooks_by_token: Dict[str, SimulatedOrderbook] = field(default_factory=dict)


def _parse_ts(ts: str) -> datetime:
    """Parse `snapshots.ts` (ISO-8601, recorder writes with `+00:00`)."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _row_to_deribit_option(row: sqlite3.Row) -> Optional[DeribitOption]:
    """Reconstruct a `DeribitOption` from a `deribit_options` row.

    Returns None when the row lacks an `underlying_price` (BTC reconstruction
    requires it). Such rows are rare and the call-spread interpolation would
    skip them anyway.
    """
    underlying = row["underlying_price"]
    if underlying is None or underlying <= 0:
        return None

    mark_usd = row["mark_usd"]
    bid_usd = row["bid_usd"]
    ask_usd = row["ask_usd"]

    return DeribitOption(
        instrument_name=row["instrument_name"],
        expiry_date=date.fromisoformat(row["expiry_date"]),
        strike=float(row["strike"]),
        option_type="C",  # recorder stores calls only
        mark_iv=row["mark_iv"] or 0.0,
        underlying_price=float(underlying),
        mark_price_btc=float(mark_usd) / float(underlying) if mark_usd is not None else 0.0,
        bid_price_btc=(float(bid_usd) / float(underlying)) if bid_usd is not None else None,
        ask_price_btc=(float(ask_usd) / float(underlying)) if ask_usd is not None else None,
        bid_iv=row["bid_iv"],
        ask_iv=row["ask_iv"],
        volume_24h=row["volume_24h"] or 0.0,
        open_interest=row["open_interest"] or 0.0,
    )


def _row_to_threshold_market(row: sqlite3.Row) -> ThresholdMarket:
    """Reconstruct a `ThresholdMarket` from a `polymarket_bins` row."""
    expiry = date.fromisoformat(row["expiry_date"])
    return ThresholdMarket(
        condition_id=row["condition_id"],
        question="",
        strike=float(row["strike"]),
        expiry_date=expiry,
        resolution_time_utc=poly_settlement_dt_utc(expiry),
        yes_token_id=row["yes_token_id"],
        no_token_id=row["no_token_id"],
        yes_price=float(row["yes_price"]) if row["yes_price"] is not None else 0.0,
        no_price=float(row["no_price"]) if row["no_price"] is not None else 0.0,
        event_id="",
        volume=row["volume"] or 0.0,
        description="",
        settlement=None,
    )


def _parse_compatibility(value: Optional[str]) -> CompatibilityClass:
    """Map the recorder's stored compat string back to the enum."""
    if value is None:
        return CompatibilityClass.REJECT
    try:
        return CompatibilityClass(value)
    except ValueError:
        logger.warning("Unknown compatibility value %r — treating as REJECT", value)
        return CompatibilityClass.REJECT


class SqliteMarketDataProvider:
    """Iterate recorded snapshots between `start_ts` and `end_ts` (inclusive).

    Opens the SQLite database read-only. The iterator does two batched
    fetches per snapshot (one for `deribit_options`, one for `polymarket_bins`)
    so end-to-end iteration is O(snapshots), not O(snapshots × markets).
    """

    def __init__(
        self,
        db_path: str,
        start_ts: datetime,
        end_ts: datetime,
        tick_stride: int = 1,
    ):
        if tick_stride < 1:
            raise ValueError(f"tick_stride must be ≥ 1, got {tick_stride}")
        self._db_path = db_path
        self._start_ts = start_ts.astimezone(timezone.utc)
        self._end_ts = end_ts.astimezone(timezone.utc)
        self._tick_stride = tick_stride

        # Use URI form for read-only access.
        self._conn = sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
            detect_types=0,
        )
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SqliteMarketDataProvider":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def iter_snapshots(self) -> Iterator[TickSnapshot]:
        start_epoch = self._start_ts.timestamp()
        end_epoch = self._end_ts.timestamp()

        snap_cursor = self._conn.execute(
            """SELECT id, ts, epoch_ts, spot_price
               FROM snapshots
               WHERE epoch_ts BETWEEN ? AND ?
               ORDER BY epoch_ts ASC""",
            (start_epoch, end_epoch),
        )

        emitted = 0
        for snap_idx, snap in enumerate(snap_cursor):
            if snap_idx % self._tick_stride != 0:
                continue

            snapshot_id = snap["id"]
            ts = _parse_ts(snap["ts"])
            spot = snap["spot_price"]

            options = self._load_options(snapshot_id)
            markets, compat_map, orderbooks = self._load_markets(snapshot_id)

            yield TickSnapshot(
                timestamp=ts,
                snapshot_id=snapshot_id,
                spot=float(spot) if spot is not None else None,
                options=options,
                markets=markets,
                compat_map=compat_map,
                orderbooks_by_token=orderbooks,
            )
            emitted += 1

        logger.info(
            "SqliteMarketDataProvider: emitted %d snapshots in [%s, %s] (stride=%d)",
            emitted, self._start_ts.isoformat(), self._end_ts.isoformat(), self._tick_stride,
        )

    def _load_options(self, snapshot_id: int) -> List[DeribitOption]:
        rows = self._conn.execute(
            """SELECT instrument_name, expiry_date, strike, bid_usd, ask_usd,
                      mid_usd, mark_usd, mark_iv, bid_iv, ask_iv,
                      underlying_price, volume_24h, open_interest
               FROM deribit_options
               WHERE snapshot_id = ?""",
            (snapshot_id,),
        ).fetchall()
        out: List[DeribitOption] = []
        for row in rows:
            opt = _row_to_deribit_option(row)
            if opt is not None:
                out.append(opt)
        return out

    def _load_markets(
        self,
        snapshot_id: int,
    ) -> tuple[List[ThresholdMarket], Dict[BinKey, CompatibilityClass], Dict[str, SimulatedOrderbook]]:
        rows = self._conn.execute(
            """SELECT condition_id, yes_token_id, no_token_id, expiry_date, strike,
                      yes_price, no_price, volume, compatibility,
                      yes_best_bid, yes_best_ask
               FROM polymarket_bins
               WHERE snapshot_id = ?""",
            (snapshot_id,),
        ).fetchall()

        markets: List[ThresholdMarket] = []
        compat_map: Dict[BinKey, CompatibilityClass] = {}
        orderbooks: Dict[str, SimulatedOrderbook] = {}

        for row in rows:
            market = _row_to_threshold_market(row)
            markets.append(market)

            key = BinKey(market.expiry_date, market.strike)
            compat_map[key] = _parse_compatibility(row["compatibility"])

            yes_bid = row["yes_best_bid"]
            yes_ask = row["yes_best_ask"]
            orderbooks[market.yes_token_id] = SimulatedOrderbook(
                best_yes_bid=float(yes_bid) if yes_bid is not None else None,
                best_yes_ask=float(yes_ask) if yes_ask is not None else None,
            )

        return markets, compat_map, orderbooks
