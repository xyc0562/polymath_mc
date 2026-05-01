"""Tests for SqliteMarketDataProvider — reconstruction + iteration on a synthetic DB."""

import os
import sqlite3
import tempfile
from datetime import datetime, timezone

from src.algo.deribit_leadlag.backtest.data_provider import SqliteMarketDataProvider
from src.algo.deribit_leadlag.position_manager import BinKey
from src.algo.deribit_leadlag.settlement import CompatibilityClass


def _seed_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            epoch_ts REAL NOT NULL,
            fetched_at REAL,
            spot_price REAL
        );
        CREATE TABLE deribit_options (
            snapshot_id INTEGER NOT NULL,
            instrument_name TEXT NOT NULL,
            expiry_date TEXT NOT NULL,
            strike REAL NOT NULL,
            bid_usd REAL,
            ask_usd REAL,
            mid_usd REAL,
            mark_usd REAL NOT NULL,
            mark_iv REAL,
            bid_iv REAL,
            ask_iv REAL,
            underlying_price REAL,
            volume_24h REAL,
            open_interest REAL
        );
        CREATE TABLE polymarket_bins (
            snapshot_id INTEGER NOT NULL,
            condition_id TEXT NOT NULL,
            yes_token_id TEXT NOT NULL,
            no_token_id TEXT NOT NULL,
            expiry_date TEXT NOT NULL,
            strike REAL NOT NULL,
            yes_price REAL NOT NULL,
            no_price REAL NOT NULL,
            volume REAL,
            compatibility TEXT,
            yes_best_bid REAL,
            yes_best_ask REAL
        );
    """)

    # Two snapshots, 60 seconds apart. Compute epoch_ts from the ts string
    # so we never desync: anchor = 2026-04-22 12:00 UTC.
    t0 = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc).timestamp()
    conn.execute(
        "INSERT INTO snapshots (id, ts, epoch_ts, fetched_at, spot_price) VALUES (1, ?, ?, ?, ?)",
        ("2026-04-22T12:00:00+00:00", t0, t0, 78000.0),
    )
    conn.execute(
        "INSERT INTO snapshots (id, ts, epoch_ts, fetched_at, spot_price) VALUES (2, ?, ?, ?, ?)",
        ("2026-04-22T12:01:00+00:00", t0 + 60, t0 + 60, 78050.0),
    )

    # Two calls per snapshot (bracketing strikes).
    for snap_id, underlying in [(1, 78000.0), (2, 78050.0)]:
        conn.execute(
            "INSERT INTO deribit_options VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (snap_id, "BTC-22APR26-78000-C", "2026-04-22", 78000.0,
             400.0, 420.0, 410.0, 410.0, 0.5, 0.49, 0.51, underlying, 100.0, 1000.0),
        )
        conn.execute(
            "INSERT INTO deribit_options VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (snap_id, "BTC-22APR26-80000-C", "2026-04-22", 80000.0,
             100.0, 110.0, 105.0, 105.0, 0.55, 0.54, 0.56, underlying, 50.0, 500.0),
        )

    # One Polymarket bin per snapshot, time_adjusted compatibility.
    conn.execute(
        "INSERT INTO polymarket_bins VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (1, "0xabc", "yes-tok", "no-tok", "2026-04-22", 78000.0,
         0.55, 0.45, 1234.0, "time_adjusted", 0.54, 0.56),
    )
    conn.execute(
        "INSERT INTO polymarket_bins VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (2, "0xabc", "yes-tok", "no-tok", "2026-04-22", 78000.0,
         0.56, 0.44, 1300.0, "time_adjusted", 0.55, 0.57),
    )
    conn.commit()
    conn.close()


def test_provider_iterates_in_order_and_reconstructs_dataclasses():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        _seed_db(path)
        start = datetime(2026, 4, 22, 11, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 22, 13, 0, tzinfo=timezone.utc)
        with SqliteMarketDataProvider(path, start, end) as p:
            ticks = list(p.iter_snapshots())

        assert len(ticks) == 2
        # Ascending order
        assert ticks[0].timestamp < ticks[1].timestamp
        assert ticks[0].snapshot_id == 1
        assert ticks[1].snapshot_id == 2

        first = ticks[0]
        assert first.spot == 78000.0
        assert len(first.options) == 2
        assert len(first.markets) == 1
        assert len(first.orderbooks_by_token) == 1

        # DeribitOption fields populated and *_price_usd properties correct.
        opt = next(o for o in first.options if o.strike == 78000.0)
        assert opt.option_type == "C"
        assert opt.underlying_price == 78000.0
        assert abs(opt.mark_price_usd - 410.0) < 1e-6
        assert abs(opt.bid_price_usd - 400.0) < 1e-6
        assert abs(opt.ask_price_usd - 420.0) < 1e-6
        assert abs(opt.mid_price_usd - 410.0) < 1e-6  # property recomputes

        # ThresholdMarket has resolution_time_utc synthesized at 16:00 UTC.
        m = first.markets[0]
        assert m.resolution_time_utc == datetime(2026, 4, 22, 16, 0, tzinfo=timezone.utc)
        assert m.condition_id == "0xabc"
        assert m.yes_token_id == "yes-tok"
        assert m.settlement is None
        assert m.event_id == ""

        # Compat map keyed by BinKey.
        key = BinKey(m.expiry_date, m.strike)
        assert first.compat_map[key] == CompatibilityClass.TIME_ADJUSTED

        # Orderbook attributes — NO mirrored from YES.
        ob = first.orderbooks_by_token["yes-tok"]
        assert ob.best_yes_bid == 0.54
        assert ob.best_yes_ask == 0.56
        assert abs(ob.best_no_bid - (1.0 - 0.56)) < 1e-9
        assert abs(ob.best_no_ask - (1.0 - 0.54)) < 1e-9
    finally:
        os.unlink(path)


def test_provider_drops_options_without_underlying_price():
    """Rows with NULL underlying_price can't reconstruct BTC prices — should skip."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        _seed_db(path)
        # Inject a snapshot 3 with a broken option.
        conn = sqlite3.connect(path)
        t2 = datetime(2026, 4, 22, 12, 2, tzinfo=timezone.utc).timestamp()
        conn.execute(
            "INSERT INTO snapshots (id, ts, epoch_ts, fetched_at, spot_price) VALUES (3, ?, ?, ?, ?)",
            ("2026-04-22T12:02:00+00:00", t2, t2, 78100.0),
        )
        conn.execute(
            "INSERT INTO deribit_options VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (3, "BTC-22APR26-78000-C", "2026-04-22", 78000.0,
             None, None, None, 410.0, 0.5, None, None, None, 0.0, 0.0),
        )
        conn.execute(
            "INSERT INTO polymarket_bins VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (3, "0xabc", "yes-tok", "no-tok", "2026-04-22", 78000.0,
             0.55, 0.45, 1234.0, "time_adjusted", 0.54, 0.56),
        )
        conn.commit()
        conn.close()

        start = datetime(2026, 4, 22, 12, 1, 30, tzinfo=timezone.utc)
        end = datetime(2026, 4, 22, 12, 5, tzinfo=timezone.utc)
        with SqliteMarketDataProvider(path, start, end) as p:
            ticks = list(p.iter_snapshots())
        assert len(ticks) == 1
        assert ticks[0].snapshot_id == 3
        assert len(ticks[0].options) == 0  # broken option dropped
        # But markets were still emitted.
        assert len(ticks[0].markets) == 1
    finally:
        os.unlink(path)


def test_provider_excludes_endpoint_at_end_ts():
    """Regression: half-open interval [start_ts, end_ts).

    The CLI passes `end_ts = midnight-of-(end_date+1)`. With the previous
    `BETWEEN ... AND ...` (inclusive) query, the very first snapshot of
    end_date+1 (at exactly 00:00:00) leaked into the run. The fix uses
    `epoch_ts >= start AND epoch_ts < end`.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        _seed_db(path)
        # Append a snapshot at exactly the end_ts (boundary moment).
        boundary_ts = datetime(2026, 4, 23, 0, 0, tzinfo=timezone.utc)
        boundary_epoch = boundary_ts.timestamp()
        conn = sqlite3.connect(path)
        conn.execute(
            "INSERT INTO snapshots (id, ts, epoch_ts, fetched_at, spot_price) VALUES (99, ?, ?, ?, ?)",
            (boundary_ts.isoformat(), boundary_epoch, boundary_epoch, 78200.0),
        )
        conn.commit()
        conn.close()

        # Range up to (but not including) the boundary: only the original
        # two snapshots should appear — never snapshot id=99.
        with SqliteMarketDataProvider(
            path,
            start_ts=datetime(2026, 4, 22, 0, 0, tzinfo=timezone.utc),
            end_ts=boundary_ts,  # exclusive
        ) as p:
            ticks = list(p.iter_snapshots())
        assert all(t.snapshot_id != 99 for t in ticks)
        assert len(ticks) == 2
    finally:
        os.unlink(path)


def test_provider_tick_stride_subsamples():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        _seed_db(path)
        start = datetime(2026, 4, 22, 11, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 22, 13, 0, tzinfo=timezone.utc)
        with SqliteMarketDataProvider(path, start, end, tick_stride=2) as p:
            ticks = list(p.iter_snapshots())
        assert len(ticks) == 1
        assert ticks[0].snapshot_id == 1
    finally:
        os.unlink(path)
