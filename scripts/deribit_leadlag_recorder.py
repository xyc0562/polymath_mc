"""
Data recorder for Deribit-Polymarket lead-lag backtesting.

Snapshots both venues every 60s into SQLite. Also computes and stores
adjusted reference probabilities so backtest can compare signal vs outcome.

Usage:
    python -m scripts.deribit_leadlag_recorder [--db data/leadlag_recorder.db] [--interval 60]

Tables:
    snapshots           — one row per snapshot cycle (with epoch_ts for range queries)
    deribit_options      — call option chain per snapshot
    polymarket_bins      — market prices + YES best bid/ask per snapshot
    adjusted_probs       — computed time-adjusted probs + BOTH sides' edges per snapshot
    settlements          — resolution outcomes (updated via gamma API closed events)

Note on Polymarket books:
    YES and NO books are exact mirrors (YES bid@p = NO ask@(1-p)). We store
    only YES best bid/ask. To get NO prices: NO best ask = 1 - YES best bid.

Backtest query:
    SELECT a.*, s.resolved
    FROM adjusted_probs a
    JOIN settlements s ON a.condition_id = s.condition_id
    WHERE s.resolved != 0
    ORDER BY a.snapshot_id, a.expiry_date, a.strike;
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algo.deribit_leadlag.deribit_client import fetch_btc_options_summary, fetch_spot_price
from src.algo.deribit_leadlag.implied_probs import build_call_price_curve
from src.algo.deribit_leadlag.polymarket_discovery import (
    discover_btc_threshold_markets,
    BTC_TAG_ID,
)
from src.algo.deribit_leadlag.settlement import classify_compatibility
from src.algo.deribit_leadlag.signal_comparator import (
    compute_polymarket_fee,
    interpolate_call_spread_to_poly_time,
)
from src.const import GAMMA_API_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CLOB_API_URL = "https://clob.polymarket.com"

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,              -- canonical wall-clock tick (exact interval boundary)
    epoch_ts REAL NOT NULL,         -- canonical epoch (matches ts)
    fetched_at REAL,                -- actual wall time when fetches completed
    spot_price REAL
);

CREATE TABLE IF NOT EXISTS deribit_options (
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
    open_interest REAL,
    FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
);

CREATE TABLE IF NOT EXISTS polymarket_bins (
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
    yes_best_ask REAL,
    FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
);

CREATE TABLE IF NOT EXISTS adjusted_probs (
    snapshot_id INTEGER NOT NULL,
    condition_id TEXT NOT NULL,
    expiry_date TEXT NOT NULL,
    strike REAL NOT NULL,
    prob_conservative REAL,
    prob_mid REAL,
    prob_aggressive REAL,
    method TEXT,
    has_next_day INTEGER,
    dk_same REAL,
    dk_next REAL,
    bounds_width REAL,
    yes_price REAL,
    no_price REAL,
    haircut REAL,
    buy_yes_gross_edge REAL,
    buy_yes_fee REAL,
    buy_yes_effective_edge REAL,
    buy_no_gross_edge REAL,
    buy_no_fee REAL,
    buy_no_effective_edge REAL,
    best_side TEXT,
    FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
);

CREATE TABLE IF NOT EXISTS settlements (
    condition_id TEXT PRIMARY KEY,
    expiry_date TEXT NOT NULL,
    strike REAL NOT NULL,
    question TEXT,
    resolved INTEGER NOT NULL DEFAULT 0,
    resolved_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_deribit_snapshot ON deribit_options(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_poly_snapshot ON polymarket_bins(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_adj_snapshot ON adjusted_probs(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_adj_condition ON adjusted_probs(condition_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_epoch ON snapshots(epoch_ts);
"""

FEE_RATE = 0.072
BASIS_HAIRCUT = 0.02
NO_NEXT_DAY_HAIRCUT = 0.03


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_db_path(db_path: str) -> str:
    """Anchor relative paths at repo root so cwd changes don't fork the DB file."""
    if os.path.isabs(db_path):
        return db_path
    return os.path.join(REPO_ROOT, db_path)


def init_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    # Set WAL before running schema so the journal mode sticks across the
    # first transaction (and survives across restarts).
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    # Additive migration: add fetched_at column if this is an older DB.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(snapshots)").fetchall()}
    if "fetched_at" not in cols:
        conn.execute("ALTER TABLE snapshots ADD COLUMN fetched_at REAL")
    conn.commit()
    return conn


def fetch_clob_books(token_ids: list[str]) -> dict[str, tuple]:
    """
    Batch-fetch CLOB orderbooks via POST /books (public, no auth).

    Returns dict mapping token_id -> (best_bid, best_ask) or (None, None).
    """
    if not token_ids:
        return {}

    result = {}
    for i in range(0, len(token_ids), 50):
        chunk = token_ids[i : i + 50]
        payload = [{"token_id": tid} for tid in chunk]
        try:
            resp = requests.post(
                f"{CLOB_API_URL}/books",
                json=payload,
                timeout=15,
            )
            resp.raise_for_status()
            books = resp.json()
        except Exception as e:
            logger.warning(f"CLOB books fetch failed (chunk {i}): {e}")
            continue

        if not isinstance(books, list):
            continue

        for book in books:
            asset_id = book.get("asset_id", "")
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            # Polymarket /books returns bids ascending and asks descending,
            # so bids[0] is the WORST bid and asks[0] is the WORST ask. Pick
            # the inside quote explicitly via max/min over the price field
            # so this stays correct regardless of any future ordering change.
            best_bid = max((float(b["price"]) for b in bids), default=None)
            best_ask = min((float(a["price"]) for a in asks), default=None)
            result[asset_id] = (best_bid, best_ask)

    return result


def fetch_closed_btc_events() -> list[dict]:
    """Fetch recently closed BTC threshold events from gamma API for settlement resolution."""
    all_events = []
    offset = 0
    limit = 50

    while True:
        try:
            resp = requests.get(
                f"{GAMMA_API_URL}/events",
                params={
                    "tag_id": BTC_TAG_ID,
                    "closed": "true",
                    "limit": limit,
                    "offset": offset,
                },
                timeout=15,
            )
            resp.raise_for_status()
            events = resp.json()
        except Exception as e:
            logger.warning(f"Gamma API closed events error at offset {offset}: {e}")
            break

        if not events:
            break

        all_events.extend(events)
        if len(events) < limit:
            break
        offset += limit
        if offset >= 200:
            break

    return all_events


def update_settlements_from_gamma(conn: sqlite3.Connection) -> int:
    """
    Query gamma API for closed BTC threshold events and update settlement outcomes.

    Uses the definitive outcomePrices field: [1,0] = YES won, [0,1] = NO won.
    """
    events = fetch_closed_btc_events()
    newly_resolved = 0

    for event in events:
        title = event.get("title", "")
        if "bitcoin" not in title.lower() or "above" not in title.lower():
            continue

        for mkt in event.get("markets", []):
            condition_id = mkt.get("conditionId", "")
            if not condition_id:
                continue

            row = conn.execute(
                "SELECT resolved FROM settlements WHERE condition_id = ? AND resolved != 0",
                (condition_id,),
            ).fetchone()
            if row is not None:
                continue

            outcome_prices = mkt.get("outcomePrices", "")
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except (json.JSONDecodeError, TypeError):
                    continue

            if not outcome_prices or len(outcome_prices) < 2:
                continue

            try:
                yes_final = float(outcome_prices[0])
            except (ValueError, TypeError):
                continue

            if yes_final > 0.99:
                resolved = 1
            elif yes_final < 0.01:
                resolved = -1
            else:
                continue

            question = mkt.get("question", "") or mkt.get("groupItemTitle", "")
            strike_match = re.search(r'\$([0-9,]+)', question)
            strike = float(strike_match.group(1).replace(",", "")) if strike_match else 0.0
            end_date = event.get("endDate", "")
            expiry_date = end_date[:10] if end_date else ""
            closed_time = event.get("closedTime") or datetime.now(timezone.utc).isoformat()

            conn.execute(
                """INSERT INTO settlements (condition_id, expiry_date, strike, question, resolved, resolved_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(condition_id) DO UPDATE SET
                       resolved = excluded.resolved,
                       resolved_at = excluded.resolved_at
                """,
                (condition_id, expiry_date, strike, question, resolved, closed_time),
            )
            newly_resolved += 1

    return newly_resolved


def record_snapshot(conn: sqlite3.Connection, tick_epoch: float) -> None:
    """
    Run one full snapshot cycle.

    tick_epoch is the canonical wall-clock tick this snapshot represents
    (the exact interval boundary we woke up for). The actual fetch time
    is recorded separately in `fetched_at` for latency diagnostics.
    """
    canonical_dt = datetime.fromtimestamp(tick_epoch, tz=timezone.utc)
    ts = canonical_dt.isoformat()
    epoch_ts = tick_epoch

    # Fetch Polymarket markets
    try:
        poly_markets = discover_btc_threshold_markets()
    except Exception as e:
        logger.error(f"Polymarket fetch failed: {e}")
        return
    if not poly_markets:
        logger.warning("No Polymarket markets found")
        return

    # Fetch Deribit options
    try:
        options = fetch_btc_options_summary()
    except Exception as e:
        logger.error(f"Deribit options fetch failed: {e}")
        return
    if not options:
        logger.warning("No Deribit options fetched")
        return

    try:
        spot = fetch_spot_price()
    except Exception:
        spot = None

    # Classify settlement compatibility
    compat_map = {}
    for m in poly_markets:
        if m.settlement is None:
            compat_map[m.condition_id] = "reject"
        else:
            cls, _ = classify_compatibility(m.expiry_date, m.settlement)
            compat_map[m.condition_id] = cls.value

    # Build call curves
    expiry_dates = set()
    for m in poly_markets:
        expiry_dates.add(m.expiry_date)
        expiry_dates.add(m.expiry_date + timedelta(days=1))

    curves = {}
    for exp in expiry_dates:
        c = build_call_price_curve(options, exp)
        if c:
            curves[exp] = c

    # Fetch CLOB best bid/ask (YES tokens only; NO is the mirror)
    yes_token_ids = [m.yes_token_id for m in poly_markets]
    book_data = fetch_clob_books(yes_token_ids)

    # Filter to calls for storage
    call_options = [
        o for o in options
        if o.option_type == "C" and o.expiry_date in expiry_dates
    ]

    # --- Insert snapshot ---
    # fetched_at is recorded AFTER all venue fetches finish so the delta
    # `fetched_at - epoch_ts` gives total fetch latency for this tick.
    fetched_at = datetime.now(timezone.utc).timestamp()
    cur = conn.execute(
        "INSERT INTO snapshots (ts, epoch_ts, fetched_at, spot_price) VALUES (?, ?, ?, ?)",
        (ts, epoch_ts, fetched_at, spot),
    )
    snap_id = cur.lastrowid

    # --- Deribit options ---
    deribit_rows = []
    for o in call_options:
        deribit_rows.append((
            snap_id, o.instrument_name, str(o.expiry_date), o.strike,
            o.bid_price_usd, o.ask_price_usd, o.mid_price_usd, o.mark_price_usd,
            o.mark_iv, o.bid_iv, o.ask_iv, o.underlying_price,
            o.volume_24h, o.open_interest,
        ))
    conn.executemany(
        "INSERT INTO deribit_options VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        deribit_rows,
    )

    # --- Polymarket bins with best bid/ask ---
    poly_rows = []
    books_recorded = 0
    for m in poly_markets:
        best_bid, best_ask = book_data.get(m.yes_token_id, (None, None))
        if best_bid is not None:
            books_recorded += 1
        poly_rows.append((
            snap_id, m.condition_id, m.yes_token_id, m.no_token_id,
            str(m.expiry_date), m.strike,
            m.yes_price, m.no_price, m.volume,
            compat_map.get(m.condition_id, "reject"),
            best_bid, best_ask,
        ))
    conn.executemany(
        "INSERT INTO polymarket_bins VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        poly_rows,
    )

    # --- Adjusted probs with BOTH sides' edges ---
    adj_rows = []
    eligible_count = 0
    for m in poly_markets:
        if compat_map.get(m.condition_id) == "reject":
            continue

        curve_same = curves.get(m.expiry_date)
        if not curve_same:
            continue

        curve_next = curves.get(m.expiry_date + timedelta(days=1))
        adj = interpolate_call_spread_to_poly_time(
            curve_same_day=curve_same,
            curve_next_day=curve_next,
            target_strike=m.strike,
            min_spread_usd=15.0,
        )
        if adj is None:
            continue

        eligible_count += 1
        haircut = BASIS_HAIRCUT + (NO_NEXT_DAY_HAIRCUT if not adj.has_next_day else 0.0)

        # Edge computation uses indicative prices (gamma API).
        # The backtest can re-derive execution prices from yes_best_bid/ask.
        buy_yes_gross = adj.prob_conservative - m.yes_price
        buy_yes_fee = compute_polymarket_fee(m.yes_price, FEE_RATE)
        buy_yes_eff = buy_yes_gross - buy_yes_fee - haircut

        buy_no_gross = (1.0 - adj.prob_aggressive) - m.no_price
        buy_no_fee = compute_polymarket_fee(m.no_price, FEE_RATE)
        buy_no_eff = buy_no_gross - buy_no_fee - haircut

        best_side = "BUY_YES" if buy_yes_eff >= buy_no_eff else "BUY_NO"

        adj_rows.append((
            snap_id, m.condition_id, str(m.expiry_date), m.strike,
            adj.prob_conservative, adj.prob_mid, adj.prob_aggressive,
            adj.method, 1 if adj.has_next_day else 0,
            adj.dk_same, adj.dk_next, adj.bounds_width,
            m.yes_price, m.no_price, round(haircut, 6),
            round(buy_yes_gross, 6), round(buy_yes_fee, 6), round(buy_yes_eff, 6),
            round(buy_no_gross, 6), round(buy_no_fee, 6), round(buy_no_eff, 6),
            best_side,
        ))

    conn.executemany(
        "INSERT INTO adjusted_probs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        adj_rows,
    )

    # Upsert pending settlements for active markets
    for m in poly_markets:
        conn.execute(
            """INSERT INTO settlements (condition_id, expiry_date, strike, question, resolved, resolved_at)
               VALUES (?, ?, ?, ?, 0, NULL)
               ON CONFLICT(condition_id) DO NOTHING
            """,
            (m.condition_id, str(m.expiry_date), m.strike, m.question),
        )

    conn.commit()

    spot_str = f"spot=${spot:,.0f}" if spot else "spot=N/A"
    latency_ms = (fetched_at - epoch_ts) * 1000
    logger.info(
        f"Snapshot {snap_id} @ {canonical_dt.strftime('%H:%M:%S')}: "
        f"{len(call_options)} deribit, {len(poly_markets)} poly, "
        f"{eligible_count} adj_probs, books={books_recorded}/{len(poly_markets)}, "
        f"{spot_str}, latency={latency_ms:.0f}ms"
    )


def main():
    parser = argparse.ArgumentParser(description="Deribit-Polymarket data recorder")
    parser.add_argument("--db", default="data/leadlag_recorder.db", help="SQLite database path")
    parser.add_argument("--interval", type=int, default=60, help="Snapshot interval in seconds")
    parser.add_argument("--settle-interval", type=int, default=600,
                        help="Settlement check interval in seconds")
    args = parser.parse_args()

    db_path = resolve_db_path(args.db)
    conn = init_db(db_path)
    snap_count = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM settlements WHERE resolved = 0").fetchone()[0]
    resolved = conn.execute("SELECT COUNT(*) FROM settlements WHERE resolved != 0").fetchone()[0]
    logger.info(f"Database: {db_path} ({snap_count} snapshots, {resolved} resolved, {pending} pending)")
    logger.info(
        f"Recording every {args.interval}s aligned to wall-clock, "
        f"settlement check every {args.settle_interval}s. Ctrl+C to stop."
    )

    last_settle_check = 0.0

    try:
        while True:
            # Wall-clock alignment: sleep until the next exact boundary
            # (e.g. interval=60 → wake at :00 of each minute, deterministic
            # across restarts so successive snapshots always land on the
            # same cadence).
            now = time.time()
            next_tick = (int(now) // args.interval + 1) * args.interval
            sleep_time = next_tick - now
            if sleep_time > 0:
                time.sleep(sleep_time)

            try:
                record_snapshot(conn, tick_epoch=float(next_tick))
            except Exception as e:
                logger.error(f"Snapshot failed: {e}", exc_info=True)

            # Periodically check for settled markets via gamma API
            if time.monotonic() - last_settle_check > args.settle_interval:
                try:
                    newly = update_settlements_from_gamma(conn)
                    conn.commit()
                    if newly > 0:
                        logger.info(f"Settlement update: {newly} newly resolved markets")
                    last_settle_check = time.monotonic()
                except Exception as e:
                    logger.error(f"Settlement check failed: {e}", exc_info=True)
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        final_count = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
        resolved = conn.execute("SELECT COUNT(*) FROM settlements WHERE resolved != 0").fetchone()[0]
        logger.info(f"Final: {final_count} snapshots, {resolved} resolved settlements")
        conn.close()


if __name__ == "__main__":
    main()
