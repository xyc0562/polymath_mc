"""
Backfill settlement outcomes for the lead-lag recorder DB.

Two-pass strategy:

  1. Sweep gamma /events with closed=true, restricted to a date window, and
     pull definitive outcomes from `outcomePrices` ([1,0]=YES, [0,1]=NO).
  2. For any condition_id in the window still flagged resolved=0 after step 1,
     fall back to Binance BTCUSDT 1m kline at the expiry's 16:00 UTC minute and
     decide YES if close > strike, NO otherwise.

After backfilling, cross-checks the Binance close against the Deribit BTC
index recorded in `snapshots.spot_price` at the same minute and warns on any
strike that would flip outcomes between the two venues — those rows are within
the cross-venue basis and any modeled edge is fragile.

Usage:
    python -m scripts.backfill_leadlag_settlements \\
        --db data/leadlag_recorder.db \\
        --start 2026-04-11 --end 2026-04-28
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algo.deribit_leadlag.polymarket_discovery import (
    BTC_TAG_ID,
    _extract_strike_from_title,
    _is_btc_above_event,
)
from src.const import GAMMA_API_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

# Polymarket BTC dailies resolve at 12:00 ET (noon ET). The UTC equivalent is
# 16:00 in EDT and 17:00 in EST — we let `zoneinfo` track DST so any date works.
POLY_SETTLE_TZ = ZoneInfo("America/New_York")
POLY_SETTLE_HOUR_ET = 12


def poly_settlement_dt_utc(expiry_date: date) -> datetime:
    """Return the exact UTC minute Polymarket resolves a daily BTC market on `expiry_date`."""
    et_settle = datetime.combine(
        expiry_date,
        dtime(POLY_SETTLE_HOUR_ET, 0, 0),
        tzinfo=POLY_SETTLE_TZ,
    )
    return et_settle.astimezone(timezone.utc)


def _iso_z(dt: datetime) -> str:
    """ISO-8601 with trailing 'Z' (gamma's expected format)."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_gamma_window(start_dt: datetime, end_dt: datetime, batch: int = 100):
    """Yield closed BTC events with endDate in [start_dt, end_dt]."""
    offset = 0
    while True:
        params = {
            "tag_id": BTC_TAG_ID,
            "closed": "true",
            "limit": batch,
            "offset": offset,
            "end_date_min": _iso_z(start_dt),
            "end_date_max": _iso_z(end_dt),
            "order": "endDate",
            "ascending": "true",
        }
        try:
            r = requests.get(f"{GAMMA_API_URL}/events", params=params, timeout=20)
            r.raise_for_status()
        except Exception as exc:
            logger.error(f"Gamma /events error at offset={offset}: {exc}")
            return
        events = r.json()
        if not events:
            return
        yield from events
        if len(events) < batch:
            return
        offset += batch
        time.sleep(0.1)


def _parse_outcome(mkt: dict) -> int:
    """Map gamma `outcomePrices` to {+1: YES, -1: NO, 0: indeterminate}."""
    op = mkt.get("outcomePrices", "")
    if isinstance(op, str):
        try:
            op = json.loads(op)
        except (json.JSONDecodeError, TypeError):
            return 0
    if not op or len(op) < 2:
        return 0
    try:
        yes = float(op[0])
    except (ValueError, TypeError):
        return 0
    if yes > 0.99:
        return 1
    if yes < 0.01:
        return -1
    return 0


def update_via_gamma(conn: sqlite3.Connection, start_dt: datetime, end_dt: datetime) -> int:
    """Sweep gamma for closed BTC threshold events and upsert settlements."""
    n_updated = 0
    n_skipped_hourly = 0
    n_indeterminate = 0
    for event in fetch_gamma_window(start_dt, end_dt):
        title = event.get("title", "")
        if not _is_btc_above_event(title):
            n_skipped_hourly += 1
            continue
        end_date = (event.get("endDate") or "")[:10]
        closed_time = event.get("closedTime") or datetime.now(timezone.utc).isoformat()
        for mkt in event.get("markets", []):
            cid = mkt.get("conditionId", "")
            if not cid:
                continue
            outcome = _parse_outcome(mkt)
            if outcome == 0:
                n_indeterminate += 1
                continue
            question = mkt.get("question", "") or mkt.get("groupItemTitle", "")
            strike = _extract_strike_from_title(question) or 0.0
            conn.execute(
                """INSERT INTO settlements
                       (condition_id, expiry_date, strike, question, resolved, resolved_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(condition_id) DO UPDATE SET
                       resolved = excluded.resolved,
                       resolved_at = excluded.resolved_at""",
                (cid, end_date, strike, question, outcome, closed_time),
            )
            n_updated += 1
    conn.commit()
    logger.info(
        "Gamma pass: %d resolutions written (%d hourly events skipped, "
        "%d markets indeterminate)",
        n_updated, n_skipped_hourly, n_indeterminate,
    )
    return n_updated


def fetch_binance_close(symbol: str, dt_utc: datetime) -> float:
    """Close of the 1m candle whose openTime equals dt_utc (UTC, second-aligned)."""
    open_ms = int(dt_utc.timestamp() * 1000)
    r = requests.get(
        BINANCE_KLINES_URL,
        params={
            "symbol": symbol,
            "interval": "1m",
            "startTime": open_ms,
            "endTime": open_ms + 60_000,
            "limit": 1,
        },
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError(f"No Binance kline at {dt_utc.isoformat()}")
    # Kline format: [openTime, open, high, low, close, volume, closeTime, ...]
    return float(data[0][4])


def fill_via_binance(
    conn: sqlite3.Connection,
    start_date: date,
    end_date: date,
) -> int:
    """For any unresolved row in [start_date, end_date], compute outcome from Binance."""
    rows = conn.execute(
        """SELECT condition_id, expiry_date, strike
           FROM settlements
           WHERE resolved = 0
             AND expiry_date BETWEEN ? AND ?
           ORDER BY expiry_date, strike""",
        (start_date.isoformat(), end_date.isoformat()),
    ).fetchall()

    n_updated = 0
    cache: dict[str, float] = {}
    for cid, expiry_date_str, strike in rows:
        dt = poly_settlement_dt_utc(date.fromisoformat(expiry_date_str))
        cache_key = expiry_date_str
        if cache_key not in cache:
            try:
                cache[cache_key] = fetch_binance_close("BTCUSDT", dt)
                time.sleep(0.05)  # gentle on Binance
            except Exception as exc:
                logger.warning(f"Binance fetch failed for {expiry_date_str}: {exc}")
                continue
        close = cache[cache_key]
        outcome = 1 if close > strike else -1
        conn.execute(
            """UPDATE settlements
               SET resolved = ?, resolved_at = ?
               WHERE condition_id = ?""",
            (outcome, dt.isoformat(), cid),
        )
        n_updated += 1
    conn.commit()
    logger.info(
        "Binance pass: %d resolutions written across %d distinct expiries",
        n_updated, len(cache),
    )
    return n_updated


def cross_check_against_deribit_spot(
    conn: sqlite3.Connection,
    start_date: date,
    end_date: date,
) -> int:
    """Warn on resolved rows where Deribit BTC index would flip the outcome."""
    rows = conn.execute(
        """SELECT s.condition_id, s.expiry_date, s.strike, s.resolved
           FROM settlements s
           WHERE s.resolved != 0
             AND s.expiry_date BETWEEN ? AND ?""",
        (start_date.isoformat(), end_date.isoformat()),
    ).fetchall()

    n_warn = 0
    for cid, expiry_date_str, strike, resolved in rows:
        settle_dt = poly_settlement_dt_utc(date.fromisoformat(expiry_date_str))
        spot_row = conn.execute(
            """SELECT spot_price
               FROM snapshots
               WHERE epoch_ts = ?
               ORDER BY ts ASC LIMIT 1""",
            (settle_dt.timestamp(),),
        ).fetchone()
        if spot_row is None or spot_row[0] is None:
            continue
        deribit_spot = float(spot_row[0])
        deribit_outcome = 1 if deribit_spot > strike else -1
        if deribit_outcome != resolved:
            logger.warning(
                f"  basis flip {expiry_date_str} K=${strike:,.0f}: "
                f"Binance→{('YES' if resolved == 1 else 'NO')} vs "
                f"Deribit spot=${deribit_spot:,.2f}→{('YES' if deribit_outcome == 1 else 'NO')} "
                f"(Δ=${deribit_spot - strike:+,.2f})"
            )
            n_warn += 1
    if n_warn == 0:
        logger.info("Cross-check: no Binance/Deribit outcome disagreements")
    else:
        logger.warning(
            "Cross-check: %d rows would flip under Deribit BTC index — "
            "treat their settlement labels as fragile in the backtest",
            n_warn,
        )
    return n_warn


def summarize(conn: sqlite3.Connection, start_date: date, end_date: date) -> None:
    rows = conn.execute(
        """SELECT expiry_date,
                  COUNT(*) AS total,
                  SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) AS yes_won,
                  SUM(CASE WHEN resolved = -1 THEN 1 ELSE 0 END) AS no_won,
                  SUM(CASE WHEN resolved = 0 THEN 1 ELSE 0 END) AS pending
           FROM settlements
           WHERE expiry_date BETWEEN ? AND ?
           GROUP BY expiry_date
           ORDER BY expiry_date""",
        (start_date.isoformat(), end_date.isoformat()),
    ).fetchall()
    logger.info("Per-expiry resolved counts:")
    for expiry, total, yes_won, no_won, pending in rows:
        logger.info(f"  {expiry}: total={total} YES={yes_won} NO={no_won} pending={pending}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill lead-lag settlements")
    parser.add_argument("--db", default="data/leadlag_recorder.db")
    parser.add_argument("--start", default="2026-04-11", help="YYYY-MM-DD inclusive")
    parser.add_argument("--end", default="2026-04-28", help="YYYY-MM-DD inclusive")
    parser.add_argument("--skip-gamma", action="store_true", help="Skip the gamma sweep")
    parser.add_argument("--skip-binance", action="store_true", help="Skip the Binance fallback")
    parser.add_argument("--skip-cross-check", action="store_true")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    # Gamma date filter is ON endDate (the resolution moment, 16:00 UTC),
    # so widen by one day either side to be inclusive of edge cases.
    gamma_start = datetime.combine(start_date, dtime.min, tzinfo=timezone.utc)
    gamma_end = datetime.combine(end_date + timedelta(days=1), dtime.min, tzinfo=timezone.utc)

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL")

    if not args.skip_gamma:
        update_via_gamma(conn, gamma_start, gamma_end)
    if not args.skip_binance:
        fill_via_binance(conn, start_date, end_date)
    if not args.skip_cross_check:
        cross_check_against_deribit_spot(conn, start_date, end_date)

    summarize(conn, start_date, end_date)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
