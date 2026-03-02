"""
Compare unified backtest dump files on a chronological holdout.

Usage:
    python3 -m scripts.compare_holdout \
        --baseline data/dumps/holdout/baseline.txt \
        --candidate data/dumps/holdout/late_018.txt \
        --candidate data/dumps/holdout/start12_020.txt \
        --cutoff-date 2026-02-27 \
        --cutoff-field counting_start

Notes:
    - The unified backtest summary prints trading-period start/end dates.
    - This script reconstructs counting-period dates from the event short name
      so the holdout can be defined on the actual counting window.
"""

from __future__ import annotations

import argparse
import re
import statistics
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List

from src.algo.musk_tweet_count.backtest.data_provider import parse_counting_dates


SUMMARY_ROW_RE = re.compile(
    r"^(?P<event>.+?\S)\s{2,}"
    r"(?P<trading_start>\d{4}-\d{2}-\d{2}) - (?P<settlement>\d{4}-\d{2}-\d{2})\s+"
    r"(?P<trades>\d+)\s+\$\s*(?P<pnl>[+\-]\d+\.\d+)\s+(?P<ret>[+\-]\d+\.\d+)%$"
)


@dataclass(frozen=True)
class EventSummary:
    event_name: str
    trading_start: date
    settlement_date: date
    counting_start: date
    counting_end: date
    duration_group: str
    trades: int
    pnl: float
    total_return: float


def _parse_iso_date(raw: str) -> date:
    return date.fromisoformat(raw)


def _duration_group(counting_start: date, counting_end: date) -> str:
    duration_days = (counting_end - counting_start).days
    if duration_days < 4:
        return "short"
    if duration_days < 9:
        return "weekly"
    return "monthly"


def parse_dump(path: Path) -> Dict[str, EventSummary]:
    rows: Dict[str, EventSummary] = {}
    for line in path.read_text().splitlines():
        match = SUMMARY_ROW_RE.match(line)
        if not match:
            continue

        event_name = match.group("event")
        if event_name == "TOTAL":
            continue

        trading_start = _parse_iso_date(match.group("trading_start"))
        settlement_date = _parse_iso_date(match.group("settlement"))
        counting_start, counting_end = parse_counting_dates(event_name, settlement_date)

        rows[event_name] = EventSummary(
            event_name=event_name,
            trading_start=trading_start,
            settlement_date=settlement_date,
            counting_start=counting_start,
            counting_end=counting_end,
            duration_group=_duration_group(counting_start, counting_end),
            trades=int(match.group("trades")),
            pnl=float(match.group("pnl")),
            total_return=float(match.group("ret")),
        )

    if not rows:
        raise ValueError(f"No backtest summary rows found in {path}")

    return rows


def filter_holdout(
    rows: Dict[str, EventSummary],
    cutoff_date: date,
    cutoff_field: str,
) -> Dict[str, EventSummary]:
    out: Dict[str, EventSummary] = {}
    for event_name, row in rows.items():
        row_date = {
            "counting_start": row.counting_start,
            "counting_end": row.counting_end,
            "settlement": row.settlement_date,
        }[cutoff_field]
        if row_date > cutoff_date:
            out[event_name] = row
    return out


def summarize_group(
    baseline_rows: Dict[str, EventSummary],
    candidate_rows: Dict[str, EventSummary],
    event_names: Iterable[str],
) -> str:
    names = list(event_names)
    if not names:
        return "count=0"

    base_total = sum(baseline_rows[name].pnl for name in names)
    cand_total = sum(candidate_rows[name].pnl for name in names)
    delta = cand_total - base_total
    improved = sum(candidate_rows[name].pnl > baseline_rows[name].pnl for name in names)
    regressed = sum(candidate_rows[name].pnl < baseline_rows[name].pnl for name in names)

    return (
        f"count={len(names)} base=${base_total:+.2f} cand=${cand_total:+.2f} "
        f"delta=${delta:+.2f} improved={improved} regressed={regressed}"
    )


def print_comparison(
    baseline_path: Path,
    candidate_path: Path,
    cutoff_date: date,
    cutoff_field: str,
    top_n: int,
) -> None:
    baseline_all = parse_dump(baseline_path)
    candidate_all = parse_dump(candidate_path)

    baseline = filter_holdout(baseline_all, cutoff_date, cutoff_field)
    candidate = filter_holdout(candidate_all, cutoff_date, cutoff_field)

    common_names = sorted(
        set(baseline) & set(candidate),
        key=lambda name: (
            baseline[name].settlement_date,
            baseline[name].event_name,
        ),
    )
    missing = sorted(set(baseline) - set(candidate))

    print()
    print(f"Candidate: {candidate_path.stem}")
    print(f"Baseline:  {baseline_path.stem}")
    print(f"Holdout:   {cutoff_field} > {cutoff_date.isoformat()}")

    if missing:
        print(f"Missing candidate events: {len(missing)}")
        for name in missing[:top_n]:
            print(f"  - {name}")

    if not common_names:
        print("No common holdout events selected.")
        return

    deltas = [candidate[name].pnl - baseline[name].pnl for name in common_names]
    base_total = sum(baseline[name].pnl for name in common_names)
    cand_total = sum(candidate[name].pnl for name in common_names)
    improved = sum(delta > 0 for delta in deltas)
    regressed = sum(delta < 0 for delta in deltas)
    unchanged = len(deltas) - improved - regressed

    print(
        f"Events:    {len(common_names)} | base=${base_total:+.2f} "
        f"cand=${cand_total:+.2f} delta=${cand_total - base_total:+.2f}"
    )
    print(
        f"Breadth:   improved={improved} regressed={regressed} unchanged={unchanged} | "
        f"median_delta=${statistics.median(deltas):+.2f} "
        f"mean_delta=${statistics.fmean(deltas):+.2f}"
    )

    for group in ("short", "weekly", "monthly"):
        names = [name for name in common_names if baseline[name].duration_group == group]
        print(f"{group:>7}: {summarize_group(baseline, candidate, names)}")

    ranked = sorted(
        (
            (
                candidate[name].pnl - baseline[name].pnl,
                name,
                baseline[name],
                candidate[name],
            )
            for name in common_names
        ),
        key=lambda item: item[0],
        reverse=True,
    )

    print(f"Top improved ({min(top_n, len(ranked))}):")
    for delta, name, base_row, cand_row in ranked[:top_n]:
        print(
            f"  {name:<25} delta=${delta:+.2f} "
            f"base=${base_row.pnl:+.2f} cand=${cand_row.pnl:+.2f} "
            f"group={base_row.duration_group} counting={base_row.counting_start}..{base_row.counting_end}"
        )

    print(f"Top regressed ({min(top_n, len(ranked))}):")
    for delta, name, base_row, cand_row in ranked[-top_n:]:
        print(
            f"  {name:<25} delta=${delta:+.2f} "
            f"base=${base_row.pnl:+.2f} cand=${cand_row.pnl:+.2f} "
            f"group={base_row.duration_group} counting={base_row.counting_start}..{base_row.counting_end}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare unified backtest dumps on a chronological holdout.")
    parser.add_argument("--baseline", required=True, help="Path to baseline backtest dump.")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Path to candidate backtest dump. May be passed multiple times.",
    )
    parser.add_argument(
        "--cutoff-date",
        required=True,
        help="Holdout cutoff in YYYY-MM-DD. Events strictly after this date are selected.",
    )
    parser.add_argument(
        "--cutoff-field",
        choices=("counting_start", "counting_end", "settlement"),
        default="counting_start",
        help="Event date field used to define the holdout.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of best/worst events to print for each candidate.",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    cutoff_date = _parse_iso_date(args.cutoff_date)

    for candidate in args.candidate:
        print_comparison(
            baseline_path=baseline_path,
            candidate_path=Path(candidate),
            cutoff_date=cutoff_date,
            cutoff_field=args.cutoff_field,
            top_n=args.top_n,
        )


if __name__ == "__main__":
    main()
