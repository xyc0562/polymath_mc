"""
One-tick shadow comparison between a live production snapshot and replayed backtest.

This tool seeds the unified backtest from a production log snapshot, advances to the
first eligible replay tick at or after that timestamp, runs exactly one Kelly tick,
and prints the live vs replay blocker/trade state side by side.

Usage examples:

    # Compare one production snapshot against a seeded replay tick
    python -m src.algo.musk_tweet_count.backtest.shadow_compare \
        --mode compare \
        --event 2026-03-13_Mar_6_-_Mar_13 \
        --seed-from-log data/dumps/new_idea.log \
        --seed-log-ts 2026-03-13T03:01:03Z \
        --capital 1000 \
        --projection asymmetric \
        --intraday-mode bucket \
        --historical-bootstrap

    # Sweep production snapshots across a window and flag the first divergence
    python -m src.algo.musk_tweet_count.backtest.shadow_compare \
        --mode sweep \
        --event 2026-03-13_Mar_6_-_Mar_13 \
        --seed-from-log data/dumps/new_idea.log \
        --sweep-start-ts 2026-03-13T03:00:00Z \
        --sweep-end-ts 2026-03-13T15:16:03Z \
        --sweep-min-gap-seconds 1800 \
        --capital 1000 \
        --projection asymmetric \
        --intraday-mode bucket \
        --historical-bootstrap
"""

from __future__ import annotations

import argparse
import copy
import io
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .replay_seed import extract_seed_state_from_log
from .unified_runner import UnifiedBacktestConfig, UnifiedBacktestRunner
from ..kelly.backtest_backend import (
    BacktestOrderbookProvider,
    BacktestTradeExecutor,
    SimulationConfig,
    create_backtest_portfolio,
)
from ..kelly.config import CollateralConfig, EdgeBufferConfig, KellyConfig, RateLimitConfig
from ..kelly.executor import KellyExecutor

logger = logging.getLogger(__name__)


@dataclass
class LiveSnapshot:
    """Structured live snapshot parsed from the production log."""

    event_name: str
    requested_timestamp: int
    raw_header: str = ""
    current_count: Optional[int] = None
    time_left_hours: Optional[float] = None
    forecast_wallclock: Optional[str] = None
    available_capital: Optional[float] = None
    invested_capital: Optional[float] = None
    candidate_reason_lines: List[str] = field(default_factory=list)
    all_buy_candidate_lines: List[str] = field(default_factory=list)
    reject_lines: List[str] = field(default_factory=list)


@dataclass
class ReplaySnapshot:
    """Result of a single seeded replay tick."""

    event_dir: str
    event_name: str
    requested_timestamp: int
    seed_snapshot_timestamp: int
    replay_timestamp: int
    replay_datetime: datetime
    current_count: int
    forecast_mean: float
    forecast_std: float
    hours_to_settlement: float
    available_capital_before: float
    invested_before: float
    available_capital_after: float
    invested_after: float
    num_candidates: int
    num_executed: int
    total_utility_gain: float
    trade_lines: List[str] = field(default_factory=list)
    candidate_reason_lines: List[str] = field(default_factory=list)
    all_buy_candidate_lines: List[str] = field(default_factory=list)
    reject_lines: List[str] = field(default_factory=list)


@dataclass
class ShadowSweepRow:
    """Summary row for a live-vs-replay sweep point."""

    requested_timestamp: int
    time_left_hours: Optional[float]
    seed_snapshot_timestamp: int
    replay_timestamp: int
    live_reason_counts: Dict[str, int]
    live_reject_count: int
    live_buy_candidate_count: int
    replay_reason_counts: Dict[str, int]
    replay_reject_count: int
    replay_buy_candidate_count: int
    replay_num_candidates: int
    replay_num_executed: int
    replay_trade_count: int
    replay_total_utility_gain: float


def extract_live_snapshot_from_log(
    log_path: Path,
    event_name: str,
    snapshot_timestamp: int,
    max_window_seconds: int = 15,
) -> LiveSnapshot:
    """
    Extract a structured live snapshot from a production log for one timestamp.

    The parser intentionally stays narrow: it only captures the header, capital line,
    candidate rejection block, buy candidate block, and explicit reject lines from the
    exact second requested.
    """

    snapshot = LiveSnapshot(event_name=event_name, requested_timestamp=snapshot_timestamp)

    with log_path.open("r", encoding="utf-8") as f:
        matching_contents = []
        for raw_line in f:
            line_ts = _extract_log_timestamp(raw_line)
            if line_ts is None:
                continue
            if snapshot_timestamp <= line_ts <= snapshot_timestamp + max_window_seconds:
                matching_contents.append(_strip_log_prefix(raw_line))

    if not matching_contents:
        target_prefix = datetime.fromtimestamp(snapshot_timestamp, tz=timezone.utc).strftime("%y-%m-%d %H:%M:%S")
        raise ValueError(f"No log lines found from {target_prefix} (+{max_window_seconds}s) in {log_path}")

    capture_reasons = False
    capture_candidates = False
    seen_target_context = False
    for content in matching_contents:
        header_match = _match_event_header(content, event_name)
        if header_match:
            snapshot.raw_header = content
            snapshot.current_count = int(header_match["count"])
            snapshot.time_left_hours = float(header_match["time_left_hours"])
            snapshot.forecast_wallclock = header_match["forecast_wallclock"]
            continue

        capital_match = _match_event_capital(content, event_name)
        if capital_match:
            seen_target_context = True
            snapshot.invested_capital = float(capital_match["invested"])
            snapshot.available_capital = float(capital_match["available"])
            continue

        bracket_event = _extract_bracket_event_name(content)
        if (
            bracket_event
            and _normalize_event_name(bracket_event) == _normalize_event_name(event_name)
            and "Synced from API" in content
        ):
            seen_target_context = True
            continue

        if content == "Candidate rejection reasons:":
            if seen_target_context:
                capture_reasons = True
                capture_candidates = False
            continue

        if content == "All BUY candidates (priority order):":
            if seen_target_context:
                capture_reasons = False
                capture_candidates = True
            continue

        if capture_reasons:
            if content.startswith("  Bin "):
                snapshot.candidate_reason_lines.append(content.strip())
                continue
            capture_reasons = False

        if capture_candidates:
            if content.startswith("  Bin "):
                snapshot.all_buy_candidate_lines.append(content.strip())
                continue
            capture_candidates = False

        if (
            bracket_event
            and _normalize_event_name(bracket_event) == _normalize_event_name(event_name)
            and "[SIM iter=" in content
            and ("Reject " in content or "All candidates exhausted" in content)
        ):
            snapshot.reject_lines.append(content)

    if not snapshot.raw_header:
        target_prefix = datetime.fromtimestamp(snapshot_timestamp, tz=timezone.utc).strftime("%y-%m-%d %H:%M:%S")
        raise ValueError(
            f"Could not find event header for '{event_name}' at {target_prefix} in {log_path}"
        )

    return snapshot


def extract_replay_sections(log_text: str) -> tuple[List[str], List[str], List[str], List[str]]:
    """Extract candidate reasons, buy candidates, rejects, and trade lines from captured replay logs."""

    reason_lines: List[str] = []
    candidate_lines: List[str] = []
    reject_lines: List[str] = []
    trade_lines: List[str] = []

    capture_reasons = False
    capture_candidates = False

    for raw_line in log_text.splitlines():
        content = _strip_backtest_prefix(raw_line)

        if content == "Candidate rejection reasons:":
            capture_reasons = True
            capture_candidates = False
            continue

        if content == "All BUY candidates (priority order):":
            capture_reasons = False
            capture_candidates = True
            continue

        if capture_reasons:
            if content.startswith("  Bin "):
                reason_lines.append(content.strip())
                continue
            capture_reasons = False

        if capture_candidates:
            if content.startswith("  Bin "):
                candidate_lines.append(content.strip())
                continue
            capture_candidates = False

        if "[SIM iter=" in content and ("Reject " in content or "All candidates exhausted" in content):
            reject_lines.append(content)

        if _looks_like_trade_line(content):
            trade_lines.append(content.strip())

    return reason_lines, candidate_lines, reject_lines, trade_lines


def run_shadow_compare(
    event_dir: str,
    log_path: Path,
    seed_log_timestamp: int,
    config: UnifiedBacktestConfig,
    price_data_dir: Path,
    cache_dir: Path,
) -> tuple[LiveSnapshot, ReplaySnapshot]:
    """Run a live-vs-replay comparison for one event snapshot."""

    config = copy.deepcopy(config)
    runner = UnifiedBacktestRunner(
        config=config,
        price_data_dir=price_data_dir,
        cache_dir=cache_dir,
    )
    event = runner.price_provider.load_event(event_dir)
    if event is None:
        raise ValueError(f"Event not found: {event_dir}")

    live_event_name = config.seed_log_event_name or event.short_name
    live_snapshot = extract_live_snapshot_from_log(
        log_path=log_path,
        event_name=live_event_name,
        snapshot_timestamp=seed_log_timestamp,
    )
    seed_snapshot_timestamp = _resolve_seed_snapshot_timestamp(
        log_path=log_path,
        event_name=live_event_name,
        requested_timestamp=seed_log_timestamp,
    )
    config.resume_from_timestamp = seed_snapshot_timestamp
    config.seed_log_snapshot_timestamp = seed_snapshot_timestamp

    with _capture_logs() as captured:
        replay_snapshot = _run_single_replay_tick(
            runner=runner,
            event_dir=event_dir,
            event_name=live_event_name,
            requested_timestamp=seed_log_timestamp,
            seed_snapshot_timestamp=seed_snapshot_timestamp,
        )

    reason_lines, candidate_lines, reject_lines, trade_lines = extract_replay_sections(captured.getvalue())
    replay_snapshot.candidate_reason_lines = reason_lines
    replay_snapshot.all_buy_candidate_lines = candidate_lines
    replay_snapshot.reject_lines = reject_lines
    replay_snapshot.trade_lines = trade_lines
    return live_snapshot, replay_snapshot


def extract_event_snapshot_timestamps(
    log_path: Path,
    event_name: str,
    start_timestamp: Optional[int] = None,
    end_timestamp: Optional[int] = None,
    min_gap_seconds: int = 0,
) -> List[int]:
    """Extract live snapshot header timestamps for an event from the production log."""

    timestamps: List[int] = []
    last_kept: Optional[int] = None

    with log_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            ts = _extract_log_timestamp(raw_line)
            if ts is None:
                continue
            if start_timestamp is not None and ts < start_timestamp:
                continue
            if end_timestamp is not None and ts > end_timestamp:
                continue

            content = _strip_log_prefix(raw_line)
            if not _match_event_header(content, event_name):
                continue

            if last_kept is not None and ts - last_kept < min_gap_seconds:
                continue

            timestamps.append(ts)
            last_kept = ts

    return timestamps


def run_shadow_sweep(
    event_dir: str,
    log_path: Path,
    timestamps: List[int],
    config: UnifiedBacktestConfig,
    price_data_dir: Path,
    cache_dir: Path,
) -> List[ShadowSweepRow]:
    """Run shadow compare across multiple timestamps and summarize each point."""

    rows: List[ShadowSweepRow] = []
    for ts in timestamps:
        live, replay = run_shadow_compare(
            event_dir=event_dir,
            log_path=log_path,
            seed_log_timestamp=ts,
            config=config,
            price_data_dir=price_data_dir,
            cache_dir=cache_dir,
        )
        rows.append(
            ShadowSweepRow(
                requested_timestamp=ts,
                time_left_hours=live.time_left_hours,
                seed_snapshot_timestamp=replay.seed_snapshot_timestamp,
                replay_timestamp=replay.replay_timestamp,
                live_reason_counts=_reason_category_counts(live.candidate_reason_lines),
                live_reject_count=len(live.reject_lines),
                live_buy_candidate_count=len(live.all_buy_candidate_lines),
                replay_reason_counts=_reason_category_counts(replay.candidate_reason_lines),
                replay_reject_count=len(replay.reject_lines),
                replay_buy_candidate_count=len(replay.all_buy_candidate_lines),
                replay_num_candidates=replay.num_candidates,
                replay_num_executed=replay.num_executed,
                replay_trade_count=len(replay.trade_lines),
                replay_total_utility_gain=replay.total_utility_gain,
            )
        )
    return rows


def _run_single_replay_tick(
    runner: UnifiedBacktestRunner,
    event_dir: str,
    event_name: str,
    requested_timestamp: int,
    seed_snapshot_timestamp: int,
) -> ReplaySnapshot:
    """Run exactly one seeded replay tick at or after the configured resume timestamp."""

    event = runner.price_provider.load_event(event_dir)
    if event is None:
        raise ValueError(f"Event not found: {event_dir}")

    training_start_needed = event.start_date - timedelta(days=runner.config.training_days)
    days_available = (event.start_date - runner.config.tweet_data_start_date).days
    if days_available < runner.config.training_days:
        raise ValueError(
            f"Insufficient training data for {event.short_name}: need {runner.config.training_days} days, "
            f"only have {days_available} days (training would start {training_start_needed})"
        )

    runner.posts_provider.load_or_fetch(
        start_date=event.start_date - timedelta(days=runner.config.training_days),
        end_date=event.end_date + timedelta(days=1),
    )

    forecaster, backtest_posts = runner._create_forecaster(event)
    backtest_posts_idx = 0

    kelly_config = runner._create_kelly_config()
    token_ids = {b.bin_index: f"token_{b.bin_index}" for b in event.bins}
    num_bins = len(event.bins)
    initial_probs = [1.0 / num_bins] * num_bins
    bin_upper_bounds = [b.upper_bound for b in event.bins]

    portfolio = create_backtest_portfolio(
        initial_capital=runner.config.initial_capital,
        probabilities=initial_probs,
        bin_upper_bounds=bin_upper_bounds,
    )
    seed_state = runner._load_replay_seed_state(event)
    if seed_state is not None:
        runner._apply_replay_seed_state(
            portfolio=portfolio,
            token_ids=token_ids,
            event=event,
            seed_state=seed_state,
        )

    multiplier = kelly_config.collateral.capital_multiplier
    if multiplier > 1.0:
        portfolio.phantom_capital = runner.config.initial_capital * (multiplier - 1.0)
        kelly_config.collateral.capital_multiplier = 1.0

    backend_config = SimulationConfig(
        spread=runner.config.spread,
        slippage=runner.config.slippage,
        log_trades=False,
    )
    orderbook_provider = BacktestOrderbookProvider(
        config=backend_config,
        token_ids=token_ids,
    )
    trade_executor = BacktestTradeExecutor(
        config=backend_config,
        portfolio=portfolio,
    )
    executor = KellyExecutor(
        config=kelly_config,
        portfolio=portfolio,
        orderbook_provider=orderbook_provider,
        trade_executor=trade_executor,
        token_ids=token_ids,
        on_trade=lambda result: runner._record_trade(result),
        event_name=event.short_name,
    )

    sampled_timestamps = runner._sample_timestamps(
        event.all_timestamps,
        runner.config.tick_interval_seconds,
        start_at_ts=runner.config.resume_from_timestamp,
    )
    if not sampled_timestamps:
        raise ValueError(
            f"No replay timestamps available at or after {runner.config.resume_from_timestamp} for {event.short_name}"
        )
    ts = sampled_timestamps[0]
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)

    while backtest_posts_idx < len(backtest_posts):
        post_ts, tweet_event = backtest_posts[backtest_posts_idx]
        if post_ts <= ts:
            forecaster.event_store.add_event(tweet_event)
            backtest_posts_idx += 1
        else:
            break

    from zoneinfo import ZoneInfo

    est_tz = ZoneInfo("America/New_York")
    settlement_dt = datetime.combine(
        event.counting_end_date,
        datetime.min.time().replace(hour=12),
        tzinfo=est_tz,
    )
    hours_to_settlement = (settlement_dt - dt).total_seconds() / 3600.0

    forecast = forecaster.forecast_for_event_window(
        market_start_date=event.counting_start_date,
        settlement_date=event.counting_end_date,
        now=dt,
    )
    counting_start_dt, _ = forecaster.contract_utils.get_contract_day_bounds(event.counting_start_date)
    counting_start_ts = int(counting_start_dt.timestamp())
    current_count = runner.posts_provider.count_posts_in_range(
        start_ts=counting_start_ts,
        end_ts=ts,
    )

    probabilities = runner._compute_bin_probabilities(
        event=event,
        forecast=forecast,
        current_count=current_count,
    )
    dead_bins = [b.bin_index for b in event.bins if b.upper_bound < current_count]
    simulated_obs = runner.price_provider.get_all_orderbooks(event, ts)
    orderbook_provider.update_from_simulated(simulated_obs, ts)

    from ..kelly.market_signals import compute_market_consensus_blend

    probabilities, _ = compute_market_consensus_blend(
        probabilities=probabilities,
        dead_bins=dead_bins,
        orderbooks=simulated_obs,
        consensus_config=runner.config.trading.market_consensus,
        hours_remaining=hours_to_settlement,
    )

    available_before = portfolio.capital
    invested_before = portfolio.total_collateral_used

    portfolio.probabilities = probabilities
    portfolio.dead_bins = dead_bins
    portfolio.num_bins = num_bins

    runner._trades = []
    runner._trade_table_header_printed = False
    runner._tick_header_printed = False
    runner._current_tick_context = {
        "timestamp": ts,
        "datetime": dt,
        "current_count": current_count,
        "forecast_mean": forecast.mean,
        "forecast_std": forecast.std,
        "hours_to_settlement": hours_to_settlement,
        "probabilities": probabilities,
        "dead_bins": dead_bins,
        "bin_ranges": {b.bin_index: f"{b.lower_bound}-{b.upper_bound}" for b in event.bins},
        "orderbooks": simulated_obs,
        "event": event,
        "executor": executor,
    }

    tick_result = executor.run_tick_sync(hours_to_settlement)

    return ReplaySnapshot(
        event_dir=event_dir,
        event_name=event_name,
        requested_timestamp=requested_timestamp,
        seed_snapshot_timestamp=seed_snapshot_timestamp,
        replay_timestamp=ts,
        replay_datetime=dt,
        current_count=current_count,
        forecast_mean=forecast.mean,
        forecast_std=forecast.std,
        hours_to_settlement=hours_to_settlement,
        available_capital_before=available_before,
        invested_before=invested_before,
        available_capital_after=portfolio.capital,
        invested_after=portfolio.total_collateral_used,
        num_candidates=tick_result.num_candidates,
        num_executed=tick_result.num_executed,
        total_utility_gain=tick_result.total_utility_gain,
    )


@contextmanager
def _capture_logs():
    """Capture root logger output during a comparison run."""

    root_logger = logging.getLogger()
    old_handlers = list(root_logger.handlers)
    old_level = root_logger.level
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root_logger.handlers = [handler]
    root_logger.setLevel(logging.INFO)
    try:
        yield stream
    finally:
        root_logger.handlers = old_handlers
        root_logger.setLevel(old_level)


def _strip_log_prefix(line: str) -> str:
    parts = line.split("]: ", 1)
    if len(parts) == 2:
        return parts[1].rstrip("\n")
    return line.rstrip("\n")


def _extract_log_timestamp(line: str) -> Optional[int]:
    """Parse the yy-mm-dd HH:MM:SS prefix from a structured log line."""
    try:
        dt = datetime.strptime(line[:17], "%y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def _strip_backtest_prefix(line: str) -> str:
    parts = line.split("] ", 1)
    if len(parts) == 2:
        return parts[1]
    return line


def _reason_category_counts(lines: List[str]) -> Dict[str, int]:
    """Aggregate human-facing blocker lines into a small category summary."""

    counts = {
        "onesided_or_depth": 0,
        "friction": 0,
        "opposite_inventory": 0,
        "min_utility": 0,
        "bin_cap": 0,
        "other": 0,
    }

    for line in lines:
        payload = line.split(": ", 1)[1] if ": " in line else line
        clauses = [part.strip() for part in payload.split(";") if part.strip()]
        for clause in clauses:
            counts[_categorize_clause(clause)] += 1

    return counts


def _categorize_clause(clause: str) -> str:
    lower = clause.lower()
    if "one-sided liquidity" in lower or "no bid depth" in lower or "no ask depth" in lower:
        return "onesided_or_depth"
    if "sell friction" in lower or "friction" in lower:
        return "friction"
    if "sell no first" in lower or "sell yes first" in lower:
        return "opposite_inventory"
    if "sized_utility_below_min" in lower or "utility_below_min" in lower:
        return "min_utility"
    if "bin collateral limit" in lower:
        return "bin_cap"
    return "other"


def _looks_like_trade_line(content: str) -> bool:
    stripped = content.lstrip()
    if not stripped:
        return False
    first = stripped.split(" ", 1)[0]
    return first.isdigit() and any(
        token in stripped for token in ("BUY YES", "BUY NO", "SELL YES", "SELL NO")
    )


def _match_event_header(content: str, event_name: str) -> Optional[Dict[str, str]]:
    import re

    match = re.search(
        r"Event:\s+(?P<event>.+?)\s+\|\s+Count:\s+(?P<count>\d+)\s+\|\s+Time Left:\s+(?P<time_left_hours>[0-9.]+)h\s+\|\s+Forecast @(?P<forecast_wallclock>\S+)",
        content,
    )
    if not match:
        return None
    groups = match.groupdict()
    if _normalize_event_name(groups["event"]) != _normalize_event_name(event_name):
        return None
    return groups


def _match_event_capital(content: str, event_name: str) -> Optional[Dict[str, str]]:
    import re

    match = re.search(
        r"\[(?P<event>.+?)\] Event capital: budget=\$(?P<budget>[0-9.]+), invested=\$(?P<invested>[0-9.]+), available=\$(?P<available>[0-9.]+)",
        content,
    )
    if not match:
        return None
    groups = match.groupdict()
    if _normalize_event_name(groups["event"]) != _normalize_event_name(event_name):
        return None
    return groups


def _extract_bracket_event_name(content: str) -> Optional[str]:
    import re

    match = re.match(r"\[(?P<event>.+?)\]", content)
    if match:
        return match.group("event")
    return None


def _normalize_event_name(name: str) -> str:
    import re

    # Normalize day numbers like "Mar 06" and "Mar 6" to the same representation.
    def _depad(match: re.Match[str]) -> str:
        return str(int(match.group(0)))

    normalized = re.sub(r"\b0\d\b", _depad, name)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _print_shadow_report(live: LiveSnapshot, replay: ReplaySnapshot) -> None:
    requested_dt = datetime.fromtimestamp(live.requested_timestamp, tz=timezone.utc)
    seed_dt = datetime.fromtimestamp(replay.seed_snapshot_timestamp, tz=timezone.utc)
    replay_dt = replay.replay_datetime.astimezone(timezone.utc)

    print()
    print("=" * 120)
    print(f"Shadow Compare: {live.event_name}")
    print("=" * 120)
    print(f"Requested snapshot : {requested_dt.isoformat()}")
    print(f"Seed snapshot used : {seed_dt.isoformat()}")
    print(f"Replay tick used   : {replay_dt.isoformat()}")
    if replay.seed_snapshot_timestamp != live.requested_timestamp:
        print(f"Seed offset        : {replay.seed_snapshot_timestamp - live.requested_timestamp:+d}s")
    if replay.replay_timestamp != live.requested_timestamp:
        print(f"Tick offset        : {replay.replay_timestamp - live.requested_timestamp:+d}s")
    print()
    print("Live snapshot")
    print(f"  Header   : {live.raw_header}")
    if live.available_capital is not None or live.invested_capital is not None:
        print(
            f"  Capital  : available=${live.available_capital or 0.0:.2f}, "
            f"invested=${live.invested_capital or 0.0:.2f}"
        )
    print(f"  Reasons  : {len(live.candidate_reason_lines)} lines")
    for line in live.candidate_reason_lines[:12]:
        print(f"    {line}")
    if len(live.candidate_reason_lines) > 12:
        print(f"    ... ({len(live.candidate_reason_lines) - 12} more)")
    for line in live.all_buy_candidate_lines[:8]:
        print(f"    {line}")
    for line in live.reject_lines[:8]:
        print(f"    {line}")
    print()
    print("Replay tick")
    print(
        f"  Count/forecast : count={replay.current_count}, "
        f"mu={replay.forecast_mean:.1f}±{replay.forecast_std:.1f}, "
        f"T-{replay.hours_to_settlement:.1f}h"
    )
    print(
        f"  Capital        : before=${replay.available_capital_before:.2f} "
        f"(invested=${replay.invested_before:.2f}) -> after=${replay.available_capital_after:.2f} "
        f"(invested=${replay.invested_after:.2f})"
    )
    print(
        f"  Result         : candidates={replay.num_candidates}, "
        f"executed={replay.num_executed}, utility_gain={replay.total_utility_gain:.6f}"
    )
    print(f"  Reasons        : {len(replay.candidate_reason_lines)} lines")
    for line in replay.candidate_reason_lines[:12]:
        print(f"    {line}")
    if len(replay.candidate_reason_lines) > 12:
        print(f"    ... ({len(replay.candidate_reason_lines) - 12} more)")
    for line in replay.all_buy_candidate_lines[:8]:
        print(f"    {line}")
    for line in replay.reject_lines[:8]:
        print(f"    {line}")
    if replay.trade_lines:
        print("  Trades")
        for line in replay.trade_lines:
            print(f"    {line}")
    print("=" * 120)
    print()


def _print_shadow_sweep(rows: List[ShadowSweepRow]) -> None:
    print()
    print("=" * 140)
    print("Shadow Sweep")
    print("=" * 140)
    print(
        f"{'UTC':<20} {'T_left':>6} {'seed+':>6} {'tick+':>6} "
        f"{'live(os/fr/op)':>15} {'live_rej':>8} {'replay(c/e)':>12} {'replay(os/fr/op)':>18} {'util':>8}"
    )
    print("-" * 140)
    for row in rows:
        dt = datetime.fromtimestamp(row.requested_timestamp, tz=timezone.utc)
        live_counts = row.live_reason_counts
        replay_counts = row.replay_reason_counts
        print(
            f"{dt.strftime('%Y-%m-%d %H:%M'):<20} "
            f"{(row.time_left_hours if row.time_left_hours is not None else 0.0):>6.1f} "
            f"{row.seed_snapshot_timestamp - row.requested_timestamp:>+6d} "
            f"{row.replay_timestamp - row.requested_timestamp:>+6d} "
            f"{f'{live_counts['onesided_or_depth']}/{live_counts['friction']}/{live_counts['opposite_inventory']}':>15} "
            f"{row.live_reject_count:>8d} "
            f"{f'{row.replay_num_candidates}/{row.replay_num_executed}':>12} "
            f"{f'{replay_counts['onesided_or_depth']}/{replay_counts['friction']}/{replay_counts['opposite_inventory']}':>18} "
            f"{row.replay_total_utility_gain:>8.4f}"
        )
    print("=" * 140)
    print()


def _find_first_meaningful_divergence(rows: List[ShadowSweepRow]) -> Optional[ShadowSweepRow]:
    """
    Flag the earliest point where replay can still act while live is already blocked.

    This intentionally stays simple: the cases we care about are replay executing or
    surfacing candidates while the live snapshot is showing explicit rejects.
    """

    for row in rows:
        if row.replay_num_executed > 0 and row.live_reject_count > 0:
            return row
    for row in rows:
        if row.replay_num_candidates > 0 and row.live_reject_count > 0:
            return row
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed a one-tick replay from a production log snapshot and compare live vs replay blockers."
    )
    parser.add_argument("--mode", choices=["compare", "sweep"], default="compare")
    parser.add_argument("--event", required=True, help="Backtest event directory name")
    parser.add_argument("--seed-from-log", required=True, help="Production log path")
    parser.add_argument("--seed-log-ts", help="Snapshot timestamp (unix seconds or ISO-8601)")
    parser.add_argument("--seed-log-event", help="Live log event name override (defaults to event short name)")
    parser.add_argument("--price-data", default="data/price_history", help="Historical price data directory")
    parser.add_argument("--cache-dir", default="data/backtest_cache", help="Backtest cache directory")
    parser.add_argument("--capital", type=float, default=1000.0, help="Event capital")
    parser.add_argument("--capital-multiplier", type=float, default=1.0, help="Capital multiplier")
    parser.add_argument("--projection", choices=["asymmetric", "normal"], default="asymmetric")
    parser.add_argument("--intraday-mode", choices=["ridge", "bucket"], default="bucket")
    parser.add_argument("--interday-model", choices=["ewma", "gas", "pig"], default="ewma")
    parser.add_argument("--tick-interval", type=int, default=3600, help="Replay tick interval seconds")
    parser.add_argument("--spread", type=float, default=0.02, help="Backtest spread")
    parser.add_argument("--slippage", type=float, default=0.005, help="Backtest slippage")
    parser.add_argument("--historical-bootstrap", action="store_true", help="Enable intraday historical bootstrap")
    parser.add_argument("--bootstrap-start-hours", type=float, default=6.0)
    parser.add_argument("--bootstrap-full-hours", type=float, default=3.0)
    parser.add_argument("--bootstrap-max-blend", type=float, default=0.35)
    parser.add_argument("--min-buy-utility", type=float, default=KellyConfig.min_buy_utility)
    parser.add_argument("--min-sell-utility", type=float, default=KellyConfig.min_sell_utility)
    parser.add_argument("--roi", type=float, default=EdgeBufferConfig.required_roi)
    parser.add_argument("--friction-mid", type=float, default=EdgeBufferConfig.friction_mid)
    parser.add_argument("--friction-tail", type=float, default=EdgeBufferConfig.friction_tail)
    parser.add_argument("--sweep-start-ts", help="Sweep start timestamp (unix seconds or ISO-8601)")
    parser.add_argument("--sweep-end-ts", help="Sweep end timestamp (unix seconds or ISO-8601)")
    parser.add_argument("--sweep-min-gap-seconds", type=int, default=0, help="Minimum gap between live snapshots in sweep mode")
    return parser


def _parse_timestamp_arg(value: str) -> int:
    """Parse a CLI timestamp as unix seconds or ISO-8601 UTC."""

    value = value.strip()
    if value.isdigit():
        return int(value)

    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _resolve_seed_snapshot_timestamp(
    log_path: Path,
    event_name: str,
    requested_timestamp: int,
    max_lookahead_seconds: int = 15,
) -> int:
    """Resolve the nearest production sync block timestamp at or after the requested snapshot."""

    for candidate_ts in range(requested_timestamp, requested_timestamp + max_lookahead_seconds + 1):
        try:
            extract_seed_state_from_log(
                log_path=log_path,
                event_name=event_name,
                snapshot_timestamp=candidate_ts,
            )
        except ValueError:
            continue
        return candidate_ts

    requested_dt = datetime.fromtimestamp(requested_timestamp, tz=timezone.utc).isoformat()
    raise ValueError(
        f"Could not resolve a seed snapshot for '{event_name}' within +{max_lookahead_seconds}s of {requested_dt}"
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "compare" and not args.seed_log_ts:
        parser.error("--seed-log-ts is required in compare mode")

    trading_config = KellyConfig(
        min_buy_utility=args.min_buy_utility,
        min_sell_utility=args.min_sell_utility,
        edge_buffer=EdgeBufferConfig(
            required_roi=args.roi,
            friction_mid=args.friction_mid,
            friction_tail=args.friction_tail,
        ),
        rate_limit=RateLimitConfig(
            max_orders_per_tick=50,
            min_order_delay_seconds=0.0,
            max_orders_per_minute=1000,
        ),
        collateral=CollateralConfig(
            c_event_max=args.capital,
            capital_multiplier=args.capital_multiplier,
        ),
        max_iters_per_tick=50,
    )
    config = UnifiedBacktestConfig(
        initial_capital=args.capital,
        spread=args.spread,
        slippage=args.slippage,
        trading=trading_config,
        projection_model=args.projection,
        intraday_mode=args.intraday_mode,
        interday_model=args.interday_model,
        tick_interval_seconds=args.tick_interval,
        use_historical_bootstrap=args.historical_bootstrap,
        bootstrap_start_hours=args.bootstrap_start_hours,
        bootstrap_full_hours=args.bootstrap_full_hours,
        bootstrap_max_blend=args.bootstrap_max_blend,
        resume_from_timestamp=None,
        seed_log_path=args.seed_from_log,
        seed_log_snapshot_timestamp=None,
        seed_log_event_name=args.seed_log_event,
        tweet_data_start_date=date(2025, 11, 1),
    )

    if args.mode == "compare":
        seed_log_snapshot_ts = _parse_timestamp_arg(args.seed_log_ts)
        live, replay = run_shadow_compare(
            event_dir=args.event,
            log_path=Path(args.seed_from_log),
            seed_log_timestamp=seed_log_snapshot_ts,
            config=config,
            price_data_dir=Path(args.price_data),
            cache_dir=Path(args.cache_dir),
        )
        _print_shadow_report(live, replay)
        return

    event_name = args.seed_log_event
    if event_name is None:
        event = UnifiedBacktestRunner(
            config=copy.deepcopy(config),
            price_data_dir=Path(args.price_data),
            cache_dir=Path(args.cache_dir),
        ).price_provider.load_event(args.event)
        if event is None:
            raise ValueError(f"Event not found: {args.event}")
        event_name = event.short_name

    sweep_start_ts = _parse_timestamp_arg(args.sweep_start_ts) if args.sweep_start_ts else None
    sweep_end_ts = _parse_timestamp_arg(args.sweep_end_ts) if args.sweep_end_ts else None
    timestamps = extract_event_snapshot_timestamps(
        log_path=Path(args.seed_from_log),
        event_name=event_name,
        start_timestamp=sweep_start_ts,
        end_timestamp=sweep_end_ts,
        min_gap_seconds=args.sweep_min_gap_seconds,
    )
    if not timestamps:
        raise ValueError("No matching live snapshot timestamps found for sweep")

    rows = run_shadow_sweep(
        event_dir=args.event,
        log_path=Path(args.seed_from_log),
        timestamps=timestamps,
        config=config,
        price_data_dir=Path(args.price_data),
        cache_dir=Path(args.cache_dir),
    )
    _print_shadow_sweep(rows)
    divergence = _find_first_meaningful_divergence(rows)
    if divergence is not None:
        dt = datetime.fromtimestamp(divergence.requested_timestamp, tz=timezone.utc).isoformat()
        print(
            f"First meaningful divergence: {dt} | "
            f"live_rejects={divergence.live_reject_count} | "
            f"replay_candidates={divergence.replay_num_candidates} | "
            f"replay_executed={divergence.replay_num_executed}"
        )
    else:
        print("No meaningful divergence found in the sweep window.")


if __name__ == "__main__":
    main()
