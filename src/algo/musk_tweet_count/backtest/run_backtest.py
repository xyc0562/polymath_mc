"""
Command-line interface for running backtests.

Uses unified Kelly trading logic (same as production).

Usage:
    # List available events
    python -m src.algo.musk_tweet_count.backtest.run_backtest --list

    # Run backtest on specific event
    python -m src.algo.musk_tweet_count.backtest.run_backtest \
        --event "2025-11-25_Nov_18_-_Nov_25"

    # Run backtest on all events in date range
    python -m src.algo.musk_tweet_count.backtest.run_backtest \
        --start-date 2025-11-01 --end-date 2025-12-31

    # Custom parameters
    python -m src.algo.musk_tweet_count.backtest.run_backtest \
        --event "2025-11-25_Nov_18_-_Nov_25" \
        --capital 1000 --spread 0.02 --roi 0.10
"""

import argparse
import logging
import multiprocessing
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from .unified_runner import UnifiedBacktestRunner, UnifiedBacktestConfig, BacktestResult
from ..kelly.config import (
    KellyConfig,
    EdgeBufferConfig,
    RateLimitConfig,
    CollateralConfig,
    EventTradingRulesConfig,
    MarketConsensusConfig,
    RobustKellyConfig,
    MarketBuyGuardConfig,
    LateBoundaryTakeProfitConfig,
)
from ..forecaster.config import BucketNowcastConfig

# Setup logging with immediate flush to prevent interleaving with print statements
import sys


class FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit."""
    def emit(self, record):
        super().emit(record)
        self.flush()


# Configure root logger with flushing handler
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[FlushingStreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


def _resolve_consensus_mode(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[bool, bool]:
    """Resolve consensus mode presets against explicit legacy flags."""
    mode_to_flags = {
        "off": (False, False),
        "time_only": (True, False),
        "gap_only": (False, True),
        "time_gap": (True, True),
    }
    mode_time, mode_gap = mode_to_flags[args.consensus_mode]
    flag_time = bool(args.consensus_time)
    flag_gap = bool(args.consensus_gap)

    if args.consensus_mode != "off":
        if flag_time != mode_time and flag_time:
            parser.error(
                f"--consensus-mode {args.consensus_mode} conflicts with --consensus-time"
            )
        if flag_gap != mode_gap and flag_gap:
            parser.error(
                f"--consensus-mode {args.consensus_mode} conflicts with --consensus-gap"
            )
        args.consensus_time = mode_time
        args.consensus_gap = mode_gap

    return bool(args.consensus_time), bool(args.consensus_gap)


def list_events(price_data_dir: Path) -> List[str]:
    """List all available events for backtesting."""
    from .data_provider import HistoricalDataProvider

    provider = HistoricalDataProvider(price_data_dir)
    events = provider.list_available_events()

    print()
    print("=" * 70)
    print("Available Events for Backtesting")
    print("=" * 70)
    print(f"{'#':<4} {'Event Directory':<45} {'Date':<12}")
    print("-" * 70)

    for i, event_dir in enumerate(events, 1):
        # Extract date from directory name
        event_date = event_dir[:10] if len(event_dir) >= 10 else "N/A"
        print(f"{i:<4} {event_dir:<45} {event_date:<12}")

    print("=" * 70)
    print(f"Total: {len(events)} events")
    print()

    return events


def filter_events_by_date(
    events: List[str],
    start_date: Optional[date],
    end_date: Optional[date],
) -> List[str]:
    """Filter events by date range."""
    filtered = []

    for event_dir in events:
        try:
            # Extract date from directory name (format: YYYY-MM-DD_...)
            event_date = date.fromisoformat(event_dir[:10])

            if start_date and event_date < start_date:
                continue
            if end_date and event_date > end_date:
                continue

            filtered.append(event_dir)
        except ValueError:
            continue

    return filtered


def filter_events_by_duration(
    events: List[str],
    price_data_dir: Path,
    duration_days: Optional[int],
) -> List[str]:
    """Filter events by counting window duration."""
    if duration_days is None:
        return events

    from .data_provider import HistoricalDataProvider

    provider = HistoricalDataProvider(price_data_dir)
    filtered = []

    for event_dir in events:
        event = provider.load_event(event_dir)
        if event and event.counting_start_date and event.counting_end_date:
            # Calculate duration (matches trading rules convention)
            # "Dec 19 - Dec 26" = 7 days (noon Dec 19 to noon Dec 26)
            event_duration = (event.counting_end_date - event.counting_start_date).days
            if event_duration <= duration_days:
                filtered.append(event_dir)

    return filtered


def run_single_backtest(
    runner: UnifiedBacktestRunner,
    event_dir: str,
) -> Optional[BacktestResult]:
    """Run backtest on a single event."""
    logger.info(f"Running backtest: {event_dir}")
    return runner.run(event_dir)


def parse_timestamp_arg(value: str) -> int:
    """
    Parse a CLI timestamp argument as Unix seconds.

    Accepts either:
    - Unix seconds as an integer string
    - ISO-8601 datetime, with or without timezone (naive values are treated as UTC)
    """
    value = value.strip()
    if value.isdigit():
        return int(value)

    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _preload_posts(runner, event_dirs: List[str]):
    """Pre-load posts data for all events to ensure cache is warm."""
    from datetime import timedelta

    earliest_start = None
    latest_end = None

    for event_dir in event_dirs:
        event = runner.price_provider.load_event(event_dir)
        if event:
            training_start = event.start_date - timedelta(days=runner.config.training_days)
            if earliest_start is None or training_start < earliest_start:
                earliest_start = training_start
            if latest_end is None or event.end_date > latest_end:
                latest_end = event.end_date

    if earliest_start and latest_end:
        logger.info(f"Pre-loading posts for full date range: {earliest_start} to {latest_end}")
        runner.posts_provider.load_or_fetch(
            start_date=earliest_start,
            end_date=latest_end + timedelta(days=1),
        )


def _worker_run_event(args: Tuple) -> Tuple[str, Optional[BacktestResult], str]:
    """Worker function for multiprocessing. Runs a single event with its own runner.

    Returns (event_dir, result, captured_logs).
    """
    import io
    event_dir, config, price_data_dir, cache_dir, log_level = args

    # Capture all logging output to a string buffer
    log_buffer = io.StringIO()
    handler = logging.StreamHandler(log_buffer)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # Replace all handlers on root logger with our buffer handler
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    try:
        runner = UnifiedBacktestRunner(
            config=config,
            price_data_dir=price_data_dir,
            cache_dir=cache_dir,
        )

        result = runner.run(event_dir)
        return (event_dir, result, log_buffer.getvalue())
    except Exception as e:
        logging.error(f"Worker error for {event_dir}: {e}")
        return (event_dir, None, log_buffer.getvalue())


def run_multiple_backtests(
    runner: UnifiedBacktestRunner,
    event_dirs: List[str],
    parallel: int = 0,
) -> List[BacktestResult]:
    """Run backtest on multiple events, optionally in parallel."""
    # Pre-load all data for the full date range to avoid cache issues
    if event_dirs:
        _preload_posts(runner, event_dirs)

    if parallel > 1 and len(event_dirs) > 1:
        return _run_parallel(runner, event_dirs, parallel)

    # Sequential fallback
    results = []
    for i, event_dir in enumerate(event_dirs, 1):
        logger.info(f"\n[{i}/{len(event_dirs)}] Running backtest: {event_dir}")
        result = runner.run(event_dir)
        if result:
            results.append(result)
    return results


def _run_parallel(
    runner: UnifiedBacktestRunner,
    event_dirs: List[str],
    num_processes: int,
) -> List[BacktestResult]:
    """Run backtests in parallel using multiprocessing.

    Streams logs as each event completes (not necessarily in order).
    Results are collected and returned sorted by event order.
    """
    import sys

    log_level = logging.getLogger().level
    worker_args = [
        (event_dir, runner.config, runner.price_data_dir, runner.cache_dir, log_level)
        for event_dir in event_dirs
    ]

    num_procs = min(num_processes, len(event_dirs))
    logger.info(f"Running {len(event_dirs)} events across {num_procs} processes")

    # Use imap_unordered to stream results as they complete
    results_by_event = {}
    completed = 0
    with multiprocessing.Pool(processes=num_procs) as pool:
        for event_dir, result, logs in pool.imap_unordered(_worker_run_event, worker_args):
            completed += 1
            # Stream logs immediately as each event finishes
            if logs:
                sys.stderr.write(logs)
                sys.stderr.flush()
            logger.info(f"[{completed}/{len(event_dirs)}] Finished: {event_dir}")
            if result is not None:
                results_by_event[event_dir] = result

    # Return results in original event order
    results = [results_by_event[ed] for ed in event_dirs if ed in results_by_event]
    logger.info(f"Completed: {len(results)}/{len(event_dirs)} events returned results")
    return results


def print_summary(results: List[BacktestResult]) -> None:
    """Print summary of all backtest results."""
    if not results:
        logger.info("No results to summarize")
        return

    print()
    print("=" * 90)
    print("Backtest Summary")
    print("=" * 90)
    print(f"{'Event':<25} {'Period':<25} {'Trades':>7} {'P&L':>12} {'Return':>10}")
    print("-" * 90)

    total_pnl = 0.0
    total_trades = 0
    total_initial = 0.0

    for r in results:
        period = f"{r.start_date} - {r.end_date}"
        print(
            f"{r.event_name:<25} {period:<25} "
            f"{r.num_trades:>7} ${r.total_pnl:>+10.2f} {r.total_return:>+9.1f}%"
        )
        total_pnl += r.total_pnl
        total_trades += r.num_trades
        total_initial += r.initial_capital

    print("-" * 90)
    avg_return = (total_pnl / total_initial * 100) if total_initial > 0 else 0
    print(
        f"{'TOTAL':<25} {'':<25} "
        f"{total_trades:>7} ${total_pnl:>+10.2f} {avg_return:>+9.1f}%"
    )
    print("=" * 90)
    print()

    # Win rate stats
    total_winning = sum(r.num_winning_trades for r in results)
    total_losing = sum(r.num_losing_trades for r in results)
    total_bins = total_winning + total_losing
    overall_win_rate = total_winning / max(1, total_bins)

    print(f"Overall Statistics:")
    print(f"  Events tested: {len(results)}")
    print(f"  Total trades:  {total_trades}")
    print(f"  Win rate:      {overall_win_rate:.1%} ({total_winning}/{total_bins} bins)")
    print(f"  Total P&L:     ${total_pnl:+.2f}")
    print(f"  Avg return:    {avg_return:+.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run backtests on historical Musk tweet count events"
    )

    # Event selection
    parser.add_argument(
        "--event",
        type=str,
        help="Specific event directory to backtest",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="Only test events ending after this date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="Only test events ending before this date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available events and exit",
    )

    parser.add_argument(
        "--duration",
        type=int,
        help="Only test events with this counting window duration in days (e.g., 7 for weekly events)",
    )

    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Limit to first N events after filtering (0 = no limit)",
    )

    parser.add_argument(
        "--parallel",
        type=int,
        default=8,
        help="Number of parallel processes for multi-event runs (default: 8, 1 = sequential)",
    )

    # Trading parameters
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital. Default: 10000",
    )

    parser.add_argument(
        "--spread",
        type=float,
        default=0.02,
        help="Simulated bid-ask spread. Default: 0.02 (2%%)",
    )

    parser.add_argument(
        "--slippage",
        type=float,
        default=0.005,
        help="Simulated slippage. Default: 0.005 (0.5%%)",
    )

    parser.add_argument(
        "--tail-spread",
        type=float,
        default=None,
        help="Wider simulated spread when the mid is in the tail zone (mid < tail-zone or > 1-tail-zone). Default: None (uniform --spread; legacy).",
    )

    parser.add_argument(
        "--tail-zone",
        type=float,
        default=0.10,
        help="Price band defining tail bins for --tail-spread. Default: 0.10.",
    )

    parser.add_argument(
        "--max-fill-usd",
        type=float,
        default=0.0,
        help="Max notional USD at the top of the synthetic book per side (depth proxy; live fill p90 ~$250). Default: 0 = unlimited (legacy).",
    )

    parser.add_argument(
        "--roi",
        type=float,
        default=EdgeBufferConfig.required_roi,
        help=f"Required ROI for trades. Default: {EdgeBufferConfig.required_roi}",
    )

    parser.add_argument(
        "--friction-mid",
        type=float,
        default=EdgeBufferConfig.friction_mid,
        help=f"Friction for mid-range prices (10%% < p < 90%%). Default: {EdgeBufferConfig.friction_mid}",
    )

    parser.add_argument(
        "--friction-tail",
        type=float,
        default=EdgeBufferConfig.friction_tail,
        help=f"Friction for tail prices (p <= 10%% or p >= 90%%). Default: {EdgeBufferConfig.friction_tail}",
    )

    parser.add_argument(
        "--exit-hours",
        type=float,
        default=0.0,
        help="Close all positions X hours before settlement. Default: 0 (disabled)",
    )

    # Paths
    parser.add_argument(
        "--price-data",
        type=str,
        default="data/price_history",
        help="Directory containing price history data",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/backtest_cache",
        help="Directory for caching posts data",
    )

    # Optional seeded replay controls
    parser.add_argument(
        "--resume-from-ts",
        type=str,
        help="Resume replay from this UTC timestamp (ISO-8601 or Unix seconds).",
    )
    parser.add_argument(
        "--seed-state",
        type=str,
        help="Path to JSON file with seeded portfolio state for replay.",
    )
    parser.add_argument(
        "--seed-from-log",
        type=str,
        help="Path to production log file to extract seeded portfolio state from.",
    )
    parser.add_argument(
        "--seed-log-ts",
        type=str,
        help="UTC timestamp of the production log snapshot to seed from (ISO-8601 or Unix seconds).",
    )
    parser.add_argument(
        "--seed-log-event",
        type=str,
        help="Event name inside the production log. Defaults to the event short name.",
    )

    # Logging
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    # Kelly parameters
    parser.add_argument(
        "--kappa",
        type=float,
        default=KellyConfig.kappa,
        help=f"Fractional Kelly multiplier. Default: {KellyConfig.kappa}",
    )

    parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=KellyConfig.kelly_fraction,
        help=f"Fractional Kelly parameter α ∈ (0, 1]. "
             f"1.0 = full Kelly (log utility). "
             f"0.5 = half Kelly. 0.25 = quarter Kelly. Default: {KellyConfig.kelly_fraction}",
    )

    parser.add_argument(
        "--min-buy-utility",
        type=float,
        default=KellyConfig.min_buy_utility,
        help=f"Minimum utility gain for buys. Default: {KellyConfig.min_buy_utility}",
    )
    parser.add_argument(
        "--min-sell-utility",
        type=float,
        default=KellyConfig.min_sell_utility,
        help=f"Minimum utility gain for sells. Default: {KellyConfig.min_sell_utility}",
    )

    parser.add_argument(
        "--t-stop",
        type=float,
        default=None,
        help="Stop trading X hours before settlement (default: from KellyConfig)",
    )

    parser.add_argument(
        "--c-bin-max-ratio",
        type=float,
        default=CollateralConfig.c_bin_max_ratio,
        help=f"Max collateral per bin as ratio of c_event_max. Default: {CollateralConfig.c_bin_max_ratio}",
    )

    parser.add_argument(
        "--trade-verbose",
        action="store_true",
        help="Print detailed information about each trade. "
             "Shows price, size, edge, odds, Kelly reservation prices, etc.",
    )

    parser.add_argument(
        "--min-perceived-prob",
        type=float,
        default=EdgeBufferConfig.min_perceived_prob,
        help="Minimum perceived probability (from model) to trade. "
             "Don't trade if our model assigns probability below this. Default: EdgeBufferConfig.min_perceived_prob",
    )

    parser.add_argument(
        "--min-market-price",
        type=float,
        default=EdgeBufferConfig.min_market_price,
        help="Minimum market price to trade. Don't buy YES or NO if market price is below this. "
             "Recommended: 0.15-0.20 based on backtest analysis. Default: EdgeBufferConfig.min_market_price",
    )

    parser.add_argument(
        "--max-spread-ratio",
        type=float,
        default=EdgeBufferConfig.max_spread_ratio,
        help=f"Maximum spread ratio (ask-bid)/bid to trade. Default: {EdgeBufferConfig.max_spread_ratio}. Set to 0 to disable.",
    )

    parser.add_argument(
        "--no-require-two-sided",
        action="store_true",
        help="Disable requirement for two-sided liquidity (both bid and ask). Default: require two-sided.",
    )

    parser.add_argument(
        "--max-orders",
        type=int,
        default=1000,
        help="Maximum orders per tick. Default: 1000.",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Deprecated, no-op. Kept for backward compatibility.",
    )

    parser.add_argument(
        "--tick-interval",
        type=int,
        default=3600,
        help="Tick interval in seconds (default: 3600 = 1 hour).",
    )

    parser.add_argument(
        "--late-tick-interval",
        type=int,
        default=None,
        help="Optional higher-fidelity tick interval in seconds for the final settlement window. "
             "Example: 300 for 5-minute replay near settlement.",
    )

    parser.add_argument(
        "--late-tick-start-hours",
        type=float,
        default=None,
        help="Optional hours-before-settlement window where --late-tick-interval takes over. "
             "Example: 24 means use the late interval inside the final 24 hours.",
    )

    parser.add_argument(
        "--projection",
        type=str,
        default="asymmetric",
        choices=["asymmetric", "normal", "skew_normal", "gamma"],
        help="Projection model for computing bin probabilities. "
             "'asymmetric' (default) uses actual Monte Carlo samples. "
             "'normal' uses symmetric Normal CDF. "
             "'skew_normal' uses Skew-Normal CDF (captures right-skew). "
             "'gamma' uses Gamma CDF (natural for positive sums).",
    )

    parser.add_argument(
        "--intraday-mode",
        type=str,
        default="ridge",
        choices=["ridge", "bucket"],
        help="Intraday forecaster mode. "
             "'ridge' (default) uses Ridge regression with F(τ) progress curve. "
             "'bucket' uses bucket-based forecaster with 8 time buckets.",
    )

    parser.add_argument(
        "--interday-model",
        type=str,
        default="ewma",
        choices=["ewma", "gas", "pig"],
        help="Interday forecaster mode. "
             "'ewma' (default) uses the original EWMA regime model. "
             "'gas' uses NB-GAS regime dynamics. "
             "'pig' uses PIG-GAS with heavier-tailed future-day sampling.",
    )

    parser.add_argument(
        "--capital-multiplier",
        type=float,
        default=1.0,
        help="Capital multiplier for phantom capital injection. "
             "1.0 = standard Kelly. 2.0 = Kelly sees 2x capital → bigger positions. "
             "Real capital still hard-gates execution. Default: 1.0",
    )

    parser.add_argument(
        "--exit-mode",
        type=str,
        default="kelly_only",
        choices=["kelly_or_fairprice", "kelly_only"],
        help="Exit mode for closing positions. "
             "'kelly_or_fairprice' exits when market >= fair value AND utility >= 0. "
             "'kelly_only' exits based only on utility (skips fair value check).",
    )

    parser.add_argument(
        "--event-rules",
        type=str,
        default="config/event_trading_rules.yaml",
        help="Path to event trading rules YAML config (default: config/event_trading_rules.yaml). "
             "Controls when trading is allowed based on event duration and counting status.",
    )

    parser.add_argument(
        "--nu-scale",
        type=float,
        default=1.0,
        help="CMP nu_scale multiplier for COM-Poisson distribution. "
             "Higher = thinner left tail. Default: 1.0",
    )

    parser.add_argument(
        "--historical-bootstrap",
        action="store_true",
        help="Blend late intraday bucket forecasts with weighted historical suffix samples.",
    )

    parser.add_argument(
        "--bootstrap-start-hours",
        type=float,
        default=6.0,
        help="Hours left threshold where historical bootstrap starts blending in. Default: 6.0.",
    )

    parser.add_argument(
        "--bootstrap-full-hours",
        type=float,
        default=3.0,
        help="Hours left threshold where historical bootstrap reaches max blend. Default: 3.0.",
    )

    parser.add_argument(
        "--bootstrap-max-blend",
        type=float,
        default=0.35,
        help="Maximum mixture weight for historical bootstrap samples. Default: 0.35.",
    )

    parser.add_argument(
        "--boundary-silence-overlay",
        action="store_true",
        help="Enable the late-boundary silence overlay in the final hours before settlement.",
    )

    parser.add_argument(
        "--boundary-silence-hours",
        type=float,
        default=BucketNowcastConfig.LateBoundarySilenceConfig.start_hours,
        help="Hours before settlement where the late-boundary silence overlay becomes eligible.",
    )

    parser.add_argument(
        "--boundary-silence-max-distance",
        type=int,
        default=BucketNowcastConfig.LateBoundarySilenceConfig.max_distance_to_next_bin,
        help="Maximum tweets from the next bin edge for the late-boundary silence overlay.",
    )

    parser.add_argument(
        "--boundary-silence-threshold-start",
        type=int,
        default=BucketNowcastConfig.LateBoundarySilenceConfig.silence_threshold_start_minutes,
        help="Adaptive silence threshold at the overlay start window in minutes.",
    )

    parser.add_argument(
        "--boundary-silence-threshold-floor",
        type=int,
        default=BucketNowcastConfig.LateBoundarySilenceConfig.silence_threshold_floor_minutes,
        help="Minimum adaptive silence threshold in minutes once the overlay is active.",
    )

    parser.add_argument(
        "--boundary-silence-threshold-step",
        type=float,
        default=BucketNowcastConfig.LateBoundarySilenceConfig.silence_threshold_step_per_hour,
        help="Minutes to reduce the silence threshold by per hour inside the overlay window.",
    )

    parser.add_argument(
        "--boundary-silence-min-effective-n",
        type=float,
        default=BucketNowcastConfig.LateBoundarySilenceConfig.min_effective_n,
        help="Minimum effective analog sample size required for the late-boundary silence overlay.",
    )

    parser.add_argument(
        "--late-boundary-take-profit",
        action="store_true",
        help="Enable late-boundary majority YES take-profit behavior near settlement.",
    )

    parser.add_argument(
        "--late-boundary-trigger-price",
        type=float,
        default=LateBoundaryTakeProfitConfig.trigger_price,
        help=f"Minimum majority-exit YES VWAP to trigger late-boundary take profit. Default: {LateBoundaryTakeProfitConfig.trigger_price}.",
    )

    parser.add_argument(
        "--late-boundary-min-sell-fraction",
        type=float,
        default=LateBoundaryTakeProfitConfig.min_sell_fraction,
        help=f"Minimum fraction of YES shares to sell when late-boundary take profit triggers. Default: {LateBoundaryTakeProfitConfig.min_sell_fraction}.",
    )

    parser.add_argument(
        "--late-boundary-max-sell-fraction",
        type=float,
        default=LateBoundaryTakeProfitConfig.max_sell_fraction,
        help=f"Maximum fraction of YES shares to sell under full late-boundary take-profit strength. Default: {LateBoundaryTakeProfitConfig.max_sell_fraction}.",
    )

    parser.add_argument(
        "--use-unbox-rotations",
        action="store_true",
        help="Enable same-bin unbox rotations for boxed inventory.",
    )

    parser.add_argument(
        "--unbox-start-hours-to-settlement",
        type=float,
        default=KellyConfig.unbox_start_hours_to_settlement,
        help=f"Hours-to-settlement window where unbox rotations become eligible. Default: {KellyConfig.unbox_start_hours_to_settlement}.",
    )

    parser.add_argument(
        "--unbox-min-blocked-ticks",
        type=int,
        default=KellyConfig.unbox_min_blocked_ticks,
        help=f"Minimum consecutive blocked ticks before an unbox can trigger. Default: {KellyConfig.unbox_min_blocked_ticks}.",
    )

    parser.add_argument(
        "--unbox-min-net-utility",
        type=float,
        default=KellyConfig.unbox_min_net_utility,
        help=f"Minimum net package utility required for an unbox. Default: {KellyConfig.unbox_min_net_utility}.",
    )

    parser.add_argument(
        "--unbox-late-relax-start-hours",
        type=float,
        default=KellyConfig.unbox_late_relax_start_hours_to_settlement,
        help=f"Hours-to-settlement window where the unbox min net utility starts relaxing. Default: {KellyConfig.unbox_late_relax_start_hours_to_settlement}.",
    )

    parser.add_argument(
        "--unbox-late-net-utility-relax",
        type=float,
        default=KellyConfig.unbox_late_net_utility_relax,
        help=f"Reduction applied to the unbox min net utility inside the late-relax window. Default: {KellyConfig.unbox_late_net_utility_relax}.",
    )

    parser.add_argument(
        "--unbox-repeat-net-utility-step",
        type=float,
        default=KellyConfig.unbox_repeat_net_utility_step,
        help=f"Additional net utility required per prior unbox on the same bin. Default: {KellyConfig.unbox_repeat_net_utility_step}.",
    )

    parser.add_argument(
        "--unbox-repeat-net-utility-cap",
        type=float,
        default=KellyConfig.unbox_repeat_net_utility_cap,
        help=f"Maximum repeat-unbox utility uplift. Default: {KellyConfig.unbox_repeat_net_utility_cap}.",
    )

    parser.add_argument(
        "--unbox-multi-bin-start-count",
        type=int,
        default=KellyConfig.unbox_multi_bin_start_count,
        help=f"Distinct-bin count where the multi-bin unbox utility uplift starts. Default: {KellyConfig.unbox_multi_bin_start_count}.",
    )

    parser.add_argument(
        "--unbox-multi-bin-net-utility-step",
        type=float,
        default=KellyConfig.unbox_multi_bin_net_utility_step,
        help=f"Additional net utility required per prior unbox in other bins. Default: {KellyConfig.unbox_multi_bin_net_utility_step}.",
    )

    parser.add_argument(
        "--unbox-multi-bin-net-utility-cap",
        type=float,
        default=KellyConfig.unbox_multi_bin_net_utility_cap,
        help=f"Maximum multi-bin utility uplift. Default: {KellyConfig.unbox_multi_bin_net_utility_cap}.",
    )

    parser.add_argument(
        "--unbox-turnover-penalty",
        type=float,
        default=KellyConfig.unbox_turnover_penalty,
        help=f"Turnover penalty subtracted from gross package utility for unbox rotations. Default: {KellyConfig.unbox_turnover_penalty}.",
    )

    parser.add_argument(
        "--unbox-bin-cooldown-seconds",
        type=int,
        default=KellyConfig.unbox_bin_cooldown_seconds,
        help=f"Cooldown after executing an unbox on the same bin. Default: {KellyConfig.unbox_bin_cooldown_seconds}.",
    )

    parser.add_argument(
        "--consensus-time",
        action="store_true",
        help="Enable time-based trusted-quote consensus blending near settlement.",
    )

    parser.add_argument(
        "--consensus-mode",
        type=str,
        choices=["off", "time_only", "gap_only", "time_gap"],
        default="off",
        help="Consensus preset mode. 'time_only' enables the recommended time-based blend without gap-based damping.",
    )

    parser.add_argument(
        "--consensus-gap",
        action="store_true",
        help="Enable gap-based trusted-quote consensus blending on large model-market disagreement.",
    )

    parser.add_argument(
        "--consensus-time-tau",
        type=float,
        default=MarketConsensusConfig.time_tau,
        help=f"Time constant in hours for time-based consensus alpha. Default: {MarketConsensusConfig.time_tau}.",
    )

    parser.add_argument(
        "--consensus-gap-scale",
        type=float,
        default=MarketConsensusConfig.gap_scale,
        help=f"Half-L1 disagreement scale for gap-based consensus alpha. Default: {MarketConsensusConfig.gap_scale}.",
    )

    parser.add_argument(
        "--consensus-gap-gamma",
        type=float,
        default=MarketConsensusConfig.gap_gamma,
        help=f"Curvature for gap-based consensus alpha. Default: {MarketConsensusConfig.gap_gamma}.",
    )

    parser.add_argument(
        "--consensus-gap-floor",
        type=float,
        default=MarketConsensusConfig.gap_floor,
        help=f"Minimum gap-based model weight before the combined floor. Default: {MarketConsensusConfig.gap_floor}.",
    )

    parser.add_argument(
        "--consensus-min-model-weight",
        type=float,
        default=MarketConsensusConfig.min_model_weight,
        help=f"Global minimum model weight after consensus blending. Default: {MarketConsensusConfig.min_model_weight}.",
    )

    parser.add_argument(
        "--consensus-min-coverage",
        type=float,
        default=MarketConsensusConfig.min_coverage_ratio,
        help=f"Minimum live-bin trusted-quote coverage for consensus blending. Default: {MarketConsensusConfig.min_coverage_ratio}.",
    )

    parser.add_argument(
        "--consensus-max-avg-spread",
        type=float,
        default=MarketConsensusConfig.max_avg_spread,
        help=f"Maximum average YES spread across trusted bins for consensus blending. Default: {MarketConsensusConfig.max_avg_spread}.",
    )

    parser.add_argument(
        "--consensus-max-bin-spread",
        type=float,
        default=MarketConsensusConfig.max_bin_spread,
        help=f"Maximum YES spread for a bin to count as trusted by consensus. Default: {MarketConsensusConfig.max_bin_spread}.",
    )

    parser.add_argument(
        "--consensus-allow-untrusted-buys",
        action="store_true",
        help="Allow fresh BUY entries in bins without trusted quotes even when consensus mode is enabled.",
    )

    parser.add_argument(
        "--robust-kelly",
        action="store_true",
        help="Reduce effective Kelly fraction when the market strongly disagrees and quote quality is good.",
    )

    parser.add_argument(
        "--robust-kelly-min-fraction-multiplier",
        type=float,
        default=RobustKellyConfig.min_fraction_multiplier,
        help=f"Minimum multiplier on Kelly fraction under full robust-Kelly haircut. Default: {RobustKellyConfig.min_fraction_multiplier}.",
    )

    parser.add_argument(
        "--robust-kelly-min-coverage",
        type=float,
        default=RobustKellyConfig.min_coverage_ratio,
        help=f"Minimum live-bin quote coverage for robust Kelly haircuting. Default: {RobustKellyConfig.min_coverage_ratio}.",
    )

    parser.add_argument(
        "--robust-kelly-max-avg-spread",
        type=float,
        default=RobustKellyConfig.max_avg_spread,
        help=f"Maximum average YES mid spread to allow robust Kelly haircuting. Default: {RobustKellyConfig.max_avg_spread}.",
    )

    parser.add_argument(
        "--robust-kelly-disagreement-scale",
        type=float,
        default=RobustKellyConfig.disagreement_scale,
        help=f"Half-L1 model-vs-market disagreement scale for full robust-Kelly haircut. Default: {RobustKellyConfig.disagreement_scale}.",
    )

    parser.add_argument(
        "--market-buy-guard",
        action="store_true",
        help="Widen buy-entry thresholds when the market strongly disagrees and quote quality is good.",
    )

    parser.add_argument(
        "--market-buy-guard-max-widening",
        type=float,
        default=MarketBuyGuardConfig.max_threshold_widening,
        help=f"Maximum extra buy-threshold widening in probability points. Default: {MarketBuyGuardConfig.max_threshold_widening}.",
    )

    parser.add_argument(
        "--market-buy-guard-min-coverage",
        type=float,
        default=MarketBuyGuardConfig.min_coverage_ratio,
        help=f"Minimum live-bin quote coverage for buy guard. Default: {MarketBuyGuardConfig.min_coverage_ratio}.",
    )

    parser.add_argument(
        "--market-buy-guard-max-avg-spread",
        type=float,
        default=MarketBuyGuardConfig.max_avg_spread,
        help=f"Maximum average YES mid spread to allow buy guard. Default: {MarketBuyGuardConfig.max_avg_spread}.",
    )

    parser.add_argument(
        "--market-buy-guard-disagreement-scale",
        type=float,
        default=MarketBuyGuardConfig.disagreement_scale,
        help=f"Half-L1 model-vs-market disagreement scale for full buy guard. Default: {MarketBuyGuardConfig.disagreement_scale}.",
    )

    args = parser.parse_args()
    _resolve_consensus_mode(parser, args)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    replay_requested = any([
        args.resume_from_ts,
        args.seed_state,
        args.seed_from_log,
        args.seed_log_ts,
        args.seed_log_event,
    ])
    if replay_requested and not args.event:
        logger.error("Replay resume/seed options currently require a single --event")
        return
    if args.seed_state and args.seed_from_log:
        logger.error("Use either --seed-state or --seed-from-log, not both")
        return
    if args.seed_from_log and not args.seed_log_ts:
        logger.error("--seed-from-log requires --seed-log-ts")
        return
    if args.seed_log_ts and not args.seed_from_log:
        logger.error("--seed-log-ts requires --seed-from-log")
        return
    if args.seed_log_event and not args.seed_from_log:
        logger.error("--seed-log-event requires --seed-from-log")
        return
    if (args.late_tick_interval is None) != (args.late_tick_start_hours is None):
        logger.error("--late-tick-interval and --late-tick-start-hours must be set together")
        return
    if args.late_tick_interval is not None and args.late_tick_interval <= 0:
        logger.error("--late-tick-interval must be positive")
        return
    if args.late_tick_start_hours is not None and args.late_tick_start_hours <= 0:
        logger.error("--late-tick-start-hours must be positive")
        return

    resume_from_ts = None
    if args.resume_from_ts:
        try:
            resume_from_ts = parse_timestamp_arg(args.resume_from_ts)
        except ValueError:
            logger.error(f"Invalid replay resume timestamp: {args.resume_from_ts}")
            return

    seed_log_snapshot_ts = None
    if args.seed_log_ts:
        try:
            seed_log_snapshot_ts = parse_timestamp_arg(args.seed_log_ts)
        except ValueError:
            logger.error(f"Invalid seed log timestamp: {args.seed_log_ts}")
            return

    price_data_dir = Path(args.price_data)
    cache_dir = Path(args.cache_dir)

    # List events mode
    if args.list:
        list_events(price_data_dir)
        return

    # Parse date filters
    start_date = None
    end_date = None

    if args.start_date:
        try:
            start_date = date.fromisoformat(args.start_date)
        except ValueError:
            logger.error(f"Invalid start date: {args.start_date}")
            return

    if args.end_date:
        try:
            end_date = date.fromisoformat(args.end_date)
        except ValueError:
            logger.error(f"Invalid end date: {args.end_date}")
            return

    max_orders = args.max_orders

    # Load event trading rules
    event_trading_rules = None
    if args.event_rules:
        try:
            rules_path = Path(args.event_rules)
            if rules_path.exists():
                event_trading_rules = EventTradingRulesConfig.from_yaml(str(rules_path))
                logger.info(f"Loaded event trading rules from {args.event_rules}")
                for cat in event_trading_rules.categories:
                    logger.info(f"  {cat.name}: duration=[{cat.duration_min_days}, {cat.duration_max_days}) days")
            else:
                logger.warning(f"Event rules file not found: {args.event_rules}, using defaults")
                event_trading_rules = EventTradingRulesConfig.default()
        except Exception as e:
            logger.error(f"Failed to load event rules from {args.event_rules}: {e}")
            logger.info("Using default event trading rules")
            event_trading_rules = EventTradingRulesConfig.default()

    # Build KellyConfig with CLI overrides
    _default_kelly = KellyConfig()
    trading_config = KellyConfig(
        kappa=args.kappa,
        kelly_fraction=args.kelly_fraction,
        min_buy_utility=args.min_buy_utility,
        min_sell_utility=args.min_sell_utility,
        t_stop_hours=args.t_stop if args.t_stop is not None else _default_kelly.t_stop_hours,
        kelly_only_exit=(args.exit_mode == "kelly_only"),
        edge_buffer=EdgeBufferConfig(
            required_roi=args.roi,
            friction_mid=args.friction_mid,
            friction_tail=args.friction_tail,
            min_perceived_prob=args.min_perceived_prob,
            min_market_price=args.min_market_price,
            max_spread_ratio=args.max_spread_ratio,
            require_two_sided_liquidity=not args.no_require_two_sided,
        ),
        rate_limit=RateLimitConfig(
            max_orders_per_tick=max_orders,
            min_order_delay_seconds=0.0,
            max_orders_per_minute=1000,
        ),
        collateral=CollateralConfig(
            c_event_max=args.capital,
            c_bin_max_ratio=args.c_bin_max_ratio,
            capital_multiplier=args.capital_multiplier,
        ),
        market_consensus=MarketConsensusConfig(
            enabled=args.consensus_time or args.consensus_gap,
            time_enabled=args.consensus_time,
            time_tau=args.consensus_time_tau,
            gap_enabled=args.consensus_gap,
            gap_scale=args.consensus_gap_scale,
            gap_gamma=args.consensus_gap_gamma,
            gap_floor=args.consensus_gap_floor,
            min_model_weight=args.consensus_min_model_weight,
            min_coverage_ratio=args.consensus_min_coverage,
            max_avg_spread=args.consensus_max_avg_spread,
            max_bin_spread=args.consensus_max_bin_spread,
            require_trusted_quote_for_buys=not args.consensus_allow_untrusted_buys,
        ),
        robust_kelly=RobustKellyConfig(
            enabled=args.robust_kelly,
            min_fraction_multiplier=args.robust_kelly_min_fraction_multiplier,
            min_coverage_ratio=args.robust_kelly_min_coverage,
            max_avg_spread=args.robust_kelly_max_avg_spread,
            disagreement_scale=args.robust_kelly_disagreement_scale,
        ),
        market_buy_guard=MarketBuyGuardConfig(
            enabled=args.market_buy_guard,
            max_threshold_widening=args.market_buy_guard_max_widening,
            min_coverage_ratio=args.market_buy_guard_min_coverage,
            max_avg_spread=args.market_buy_guard_max_avg_spread,
            disagreement_scale=args.market_buy_guard_disagreement_scale,
        ),
        late_boundary_take_profit=LateBoundaryTakeProfitConfig(
            enabled=args.late_boundary_take_profit,
            trigger_price=args.late_boundary_trigger_price,
            min_sell_fraction=args.late_boundary_min_sell_fraction,
            max_sell_fraction=args.late_boundary_max_sell_fraction,
        ),
        use_unbox_rotations=args.use_unbox_rotations,
        unbox_start_hours_to_settlement=args.unbox_start_hours_to_settlement,
        unbox_min_blocked_ticks=args.unbox_min_blocked_ticks,
        unbox_min_net_utility=args.unbox_min_net_utility,
        unbox_late_relax_start_hours_to_settlement=args.unbox_late_relax_start_hours,
        unbox_late_net_utility_relax=args.unbox_late_net_utility_relax,
        unbox_repeat_net_utility_step=args.unbox_repeat_net_utility_step,
        unbox_repeat_net_utility_cap=args.unbox_repeat_net_utility_cap,
        unbox_multi_bin_start_count=args.unbox_multi_bin_start_count,
        unbox_multi_bin_net_utility_step=args.unbox_multi_bin_net_utility_step,
        unbox_multi_bin_net_utility_cap=args.unbox_multi_bin_net_utility_cap,
        unbox_turnover_penalty=args.unbox_turnover_penalty,
        unbox_bin_cooldown_seconds=args.unbox_bin_cooldown_seconds,
        max_iters_per_tick=50,
    )

    config = UnifiedBacktestConfig(
        initial_capital=args.capital,
        spread=args.spread,
        slippage=args.slippage,
        tail_spread=args.tail_spread,
        tail_zone=args.tail_zone,
        max_fill_usd=args.max_fill_usd,
        trading=trading_config,
        exit_hours_before_settlement=args.exit_hours,
        verbose=args.trade_verbose,
        projection_model=args.projection,
        intraday_mode=args.intraday_mode,
        interday_model=args.interday_model,
        event_trading_rules=event_trading_rules,
        tick_interval_seconds=args.tick_interval,
        late_tick_interval_seconds=args.late_tick_interval,
        late_tick_start_hours_before_settlement=args.late_tick_start_hours,
        cmp_nu_scale=args.nu_scale,
        use_historical_bootstrap=args.historical_bootstrap,
        bootstrap_start_hours=args.bootstrap_start_hours,
        bootstrap_full_hours=args.bootstrap_full_hours,
        bootstrap_max_blend=args.bootstrap_max_blend,
        boundary_silence_overlay=args.boundary_silence_overlay,
        boundary_silence_hours=args.boundary_silence_hours,
        boundary_silence_max_distance=args.boundary_silence_max_distance,
        boundary_silence_threshold_start=args.boundary_silence_threshold_start,
        boundary_silence_threshold_floor=args.boundary_silence_threshold_floor,
        boundary_silence_threshold_step=args.boundary_silence_threshold_step,
        boundary_silence_min_effective_n=args.boundary_silence_min_effective_n,
        resume_from_timestamp=resume_from_ts,
        seed_state_path=args.seed_state,
        seed_log_path=args.seed_from_log,
        seed_log_snapshot_timestamp=seed_log_snapshot_ts,
        seed_log_event_name=args.seed_log_event,
    )

    runner = UnifiedBacktestRunner(
        config=config,
        price_data_dir=price_data_dir,
        cache_dir=cache_dir,
    )
    logger.info(f"Projection model: {args.projection}")
    logger.info(f"Intraday mode: {args.intraday_mode}")
    logger.info(f"Interday model: {args.interday_model}")
    if args.late_tick_interval is not None:
        logger.info(
            "Mixed tick schedule: %ss default, %ss inside final %.1fh",
            args.tick_interval,
            args.late_tick_interval,
            args.late_tick_start_hours,
        )
    if args.exit_mode != "kelly_only":
        logger.info(f"Exit mode: {args.exit_mode}")
    if args.nu_scale != 1.0:
        logger.info(f"CMP nu_scale: {args.nu_scale}")
    if args.historical_bootstrap:
        logger.info(
            "Historical intraday bootstrap: enabled (start=%.1fh, full=%.1fh, max_blend=%.2f)",
            args.bootstrap_start_hours,
            args.bootstrap_full_hours,
            args.bootstrap_max_blend,
        )
    if args.boundary_silence_overlay:
        logger.info(
            "Late boundary silence overlay: enabled (window=%.1fh, max_distance=%d, threshold=max(%d, %d - %.1f*(%.1f-h)), min_n_eff=%.1f)",
            args.boundary_silence_hours,
            args.boundary_silence_max_distance,
            args.boundary_silence_threshold_floor,
            args.boundary_silence_threshold_start,
            args.boundary_silence_threshold_step,
            args.boundary_silence_hours,
            args.boundary_silence_min_effective_n,
        )
    if args.consensus_time or args.consensus_gap:
        logger.info(
            "Market consensus: enabled (time=%s tau=%.1fh, gap=%s scale=%.2f gamma=%.2f floor=%.2f, min_model_weight=%.2f, min_coverage=%.2f, max_avg_spread=%.3f, max_bin_spread=%.3f, require_trusted_buys=%s)",
            args.consensus_time,
            args.consensus_time_tau,
            args.consensus_gap,
            args.consensus_gap_scale,
            args.consensus_gap_gamma,
            args.consensus_gap_floor,
            args.consensus_min_model_weight,
            args.consensus_min_coverage,
            args.consensus_max_avg_spread,
            args.consensus_max_bin_spread,
            not args.consensus_allow_untrusted_buys,
        )
    if args.robust_kelly:
        logger.info(
            "Robust Kelly haircut: enabled (min_fraction_mult=%.2f, min_coverage=%.2f, max_avg_spread=%.3f, disagreement_scale=%.2f)",
            args.robust_kelly_min_fraction_multiplier,
            args.robust_kelly_min_coverage,
            args.robust_kelly_max_avg_spread,
            args.robust_kelly_disagreement_scale,
        )
    if args.market_buy_guard:
        logger.info(
            "Market buy guard: enabled (max_widening=%.3f, min_coverage=%.2f, max_avg_spread=%.3f, disagreement_scale=%.2f)",
            args.market_buy_guard_max_widening,
            args.market_buy_guard_min_coverage,
            args.market_buy_guard_max_avg_spread,
            args.market_buy_guard_disagreement_scale,
        )
    if args.use_unbox_rotations:
        logger.info(
            "Unbox rotations: enabled (start=%.1fh, blocked_ticks>=%d, min_net=%.3f, late_relax_start=%.1fh, late_relax=%.3f, repeat_step=%.3f cap=%.3f, multi_bin_start=%d step=%.3f cap=%.3f, turnover=%.3f, cooldown=%ds)",
            args.unbox_start_hours_to_settlement,
            args.unbox_min_blocked_ticks,
            args.unbox_min_net_utility,
            args.unbox_late_relax_start_hours,
            args.unbox_late_net_utility_relax,
            args.unbox_repeat_net_utility_step,
            args.unbox_repeat_net_utility_cap,
            args.unbox_multi_bin_start_count,
            args.unbox_multi_bin_net_utility_step,
            args.unbox_multi_bin_net_utility_cap,
            args.unbox_turnover_penalty,
            args.unbox_bin_cooldown_seconds,
        )
    if args.capital_multiplier != 1.0:
        logger.info(f"Capital multiplier: {args.capital_multiplier}x (phantom capital: ${args.capital * (args.capital_multiplier - 1.0):.2f})")
    if resume_from_ts is not None:
        logger.info(f"Replay resume timestamp: {datetime.fromtimestamp(resume_from_ts, tz=timezone.utc).isoformat()}")
    if args.seed_state:
        logger.info(f"Replay seed state file: {args.seed_state}")
    if args.seed_from_log:
        log_seed_dt = datetime.fromtimestamp(seed_log_snapshot_ts, tz=timezone.utc).isoformat()
        logger.info(f"Replay seed from log: {args.seed_from_log} @ {log_seed_dt}")
        if args.seed_log_event:
            logger.info(f"Replay seed log event: {args.seed_log_event}")

    # Determine which events to test
    if args.event:
        # Single event
        event_dirs = [args.event]
    else:
        # Get all events and filter by date
        all_events = runner.price_provider.list_available_events()

        if not all_events:
            logger.error(f"No events found in {price_data_dir}")
            logger.info("Run the price history scraper first:")
            logger.info("  python -m src.algo.musk_tweet_count.scrapers.price_history_scraper")
            return

        event_dirs = filter_events_by_date(all_events, start_date, end_date)

        if not event_dirs:
            logger.error("No events match the date filter")
            return

        # Filter by duration if specified
        if args.duration:
            event_dirs = filter_events_by_duration(event_dirs, price_data_dir, args.duration)
            if not event_dirs:
                logger.error(f"No events match the duration filter ({args.duration} days)")
                return

        # Limit number of events if specified
        if args.max_events and args.max_events > 0:
            event_dirs = event_dirs[:args.max_events]

    logger.info(f"Will test {len(event_dirs)} event(s)")

    # Run backtests
    if len(event_dirs) == 1:
        result = run_single_backtest(runner, event_dirs[0])
        if result:
            print_summary([result])
    else:
        results = run_multiple_backtests(runner, event_dirs, parallel=args.parallel)
        print_summary(results)


if __name__ == "__main__":
    main()
