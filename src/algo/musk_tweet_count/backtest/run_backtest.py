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
from ..kelly.config import KellyConfig, EdgeBufferConfig, RateLimitConfig, CollateralConfig, EventTradingRulesConfig

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

    args = parser.parse_args()

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
        max_iters_per_tick=50,
    )

    config = UnifiedBacktestConfig(
        initial_capital=args.capital,
        spread=args.spread,
        slippage=args.slippage,
        trading=trading_config,
        exit_hours_before_settlement=args.exit_hours,
        verbose=args.trade_verbose,
        projection_model=args.projection,
        intraday_mode=args.intraday_mode,
        event_trading_rules=event_trading_rules,
        tick_interval_seconds=args.tick_interval,
        cmp_nu_scale=args.nu_scale,
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
    if args.exit_mode != "kelly_only":
        logger.info(f"Exit mode: {args.exit_mode}")
    if args.nu_scale != 1.0:
        logger.info(f"CMP nu_scale: {args.nu_scale}")
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
