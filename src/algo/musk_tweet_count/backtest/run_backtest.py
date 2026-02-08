"""
Command-line interface for running backtests.

Usage:
    # List available events
    python -m src.algo.musk_tweet_count.backtest.run_backtest --list

    # Run backtest on specific event
    python -m src.algo.musk_tweet_count.backtest.run_backtest \
        --event "2025-11-25_Nov_18_-_Nov_25"

    # Run backtest on all events in date range
    python -m src.algo.musk_tweet_count.backtest.run_backtest \
        --start-date 2025-11-01 --end-date 2025-12-31

    # Run with unified Kelly trading logic (same as production)
    python -m src.algo.musk_tweet_count.backtest.run_backtest \
        --event "2025-11-25_Nov_18_-_Nov_25" --unified

    # Custom parameters
    python -m src.algo.musk_tweet_count.backtest.run_backtest \
        --event "2025-11-25_Nov_18_-_Nov_25" \
        --capital 1000 --spread 0.02 --roi 0.10
"""

import argparse
import logging
from datetime import date
from pathlib import Path
from typing import List, Optional

from .runner import BacktestRunner, BacktestConfig, BacktestResult
from ..kelly.config import KellyConfig, EdgeBufferConfig, AdaptiveDeltaConfig, RateLimitConfig, CollateralConfig, EventTradingRulesConfig

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
            if event_duration == duration_days:
                filtered.append(event_dir)

    return filtered


def run_single_backtest(
    runner: BacktestRunner,
    event_dir: str,
) -> Optional[BacktestResult]:
    """Run backtest on a single event."""
    logger.info(f"Running backtest: {event_dir}")
    return runner.run(event_dir)


def run_multiple_backtests(
    runner: BacktestRunner,
    event_dirs: List[str],
) -> List[BacktestResult]:
    """Run backtest on multiple events."""
    # Pre-load all data for the full date range to avoid cache issues
    if event_dirs:
        from datetime import timedelta

        # Find earliest and latest dates needed
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

    results = []

    for i, event_dir in enumerate(event_dirs, 1):
        logger.info(f"\n[{i}/{len(event_dirs)}] Running backtest: {event_dir}")

        result = runner.run(event_dir)
        if result:
            results.append(result)

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

    # Trading parameters
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital. Default: 10000",
    )

    parser.add_argument(
        "--max-position",
        type=float,
        default=100.0,
        help="Max position per bin. Default: 100",
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
        default=0.10,
        help="Required ROI for trades. Default: 0.10 (10%%)",
    )

    parser.add_argument(
        "--exit-hours",
        type=float,
        default=0.0,
        help="Close all positions X hours before settlement. Default: 0 (disabled)",
    )

    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.0,
        help="Exit position if value drops below this fraction of entry cost. 0 = disabled (default). WARNING: Stop-loss typically hurts returns in prediction markets.",
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

    # Logging
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    # Unified mode (uses same trading logic as production)
    parser.add_argument(
        "--unified",
        action="store_true",
        help="Use unified Kelly trading logic (same as production). "
             "This ensures backtest uses identical logic to live trading.",
    )

    # Kelly parameters for unified mode
    parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=1.0,
        help="Fractional Kelly parameter α ∈ (0, 1]. "
             "1.0 = full Kelly (log utility). "
             "0.5 = half Kelly. 0.25 = quarter Kelly. Default: 1.0",
    )

    parser.add_argument(
        "--min-utility",
        type=float,
        default=0.0001,
        help="Minimum utility gain threshold. Default: 0.0001",
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
        default=0.15,
        help="Max collateral per bin as ratio of c_event_max. Default: 0.15 (15%%). "
             "With $500 event max, this means $75 max per bin.",
    )

    parser.add_argument(
        "--trade-verbose",
        action="store_true",
        help="Print detailed information about each trade (unified mode only). "
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
        default=2.0,
        help="Maximum spread ratio (ask-bid)/bid to trade. Default: 2.0. Set to 0 to disable.",
    )

    parser.add_argument(
        "--no-require-two-sided",
        action="store_true",
        help="Disable requirement for two-sided liquidity (both bid and ask). Default: require two-sided.",
    )

    # Default from AdaptiveDeltaConfig
    _adaptive_defaults = AdaptiveDeltaConfig()
    parser.add_argument(
        "--base-delta-usd",
        type=float,
        default=_adaptive_defaults.base_delta_usd,
        help=f"Base trade size in USD. Default: ${_adaptive_defaults.base_delta_usd}. "
             "For faster backtests with fewer trades, use $50-100.",
    )

    parser.add_argument(
        "--max-orders",
        type=int,
        default=50,
        help="Maximum orders per tick. Default: 50. Use 1-5 for faster backtests.",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick backtest mode: base_delta_usd=$50 (unless overridden), max_orders=5. "
             "Use this for faster iteration. Combine with --base-delta-usd to customize.",
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
        default=None,
        help="Path to YAML file with event trading rules configuration. "
             "Controls when trading is allowed based on event duration. "
             "Example: config/event_trading_rules.yaml",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

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

    # Apply --quick mode overrides
    # Quick mode sets defaults for faster backtests, but user can override
    base_delta_usd = args.base_delta_usd
    max_orders = args.max_orders
    if args.quick:
        # Only override base_delta_usd if user didn't explicitly set it
        if args.base_delta_usd == _adaptive_defaults.base_delta_usd:
            base_delta_usd = 12.5  # Quick mode default
        max_orders = 5  # Fewer orders per tick
        logger.info(f"Quick mode: base_delta_usd=${base_delta_usd}, max_orders={max_orders}")

    # Load event trading rules if specified
    event_trading_rules = None
    if args.event_rules:
        event_trading_rules = EventTradingRulesConfig.from_yaml(args.event_rules)
        logger.info(f"Loaded event trading rules from: {args.event_rules}")
        for cat in event_trading_rules.categories:
            logger.info(f"  {cat.name}: duration=[{cat.duration_min_days}, {cat.duration_max_days}) days")

    # Choose between unified and legacy runners
    if args.unified:
        # Use unified runner with production Kelly logic
        from .unified_runner import UnifiedBacktestRunner, UnifiedBacktestConfig

        # Build KellyConfig with CLI overrides
        _default_kelly = KellyConfig()
        trading_config = KellyConfig(
            kelly_fraction=args.kelly_fraction,
            min_utility=args.min_utility,
            t_stop_hours=args.t_stop if args.t_stop is not None else _default_kelly.t_stop_hours,
            kelly_only_exit=(args.exit_mode == "kelly_only"),
            edge_buffer=EdgeBufferConfig(
                required_roi=args.roi,
                friction_mid=0.015,
                friction_tail=0.03,
                tail_threshold=0.09,
                min_perceived_prob=args.min_perceived_prob,
                min_market_price=args.min_market_price,
                max_spread_ratio=args.max_spread_ratio,
                require_two_sided_liquidity=not args.no_require_two_sided,
            ),
            adaptive_delta=AdaptiveDeltaConfig(
                base_delta_usd=base_delta_usd,
                max_depth_fraction=0.10,
                min_delta_usd=1.0,
            ),
            rate_limit=RateLimitConfig(
                max_orders_per_tick=max_orders,
                min_order_delay_seconds=0.0,
                max_orders_per_minute=1000,
            ),
            collateral=CollateralConfig(
                c_event_max=args.capital,
                c_bin_max_ratio=args.c_bin_max_ratio,
            ),
            max_iters_per_tick=50,
        )

        unified_config = UnifiedBacktestConfig(
            initial_capital=args.capital,
            spread=args.spread,
            slippage=args.slippage,
            trading=trading_config,
            exit_hours_before_settlement=args.exit_hours,
            verbose=args.trade_verbose,
            projection_model=args.projection,
            intraday_mode=args.intraday_mode,
            event_trading_rules=event_trading_rules,
        )

        runner = UnifiedBacktestRunner(
            config=unified_config,
            price_data_dir=price_data_dir,
            cache_dir=cache_dir,
        )
        logger.info(f"Using UNIFIED runner (same Kelly logic as production)")
        logger.info(f"Projection model: {args.projection}")
        logger.info(f"Intraday mode: {args.intraday_mode}")
        if args.exit_mode != "kelly_only":
            logger.info(f"Exit mode: {args.exit_mode}")
    else:
        # Use legacy runner
        edge_buffer = EdgeBufferConfig(
            required_roi=args.roi,
            friction_mid=0.015,  # 1.5% for >= 9% probability
            friction_tail=0.03,  # 3% for < 9% probability
            tail_threshold=0.09,  # 9% threshold
            min_perceived_prob=args.min_perceived_prob,
            min_market_price=args.min_market_price,
            max_spread_ratio=args.max_spread_ratio,
            require_two_sided_liquidity=not args.no_require_two_sided,
        )
        config = BacktestConfig(
            initial_capital=args.capital,
            max_position_per_bin=args.max_position,
            spread=args.spread,
            slippage=args.slippage,
            edge_buffer=edge_buffer,
            exit_hours_before_settlement=args.exit_hours,
            stop_loss_pct=args.stop_loss,
        )

        runner = BacktestRunner(
            config=config,
            price_data_dir=price_data_dir,
            cache_dir=cache_dir,
        )
        logger.info("Using legacy runner")

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

    logger.info(f"Will test {len(event_dirs)} event(s)")

    # Run backtests
    if len(event_dirs) == 1:
        result = run_single_backtest(runner, event_dirs[0])
        if result:
            print_summary([result])
    else:
        results = run_multiple_backtests(runner, event_dirs)
        print_summary(results)


if __name__ == "__main__":
    main()
