#!/usr/bin/env python3
"""
Run backtest on historical data (XTracker API or CSV file).

XTracker API is available from November 1, 2025 onwards.
CSV files can extend the data range for more comprehensive testing.

Usage:
    # Using XTracker API (official data)
    python -m forecaster.run_backtest
    python -m forecaster.run_backtest --quick
    python -m forecaster.run_backtest --ablation

    # Using CSV file (unofficial data)
    python -m forecaster.run_backtest --csv data/musk_posts_2024-01-01_to_2026-01-21_ET_merged.csv
    python -m forecaster.run_backtest --csv path/to/data.csv --quick
"""

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from typing import Dict, List

from .config import ForecasterConfig
from .data import ContractDayUtils, TweetEvent, XTrackerClient, load_events_from_csv
from .backtest import Backtester, BacktestConfig, AblationStudy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# XTracker data availability
XTRACKER_START_DATE = date(2025, 11, 1)


def fetch_historical_data(
    verbose: bool = True,
) -> Dict[date, List[TweetEvent]]:
    """
    Fetch all historical data from XTracker API in a single call.

    XTracker API returns all posts without pagination, so we fetch
    everything from Nov 1, 2025 to today in one request.

    Args:
        verbose: Print progress

    Returns:
        Dict mapping contract_date -> List[TweetEvent]
    """
    contract_utils = ContractDayUtils()
    client = XTrackerClient(timeout=120)  # Longer timeout for bulk fetch

    if verbose:
        logger.info(f"Fetching all historical data from XTracker API...")
        logger.info(f"Date range: {XTRACKER_START_DATE} to today")

    # Single API call to fetch all data
    events_by_date = client.fetch_events_by_contract_day(
        start_date=XTRACKER_START_DATE,
        end_date=None,  # Today
        contract_utils=contract_utils,
    )

    if verbose:
        total_events = sum(len(e) for e in events_by_date.values())
        avg_per_day = total_events / len(events_by_date) if events_by_date else 0
        logger.info(
            f"Fetched {len(events_by_date)} contract-days, "
            f"{total_events} total tweets, "
            f"{avg_per_day:.1f} avg/day"
        )

    return events_by_date


def load_csv_data(
    csv_path: str,
    verbose: bool = True,
) -> Dict[date, List[TweetEvent]]:
    """
    Load historical data from CSV file.

    CSV format:
        tweet_id,post_date,content,type
        2014182947297930000,21/1/26 22:46,Correct,original

    Args:
        csv_path: Path to CSV file
        verbose: Print progress

    Returns:
        Dict mapping contract_date -> List[TweetEvent]
    """
    if verbose:
        logger.info(f"Loading data from CSV: {csv_path}")

    contract_utils = ContractDayUtils()
    events_by_date = load_events_from_csv(
        csv_path,
        contract_utils=contract_utils,
        # filter_originals=True,  # Exclude retweets
    )

    if verbose:
        total_events = sum(len(e) for e in events_by_date.values())
        avg_per_day = total_events / len(events_by_date) if events_by_date else 0
        logger.info(
            f"Loaded {len(events_by_date)} contract-days, "
            f"{total_events} total tweets, "
            f"{avg_per_day:.1f} avg/day"
        )

    return events_by_date


def print_data_summary(events_by_date: Dict[date, List[TweetEvent]]) -> None:
    """Print summary statistics of the data."""
    if not events_by_date:
        print("No data available!")
        return

    sorted_dates = sorted(events_by_date.keys())
    counts = [len(events_by_date[d]) for d in sorted_dates]

    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Date range: {sorted_dates[0]} to {sorted_dates[-1]}")
    print(f"Total days: {len(sorted_dates)}")
    print(f"Total tweets: {sum(counts)}")
    print(f"Daily stats: min={min(counts)}, max={max(counts)}, mean={sum(counts)/len(counts):.1f}")
    print()

    # Show last 14 days
    print("Recent daily counts:")
    for d in sorted_dates[-14:]:
        count = len(events_by_date[d])
        bar = "█" * (count // 10)
        print(f"  {d}: {count:3d} {bar}")
    print("=" * 60 + "\n")


def run_full_backtest(
    events_by_date: Dict[date, List[TweetEvent]],
    n_simulations: int = 5000,
    use_ensemble: bool = False,
    use_gas: bool = False,
) -> None:
    """Run full backtest with multiple τ values."""
    logger.info(f"Running full backtest (ensemble={use_ensemble}, gas={use_gas})...")

    # Configure for available data
    # With ~85 days, use 45-day training to maximize test window
    config = BacktestConfig(
        training_window_days=45,
        min_training_days=30,
        tau_values=[360, 720, 1080],  # 6h, 12h, 18h into contract day
        n_simulations=n_simulations,
        include_naive_baseline=True,
        include_interday_baseline=True,
        verbose=True,
    )

    backtester = Backtester(backtest_config=config, use_ensemble=use_ensemble, use_gas=use_gas)
    results = backtester.run(events_by_date)

    # Print results
    print(results.summary())

    # Additional analysis
    print("\nDETAILED ANALYSIS")
    print("-" * 60)

    # Show improvement by time of day
    print("\nForecast accuracy by time of day:")
    for tau in sorted(results.metrics_by_tau.keys()):
        metrics = results.metrics_by_tau[tau]
        hours = tau / 60
        time_label = f"{12 + int(hours)}:00" if hours < 12 else f"{int(hours) - 12}:00"
        print(f"  τ={tau:4d} (~{time_label} ET): MAE={metrics['mae_7day']:.1f}, Coverage90={metrics['coverage_90']:.1%}")

    # Show worst predictions
    print("\nTop 5 largest errors:")
    sorted_forecasts = sorted(results.forecasts, key=lambda f: f.mae_7day, reverse=True)
    for f in sorted_forecasts[:5]:
        print(
            f"  {f.forecast_date} τ={f.tau}: "
            f"predicted={f.forecast.mean:.0f}, actual={f.actual_7day}, "
            f"error={f.mae_7day:.0f}"
        )

    # Calibration analysis
    print("\nCalibration analysis:")
    print(f"  50% interval coverage: {results.overall_metrics['coverage_50']:.1%} (target: 50%)")
    print(f"  90% interval coverage: {results.overall_metrics['coverage_90']:.1%} (target: 90%)")

    # Side-by-side predictions vs actuals table
    print("\n" + "=" * 100)
    print("PREDICTIONS VS ACTUALS (τ=720 only for clarity)")
    print("=" * 100)
    print(f"{'Date':<12} {'Predicted':>10} {'Actual':>10} {'Error':>10} {'P5':>8} {'P95':>8} {'In 90%':>8} {'Today_Est':>10} {'Today_Act':>10}")
    print("-" * 100)

    # Filter to τ=720 for cleaner output
    tau_720_forecasts = sorted(
        [f for f in results.forecasts if f.tau == 720],
        key=lambda f: f.forecast_date
    )

    for f in tau_720_forecasts:
        in_range = "Yes" if f.in_90_interval else "NO"
        print(
            f"{f.forecast_date!s:<12} "
            f"{f.forecast.mean:>10.0f} "
            f"{f.actual_7day:>10} "
            f"{f.actual_7day - f.forecast.mean:>+10.0f} "
            f"{f.forecast.p5:>8.0f} "
            f"{f.forecast.p95:>8.0f} "
            f"{in_range:>8} "
            f"{f.forecast.today_estimate:>10.1f} "
            f"{f.actual_today:>10}"
        )

    print("=" * 100)

    # Show daily counts to understand regime
    print_daily_counts(events_by_date)

    return results


def print_daily_counts(events_by_date: Dict[date, List[TweetEvent]]) -> None:
    """Print daily counts and 7-day rolling sums."""
    print("\n" + "=" * 80)
    print("DAILY TWEET COUNTS (to understand regime shifts)")
    print("=" * 80)

    sorted_dates = sorted(events_by_date.keys())

    # Calculate rolling 7-day sums
    print(f"{'Date':<12} {'Daily':>8} {'7-Day Sum':>12} {'Avg/Day':>10}")
    print("-" * 80)

    for i, d in enumerate(sorted_dates):
        daily_count = len(events_by_date[d])

        # Calculate 7-day sum if we have enough data
        if i >= 6:
            seven_days = sorted_dates[i-6:i+1]
            seven_day_sum = sum(len(events_by_date[day]) for day in seven_days)
            avg_per_day = seven_day_sum / 7
            print(f"{d!s:<12} {daily_count:>8} {seven_day_sum:>12} {avg_per_day:>10.1f}")
        else:
            print(f"{d!s:<12} {daily_count:>8} {'N/A':>12} {'N/A':>10}")

    print("=" * 80)


def run_quick_backtest(
    events_by_date: Dict[date, List[TweetEvent]],
    n_simulations: int = 2000,
    use_ensemble: bool = False,
    use_gas: bool = False,
) -> None:
    """Run quick backtest with single τ value."""
    logger.info(f"Running quick backtest (ensemble={use_ensemble}, gas={use_gas})...")

    config = BacktestConfig(
        training_window_days=45,
        min_training_days=30,
        tau_values=[720],  # Only at noon (12h into contract day)
        n_simulations=n_simulations,
        include_naive_baseline=True,
        include_interday_baseline=False,
        verbose=True,
    )

    backtester = Backtester(backtest_config=config, use_ensemble=use_ensemble, use_gas=use_gas)
    results = backtester.run(events_by_date)

    print(results.summary())
    return results


def run_ablation_study(
    events_by_date: Dict[date, List[TweetEvent]],
    n_simulations: int = 2000,
) -> None:
    """Run ablation study comparing different configurations."""
    logger.info("Running ablation study...")

    backtest_config = BacktestConfig(
        training_window_days=45,
        min_training_days=30,
        tau_values=[720],  # Single τ for faster ablation
        n_simulations=n_simulations,
        include_naive_baseline=False,
        verbose=False,  # Less verbose for ablation
    )

    study = AblationStudy(backtest_config=backtest_config)
    results = study.run(events_by_date)

    print(study.summary(results))

    # Additional insights
    print("\nKEY FINDINGS:")
    print("-" * 60)

    full_mae = results["full_model"].overall_metrics["mae_7day"]
    for name, result in results.items():
        if name == "full_model":
            continue
        mae = result.overall_metrics["mae_7day"]
        diff = mae - full_mae
        pct = diff / full_mae * 100
        impact = "worse" if diff > 0 else "better"
        print(f"  {name}: {mae:.1f} MAE ({abs(pct):.1f}% {impact} than full model)")

    return results


def run_dispersion_grid_search(
    events_by_date: Dict[date, List[TweetEvent]],
    n_simulations: int = 2000,
    inflation_values: List[float] = None,
    use_gas: bool = False,
) -> None:
    """
    Grid search for optimal dispersion inflation factor.

    Tests different values of dispersion_inflation_factor to find
    the value that achieves best calibration (coverage close to nominal).

    Args:
        events_by_date: Historical events
        n_simulations: MC simulations per forecast
        inflation_values: List of s values to test
    """
    from .config import ForecasterConfig

    if inflation_values is None:
        inflation_values = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

    logger.info(f"Running dispersion grid search with s in {inflation_values}")

    results_table = []

    for s in inflation_values:
        logger.info(f"\nTesting dispersion_inflation_factor = {s}")

        # Create config with this inflation factor
        forecaster_config = ForecasterConfig()
        forecaster_config.monte_carlo.dispersion_inflation_factor = s
        forecaster_config.monte_carlo.today_std_inflation_factor = s

        backtest_config = BacktestConfig(
            training_window_days=45,
            min_training_days=30,
            tau_values=[720],  # Single τ for speed
            horizons=[7],  # Focus on 7-day horizon
            n_simulations=n_simulations,
            include_naive_baseline=False,
            include_interday_baseline=False,
            verbose=False,
        )

        backtester = Backtester(
            forecaster_config=forecaster_config,
            backtest_config=backtest_config,
            use_gas=use_gas,
        )

        try:
            result = backtester.run(events_by_date)
            metrics = result.overall_metrics

            results_table.append({
                "s": s,
                "mae_7day": metrics.get("mae_7day", float("inf")),
                "rmse_7day": metrics.get("rmse_7day", float("inf")),
                "coverage_50": metrics.get("coverage_50", 0),
                "coverage_90": metrics.get("coverage_90", 0),
                "n_forecasts": metrics.get("n_forecasts", 0),
            })

            print(f"  s={s:.1f}: MAE={metrics.get('mae_7day', 0):.1f}, "
                  f"Coverage50={metrics.get('coverage_50', 0):.1%}, "
                  f"Coverage90={metrics.get('coverage_90', 0):.1%}")

        except Exception as e:
            logger.error(f"  Failed for s={s}: {e}")
            continue

    # Print summary table
    print("\n" + "=" * 80)
    print("DISPERSION INFLATION GRID SEARCH RESULTS")
    print("=" * 80)
    print(f"{'s':>6} | {'MAE':>8} | {'RMSE':>8} | {'Cov50':>8} | {'Cov90':>8} | {'Status':<20}")
    print("-" * 80)

    for r in results_table:
        # Check calibration status
        cov50 = r["coverage_50"]
        cov90 = r["coverage_90"]

        status = []
        if 0.45 <= cov50 <= 0.55:
            status.append("Cov50 OK")
        elif cov50 < 0.45:
            status.append("Cov50 LOW")
        else:
            status.append("Cov50 HIGH")

        if 0.86 <= cov90 <= 0.94:
            status.append("Cov90 OK")
        elif cov90 < 0.86:
            status.append("Cov90 LOW")
        else:
            status.append("Cov90 HIGH")

        print(f"{r['s']:>6.1f} | {r['mae_7day']:>8.1f} | {r['rmse_7day']:>8.1f} | "
              f"{cov50:>7.1%} | {cov90:>7.1%} | {', '.join(status):<20}")

    print("=" * 80)

    # Find best s
    best = None
    for r in results_table:
        if 0.86 <= r["coverage_90"] <= 0.94:
            if best is None or r["s"] < best["s"]:
                best = r

    if best:
        print(f"\nRECOMMENDED: dispersion_inflation_factor = {best['s']:.1f}")
        print(f"  Coverage90 = {best['coverage_90']:.1%} (target: 86-94%)")
        print(f"  Coverage50 = {best['coverage_50']:.1%} (target: 45-55%)")
        print(f"  MAE = {best['mae_7day']:.1f}")
    else:
        print("\nNo value achieved target coverage. Try larger s values.")

    return results_table


def main():
    parser = argparse.ArgumentParser(
        description="Run backtest on historical data (XTracker API or CSV)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick backtest (single τ, fewer simulations)",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run dispersion inflation grid search to find optimal s",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=5000,
        help="Monte Carlo simulations per forecast (default: 5000)",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only fetch and summarize data, don't run backtest",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file with historical data (format: tweet_id,post_date,content,type)",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Use ensemble forecasting (combines fast and slow EWMA models)",
    )
    parser.add_argument(
        "--gas",
        action="store_true",
        help="Use NB-GAS regime model instead of EWMA (MLE-optimized reactivity)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("MUSK TWEET COUNT FORECASTER - BACKTEST")
    print("=" * 60)

    # Load data from CSV or XTracker API
    if args.csv:
        print(f"Data source: CSV file ({args.csv})")
        print("Loading data from CSV...")
        print("=" * 60 + "\n")
        events_by_date = load_csv_data(args.csv)
    else:
        print(f"Data source: XTracker API")
        print(f"XTracker data available from: {XTRACKER_START_DATE}")
        print("Fetching all available historical data...")
        print("=" * 60 + "\n")
        events_by_date = fetch_historical_data()

    if not events_by_date:
        logger.error("No data fetched! Check API connectivity.")
        sys.exit(1)

    # Print data summary
    print_data_summary(events_by_date)

    if args.data_only:
        return

    # Run appropriate backtest
    if args.grid_search:
        run_dispersion_grid_search(events_by_date, n_simulations=args.n_simulations, use_gas=args.gas)
    elif args.ablation:
        run_ablation_study(events_by_date, n_simulations=args.n_simulations)
    elif args.quick:
        run_quick_backtest(events_by_date, n_simulations=args.n_simulations, use_ensemble=args.ensemble, use_gas=args.gas)
    else:
        run_full_backtest(events_by_date, n_simulations=args.n_simulations, use_ensemble=args.ensemble, use_gas=args.gas)


if __name__ == "__main__":
    main()
