#!/usr/bin/env python3
"""
Grid search for optimal hyperparameters on unofficial historical data.

Usage:
    python -m src.algo.musk_tweet_count.forecaster.grid_search_hyperparams
"""

import argparse
import logging
import multiprocessing as mp
from datetime import date
from typing import Dict, List, Tuple
import copy

from .config import ForecasterConfig
from .data import load_events_from_csv, TweetEvent
from .backtest import Backtester, BacktestConfig

logging.basicConfig(
    level=logging.WARNING,  # Reduce noise
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_single_config(args: Tuple) -> Dict:
    """Run backtest for a single configuration."""
    events_by_date, training_window, half_life, n_simulations = args

    try:
        # Create configs
        forecaster_config = ForecasterConfig()

        # Set half_life for all components that use it
        forecaster_config.nowcast.weight_half_life_days = half_life
        forecaster_config.dispersion.half_life_days = half_life
        forecaster_config.weekend.half_life_days = half_life
        forecaster_config.intraday_curve.half_life_days = half_life

        backtest_config = BacktestConfig(
            training_window_days=training_window,
            min_training_days=min(30, training_window - 10),
            tau_values=[720],  # Only noon
            n_simulations=n_simulations,
            include_naive_baseline=True,
            include_interday_baseline=True,
            verbose=False,
        )

        backtester = Backtester(
            forecaster_config=forecaster_config,
            backtest_config=backtest_config,
            use_gas=True,
        )

        results = backtester.run(events_by_date)

        metrics = results.overall_metrics

        return {
            "training_window": training_window,
            "half_life": half_life,
            "mae_7day": metrics.get("mae_7day", float("inf")),
            "rmse_7day": metrics.get("rmse_7day", float("inf")),
            "coverage_90": metrics.get("coverage_90", 0),
            "coverage_50": metrics.get("coverage_50", 0),
            "n_forecasts": metrics.get("n_forecasts", 0),
            "status": "success",
        }
    except Exception as e:
        return {
            "training_window": training_window,
            "half_life": half_life,
            "mae_7day": float("inf"),
            "status": f"error: {str(e)}",
        }


def main():
    parser = argparse.ArgumentParser(description="Grid search hyperparameters")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/elonmusk_tweets_utc_to_est_unofficial_2024-07-1_2026-01-21.csv",
        help="Path to CSV file",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=2000,
        help="Number of Monte Carlo simulations",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("HYPERPARAMETER GRID SEARCH")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from: {args.csv}")
    events_by_date = load_events_from_csv(args.csv)

    total_events = sum(len(e) for e in events_by_date.values())
    print(f"Loaded {len(events_by_date)} days, {total_events} total tweets")
    print(f"Date range: {min(events_by_date.keys())} to {max(events_by_date.keys())}")

    # Grid parameters
    training_windows = [45, 60, 75, 90, 120, 150]
    half_lives = [7, 14]

    print(f"\nGrid search parameters:")
    print(f"  training_window_days: {training_windows}")
    print(f"  weight_half_life_days: {half_lives}")
    print(f"  Total combinations: {len(training_windows) * len(half_lives)}")
    print(f"  Workers: {args.n_workers}")
    print(f"  Simulations: {args.n_simulations}")

    # Prepare tasks
    tasks = []
    for tw in training_windows:
        for hl in half_lives:
            tasks.append((events_by_date, tw, hl, args.n_simulations))

    print(f"\nRunning {len(tasks)} configurations...")
    print("-" * 70)

    # Run in parallel
    with mp.Pool(processes=args.n_workers) as pool:
        results = pool.map(run_single_config, tasks)

    # Sort by MAE
    results = sorted(results, key=lambda x: x.get("mae_7day", float("inf")))

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS (sorted by MAE)")
    print("=" * 70)
    print(f"{'Window':<8} {'HalfLife':<10} {'MAE':<10} {'RMSE':<10} {'Cov90':<8} {'Cov50':<8} {'N':<6}")
    print("-" * 70)

    for r in results:
        if r.get("status") == "success":
            print(
                f"{r['training_window']:<8} "
                f"{r['half_life']:<10} "
                f"{r['mae_7day']:<10.2f} "
                f"{r['rmse_7day']:<10.2f} "
                f"{r['coverage_90']*100:<8.1f} "
                f"{r['coverage_50']*100:<8.1f} "
                f"{int(r['n_forecasts']):<6}"
            )
        else:
            print(f"{r['training_window']:<8} {r['half_life']:<10} {r['status']}")

    print("-" * 70)

    # Best config
    best = results[0]
    print(f"\nBEST CONFIGURATION:")
    print(f"  training_window_days = {best['training_window']}")
    print(f"  weight_half_life_days = {best['half_life']}")
    print(f"  MAE 7-day = {best['mae_7day']:.2f}")
    print(f"  Coverage 90% = {best['coverage_90']*100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
