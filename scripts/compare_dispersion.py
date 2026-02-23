"""
Compare model probabilities across different dispersion_inflation_factor values.

Usage:
    python3 -m scripts.compare_dispersion
"""
import sys
import logging
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.WARNING)

from src.algo.musk_tweet_count.forecaster.config import ForecasterConfig, MonteCarloConfig
from src.algo.musk_tweet_count.forecaster.forecaster import TweetCountForecaster
from src.algo.musk_tweet_count.forecaster.data import EventStore, ContractDayUtils, XTrackerClient, TweetEvent
from src.algo.musk_tweet_count.forecaster.projection import AsymmetricProjection


def run_forecast(dispersion_factor: float, event_store: EventStore, contract_utils: ContractDayUtils,
                 market_start: date, settlement: date, now: datetime, bins: list, current_count: int):
    """Run forecast with a specific dispersion factor and return bin probabilities."""
    mc_config = MonteCarloConfig(
        n_simulations=25000,
        random_seed=42,
        dispersion_inflation_factor=dispersion_factor,
        today_std_inflation_factor=dispersion_factor,
    )
    config = ForecasterConfig(
        monte_carlo=mc_config,
        intraday_mode="bucket",
    )

    forecaster = TweetCountForecaster(config, event_store=event_store)
    forecaster.fit(n_days=45, skip_fetch=True, as_of_date=market_start)

    forecast = forecaster.forecast_for_event_window(
        market_start_date=market_start,
        settlement_date=settlement,
        now=now,
    )

    projection = AsymmetricProjection()
    probabilities = projection.compute_bin_probabilities(
        forecast=forecast,
        bins=bins,
        shift=0,
        floor=current_count,
    )

    # Renormalize
    total = sum(probabilities)
    if total > 0:
        probabilities = [p / total for p in probabilities]

    return forecast, probabilities


def main():
    # Set up contract utils
    contract_utils = ContractDayUtils(timezone="America/New_York", boundary_hour=12)
    xtracker = XTrackerClient()
    event_store = EventStore(contract_utils, xtracker)

    # Fetch real data via XTracker
    print("Fetching posts from XTracker...")
    fetch_start = date(2026, 1, 1)
    fetch_end = date(2026, 2, 23)
    events = xtracker.fetch_all_events(start_date=fetch_start, end_date=fetch_end)
    for ev in events:
        event_store.add_event(ev)
    print(f"Loaded {len(events)} events into store")

    # Feb 21 - Feb 23 event
    market_start = date(2026, 2, 21)
    settlement = date(2026, 2, 23)

    # Log timestamp is UTC: 15:21:13 UTC on Feb 22
    # = 10:21:13 EST on Feb 22
    est = ZoneInfo("America/New_York")
    from datetime import timezone as tz
    now_utc = datetime(2026, 2, 22, 15, 21, 13, tzinfo=tz.utc)
    now = now_utc.astimezone(est)

    # Current count at that time
    today = contract_utils.get_contract_date(now)
    today_events = event_store.get_contract_day_events(today)
    observed_before_now = len([e for e in today_events if e.timestamp < now])

    # Count past days
    past_count = 0
    d = market_start
    while d < today:
        past_count += event_store.get_contract_day_count(d)
        d += timedelta(days=1)

    current_count = past_count + observed_before_now
    print(f"Current count at {now.strftime('%Y-%m-%d %H:%M:%S %Z')}: {current_count} ({past_count} past + {observed_before_now} today)")

    # Bins for this event
    bins = [
        (40, 64), (65, 89), (90, 114), (115, 139), (140, 164),
        (165, 189), (190, 214), (215, 239), (240, 999),
    ]
    bin_labels = [
        "40-64", "65-89", "90-114", "115-139", "140-164",
        "165-189", "190-214", "215-239", "240+",
    ]

    # Test different dispersion values
    dispersion_values = [1.0, 1.5, 1.75, 2.0, 2.5]

    results = {}
    for d in dispersion_values:
        print(f"Running with dispersion={d}...")
        forecast, probs = run_forecast(d, event_store, contract_utils,
                                        market_start, settlement, now, bins, current_count)
        results[d] = {"forecast": forecast, "probs": probs}
        print(f"  Mean={forecast.mean:.0f}, Std={forecast.std:.0f}")

    # Market prices from the production log at that timestamp
    market_yes_asks = {
        1: 0.006, 2: 0.250, 3: 0.550, 4: 0.180, 5: 0.036,
        6: 0.003, 7: 0.002, 8: 0.001, 9: 0.002,
    }
    market_no_asks = {
        1: 0.995, 2: 0.770, 3: 0.460, 4: 0.830, 5: 0.965,
        6: 0.998, 7: 0.999, 8: None, 9: 0.999,
    }

    # Print one full table per dispersion value
    for d in dispersion_values:
        forecast = results[d]["forecast"]
        probs = results[d]["probs"]
        print()
        print("=" * 80)
        print(f"  dispersion_inflation_factor = {d}")
        print(f"  Event: Feb 21 - Feb 23  |  Count: {current_count}  |  Forecast: {forecast.mean:.0f}  (std: {forecast.std:.0f})")
        print("-" * 80)
        print(f"{'Bin':<5} {'Range':<10} {'Model':>7} {'Y.Ask':>7} {'N.Ask':>7}")
        print("-" * 80)
        for i, label in enumerate(bin_labels):
            prob = probs[i] * 100
            ya = market_yes_asks.get(i + 1)
            na = market_no_asks.get(i + 1)
            ya_str = f"{ya * 100:.1f}%" if ya is not None else "  N/A"
            na_str = f"{na * 100:.1f}%" if na is not None else "  N/A"
            print(f"{i+1:<5} {label:<10} {prob:>6.1f}% {ya_str:>7} {na_str:>7}")
        print("-" * 80)

    # Side-by-side model probabilities
    print()
    print("=" * 90)
    print(f"  Side-by-side model probabilities  |  Count: {current_count}  |  Mean: 98")
    print("-" * 90)
    header = f"{'Bin':<5} {'Range':<10} {'Y.Ask':>7}"
    for d in dispersion_values:
        header += f"  {'d=' + str(d):>7}"
    print(header)
    print("-" * 90)
    for i, label in enumerate(bin_labels):
        ya = market_yes_asks.get(i + 1)
        ya_str = f"{ya * 100:.1f}%" if ya is not None else "  N/A"
        row = f"{i+1:<5} {label:<10} {ya_str:>7}"
        for d in dispersion_values:
            prob = results[d]["probs"][i] * 100
            row += f"  {prob:>6.1f}%"
        print(row)
    print("-" * 90)
    stats_row = f"{'':5} {'Std':10} {'':>7}"
    for d in dispersion_values:
        stats_row += f"  {results[d]['forecast'].std:>6.0f} "
    print(stats_row)
    print("=" * 90)


if __name__ == "__main__":
    main()
