"""
Scraper for Polymarket Musk tweet event price history.

Fetches historical price data for all bins of Musk tweet count events
and stores them in a structured format for backtesting.

Usage:
    # Scrape all available events
    python -m src.algo.musk_tweet_count.scrapers.price_history_scraper

    # Scrape events within date range
    python -m src.algo.musk_tweet_count.scrapers.price_history_scraper \
        --start-date 2024-10-01 --end-date 2024-12-31

    # Specify granularity (fidelity) - lower = more data points
    python -m src.algo.musk_tweet_count.scrapers.price_history_scraper --fidelity 60

    # List available events without scraping
    python -m src.algo.musk_tweet_count.scrapers.price_history_scraper --list-only
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# API endpoints
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"

# Tag ID for Musk tweet count markets
MUSK_TWEET_TAG_ID = 972

# Default output directory
DEFAULT_OUTPUT_DIR = Path("data/price_history")


@dataclass
class BinInfo:
    """Information about a single bin (outcome) in an event."""
    bin_index: int
    outcome: str  # e.g., "300-324", "<150", "400+"
    lower_bound: int
    upper_bound: int  # Use large number for "+" bins
    token_id: str
    is_winner: bool
    final_price: float  # Resolution price (1.0 for winner, 0.0 for loser)


@dataclass
class EventInfo:
    """Information about a Musk tweet count event."""
    event_id: str
    title: str
    short_name: str  # e.g., "Oct 25 - Nov 1"
    start_date: str  # ISO format
    end_date: str  # ISO format
    num_bins: int
    bins: List[BinInfo]
    is_closed: bool = True  # Whether event has settled


@dataclass
class PricePoint:
    """A single price data point."""
    timestamp: int  # Unix timestamp
    price: float


@dataclass
class BinPriceHistory:
    """Price history for a single bin."""
    event_id: str
    bin_index: int
    outcome: str
    token_id: str
    is_winner: bool
    data_points: int
    first_timestamp: Optional[int]
    last_timestamp: Optional[int]
    min_price: Optional[float]
    max_price: Optional[float]
    prices: List[PricePoint]


def parse_bin_bounds(outcome: str) -> Tuple[int, int]:
    """
    Parse bin outcome string to get lower and upper bounds.

    Examples:
        "<150" -> (0, 149)
        "300-324" -> (300, 324)
        "400+" -> (400, 999999)
    """
    outcome = outcome.strip()

    # Handle "<X" format
    if outcome.startswith("<"):
        try:
            upper = int(outcome[1:].strip()) - 1
            return (0, upper)
        except ValueError:
            pass

    # Handle "X+" format
    if outcome.endswith("+"):
        try:
            lower = int(outcome[:-1].strip())
            return (lower, 999999)
        except ValueError:
            pass

    # Handle "X-Y" format
    if "-" in outcome:
        parts = outcome.split("-")
        if len(parts) == 2:
            try:
                lower = int(parts[0].strip())
                upper = int(parts[1].strip())
                return (lower, upper)
            except ValueError:
                pass

    # Handle single number (exact count)
    try:
        val = int(outcome)
        return (val, val)
    except ValueError:
        pass

    # Default fallback
    return (0, 999999)


def extract_short_name(title: str) -> str:
    """Extract short date range from event title."""
    import re

    # Match patterns like "January 27 - February 3" or "Jan 27 - Feb 3"
    # Also handles "Oct 25 - Nov 1", "December 6-13", "Dec 27-January 3"
    match = re.search(
        r"(\w+)\s+(\d+)\s*[-–]\s*(\w+)\s+(\d+)",
        title
    )
    if match:
        m1, d1, m2, d2 = match.groups()
        m1 = m1[:3]
        m2 = m2[:3]
        return f"{m1} {d1} - {m2} {d2}"

    # Try pattern without space before second month: "December 6-13"
    match = re.search(
        r"(\w+)\s+(\d+)\s*[-–]\s*(\d+)",
        title
    )
    if match:
        m1, d1, d2 = match.groups()
        m1 = m1[:3]
        return f"{m1} {d1} - {m1} {d2}"

    # Try pattern: "Oct 4-11"
    match = re.search(
        r"(\w{3,})\s+(\d+)\s*[-–]\s*(\d+)",
        title
    )
    if match:
        m1, d1, d2 = match.groups()
        m1 = m1[:3]
        return f"{m1} {d1} - {d2}"

    return title[:30]


def fetch_musk_events(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    closed_only: bool = True,
    active_only: bool = False,
) -> List[EventInfo]:
    """
    Fetch Musk tweet count events from Gamma API.

    Args:
        start_date: Filter events ending after this date
        end_date: Filter events ending before this date
        closed_only: Only fetch closed/settled events
        active_only: Only fetch active (ongoing) events

    Returns:
        List of EventInfo objects
    """
    logger.info("Fetching Musk tweet events from Gamma API...")

    events = []
    params = {
        "tag_id": MUSK_TWEET_TAG_ID,
        "limit": 300,  # Fetch more to get all events
    }
    # Note: Gamma API doesn't have an 'active' filter, we filter manually below

    try:
        response = requests.get(f"{GAMMA_API_URL}/events", params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Failed to fetch events: {e}")
        return []

    for event_data in data:
        title = event_data.get("title", "")

        # Filter for Elon Musk events only
        if "Elon Musk" not in title and "elon musk" not in title.lower():
            continue
        if "tweet" not in title.lower():
            continue

        event_end = event_data.get("endDate", "")
        if not event_end:
            continue

        # Parse end date for filtering
        try:
            end_dt = datetime.fromisoformat(event_end.replace("Z", "+00:00"))
            event_end_date = end_dt.date()
        except:
            continue

        # Apply date filters
        if start_date and event_end_date < start_date:
            continue
        if end_date and event_end_date > end_date:
            continue

        # Get start date (usually 7 days before end)
        event_start = event_data.get("startDate", "")

        # Parse markets (bins)
        markets = event_data.get("markets", [])
        if not markets:
            continue

        bins = []
        for i, market in enumerate(markets):
            outcome = market.get("groupItemTitle", f"Bin {i}")

            # Get token IDs
            token_ids = market.get("clobTokenIds", [])
            if isinstance(token_ids, str):
                try:
                    token_ids = json.loads(token_ids)
                except:
                    token_ids = []

            if not token_ids:
                continue

            token_id = token_ids[0]  # YES token

            # Get resolution status
            outcome_prices = market.get("outcomePrices", "")
            is_winner = False
            final_price = 0.0

            if outcome_prices:
                if isinstance(outcome_prices, str):
                    try:
                        outcome_prices = json.loads(outcome_prices)
                    except:
                        outcome_prices = []

                if outcome_prices:
                    try:
                        final_price = float(outcome_prices[0])
                        is_winner = final_price > 0.99
                    except:
                        pass

            # Parse bounds
            lower, upper = parse_bin_bounds(outcome)

            bins.append(BinInfo(
                bin_index=i,
                outcome=outcome,
                lower_bound=lower,
                upper_bound=upper,
                token_id=token_id,
                is_winner=is_winner,
                final_price=final_price,
            ))

        if not bins:
            continue

        # Sort bins by lower bound
        bins.sort(key=lambda b: b.lower_bound)

        # Re-index after sorting
        for i, bin_info in enumerate(bins):
            bin_info.bin_index = i

        event_id = event_data.get("id", f"event_{event_end_date}")

        # Check if event is closed
        is_closed = event_data.get("closed", False)

        # Filter by closed/active status
        if closed_only and not is_closed:
            continue
        if active_only and is_closed:
            continue

        events.append(EventInfo(
            event_id=event_id,
            title=title,
            short_name=extract_short_name(title),
            start_date=event_start[:10] if event_start else "",
            end_date=event_end[:10] if event_end else "",
            num_bins=len(bins),
            bins=bins,
            is_closed=is_closed,
        ))

    # Sort events by end date
    events.sort(key=lambda e: e.end_date)

    logger.info(f"Found {len(events)} Musk tweet events")
    return events


def fetch_price_history(
    token_id: str,
    start_ts: int,
    fidelity: int = 60,
) -> List[PricePoint]:
    """
    Fetch price history for a single token.

    Args:
        token_id: The YES token ID
        start_ts: Unix timestamp to start from
        fidelity: Granularity of data (lower = more points)

    Returns:
        List of PricePoint objects
    """
    url = f"{CLOB_API_URL}/prices-history"
    params = {
        "market": token_id,
        "startTs": start_ts,
        "fidelity": fidelity,
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.warning(f"Failed to fetch price history for {token_id[:20]}...: {e}")
        return []

    history = data.get("history", [])

    return [
        PricePoint(timestamp=point["t"], price=point["p"])
        for point in history
    ]


def scrape_event_prices(
    event: EventInfo,
    fidelity: int = 60,
    delay_between_requests: float = 0.1,
) -> List[BinPriceHistory]:
    """
    Scrape price history for all bins in an event.

    Args:
        event: The event to scrape
        fidelity: Granularity of data
        delay_between_requests: Seconds to wait between API calls

    Returns:
        List of BinPriceHistory objects
    """
    logger.info(f"Scraping prices for: {event.short_name} ({len(event.bins)} bins)")

    # Calculate start timestamp (event start or 7 days before end)
    if event.start_date:
        try:
            start_dt = datetime.fromisoformat(event.start_date)
            start_ts = int(start_dt.timestamp())
        except:
            start_ts = 0
    else:
        start_ts = 0

    results = []

    for bin_info in event.bins:
        prices = fetch_price_history(
            token_id=bin_info.token_id,
            start_ts=start_ts,
            fidelity=fidelity,
        )

        # Compute stats
        first_ts = prices[0].timestamp if prices else None
        last_ts = prices[-1].timestamp if prices else None
        min_price = min(p.price for p in prices) if prices else None
        max_price = max(p.price for p in prices) if prices else None

        results.append(BinPriceHistory(
            event_id=event.event_id,
            bin_index=bin_info.bin_index,
            outcome=bin_info.outcome,
            token_id=bin_info.token_id,
            is_winner=bin_info.is_winner,
            data_points=len(prices),
            first_timestamp=first_ts,
            last_timestamp=last_ts,
            min_price=min_price,
            max_price=max_price,
            prices=prices,
        ))

        logger.debug(f"  {bin_info.outcome}: {len(prices)} points")

        # Rate limiting
        if delay_between_requests > 0:
            time.sleep(delay_between_requests)

    total_points = sum(h.data_points for h in results)
    logger.info(f"  Total: {total_points} data points across {len(results)} bins")

    return results


def load_existing_prices(event_dir: Path) -> Dict[int, set]:
    """
    Load existing price timestamps from a previously scraped event.

    Returns:
        Dict mapping bin_index -> set of timestamps already saved
    """
    existing: Dict[int, set] = {}

    prices_csv = event_dir / "prices.csv"
    if not prices_csv.exists():
        return existing

    try:
        with open(prices_csv, "r") as f:
            header = f.readline()  # Skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    try:
                        bin_index = int(parts[0])
                        timestamp = int(parts[2])
                        if bin_index not in existing:
                            existing[bin_index] = set()
                        existing[bin_index].add(timestamp)
                    except ValueError:
                        continue
    except Exception as e:
        logger.warning(f"Failed to load existing prices: {e}")

    return existing


def save_event_data(
    event: EventInfo,
    price_histories: List[BinPriceHistory],
    output_dir: Path,
    format: str = "csv",
    merge: bool = True,
) -> None:
    """
    Save event data and price histories to files.

    Creates:
        output_dir/
            {event_date}_{short_name}/
                event_info.json
                bins.csv
                prices.csv (or prices.parquet)

    If merge=True and data already exists, new prices are appended
    (duplicates are skipped based on timestamp).
    """
    # Create event directory
    safe_name = event.short_name.replace(" ", "_").replace("/", "-")
    event_dir = output_dir / f"{event.end_date}_{safe_name}"
    event_dir.mkdir(parents=True, exist_ok=True)

    # Load existing timestamps for deduplication
    existing_timestamps: Dict[int, set] = {}
    if merge:
        existing_timestamps = load_existing_prices(event_dir)
        if existing_timestamps:
            total_existing = sum(len(ts) for ts in existing_timestamps.values())
            logger.info(f"  Found {total_existing} existing data points, will merge")

    # Save event info (always overwrite - may have updated resolution status)
    event_info_path = event_dir / "event_info.json"
    event_dict = {
        "event_id": event.event_id,
        "title": event.title,
        "short_name": event.short_name,
        "start_date": event.start_date,
        "end_date": event.end_date,
        "num_bins": event.num_bins,
        "last_updated": datetime.now().isoformat(),
        "bins": [
            {
                "bin_index": b.bin_index,
                "outcome": b.outcome,
                "lower_bound": b.lower_bound,
                "upper_bound": b.upper_bound,
                "token_id": b.token_id,
                "is_winner": b.is_winner,
                "final_price": b.final_price,
            }
            for b in event.bins
        ],
    }
    with open(event_info_path, "w") as f:
        json.dump(event_dict, f, indent=2)

    # Filter out already-existing prices and merge
    merged_histories = []
    new_points_count = 0

    for ph in price_histories:
        existing_ts = existing_timestamps.get(ph.bin_index, set())

        # Filter to only new prices
        new_prices = [p for p in ph.prices if p.timestamp not in existing_ts]
        new_points_count += len(new_prices)

        # Combine existing + new for stats calculation
        all_timestamps = existing_ts | {p.timestamp for p in ph.prices}
        all_prices = ph.prices  # New prices (we'll append to file)

        # For stats, we need all prices including existing ones
        # Since we only have timestamps for existing, we'll recalculate from the full file later
        # For now, use the new data's stats as approximation
        merged_histories.append(BinPriceHistory(
            event_id=ph.event_id,
            bin_index=ph.bin_index,
            outcome=ph.outcome,
            token_id=ph.token_id,
            is_winner=ph.is_winner,
            data_points=len(all_timestamps),
            first_timestamp=min(all_timestamps) if all_timestamps else None,
            last_timestamp=max(all_timestamps) if all_timestamps else None,
            min_price=ph.min_price,  # Approximation
            max_price=ph.max_price,  # Approximation
            prices=new_prices,  # Only new prices to append
        ))

    if merge and existing_timestamps:
        logger.info(f"  Adding {new_points_count} new data points")

    # Save bin summary (recalculate stats)
    bins_path = event_dir / "bins.csv"
    with open(bins_path, "w") as f:
        f.write("bin_index,outcome,lower_bound,upper_bound,token_id,is_winner,final_price,data_points,first_ts,last_ts,min_price,max_price\n")
        for ph in merged_histories:
            bin_info = next((b for b in event.bins if b.bin_index == ph.bin_index), None)
            if bin_info:
                f.write(f"{ph.bin_index},{ph.outcome},{bin_info.lower_bound},{bin_info.upper_bound},{ph.token_id},{ph.is_winner},{bin_info.final_price},{ph.data_points},{ph.first_timestamp or ''},{ph.last_timestamp or ''},{ph.min_price or ''},{ph.max_price or ''}\n")

    # Save price data
    if format == "parquet":
        try:
            import pandas as pd

            # For parquet, we need to read existing and merge
            parquet_path = event_dir / "prices.parquet"
            rows = []

            # Load existing if present
            if merge and parquet_path.exists():
                existing_df = pd.read_parquet(parquet_path)
                rows = existing_df.to_dict('records')

            # Add new prices
            for ph in merged_histories:
                for price in ph.prices:
                    rows.append({
                        "bin_index": ph.bin_index,
                        "outcome": ph.outcome,
                        "timestamp": price.timestamp,
                        "price": price.price,
                    })

            if rows:
                df = pd.DataFrame(rows)
                # Remove duplicates
                df = df.drop_duplicates(subset=["bin_index", "timestamp"])
                df = df.sort_values(["bin_index", "timestamp"])
                df.to_parquet(parquet_path, index=False)
        except ImportError:
            logger.warning("pandas not available, falling back to CSV")
            format = "csv"

    if format == "csv":
        prices_path = event_dir / "prices.csv"

        # If merging, append to existing file
        if merge and prices_path.exists() and new_points_count > 0:
            with open(prices_path, "a") as f:
                for ph in merged_histories:
                    for price in ph.prices:
                        dt = datetime.fromtimestamp(price.timestamp, tz=None).isoformat()
                        f.write(f"{ph.bin_index},{ph.outcome},{price.timestamp},{dt},{price.price}\n")
        else:
            # Write fresh file with all data
            with open(prices_path, "w") as f:
                f.write("bin_index,outcome,timestamp,datetime,price\n")
                for ph in price_histories:  # Use original, not merged
                    for price in ph.prices:
                        dt = datetime.fromtimestamp(price.timestamp, tz=None).isoformat()
                        f.write(f"{ph.bin_index},{ph.outcome},{price.timestamp},{dt},{price.price}\n")

    logger.info(f"  Saved to: {event_dir}")


def create_summary_index(output_dir: Path, events: List[EventInfo]) -> None:
    """Create an index file listing all scraped events."""
    index_path = output_dir / "index.json"

    index_data = {
        "scraped_at": datetime.now().isoformat(),
        "num_events": len(events),
        "events": [
            {
                "event_id": e.event_id,
                "title": e.title,
                "short_name": e.short_name,
                "start_date": e.start_date,
                "end_date": e.end_date,
                "num_bins": e.num_bins,
                "directory": f"{e.end_date}_{e.short_name.replace(' ', '_').replace('/', '-')}",
            }
            for e in events
        ],
    }

    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)

    logger.info(f"Created index at: {index_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape Polymarket Musk tweet event price history"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="Only scrape events ending after this date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="Only scrape events ending before this date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--fidelity",
        type=int,
        default=60,
        help="Price data granularity (lower = more points). Default: 60",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )

    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="Output format for price data. Default: csv",
    )

    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list available events, don't scrape",
    )

    parser.add_argument(
        "--include-active",
        action="store_true",
        help="Include active (ongoing) events, not just settled ones",
    )

    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Only scrape active (ongoing) events",
    )

    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Overwrite existing data instead of merging",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between API requests in seconds. Default: 0.1",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse dates
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

    # Fetch events
    closed_only = not args.include_active and not args.active_only
    events = fetch_musk_events(
        start_date=start_date,
        end_date=end_date,
        closed_only=closed_only,
        active_only=args.active_only,
    )

    if not events:
        logger.warning("No events found matching criteria")
        return

    # Display events
    print()
    print("=" * 90)
    print(f"Found {len(events)} Musk tweet events")
    print("=" * 90)
    print(f"{'#':<3} {'End Date':<12} {'Short Name':<20} {'Status':<10} {'Bins':<6} {'Winner':<15}")
    print("-" * 90)

    for i, event in enumerate(events, 1):
        status = "Settled" if event.is_closed else "ACTIVE"
        if event.is_closed:
            winner = next((b.outcome for b in event.bins if b.is_winner), "N/A")
        else:
            winner = "(ongoing)"
        print(f"{i:<3} {event.end_date:<12} {event.short_name:<20} {status:<10} {event.num_bins:<6} {winner:<15}")

    print("=" * 90)
    print()

    if args.list_only:
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scrape each event
    logger.info(f"Starting scrape with fidelity={args.fidelity}")

    scraped_events = []
    for i, event in enumerate(events, 1):
        logger.info(f"[{i}/{len(events)}] Processing: {event.short_name}")

        try:
            price_histories = scrape_event_prices(
                event=event,
                fidelity=args.fidelity,
                delay_between_requests=args.delay,
            )

            save_event_data(
                event=event,
                price_histories=price_histories,
                output_dir=output_dir,
                format=args.format,
                merge=not args.no_merge,
            )

            scraped_events.append(event)

        except Exception as e:
            logger.error(f"Failed to scrape {event.short_name}: {e}")
            continue

    # Create index
    create_summary_index(output_dir, scraped_events)

    logger.info(f"Scraping complete! {len(scraped_events)} events saved to {output_dir}")


if __name__ == "__main__":
    main()
