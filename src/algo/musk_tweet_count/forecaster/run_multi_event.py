#!/usr/bin/env python3
"""
Run multi-event trading bot with shared capital pool.

Usage:
    python -m src.algo.musk_tweet_count.forecaster.run_multi_event \
        --capital 5000 \
        --dry-run

    # With specific events (comma-separated event IDs or "auto" for discovery)
    python -m src.algo.musk_tweet_count.forecaster.run_multi_event \
        --capital 5000 \
        --events auto
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import requests

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds

from src.utils.crypto_utils import load_private_key
from src.algo.musk_tweet_count.forecaster.config import ForecasterConfig
from src.algo.musk_tweet_count.forecaster.multi_event_manager import (
    MultiEventManager,
    MultiEventConfig,
    EventInfo,
)
from src.algo.musk_tweet_count.kelly.config import KellyConfig, EdgeBufferConfig, AdaptiveDeltaConfig
from src.algo.musk_tweet_count.kelly.capital_pool import CapitalPoolConfig

logger = logging.getLogger(__name__)

# API Constants
GAMMA_API_URL = "https://gamma-api.polymarket.com"
MUSK_TWEET_TAG_ID = 972


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with compact format."""
    level = logging.DEBUG if verbose else logging.INFO

    # Custom formatter for compact output
    # Format: "26-02-03 15:20:01[I]: message"
    class CompactFormatter(logging.Formatter):
        LEVEL_MAP = {
            'DEBUG': 'D',
            'INFO': 'I',
            'WARNING': 'W',
            'ERROR': 'E',
            'CRITICAL': 'C',
        }

        def format(self, record):
            level_char = self.LEVEL_MAP.get(record.levelname, '?')
            timestamp = self.formatTime(record, "%y-%m-%d %H:%M:%S")
            return f"{timestamp}[{level_char}]: {record.getMessage()}"

    handler = logging.StreamHandler()
    handler.setFormatter(CompactFormatter())

    logging.root.handlers = []
    logging.root.addHandler(handler)
    logging.root.setLevel(level)

    # Reduce noise from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


def create_clob_client() -> ClobClient:
    """
    Create authenticated CLOB client from environment variables.

    Loads private key from:
    - POLYMARKET_PRIVATE_KEY (plain)
    - ENCRYPTED_POLYMARKET_PRIVATE_KEY + PK_PWD (encrypted)

    API credentials from:
    - CLOB_API_KEY, CLOB_API_SECRET, CLOB_API_PASSPHRASE

    For proxy wallets (Magic Link / browser connection):
    - POLY_FUNDER: Proxy wallet address (required for signature_type 1 or 2)
    - POLY_SIGNATURE_TYPE: 0=EOA, 1=POLY_PROXY (Magic Link), 2=GNOSIS_SAFE
    """
    host = os.getenv("CLOB_HOST", "https://clob.polymarket.com")
    key = os.getenv("CLOB_API_KEY")
    secret = os.getenv("CLOB_API_SECRET")
    passphrase = os.getenv("CLOB_API_PASSPHRASE")
    chain_id = int(os.getenv("CHAIN_ID", "137"))  # Polygon mainnet

    # Proxy wallet settings
    funder = os.getenv("POLY_FUNDER")  # Proxy wallet address
    signature_type = int(os.getenv("POLY_SIGNATURE_TYPE", "0"))  # 0=EOA, 1=POLY_PROXY, 2=GNOSIS_SAFE

    if not all([key, secret, passphrase]):
        raise ValueError(
            "Missing CLOB credentials. Set CLOB_API_KEY, CLOB_API_SECRET, CLOB_API_PASSPHRASE"
        )

    # Load private key (handles encrypted keys automatically)
    try:
        private_key = load_private_key()
        logger.info("Private key loaded successfully")
    except ValueError as e:
        raise ValueError(f"Failed to load private key: {e}")

    creds = ApiCreds(
        api_key=key,
        api_secret=secret,
        api_passphrase=passphrase,
    )

    # Create client with proxy wallet support
    if funder and signature_type > 0:
        logger.info(f"Using proxy wallet: {funder} (signature_type={signature_type})")
        return ClobClient(
            host,
            key=private_key,
            chain_id=chain_id,
            creds=creds,
            signature_type=signature_type,
            funder=funder,
        )
    else:
        logger.info("Using EOA wallet (signature_type=0)")
        return ClobClient(host, key=private_key, chain_id=chain_id, creds=creds)


def get_wallet_address_for_positions() -> str:
    """
    Get wallet address for position queries.

    IMPORTANT: When using a proxy wallet (signature_type=1 or 2),
    positions are held by the PROXY wallet (POLY_FUNDER), not the main wallet.
    This is because trades are executed through the proxy.

    Returns:
        Wallet address that holds positions (proxy if configured, else main wallet)
    """
    # Check if proxy wallet is configured
    funder = os.getenv("POLY_FUNDER")
    signature_type = int(os.getenv("POLY_SIGNATURE_TYPE", "0"))

    if funder and signature_type in (1, 2):
        # Proxy wallet holds the positions
        logger.debug(f"Using proxy wallet for position queries: {funder}")
        return funder

    # No proxy - use main wallet
    wallet = os.getenv("WALLET_ADDRESS")
    if wallet:
        return wallet

    raise ValueError(
        "No wallet address configured. "
        "Set WALLET_ADDRESS or POLY_FUNDER (for proxy wallets)."
    )


def get_wallet_address(clob_client: ClobClient) -> str:
    """
    Get wallet address from CLOB client (for signing).

    NOTE: For position queries, use get_wallet_address_for_positions() instead,
    as positions may be held by a proxy wallet.
    """
    try:
        # Derive from CLOB client's signer
        address = clob_client.get_address()
        if address:
            return address
    except Exception as e:
        logger.debug(f"Could not derive address from CLOB client: {e}")

    # Fallback to env var
    wallet = os.getenv("WALLET_ADDRESS")
    if wallet:
        return wallet

    raise ValueError(
        "Could not derive wallet address from CLOB client. "
        "Ensure CLOB credentials are correct."
    )


def parse_counting_dates_from_title(title: str) -> Tuple[Optional[date], Optional[date]]:
    """
    Parse tweet counting dates from event title.

    Matches patterns like:
    - "Elon Musk # tweets January 13 - January 20, 2026?"

    Returns:
        Tuple of (start_date, end_date) as date objects, or (None, None) if not parsed
    """
    # Pattern: "Month Day - Month Day, Year"
    pattern = r"(\w+)\s+(\d{1,2})\s*[-–]\s*(\w+)\s+(\d{1,2}),?\s*(\d{4})"
    match = re.search(pattern, title)

    if not match:
        return None, None

    try:
        start_month_str = match.group(1)
        start_day = int(match.group(2))
        end_month_str = match.group(3)
        end_day = int(match.group(4))
        year = int(match.group(5))

        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }

        start_month = month_map.get(start_month_str.lower())
        end_month = month_map.get(end_month_str.lower())

        if not start_month or not end_month:
            return None, None

        start_date = date(year, start_month, start_day)
        end_date = date(year, end_month, end_day)

        # Validate 7-day period
        if (end_date - start_date).days != 7:
            logger.debug(f"Skipping non-7-day event: {title}")
            return None, None

        return start_date, end_date

    except (ValueError, AttributeError) as e:
        logger.debug(f"Failed to parse dates from '{title}': {e}")
        return None, None


def parse_bin_bounds(outcome: str) -> Tuple[int, int]:
    """
    Parse bin bounds from outcome string.

    Examples:
        "0-74" -> (0, 74)
        "75-99" -> (75, 99)
        "500+" -> (500, inf)

    Returns:
        Tuple of (lower, upper) bounds
    """
    outcome = outcome.strip()

    # Handle "X+" format (e.g., "500+")
    if outcome.endswith("+"):
        lower = int(outcome[:-1])
        return lower, float('inf')

    # Handle "X-Y" format
    if "-" in outcome:
        parts = outcome.split("-")
        lower = int(parts[0])
        upper = int(parts[1])
        return lower, upper

    # Single number
    try:
        val = int(outcome)
        return val, val
    except ValueError:
        return 0, 0


async def discover_musk_tweet_events(clob_client: ClobClient) -> List[EventInfo]:
    """
    Discover active Musk tweet count events from Polymarket.

    Uses the Gamma API to find active events with tag_id=972 (Musk tweets).

    Returns:
        List of EventInfo for active events
    """
    logger.info("Discovering Musk tweet count events from Gamma API...")

    try:
        response = requests.get(
            f"{GAMMA_API_URL}/events",
            params={
                "tag_id": MUSK_TWEET_TAG_ID,
                "active": "true",
                "closed": "false",
                "limit": 100,
            },
            timeout=30,
        )
        response.raise_for_status()
        events_data = response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch events from Gamma API: {e}")
        return []

    events: List[EventInfo] = []

    for event_data in events_data:
        event_title = event_data.get("title", "")
        event_id = event_data.get("id", "")

        # Filter for Musk tweet events only (API may return other events under same tag)
        title_lower = event_title.lower()
        if "musk" not in title_lower and "elon" not in title_lower:
            logger.debug(f"Skipping non-Musk event: {event_title}")
            continue

        # Parse counting dates from title
        market_start, settlement = parse_counting_dates_from_title(event_title)

        if not market_start or not settlement:
            logger.debug(f"Skipping event with unparseable dates: {event_title}")
            continue

        # Check if event hasn't settled yet
        now = datetime.now(timezone.utc)
        settlement_dt = datetime.combine(
            settlement,
            datetime.min.time().replace(hour=17),  # Noon ET = 17:00 UTC
            tzinfo=timezone.utc,
        )

        if now >= settlement_dt:
            logger.debug(f"Skipping settled event: {event_title}")
            continue

        # Parse markets (bins) from event
        markets = event_data.get("markets", [])
        bins = []

        for market in markets:
            outcome = market.get("groupItemTitle", "")
            if not outcome:
                continue

            lower, upper = parse_bin_bounds(outcome)

            # Get token IDs (index 0 = YES, index 1 = NO)
            clob_token_ids = market.get("clobTokenIds", [])
            if isinstance(clob_token_ids, str):
                try:
                    clob_token_ids = json.loads(clob_token_ids)
                except json.JSONDecodeError:
                    clob_token_ids = []

            yes_token_id = clob_token_ids[0] if len(clob_token_ids) > 0 else None
            no_token_id = clob_token_ids[1] if len(clob_token_ids) > 1 else None

            if yes_token_id:
                bins.append({
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "token_id": yes_token_id,
                    "no_token_id": no_token_id,
                    "outcome": outcome,
                    "condition_id": market.get("conditionId"),
                })

        # Sort bins by lower bound
        bins.sort(key=lambda b: b.get("lower_bound", 0))

        if not bins:
            logger.debug(f"Skipping event with no valid bins: {event_title}")
            continue

        # Create short name from dates
        short_name = f"{market_start.strftime('%b %d')} - {settlement.strftime('%b %d')}"

        event_info = EventInfo(
            event_id=event_id,
            title=event_title,
            short_name=short_name,
            settlement_date=settlement,
            market_start_date=market_start,
            bins=bins,
            condition_id=event_data.get("conditionId"),
        )

        events.append(event_info)
        logger.info(f"Discovered event: {short_name} ({len(bins)} bins)")

    logger.info(f"Discovered {len(events)} active Musk tweet events")
    return events


async def fetch_event_details(
    clob_client: ClobClient,
    event_id: str,
) -> EventInfo:
    """
    Fetch event details from Polymarket by event ID.

    Args:
        clob_client: CLOB client (not used, keeping for interface compatibility)
        event_id: Event ID to fetch

    Returns:
        EventInfo with full details
    """
    logger.info(f"Fetching details for event {event_id}...")

    try:
        response = requests.get(
            f"{GAMMA_API_URL}/events/{event_id}",
            timeout=30,
        )
        response.raise_for_status()
        event_data = response.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch event {event_id}: {e}")

    event_title = event_data.get("title", "")

    # Parse counting dates from title
    market_start, settlement = parse_counting_dates_from_title(event_title)

    if not market_start or not settlement:
        raise ValueError(f"Could not parse dates from event title: {event_title}")

    # Parse markets (bins) from event
    markets = event_data.get("markets", [])
    bins = []

    for market in markets:
        outcome = market.get("groupItemTitle", "")
        if not outcome:
            continue

        lower, upper = parse_bin_bounds(outcome)

        # Get token IDs (index 0 = YES, index 1 = NO)
        clob_token_ids = market.get("clobTokenIds", [])
        if isinstance(clob_token_ids, str):
            try:
                clob_token_ids = json.loads(clob_token_ids)
            except json.JSONDecodeError:
                clob_token_ids = []

        yes_token_id = clob_token_ids[0] if len(clob_token_ids) > 0 else None
        no_token_id = clob_token_ids[1] if len(clob_token_ids) > 1 else None

        if yes_token_id:
            bins.append({
                "lower_bound": lower,
                "upper_bound": upper,
                "token_id": yes_token_id,
                "no_token_id": no_token_id,
                "outcome": outcome,
                "condition_id": market.get("conditionId"),
            })

    # Sort bins by lower bound
    bins.sort(key=lambda b: b.get("lower_bound", 0))

    if not bins:
        raise ValueError(f"No valid bins found for event: {event_title}")

    # Create short name from dates
    short_name = f"{market_start.strftime('%b %d')} - {settlement.strftime('%b %d')}"

    return EventInfo(
        event_id=event_id,
        title=event_title,
        short_name=short_name,
        settlement_date=settlement,
        market_start_date=market_start,
        bins=bins,
        condition_id=event_data.get("conditionId"),
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run multi-event trading bot with shared capital pool"
    )

    # Capital configuration
    parser.add_argument(
        "--capital",
        type=float,
        default=0.0,
        help="Total capital in pool (default: 0 = auto-detect from API)",
    )
    parser.add_argument(
        "--max-per-event",
        type=float,
        default=None,
        help="Maximum capital per event (default: from KellyConfig.collateral.c_event_max)",
    )
    parser.add_argument(
        "--min-allocation",
        type=float,
        default=100.0,
        help="Minimum capital to start an event (default: 100)",
    )

    # Trading configuration
    parser.add_argument(
        "--tick-interval",
        type=int,
        default=300,
        help="Seconds between trading ticks (default: 300)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no real orders)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live mode (real orders)",
    )

    # Event selection
    parser.add_argument(
        "--events",
        type=str,
        default="auto",
        help="Event IDs to trade (comma-separated) or 'auto' for discovery",
    )
    parser.add_argument(
        "--max-duration-days",
        type=float,
        default=0.0,
        help="Trading window threshold in days (default: 0 = no limit). "
             "For events longer than this, wait until remaining time < threshold. "
             "E.g., 7 = for >7-day events, start trading when T-168h remaining",
    )

    # Kelly configuration
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.25,
        help="Fractional Kelly multiplier (default: 0.25)",
    )
    parser.add_argument(
        "--required-roi",
        type=float,
        default=0.10,
        help="Required ROI for edge buffer (default: 0.10)",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.05,
        help="Minimum model probability to trade a bin (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--min-market-price",
        type=float,
        default=0.03,
        help="Minimum market price to trade (default: 0.03 = 3%%)",
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
    parser.add_argument(
        "--min-utility",
        type=float,
        default=0.0001,
        help="Minimum utility gain threshold (default: 0.0001)",
    )

    # Chunk sizing in USD (defaults from AdaptiveDeltaConfig)
    _adaptive_defaults = AdaptiveDeltaConfig()
    parser.add_argument(
        "--base-delta-usd",
        type=float,
        default=_adaptive_defaults.base_delta_usd,
        help=f"Base chunk size in USD per order (default: ${_adaptive_defaults.base_delta_usd})",
    )
    parser.add_argument(
        "--min-delta-usd",
        type=float,
        default=_adaptive_defaults.min_delta_usd,
        help=f"Minimum chunk size in USD (default: ${_adaptive_defaults.min_delta_usd})",
    )

    # Projection model
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

    # Other
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--list-events",
        action="store_true",
        help="List all active Musk tweet events with IDs and exit",
    )

    return parser.parse_args()


async def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    # Handle --list-events: list events and exit
    if args.list_events:
        # Need a minimal CLOB client just for API discovery
        try:
            clob_client = create_clob_client()
        except Exception as e:
            logger.error(f"Failed to create CLOB client: {e}")
            sys.exit(1)

        events = await discover_musk_tweet_events(clob_client)

        if not events:
            print("\nNo active Musk tweet events found.")
            sys.exit(0)

        print(f"\n{'='*80}")
        print(f"Active Musk Tweet Events ({len(events)} found)")
        print(f"{'='*80}\n")

        for event in sorted(events, key=lambda e: e.settlement_date):
            now = datetime.now(timezone.utc)
            settlement_dt = datetime.combine(
                event.settlement_date,
                datetime.min.time().replace(hour=17),
                tzinfo=timezone.utc,
            )
            hours_remaining = (settlement_dt - now).total_seconds() / 3600

            print(f"Event ID: {event.event_id}")
            print(f"  Title:      {event.title}")
            print(f"  Short Name: {event.short_name}")
            print(f"  Dates:      {event.market_start_date} to {event.settlement_date}")
            print(f"  Hours Left: {hours_remaining:.1f}h")
            print(f"  Bins:       {len(event.bins)}")
            print()

        print(f"{'='*80}")
        print("Usage: --events <id1>,<id2>,... to trade specific events")
        print(f"{'='*80}")
        sys.exit(0)

    # Validate dry-run vs live
    if args.live and args.dry_run:
        logger.error("Cannot specify both --dry-run and --live")
        sys.exit(1)

    dry_run = not args.live
    if dry_run:
        logger.info("Running in DRY-RUN mode (no real orders)")
    else:
        logger.warning("Running in LIVE mode - real orders will be placed!")

    logger.info(f"Projection model: {args.projection}")

    # Create CLOB client
    try:
        clob_client = create_clob_client()
        logger.info("CLOB client created successfully")
    except Exception as e:
        logger.error(f"Failed to create CLOB client: {e}")
        sys.exit(1)

    # Create configurations
    # Kelly config first (source of truth for c_event_max)
    edge_buffer_config = EdgeBufferConfig(
        required_roi=args.required_roi,
        friction_mid=0.015,
        friction_tail=0.03,
        min_perceived_prob=args.min_prob,
        min_market_price=args.min_market_price,
        max_spread_ratio=args.max_spread_ratio,
        require_two_sided_liquidity=not args.no_require_two_sided,
    )

    adaptive_delta_config = AdaptiveDeltaConfig(
        base_delta_usd=args.base_delta_usd,
        min_delta_usd=args.min_delta_usd,
    )

    kelly_config = KellyConfig(
        kappa=args.kappa,
        min_utility=args.min_utility,
        edge_buffer=edge_buffer_config,
        adaptive_delta=adaptive_delta_config,
    )

    # max_per_event: CLI override or kelly_config default
    max_per_event = args.max_per_event if args.max_per_event is not None else kelly_config.collateral.c_event_max
    logger.info(f"Max capital per event: ${max_per_event:.2f}")

    capital_pool_config = CapitalPoolConfig(
        total_capital=args.capital,
        min_allocation=args.min_allocation,
    )

    forecaster_config = ForecasterConfig()

    multi_event_config = MultiEventConfig(
        capital_pool=capital_pool_config,
        max_per_event=max_per_event,
        tick_interval_seconds=args.tick_interval,
        dry_run=dry_run,
        max_event_duration_days=args.max_duration_days,
        projection_model=args.projection,
    )

    # Get wallet address for position fetching
    # IMPORTANT: Use proxy wallet if configured (positions are held there)
    try:
        wallet_address = get_wallet_address_for_positions()
        logger.info(f"Using wallet address for positions: {wallet_address}")
    except ValueError as e:
        logger.error(f"{e}")
        sys.exit(1)

    # Determine events to trade
    initial_events: List[EventInfo] = []
    discovery_callback = None

    if args.events == "auto":
        # Auto mode: use discovery callback
        async def discovery_callback() -> List[EventInfo]:
            return await discover_musk_tweet_events(clob_client)
        logger.info("Using auto event discovery mode")
    else:
        # Manual mode: fetch specific event IDs upfront
        event_ids = [e.strip() for e in args.events.split(",")]
        logger.info(f"Loading {len(event_ids)} specific events: {event_ids}")
        for event_id in event_ids:
            try:
                event_info = await fetch_event_details(clob_client, event_id)
                initial_events.append(event_info)
                logger.info(f"Loaded event: {event_info.short_name}")
            except Exception as e:
                logger.error(f"Failed to fetch event {event_id}: {e}")

        if not initial_events:
            logger.error("No events could be loaded. Exiting.")
            sys.exit(1)

    # Create manager with initial events or discovery callback
    manager = MultiEventManager(
        clob_client=clob_client,
        kelly_config=kelly_config,
        forecaster_config=forecaster_config,
        config=multi_event_config,
        wallet_address=wallet_address,
        event_discovery_callback=discovery_callback,
        initial_events=initial_events if initial_events else None,
    )

    # Reconstruct state from Polymarket API (positions, balances)
    # This recovers state after a crash/restart
    logger.info("Reconstructing state from Polymarket API...")
    await manager.reconstruct_state_from_api()

    # Run manager
    async with manager._events_lock:
        num_events = len(manager._pending_events) + len(manager._active_events)
    logger.info(f"Starting multi-event manager with {num_events} events")

    # Setup signal handlers for graceful shutdown
    import signal
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal (Ctrl+C)")
        manager.stop()
        shutdown_event.set()

    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await manager.run()
    except asyncio.CancelledError:
        logger.info("Manager cancelled")
    except Exception as e:
        logger.error(f"Manager error: {e}", exc_info=True)
    finally:
        # Remove signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)

        # Ensure manager is stopped
        manager.stop()

        # Give tasks time to clean up
        await asyncio.sleep(0.5)

        # Log final status
        try:
            status = manager.get_status()
            logger.info(f"Final status: {status}")

            performance = manager.capital_pool.get_performance_summary()
            logger.info(
                f"Performance: {performance['num_events']} events, "
                f"total P&L: ${performance['total_pnl']:.2f}, "
                f"win rate: {performance['win_rate']:.1%}"
            )
        except Exception as e:
            logger.debug(f"Error logging final status: {e}")

        logger.info("Shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Already handled by signal handler
