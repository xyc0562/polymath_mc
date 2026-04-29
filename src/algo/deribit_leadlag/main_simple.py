"""
Deribit-reference -> Polymarket lead-lag trading bot.

Polls Deribit for BTC option implied probabilities, compares against
Polymarket BTC binary threshold events, and places FAK orders when
mispricing exceeds threshold after fees.

Usage:
    python -m src.algo.deribit_leadlag.main --config config/deribit_leadlag.yaml [--dry-run]
"""

import argparse
import logging
import time
from datetime import datetime, timezone

from py_clob_client_v2.client import ClobClient
from py_clob_client_v2.constants import POLYGON

from src.const import CLOB_API_URL
from src.utils.crypto_utils import load_private_key

from .config import LeadLagConfig
from .deribit_client import fetch_btc_options_summary
from .implied_probs import build_implied_prob_map
from .polymarket_discovery import discover_btc_threshold_markets, build_target_strikes
from .signal_comparator import match_markets, generate_signals
from .executor import LeadLagExecutor

logger = logging.getLogger(__name__)


def setup_clob_client(private_key: str) -> ClobClient:
    """Initialize and authenticate ClobClient."""
    client = ClobClient(
        host=CLOB_API_URL,
        key=private_key,
        chain_id=POLYGON,
    )

    try:
        api_creds = client.derive_api_key()
        client.set_api_creds(api_creds)
        logger.info("Derived existing API credentials")
    except Exception:
        try:
            api_creds = client.create_api_key()
            client.set_api_creds(api_creds)
            logger.info("Created new API credentials")
        except Exception as e:
            logger.error(f"Failed to setup API credentials: {e}")
            raise

    return client


def run_poll_cycle(
    config: LeadLagConfig,
    executor: LeadLagExecutor,
    poly_markets: list,
    recent_signals: dict,
) -> int:
    """
    Run a single poll cycle: fetch Deribit -> compute probs -> generate signals -> execute.

    Returns number of signals generated.
    """
    now = datetime.now(timezone.utc)

    # 1. Fetch Deribit options
    try:
        options = fetch_btc_options_summary(config.deribit.base_url)
    except Exception as e:
        logger.error(f"Deribit fetch failed: {e}")
        return 0

    if not options:
        logger.warning("No options fetched from Deribit")
        return 0

    # 2. Build target strikes from Polymarket markets
    target_strikes = build_target_strikes(poly_markets)
    if not target_strikes:
        logger.warning("No target strikes to evaluate")
        return 0

    # 3. Compute implied probabilities
    implied_probs = build_implied_prob_map(
        options=options,
        now=now,
        target_strikes=target_strikes,
        min_T_hours=config.signal.min_time_to_expiry_hours,
        min_call_spread_usd=config.signal.min_call_spread_usd,
    )

    if not implied_probs:
        logger.info("No implied probabilities computed")
        return 0

    # 4. Match Deribit probs with Polymarket markets
    matched = match_markets(implied_probs, poly_markets)

    # 5. Generate signals
    signals = generate_signals(
        matched_pairs=matched,
        signal_config=config.signal,
        exec_config=config.execution,
        current_exposure_usd=executor.total_exposure,
        recent_signals=recent_signals,
    )

    # 6. Execute signals
    executed = 0
    for signal in signals:
        result = executor.execute_signal(signal)
        if result is not None:
            executed += 1
            # Track for cooldown
            key = f"{signal.matched.expiry_date}_{signal.matched.strike}_{signal.side}"
            recent_signals[key] = signal.timestamp

    if executed > 0:
        logger.info(f"Executed {executed}/{len(signals)} signals")

    return len(signals)


def main():
    parser = argparse.ArgumentParser(description="Deribit-Polymarket lead-lag bot")
    parser.add_argument(
        "--config",
        default="config/deribit_leadlag.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Override config to dry-run mode",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single cycle and exit (for testing)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    config = LeadLagConfig.from_yaml(args.config)
    if args.dry_run:
        config.dry_run = True
        config.execution.dry_run = True

    logger.info(f"Config loaded: dry_run={config.dry_run}, poll={config.poll_interval_seconds}s")

    # Load private key and init CLOB client
    private_key = load_private_key()
    clob_client = setup_clob_client(private_key)

    # Init executor
    executor = LeadLagExecutor(clob_client, config.execution)
    executor.reconcile_positions()

    # Discover Polymarket markets
    logger.info("Discovering Polymarket BTC threshold markets...")
    poly_markets = discover_btc_threshold_markets()

    if not poly_markets:
        logger.error("No Polymarket BTC threshold markets found")
        return 1

    logger.info(f"Found {len(poly_markets)} markets across dates: "
                f"{sorted(set(m.expiry_date for m in poly_markets))}")

    # Signal cooldown tracking
    recent_signals: dict = {}

    # Main loop
    last_market_refresh = time.time()

    while True:
        try:
            cycle_start = time.time()
            n_signals = run_poll_cycle(config, executor, poly_markets, recent_signals)

            # Periodic market refresh
            if time.time() - last_market_refresh > config.market_refresh_interval_seconds:
                logger.info("Refreshing Polymarket markets...")
                new_markets = discover_btc_threshold_markets()
                if new_markets:
                    poly_markets = new_markets
                    last_market_refresh = time.time()

            cycle_duration = time.time() - cycle_start
            logger.info(
                f"Cycle complete: {n_signals} signals, "
                f"exposure=${executor.total_exposure:.2f}, "
                f"took {cycle_duration:.1f}s"
            )

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)

        if args.once:
            break

        # Sleep until next cycle
        elapsed = time.time() - cycle_start
        sleep_time = max(0, config.poll_interval_seconds - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    return 0


if __name__ == "__main__":
    exit(main())
