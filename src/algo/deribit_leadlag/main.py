"""
Deribit-reference -> Polymarket lead-lag trading bot (async version).

Real-time hybrid maker/taker system with settlement compatibility checking.

Architecture:
  Task 1: DeribitWS → PriceStore
  Task 2: StrategyTick (1.5s) reads all stores → acts
  Task 3: PositionPoller (10s) → PositionManager
  Task 4: MarketRefresh (10min)
  Task 5: DeribitWatchdog (built into DeribitWebSocket)
  PolymarketStreams (orderbook WS + user stream)

Usage:
    python -m src.algo.deribit_leadlag.main --config config/deribit_leadlag.yaml [--dry-run]
"""

import argparse
import asyncio
import logging
import signal
import time
from datetime import date, datetime, timedelta, timezone

from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON

from src.const import CLOB_API_URL
from src.utils.crypto_utils import load_private_key

from .config import LeadLagConfig
from .deribit_client import fetch_btc_options_summary
from .deribit_ws import DeribitWebSocket, PriceStore
from .order_manager import OrderManager
from .poly_ws import PolymarketStreams
from .polymarket_discovery import ThresholdMarket, discover_btc_threshold_markets
from .position_manager import BinKey, PositionManager
from .settlement import CompatibilityClass, classify_compatibility
from .signal_comparator import build_adjusted_prob_map

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


def classify_markets(
    markets: list[ThresholdMarket],
) -> dict[BinKey, CompatibilityClass]:
    """Classify settlement compatibility for all markets."""
    result = {}
    for m in markets:
        key = BinKey(expiry_date=m.expiry_date, strike=m.strike)
        if m.settlement is None:
            result[key] = CompatibilityClass.REJECT
            continue
        compat, reason = classify_compatibility(m.expiry_date, m.settlement)
        result[key] = compat
        if compat != CompatibilityClass.REJECT:
            logger.debug(f"  {m.expiry_date}/{m.strike:.0f}: {compat.value} — {reason}")
    eligible = sum(1 for v in result.values() if v != CompatibilityClass.REJECT)
    logger.info(
        f"Settlement compatibility: {eligible}/{len(result)} eligible "
        f"({sum(1 for v in result.values() if v == CompatibilityClass.TIME_ADJUSTED)} time_adjusted)"
    )
    return result


def get_deribit_instruments(
    markets: list[ThresholdMarket],
    options: list,
) -> list[str]:
    """Determine which Deribit instruments to subscribe to via WS."""
    target_dates = set()
    for m in markets:
        target_dates.add(m.expiry_date)
        target_dates.add(m.expiry_date + timedelta(days=1))

    instruments = []
    for opt in options:
        if opt.option_type == "C" and opt.expiry_date in target_dates:
            instruments.append(opt.instrument_name)
    return instruments


class LeadLagBot:
    """Async lead-lag trading bot."""

    def __init__(self, config: LeadLagConfig, clob_client: ClobClient):
        self._config = config
        self._client = clob_client
        self._running = False

        # Components
        self._price_store = PriceStore()
        self._deribit_ws = DeribitWebSocket(
            price_store=self._price_store,
            on_emergency=self._emergency_cancel,
            ws_url=config.deribit_ws.ws_url,
            reconnect_delay=config.deribit_ws.reconnect_delay_seconds,
            max_reconnect_delay=config.deribit_ws.max_reconnect_delay_seconds,
            stale_threshold=config.deribit_ws.stale_threshold_seconds,
        )

        wallet_address = clob_client.get_address() if hasattr(clob_client, 'get_address') else ""
        self._position_mgr = PositionManager(
            alloc_config=config.allocation,
            order_config=config.order,
            signal_config=config.signal,
            wallet_address=wallet_address,
        )
        self._order_mgr = OrderManager(
            clob_client=clob_client,
            order_config=config.order,
            dry_run=config.dry_run,
        )

        # State
        self._markets: list[ThresholdMarket] = []
        self._compat_map: dict[BinKey, CompatibilityClass] = {}
        self._poly_streams: PolymarketStreams | None = None

    async def _emergency_cancel(self) -> None:
        """Emergency: cancel all Polymarket orders (data stale)."""
        logger.error("[EMERGENCY] Cancelling all orders due to stale data")
        self._order_mgr.cancel_all_orders()

    def _on_fill(self, fill_event) -> None:
        """Handle fill from user stream."""
        logger.info(
            f"[FILL] order={fill_event.order_id[:12]}... "
            f"side={fill_event.side} "
            f"token={fill_event.token_id[:12]}... "
            f"size={fill_event.size} @ {fill_event.price:.4f}"
        )
        self._position_mgr.handle_fill(
            order_id=fill_event.order_id,
            token_id=fill_event.token_id,
            side=fill_event.side,
            filled_size=fill_event.size,
        )

    async def run(self) -> None:
        """
        Main entry point. Runs the startup sequence then concurrent tasks.

        Startup Reconciliation:
        1. Connect user stream
        2. Fetch open orders
        3. Fetch positions
        4. Cancel ALL inherited orders
        5. Re-fetch to confirm clean state
        6. Connect WS feeds
        7. Seed PriceStore
        8. Start strategy tick
        """
        self._running = True

        # Discover markets
        logger.info("Discovering Polymarket BTC threshold markets...")
        self._markets = discover_btc_threshold_markets()
        if not self._markets:
            logger.error("No Polymarket BTC threshold markets found")
            return

        self._compat_map = classify_markets(self._markets)
        eligible = [
            m for m in self._markets
            if self._compat_map.get(BinKey(m.expiry_date, m.strike)) != CompatibilityClass.REJECT
        ]
        if not eligible:
            logger.info("Zero eligible pairs after settlement compatibility check. Idling.")
            return

        logger.info(f"Found {len(eligible)} eligible markets across "
                     f"{len(set(m.expiry_date for m in eligible))} dates")

        # Update position manager with eligible markets
        self._position_mgr.update_markets(self._markets, self._compat_map)

        # Step 1-3: Setup Polymarket streams + fetch state
        api_creds = getattr(self._client, "creds", None)
        if api_creds:
            self._poly_streams = PolymarketStreams(
                config=self._config.poly_ws,
                api_key=api_creds.api_key,
                api_secret=api_creds.api_secret,
                api_passphrase=api_creds.api_passphrase,
                on_fill=self._on_fill,
            )
            await self._poly_streams.start()
        else:
            logger.warning("ClobClient has no API creds attached; Polymarket WS streams disabled")

        # Steps 2-3: Fetch current state
        self._position_mgr.fetch_open_orders(self._client)
        self._position_mgr.fetch_positions()

        # Step 4: Cancel ALL inherited orders
        logger.info("Cancelling all inherited orders from prior run...")
        self._order_mgr.cancel_all_orders()

        # Step 5: Re-fetch to confirm clean state
        self._position_mgr.fetch_open_orders(self._client)
        self._position_mgr.fetch_positions()

        # Step 6-7: Deribit setup
        logger.info("Fetching Deribit options snapshot...")
        options = fetch_btc_options_summary(self._config.deribit.base_url)
        if not options:
            logger.error("No Deribit options fetched")
            return

        self._price_store.load_initial(options)

        instruments = get_deribit_instruments(self._markets, options)
        logger.info(f"Subscribing to {len(instruments)} Deribit instruments...")
        await self._deribit_ws.connect()
        await self._deribit_ws.update_subscriptions(instruments)

        # Subscribe Polymarket orderbooks
        if self._poly_streams:
            await self._poly_streams.subscribe_markets(eligible)

        # Step 8: Run concurrent tasks
        logger.info("Starting strategy loop...")
        try:
            await asyncio.gather(
                self._strategy_tick_loop(),
                self._position_poll_loop(),
                self._market_refresh_loop(),
            )
        except asyncio.CancelledError:
            pass
        finally:
            await self._shutdown()

    async def _strategy_tick_loop(self) -> None:
        """Main strategy loop — runs every tick_interval_seconds."""
        interval = self._config.strategy.tick_interval_seconds
        while self._running:
            try:
                self._strategy_tick()
            except Exception as e:
                logger.error(f"Strategy tick error: {e}", exc_info=True)
            await asyncio.sleep(interval)

    def _strategy_tick(self) -> None:
        """Single strategy tick: compute probs → signals → targets → actions."""
        now = datetime.now(timezone.utc)
        options = self._price_store.build_deribit_options()

        if not options:
            return

        # Build adjusted reference probabilities
        adjusted_probs = build_adjusted_prob_map(
            options=options,
            now=now,
            markets=self._markets,
            signal_config=self._config.signal,
            compatibility_map=self._compat_map,
        )

        if not adjusted_probs:
            return

        # Get live orderbooks
        orderbooks = {}
        if self._poly_streams:
            orderbooks = self._poly_streams.get_all_orderbooks()

        # Compute targets and deltas
        self._position_mgr.compute_targets(adjusted_probs, orderbooks)
        actions = self._position_mgr.compute_deltas()

        if actions:
            report = self._order_mgr.execute_actions(actions)
            self._position_mgr.apply_execution_report(report)
            if report.posted_actions:
                logger.info(f"Executed {len(report.posted_actions)} orders this tick")

    async def _position_poll_loop(self) -> None:
        """Periodically re-fetch positions from exchange."""
        interval = self._config.strategy.position_poll_interval_seconds
        while self._running:
            await asyncio.sleep(interval)
            try:
                self._position_mgr.fetch_positions()
                self._position_mgr.fetch_open_orders(self._client)
            except Exception as e:
                logger.error(f"Position poll error: {e}", exc_info=True)

    async def _market_refresh_loop(self) -> None:
        """Periodically re-discover Polymarket markets."""
        interval = self._config.strategy.market_refresh_interval_seconds
        while self._running:
            await asyncio.sleep(interval)
            try:
                new_markets = discover_btc_threshold_markets()
                if new_markets:
                    self._markets = new_markets
                    self._compat_map = classify_markets(self._markets)
                    self._position_mgr.update_markets(self._markets, self._compat_map)

                    # Update Deribit subscriptions
                    options = self._price_store.build_deribit_options()
                    instruments = get_deribit_instruments(self._markets, options)
                    await self._deribit_ws.update_subscriptions(instruments)

                    # Update Polymarket subscriptions
                    eligible = [
                        m for m in self._markets
                        if self._compat_map.get(BinKey(m.expiry_date, m.strike)) != CompatibilityClass.REJECT
                    ]
                    if self._poly_streams:
                        await self._poly_streams.subscribe_markets(eligible)

                    logger.info(f"Market refresh: {len(eligible)} eligible markets")
            except Exception as e:
                logger.error(f"Market refresh error: {e}", exc_info=True)

    async def _shutdown(self) -> None:
        """Graceful shutdown."""
        self._running = False
        logger.info("Shutting down...")
        self._order_mgr.cancel_all_orders()
        await self._deribit_ws.disconnect()
        if self._poly_streams:
            await self._poly_streams.stop()
        logger.info("Shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="Deribit-Polymarket lead-lag bot (async)")
    parser.add_argument("--config", default="config/deribit_leadlag.yaml", help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Override config to dry-run mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = LeadLagConfig.from_yaml(args.config)
    if args.dry_run:
        config.dry_run = True

    logger.info(f"Config loaded: dry_run={config.dry_run}")

    private_key = load_private_key()
    clob_client = setup_clob_client(private_key)

    bot = LeadLagBot(config, clob_client)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def handle_signal(sig, frame):
        logger.info(f"Received signal {sig}, requesting shutdown...")
        bot._running = False
        for task in asyncio.all_tasks(loop):
            task.cancel()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        loop.run_until_complete(bot.run())
    finally:
        loop.close()

    return 0


if __name__ == "__main__":
    exit(main())
