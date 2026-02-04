"""
Integration module for Kelly optimizer with the main trading bot.

Provides a KellyTradingBot that integrates the Kelly optimizer with
the existing PolymarketTradingBot infrastructure.

Key design decisions:
- Portfolio is ONLY updated on WebSocket fill confirmation, not on order placement
- Sync from API happens before each tick, not periodically
- UserStreamClient provides real-time fill notifications
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Tuple

import requests
from py_clob_client.client import ClobClient

# Polymarket Data API for fetching positions
POLYMARKET_DATA_API = "https://data-api.polymarket.com"

from .config import KellyConfig
from .orderbook import UnifiedOrderbook
from .portfolio import Portfolio
from .candidates import TradeCandidate
from .executor import KellyExecutor, OrderExecutor, ExecutionResult, TickResult
from .websocket_client import OrderbookManager, WebSocketConfig
from .kelly_math import identify_dead_bins, renormalize_probabilities
from .user_stream import UserStreamClient, FillEvent, PendingOrder

logger = logging.getLogger(__name__)


class KellyTradingBot:
    """
    Kelly-optimal trading bot for Polymarket tweet-count bins.

    Integrates with the existing trading infrastructure while using
    Kelly criterion optimization for position sizing.

    Key architecture:
    - Portfolio updates happen ONLY on WebSocket fill confirmation
    - Sync from API before each tick (not periodic)
    - UserStreamClient provides real-time fill/order notifications
    """

    def __init__(
        self,
        clob_client: ClobClient,
        config: KellyConfig,
        probability_model: Callable[[int, float, float], List[float]],
        dry_run: bool = False,
        wallet_address: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        api_passphrase: Optional[str] = None,
        external_user_stream: Optional[UserStreamClient] = None,
        disable_websocket: bool = False,
    ):
        """
        Initialize Kelly trading bot.

        Args:
            clob_client: Authenticated ClobClient for order execution
            config: Kelly configuration
            probability_model: Function(current_count, hours_elapsed, hours_remaining)
                              -> List[float] probabilities for each bin
            dry_run: If True, simulate trades without execution
            wallet_address: Wallet address for position queries
            api_key: Polymarket API key (for user stream)
            api_secret: Polymarket API secret (for user stream)
            api_passphrase: Polymarket API passphrase (for user stream)
            external_user_stream: Optional external UserStreamClient (from MultiEventManager).
                                 If provided, uses this instead of creating our own.
                                 Fill events are routed to this bot via handle_fill().
            disable_websocket: If True, disable orderbook WebSocket (use REST polling)
        """
        self._external_user_stream = external_user_stream
        self._owns_user_stream = False  # Will be set in setup()
        self._disable_websocket = disable_websocket
        self.clob_client = clob_client
        self.config = config
        self.probability_model = probability_model
        self.dry_run = dry_run
        self.wallet_address = wallet_address or os.getenv("WALLET_ADDRESS", "")

        # API credentials for user stream
        self._api_key = api_key or os.getenv("CLOB_API_KEY", "")
        self._api_secret = api_secret or os.getenv("CLOB_API_SECRET", "")
        self._api_passphrase = api_passphrase or os.getenv("CLOB_API_PASSPHRASE", "")

        # Bin metadata (set during setup)
        self.bin_upper_bounds: List[int] = []
        self.bin_token_ids: Dict[int, str] = {}  # bin_index -> YES token_id
        self.num_bins: int = 0

        # Components (initialized during setup)
        self.orderbook_manager: Optional[OrderbookManager] = None
        self.order_executor: Optional[OrderExecutor] = None
        self.portfolio: Optional[Portfolio] = None
        self.kelly_executor: Optional[KellyExecutor] = None
        self.user_stream: Optional[UserStreamClient] = None

        # State
        self._running = False
        self._setup_complete = False
        self._last_logged_dead_bins: Optional[List[int]] = None  # Track to avoid spam

        # Lock for thread-safe tick execution
        self._tick_lock = asyncio.Lock()

    async def setup(
        self,
        initial_capital: float,
        bins: List[Dict],  # List of {"upper_bound": int, "token_id": str}
    ) -> None:
        """
        Setup the Kelly trading bot.

        Args:
            initial_capital: Initial USDC capital
            bins: List of bin definitions with upper_bound and token_id
        """
        logger.info(f"Setting up Kelly bot with {len(bins)} bins, ${initial_capital:.2f} capital")

        # Store bin metadata
        self.num_bins = len(bins)
        self.bin_upper_bounds = [b["upper_bound"] for b in bins]
        self.bin_token_ids = {i: b["token_id"] for i, b in enumerate(bins)}

        # Initialize portfolio
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            capital=initial_capital,
            num_bins=self.num_bins,
            bin_upper_bounds=self.bin_upper_bounds,
        )

        # Initialize orderbook manager with WebSocket config
        ws_config = WebSocketConfig(enabled=not self._disable_websocket)
        self.orderbook_manager = OrderbookManager(
            config=ws_config,
            clob_client=self.clob_client,
        )

        # Start orderbook streaming
        await self.orderbook_manager.start()

        # Subscribe to all bins
        await self.orderbook_manager.subscribe_bins([
            {"token_id": b["token_id"], "bin_index": i}
            for i, b in enumerate(bins)
        ])

        # Initialize order executor
        self.order_executor = OrderExecutor(
            clob_client=self.clob_client,
            dry_run=self.dry_run,
        )

        # Initialize user stream for fill confirmations (only for live trading)
        # If external_user_stream is provided (from MultiEventManager), use it
        # Otherwise create our own if API credentials are available
        if self._external_user_stream:
            self.user_stream = self._external_user_stream
            self._owns_user_stream = False  # Don't stop it in shutdown()
            logger.info("Using external UserStreamClient for fill confirmations")
        elif not self.dry_run and self._api_key and self._api_secret:
            self.user_stream = UserStreamClient(
                api_key=self._api_key,
                api_secret=self._api_secret,
                api_passphrase=self._api_passphrase,
            )
            self._owns_user_stream = True  # We created it, we stop it
            # Start user stream
            await self.user_stream.start()
            logger.info("User stream started for fill confirmations")
        else:
            self.user_stream = None
            self._owns_user_stream = False
            if not self.dry_run:
                logger.warning("No API credentials for user stream - fill confirmations disabled")

        # Initialize Kelly executor
        self.kelly_executor = KellyExecutor(
            config=self.config,
            portfolio=self.portfolio,
            orderbook_manager=self.orderbook_manager,
            order_executor=self.order_executor,
            token_ids=self.bin_token_ids,
            on_trade=self._on_trade,
            user_stream=self.user_stream,
        )

        # Wire up user stream callbacks
        # Only set callbacks if we own the user_stream (not external)
        # External user_stream has its callbacks managed by MultiEventManager
        if self.user_stream and getattr(self, '_owns_user_stream', False):
            self.user_stream.on_fill = self._handle_fill
            self.user_stream.on_stale_order = self._handle_stale_order

        self._setup_complete = True
        logger.info("Kelly bot setup complete")

    async def shutdown(self) -> None:
        """Shutdown the Kelly trading bot."""
        logger.info("Shutting down Kelly bot")
        self._running = False

        if self.kelly_executor:
            self.kelly_executor.stop()

        # Only stop user_stream if we own it (created it ourselves)
        # External user_stream is managed by MultiEventManager
        if self.user_stream and getattr(self, '_owns_user_stream', False):
            await self.user_stream.stop()

        if self.orderbook_manager:
            await self.orderbook_manager.stop()

        self._setup_complete = False

    def _on_trade(self, result: ExecutionResult) -> None:
        """Callback for order placement (not fill)."""
        if result.success:
            logger.info(
                f"Order placed (pending fill): {result.candidate.action.value} "
                f"bin={result.candidate.bin_index} "
                f"size={result.candidate.size:.2f} @ {result.candidate.price:.4f}"
            )
        else:
            logger.warning(f"Order placement failed: {result.error}")

    def _handle_fill(self, fill_event: FillEvent) -> None:
        """
        Handle fill confirmation from WebSocket.

        Forwards to Kelly executor which updates the portfolio.
        """
        if self.kelly_executor:
            self.kelly_executor.handle_fill(fill_event)

    async def _handle_stale_order(self, pending: PendingOrder) -> None:
        """
        Handle stale order cancellation.

        Called by UserStreamClient when an order hasn't filled within timeout.
        """
        if self.kelly_executor:
            await self.kelly_executor.handle_stale_order(pending)

    def update_probabilities(
        self,
        current_count: int,
        hours_elapsed: float,
        hours_remaining: float,
    ) -> None:
        """
        Update probability estimates and identify dead bins.

        Args:
            current_count: Current tweet count from xtracker
            hours_elapsed: Hours since counting period started
            hours_remaining: Hours until settlement
        """
        if not self._setup_complete:
            raise RuntimeError("Bot not setup. Call setup() first.")

        # Identify dead bins (upper_bound < current_count)
        dead_bins = identify_dead_bins(self.bin_upper_bounds, current_count)
        self.portfolio.dead_bins = dead_bins

        # Only log when dead bins change (avoid spam)
        if dead_bins != self._last_logged_dead_bins:
            if dead_bins:
                logger.info(f"Dead bins (count={current_count}): {dead_bins}")
            self._last_logged_dead_bins = dead_bins

        # Get raw probabilities from model
        raw_probabilities = self.probability_model(
            current_count, hours_elapsed, hours_remaining
        )

        # Renormalize if configured
        if self.config.renormalize_probabilities:
            probabilities = renormalize_probabilities(raw_probabilities, dead_bins)
        else:
            probabilities = raw_probabilities

        # Update portfolio
        self.portfolio.update_probabilities(probabilities, renormalize=False)

        logger.debug(f"Updated probabilities: {probabilities}")

    async def run_tick(
        self,
        current_count: int,
        hours_elapsed: float,
        hours_to_settlement: float,
        sync_before_trade: bool = True,
        forecast_mean: float = 0,
        forecast_std: float = 0,
    ) -> TickResult:
        """
        Run a single optimization tick.

        Thread-safe: uses lock to prevent concurrent tick execution.

        IMPORTANT: Syncs portfolio from API before trading to ensure
        we have the latest position state.

        Args:
            current_count: Current tweet count
            hours_elapsed: Hours since counting started
            hours_to_settlement: Hours until settlement
            sync_before_trade: If True, sync from API before trading (default True)
            forecast_mean: Forecast mean for trade logging (optional)
            forecast_std: Forecast std for trade logging (optional)

        Returns:
            TickResult with execution summary
        """
        if not self._setup_complete:
            raise RuntimeError("Bot not setup. Call setup() first.")

        # Try to acquire lock without blocking - skip if already running
        if self._tick_lock.locked():
            logger.debug("Kelly tick already in progress, skipping")
            return TickResult(
                num_candidates=0,
                num_executed=0,
                total_utility_gain=0.0,
                executions=[],
                elapsed_seconds=0.0,
            )

        async with self._tick_lock:
            # Sync portfolio from API before trading (per user request: always sync before trade)
            if sync_before_trade and self.wallet_address and not self.dry_run:
                try:
                    await self.sync_positions_from_api(self.wallet_address)
                except Exception as e:
                    logger.warning(f"Failed to sync from API before trade: {e}")

            # Update probabilities (this also identifies dead bins)
            self.update_probabilities(current_count, hours_elapsed, hours_to_settlement)

            # Fetch fresh orderbooks only for LIVE bins (skip dead bins)
            dead_bins = set(self.portfolio.dead_bins)
            orderbooks = {}
            for bin_idx, token_id in self.bin_token_ids.items():
                if bin_idx in dead_bins:
                    continue  # Skip dead bins - no need to fetch orderbooks
                if not self.orderbook_manager.get_orderbook(token_id):
                    await self.orderbook_manager.fetch_orderbook(token_id, bin_idx)
                ob = self.orderbook_manager.get_orderbook(token_id)
                if ob:
                    orderbooks[bin_idx] = ob

            # Build bin ranges for logging
            bin_ranges = {}
            for bin_idx in range(self.num_bins):
                if bin_idx < len(self.bin_upper_bounds):
                    upper = self.bin_upper_bounds[bin_idx]
                    lower = self.bin_upper_bounds[bin_idx - 1] + 1 if bin_idx > 0 else 0
                    if upper == float('inf'):
                        bin_ranges[bin_idx] = f"{lower}+"
                    else:
                        bin_ranges[bin_idx] = f"{lower}-{upper}"

            # Set logging context for detailed trade logs
            self.kelly_executor.set_log_context(
                probabilities=self.portfolio.probabilities if self.portfolio else [],
                orderbooks=orderbooks,
                bin_ranges=bin_ranges,
                current_count=current_count,
                hours_to_settlement=hours_to_settlement,
                forecast_mean=forecast_mean,
                forecast_std=forecast_std,
            )

            # Run Kelly optimization tick
            result = await self.kelly_executor.run_tick(hours_to_settlement)

            return result

    async def run_continuous(
        self,
        get_market_state: Callable[[], Dict],
        tick_interval_seconds: float = 60.0,
    ) -> None:
        """
        Run continuous Kelly optimization.

        Args:
            get_market_state: Function returning {
                "current_count": int,
                "hours_elapsed": float,
                "hours_to_settlement": float,
            }
            tick_interval_seconds: Seconds between optimization ticks
        """
        if not self._setup_complete:
            raise RuntimeError("Bot not setup. Call setup() first.")

        self._running = True
        logger.info("Starting continuous Kelly optimization")

        while self._running:
            try:
                # Get current market state
                state = get_market_state()

                current_count = state.get("current_count", 0)
                hours_elapsed = state.get("hours_elapsed", 0)
                hours_to_settlement = state.get("hours_to_settlement", 0)

                # Check T_stop
                if hours_to_settlement <= self.config.t_stop_hours:
                    logger.info("Reached T_stop. Holding positions to settlement.")
                    break

                # Run tick
                result = await self.run_tick(
                    current_count,
                    hours_elapsed,
                    hours_to_settlement,
                )

                logger.info(
                    f"Tick: candidates={result.num_candidates}, "
                    f"executed={result.num_executed}, "
                    f"utility_gain={result.total_utility_gain:.6f}"
                )

                # Wait for next tick
                await asyncio.sleep(tick_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in tick: {e}")
                await asyncio.sleep(tick_interval_seconds)

        self._running = False
        logger.info("Continuous optimization stopped")

    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary."""
        if not self.portfolio:
            return {}
        return self.portfolio.to_summary()

    def get_status(self) -> Dict:
        """
        Get full bot status including pending orders.

        Returns:
            Dict with portfolio summary, pending orders, and user stream status
        """
        status = {
            "setup_complete": self._setup_complete,
            "running": self._running,
            "dry_run": self.dry_run,
            "portfolio": self.get_portfolio_summary(),
            "pending_orders": {
                "count": 0,
                "collateral": 0.0,
            },
            "user_stream": {
                "connected": False,
                "pending_tracked": 0,
            },
        }

        if self.kelly_executor:
            status["pending_orders"]["count"] = self.kelly_executor.get_pending_count()
            status["pending_orders"]["collateral"] = self.kelly_executor.get_pending_collateral()

        if self.user_stream:
            status["user_stream"]["connected"] = self.user_stream._ws is not None
            status["user_stream"]["pending_tracked"] = self.user_stream.get_pending_orders_count()

        return status

    def get_reservation_prices(self) -> tuple[List[float], List[float]]:
        """Get current Kelly reservation prices."""
        if not self.portfolio:
            return [], []
        return self.portfolio.get_reservation_prices(self.config.w_floor)

    async def fetch_positions_from_api(self, wallet_address: str) -> Dict[str, float]:
        """
        Fetch current positions from Polymarket Data API.

        Args:
            wallet_address: The wallet address to fetch positions for

        Returns:
            Dict mapping token_id -> shares held
        """
        try:
            response = requests.get(
                f"{POLYMARKET_DATA_API}/positions",
                params={"user": wallet_address.lower()},
                timeout=30,
            )
            response.raise_for_status()
            positions_data = response.json()

            # Parse positions: token_id -> shares
            positions = {}
            for pos in positions_data:
                token_id = pos.get("asset", {}).get("id") or pos.get("token_id")
                size = float(pos.get("size", 0))
                if token_id and size > 0:
                    positions[token_id] = size

            logger.info(f"Fetched {len(positions)} positions from Polymarket API")
            return positions

        except Exception as e:
            logger.warning(f"Failed to fetch positions from API: {e}")
            return {}

    async def fetch_usdc_balance(self) -> float:
        """
        Fetch current USDC balance from CLOB client.

        Returns:
            USDC balance
        """
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

            # Use CLOB client to get balance (COLLATERAL = USDC)
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            balance_info = self.clob_client.get_balance_allowance(params)
            # USDC has 6 decimals, so divide by 1e6
            usdc_balance = float(balance_info.get("balance", 0)) / 1e6
            logger.info(f"Fetched USDC balance: ${usdc_balance:.2f}")
            return usdc_balance
        except Exception as e:
            logger.warning(f"Failed to fetch USDC balance: {e}")
            return 0.0

    async def sync_positions_from_api(self, wallet_address: str) -> Tuple[float, Dict[int, float]]:
        """
        Sync portfolio state from Polymarket API.

        Fetches current positions and USDC balance, then updates the portfolio
        to match the on-chain state. Use this on startup to recover state after restart.

        Args:
            wallet_address: The wallet address to sync positions for

        Returns:
            Tuple of (usdc_balance, {bin_index: shares})
        """
        if not self._setup_complete:
            raise RuntimeError("Bot must be setup before syncing positions")

        # Fetch current state from API
        api_positions = await self.fetch_positions_from_api(wallet_address)
        usdc_balance = await self.fetch_usdc_balance()

        # Map token_ids back to bin indices
        token_to_bin = {token_id: bin_idx for bin_idx, token_id in self.bin_token_ids.items()}

        # Identify which positions we have and need orderbooks for
        positions_to_sync = []
        for token_id, shares in api_positions.items():
            bin_idx = token_to_bin.get(token_id)
            if bin_idx is not None:
                positions_to_sync.append((token_id, bin_idx, shares))

        # Fetch orderbooks for all positions before computing values
        if self.orderbook_manager and positions_to_sync:
            logger.info(f"Fetching orderbooks for {len(positions_to_sync)} positions...")
            for token_id, bin_idx, _ in positions_to_sync:
                if not self.orderbook_manager.get_orderbook(token_id):
                    try:
                        await self.orderbook_manager.fetch_orderbook(token_id, bin_idx)
                    except Exception as e:
                        logger.warning(f"Failed to fetch orderbook for bin {bin_idx}: {e}")

        # Update portfolio positions with fetched orderbook prices
        synced_positions = {}
        total_position_value = 0.0

        for token_id, bin_idx, shares in positions_to_sync:
            synced_positions[bin_idx] = shares

            # Get current price from orderbook (should be available now)
            orderbook = self.orderbook_manager.get_orderbook(token_id) if self.orderbook_manager else None
            if orderbook and orderbook.best_yes_bid:
                position_value = shares * orderbook.best_yes_bid
                price_used = orderbook.best_yes_bid
                price_source = "orderbook"
            else:
                # No orderbook = value position at 0 (conservative)
                position_value = 0.0
                price_used = 0.0
                price_source = "no_orderbook"
                logger.warning(f"Bin {bin_idx}: no orderbook available, valuing at $0")
            total_position_value += position_value

            # Update portfolio position
            pos = self.portfolio.ensure_position(bin_idx, token_id)
            pos.yes_shares = shares
            pos.yes_avg_cost = price_used
            pos.collateral_used = shares * price_used

            logger.info(f"Synced bin {bin_idx}: {shares:.2f} shares @ {price_used:.4f} ({price_source})")

        # Update portfolio capital
        self.portfolio.capital = usdc_balance + total_position_value
        self.portfolio.initial_capital = self.portfolio.capital

        logger.info(
            f"Portfolio synced: ${usdc_balance:.2f} USDC + ${total_position_value:.2f} positions = "
            f"${self.portfolio.capital:.2f} total"
        )

        return usdc_balance, synced_positions


def create_kelly_bot_from_config(
    clob_client: ClobClient,
    yaml_config: Dict,
    probability_model: Callable[[int, float, float], List[float]],
    dry_run: bool = False,
) -> KellyTradingBot:
    """
    Create a KellyTradingBot from YAML configuration.

    Args:
        clob_client: Authenticated ClobClient
        yaml_config: Configuration dict with 'kelly' section
        probability_model: Probability model function
        dry_run: If True, simulate trades

    Returns:
        KellyTradingBot instance
    """
    kelly_config_dict = yaml_config.get("kelly", {})

    # Add collateral config if present
    if "collateral" in yaml_config:
        kelly_config_dict["collateral"] = yaml_config["collateral"]

    kelly_config = KellyConfig.from_dict(kelly_config_dict)

    return KellyTradingBot(
        clob_client=clob_client,
        config=kelly_config,
        probability_model=probability_model,
        dry_run=dry_run,
    )
