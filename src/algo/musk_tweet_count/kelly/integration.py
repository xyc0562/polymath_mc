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
        event_name: Optional[str] = None,
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
            event_name: Optional event name for logging (e.g., "Feb 03 - Feb 10")
        """
        self._external_user_stream = external_user_stream
        self.event_name = event_name or "unknown"
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
        self._ema_probabilities: Optional[List[float]] = None

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
        self.bin_lower_bounds = [b.get("lower_bound", 0) for b in bins]
        self.bin_token_ids = {i: b["token_id"] for i, b in enumerate(bins)}  # YES token IDs
        self.bin_no_token_ids = {i: b.get("no_token_id") for i, b in enumerate(bins)}  # NO token IDs

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
            event_name=self.event_name,
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

        # Initialize Kelly executor with sync callback
        # The sync callback fetches official portfolio state from API before each decision
        self.kelly_executor = KellyExecutor(
            config=self.config,
            portfolio=self.portfolio,
            orderbook_manager=self.orderbook_manager,
            order_executor=self.order_executor,
            token_ids=self.bin_token_ids,
            no_token_ids=self.bin_no_token_ids,
            on_trade=self._on_trade,
            user_stream=self.user_stream,
            sync_portfolio=self._create_sync_callback(),
            event_name=self.event_name,
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

    def _create_sync_callback(self):
        """
        Create a callback function that syncs portfolio from API.

        This callback is called by the executor BEFORE each Kelly decision
        to ensure we have official state, not calculated estimates.
        """
        async def sync_callback():
            if self.wallet_address and not self.dry_run:
                await self.sync_positions_from_api(self.wallet_address)
            else:
                logger.debug("Skipping API sync (dry_run or no wallet)")

        return sync_callback

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
            self._ema_probabilities = None  # Reset EMA on dead bin change

        # Get raw probabilities from model
        raw_probabilities = self.probability_model(
            current_count, hours_elapsed, hours_remaining
        )

        # Renormalize if configured
        if self.config.renormalize_probabilities:
            probabilities = renormalize_probabilities(raw_probabilities, dead_bins)
        else:
            probabilities = raw_probabilities

        # EMA smooth probabilities to dampen Monte Carlo noise
        alpha = self.config.prob_ema_alpha
        if alpha < 1.0 and self._ema_probabilities is not None:
            probabilities = [
                alpha * p_new + (1 - alpha) * p_old
                for p_new, p_old in zip(probabilities, self._ema_probabilities)
            ]
            # Re-normalize after blending (EMA can drift slightly from sum=1)
            total = sum(probabilities)
            if total > 0:
                probabilities = [p / total for p in probabilities]
        self._ema_probabilities = probabilities

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
        verbose: bool = False,
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
            verbose: If True, log detailed rejection reasons for candidates

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
            # NOTE: Portfolio sync is now handled by the executor BEFORE EACH order decision
            # The executor calls sync_portfolio callback before each generate_candidates() call
            # This ensures we always use official API data, not calculated estimates

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
                    lower = self.bin_lower_bounds[bin_idx] if bin_idx < len(self.bin_lower_bounds) else 0
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
            result = await self.kelly_executor.run_tick(hours_to_settlement, verbose=verbose)

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
        return self.portfolio.get_reservation_prices(self.config.w_floor, self.config.kelly_fraction)

    async def fetch_positions_from_api(self, wallet_address: str) -> Dict[str, dict]:
        """
        Fetch current positions from Polymarket Data API.

        Args:
            wallet_address: The wallet address to fetch positions for

        Returns:
            Dict mapping token_id -> {shares, avg_price, value}
        """
        try:
            response = requests.get(
                f"{POLYMARKET_DATA_API}/positions",
                params={"user": wallet_address.lower()},
                timeout=30,
            )
            response.raise_for_status()
            positions_data = response.json()

            # Parse positions: token_id -> {shares, avg_price, value}
            # Handle different response formats
            positions = {}

            def parse_position(pos: dict) -> tuple:
                """Parse position dict, returns (token_id, pos_info) or (None, None)."""
                asset = pos.get("asset")
                if isinstance(asset, str):
                    token_id = asset
                elif isinstance(asset, dict):
                    token_id = asset.get("id")
                else:
                    token_id = pos.get("token_id") or pos.get("asset_id")

                size = float(pos.get("size", 0))
                if not token_id or size <= 0:
                    return None, None

                avg_price = float(pos.get("avgPrice", 0))
                initial_value = float(pos.get("initialValue", 0))
                value = initial_value if initial_value > 0 else (size * avg_price)

                return token_id, {
                    "shares": size,
                    "avg_price": avg_price,
                    "value": value,
                }

            if isinstance(positions_data, list):
                for pos in positions_data:
                    if isinstance(pos, dict):
                        token_id, pos_info = parse_position(pos)
                        if token_id:
                            positions[token_id] = pos_info
            elif isinstance(positions_data, dict):
                pos_list = positions_data.get("positions", positions_data.get("data", []))
                for pos in pos_list:
                    if isinstance(pos, dict):
                        token_id, pos_info = parse_position(pos)
                        if token_id:
                            positions[token_id] = pos_info

            logger.debug(f"Fetched {len(positions)} positions from Polymarket API")
            return positions

        except Exception as e:
            logger.warning(f"Failed to fetch positions from API: {e}", exc_info=True)
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
            logger.debug(f"Fetched USDC balance: ${usdc_balance:.2f}")
            return usdc_balance
        except Exception as e:
            logger.warning(f"Failed to fetch USDC balance: {e}")
            return 0.0

    async def sync_positions_from_api(self, wallet_address: str) -> Tuple[float, Dict[int, float]]:
        """
        Sync portfolio state from Polymarket API.

        The API is the authoritative source of truth for positions.
        This sync unconditionally overwrites local state with API data:
        - Updates positions to match API (both increases and decreases)
        - Clears positions the API no longer reports (sold/closed)
        - Recalculates collateral_used from synced positions

        Note: The API has some latency (seconds to minutes), so within a
        single tick we use local state for iterative Kelly optimization.
        WebSocket fills are logged but don't update portfolio (avoids
        double-counting from MATCHED/MINED/CONFIRMED callbacks).

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

        # Map token_ids back to bin indices (both YES and NO tokens)
        token_to_bin = {}
        token_is_no = {}  # Track which tokens are NO tokens
        for bin_idx, token_id in self.bin_token_ids.items():
            token_to_bin[token_id] = bin_idx
            token_is_no[token_id] = False
        for bin_idx, token_id in self.bin_no_token_ids.items():
            if token_id:
                token_to_bin[token_id] = bin_idx
                token_is_no[token_id] = True

        # IMPORTANT: Trust the API as the source of truth for positions.
        # This handles manual trades, position closures, and ensures consistency.
        # The API has some latency, but it's the authoritative source.

        # First, collect all bin indices we're tracking
        tracked_bins = set(self.bin_token_ids.keys())

        # Identify which positions API reports
        api_bin_positions = {}  # bin_idx -> (pos_info, is_no)
        for token_id, pos_info in api_positions.items():
            bin_idx = token_to_bin.get(token_id)
            if bin_idx is not None:
                is_no = token_is_no.get(token_id, False)
                api_bin_positions[(bin_idx, is_no)] = pos_info

        # Update positions from API - API is the source of truth
        synced_positions = {}

        # First, handle positions that API reports
        for (bin_idx, is_no), pos_info in api_bin_positions.items():
            api_shares = pos_info["shares"]
            api_avg_price = pos_info["avg_price"]

            yes_token = self.bin_token_ids.get(bin_idx)
            pos = self.portfolio.ensure_position(bin_idx, yes_token or "")

            local_shares = pos.no_shares if is_no else pos.yes_shares
            price_used = api_avg_price if api_avg_price > 0 else 0.5

            # Update to API value if different
            if abs(api_shares - local_shares) > 0.01:
                if is_no:
                    pos.no_shares = api_shares
                    pos.no_avg_cost = price_used
                    # Recalculate collateral as sum of YES + NO collateral
                    pos.collateral_used = (pos.yes_shares * pos.yes_avg_cost) + (pos.no_shares * pos.no_avg_cost)
                    logger.info(f"API sync: bin {bin_idx} NO updated {local_shares:.2f} -> {api_shares:.2f} @ ${price_used:.4f}")
                else:
                    pos.yes_shares = api_shares
                    pos.yes_avg_cost = price_used
                    # Recalculate collateral as sum of YES + NO collateral
                    pos.collateral_used = (pos.yes_shares * pos.yes_avg_cost) + (pos.no_shares * pos.no_avg_cost)
                    logger.info(f"API sync: bin {bin_idx} YES updated {local_shares:.2f} -> {api_shares:.2f} @ ${price_used:.4f}")

            synced_positions[bin_idx] = api_shares

        # Second, clear positions that API doesn't report (they were sold/closed)
        for bin_idx in tracked_bins:
            pos = self.portfolio.positions.get(bin_idx)
            if not pos:
                continue

            # Check YES position - if API doesn't have it and we do, clear it
            if pos.yes_shares > 0.01 and (bin_idx, False) not in api_bin_positions:
                logger.info(f"API sync: bin {bin_idx} YES cleared {pos.yes_shares:.2f} -> 0 (position closed)")
                pos.yes_shares = 0
                pos.yes_avg_cost = 0
                # Recalculate collateral (only NO remains if any)
                pos.collateral_used = pos.no_shares * pos.no_avg_cost

            # Check NO position - if API doesn't have it and we do, clear it
            if pos.no_shares > 0.01 and (bin_idx, True) not in api_bin_positions:
                logger.info(f"API sync: bin {bin_idx} NO cleared {pos.no_shares:.2f} -> 0 (position closed)")
                pos.no_shares = 0
                pos.no_avg_cost = 0
                # Recalculate collateral (only YES remains if any)
                pos.collateral_used = pos.yes_shares * pos.yes_avg_cost

        # IMPORTANT: Use event capital budget (c_event_max), NOT wallet USDC balance
        # In multi-event scenarios, each event has its own capital allocation.
        # The wallet USDC balance is shared across all events and is NOT this event's capital.
        #
        # Event capital model:
        #   event_budget = c_event_max (maximum capital for this event)
        #   capital = event_budget - collateral_used (available for new trades)
        #   total_value = capital + collateral_used = event_budget (constant)
        #
        # This ensures Kelly utility calculations use the correct capital base.
        event_budget = self.config.collateral.c_event_max
        total_collateral = self.portfolio.total_collateral_used

        if event_budget > 0:
            # Compute available capital as event budget minus collateral in use
            available_capital = max(0.0, event_budget - total_collateral)
            old_capital = self.portfolio.capital
            self.portfolio.capital = available_capital
            logger.info(
                f"[{self.event_name}] Event capital: budget=${event_budget:.2f}, invested=${total_collateral:.2f}, "
                f"available=${available_capital:.2f} (was ${old_capital:.2f})"
            )
        else:
            # Fallback if c_event_max not set: use USDC balance (legacy behavior)
            local_capital = self.portfolio.capital
            if abs(usdc_balance - local_capital) > 1.0:
                logger.warning(
                    f"Capital mismatch (no event budget): local=${local_capital:.2f}, "
                    f"API USDC=${usdc_balance:.2f}. Using API value."
                )
                self.portfolio.capital = usdc_balance

        # Log current state (use info level for visibility)
        num_positions = len([p for p in self.portfolio.positions.values()
                            if p.yes_shares > 0 or p.no_shares > 0])
        logger.info(
            f"[{self.event_name}] Portfolio synced: {num_positions} positions, "
            f"invested=${total_collateral:.2f}, capital=${self.portfolio.capital:.2f}"
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
