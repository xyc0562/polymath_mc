"""
Integration module for Kelly optimizer with the main trading bot.

Provides a KellyTradingBot that integrates the Kelly optimizer with
the existing PolymarketTradingBot infrastructure.

Key design decisions:
- Portfolio base state is updated only from the positions API
- Sync from API happens before every Kelly iteration, not periodically
- Confirmed WebSocket fills feed a temporary planning overlay until the API catches up
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
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
from .websocket_client import OrderbookManager, OrderbookWebSocket, WebSocketConfig
from .kelly_math import identify_dead_bins, renormalize_probabilities
from .market_signals import compute_market_consensus_blend
from .user_stream import UserStreamClient, FillEvent, PendingOrder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PositionSyncWarningContext:
    """Warning context when positions API omits a still-live position side."""

    bin_index: int
    side: str
    token_id: str
    local_shares: float
    verified_balance_shares: Optional[float]
    missing_since: float
    missing_count: int
    reason: str


class KellyTradingBot:
    """
    Kelly-optimal trading bot for Polymarket tweet-count bins.

    Integrates with the existing trading infrastructure while using
    Kelly criterion optimization for position sizing.

    Key architecture:
    - Portfolio base state comes from the positions API only
    - Sync from API before each Kelly iteration
    - Confirmed fills feed a temporary overlay for planning, not base-state mutation
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
        external_orderbook_ws: Optional[OrderbookWebSocket] = None,
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
        self._external_orderbook_ws = external_orderbook_ws
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
        self._fresh_start_started_at: Optional[float] = None

        # Lock for thread-safe tick execution
        self._tick_lock = asyncio.Lock()
        self.on_position_sync_warning: Optional[Callable[[PositionSyncWarningContext], None]] = None

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
        logger.info(f"[{self.event_name}] Setting up Kelly bot with {len(bins)} bins, ${initial_capital:.2f} capital")

        # Store bin metadata
        self.num_bins = len(bins)
        self.bin_upper_bounds = [b["upper_bound"] for b in bins]
        self.bin_lower_bounds = [b.get("lower_bound", 0) for b in bins]
        self.bin_token_ids = {i: b["token_id"] for i, b in enumerate(bins)}  # YES token IDs
        self.bin_no_token_ids = {i: b.get("no_token_id") for i, b in enumerate(bins)}  # NO token IDs

        # Initialize portfolio
        # Phantom capital is based on event budget (c_event_max), not initial_capital.
        # initial_capital may be a small restored allocation, but the event budget
        # is the true capital base Kelly should perceive.
        multiplier = self.config.collateral.capital_multiplier
        event_budget = self.config.collateral.c_event_max
        phantom = event_budget * (multiplier - 1.0) if multiplier > 1.0 else 0.0
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            capital=initial_capital,
            num_bins=self.num_bins,
            bin_upper_bounds=self.bin_upper_bounds,
            phantom_capital=phantom,
        )
        if phantom > 0:
            logger.info(f"[{self.event_name}] Phantom capital: ${phantom:.2f} (budget=${event_budget:.2f}, multiplier={multiplier}x)")

        # Initialize orderbook manager with WebSocket config
        ws_config = WebSocketConfig(enabled=not self._disable_websocket)
        self.orderbook_manager = OrderbookManager(
            config=ws_config,
            clob_client=self.clob_client,
            ws_client=self._external_orderbook_ws,
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
            logger.info(f"[{self.event_name}] Using external UserStreamClient for fill confirmations")
        elif not self.dry_run and self._api_key and self._api_secret:
            self.user_stream = UserStreamClient(
                api_key=self._api_key,
                api_secret=self._api_secret,
                api_passphrase=self._api_passphrase,
            )
            self._owns_user_stream = True  # We created it, we stop it
            # Start user stream
            await self.user_stream.start()
            logger.info(f"[{self.event_name}] User stream started for fill confirmations")
        else:
            self.user_stream = None
            self._owns_user_stream = False
            if not self.dry_run:
                logger.warning(f"[{self.event_name}] No API credentials for user stream - fill confirmations disabled")

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
        logger.info(f"[{self.event_name}] Kelly bot setup complete")

    async def shutdown(self) -> None:
        """Shutdown the Kelly trading bot."""
        logger.info(f"[{self.event_name}] Shutting down Kelly bot")
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
                f"[{self.event_name}] Order placed (pending fill): {result.candidate.action.value} "
                f"bin={result.candidate.bin_index} "
                f"size={result.candidate.size:.2f} @ {result.candidate.price:.4f}"
            )
        else:
            logger.warning(f"[{self.event_name}] Order placement failed: {result.error}")

    def _handle_fill(self, fill_event: FillEvent) -> None:
        """
        Handle fill confirmation from WebSocket.

        Forwards to Kelly executor which records the confirmed fill overlay.
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
        orderbooks: Optional[Dict[int, UnifiedOrderbook]] = None,
    ) -> None:
        """
        Update probability estimates and identify dead bins.

        Args:
            current_count: Current tweet count from xtracker
            hours_elapsed: Hours since counting period started
            hours_remaining: Hours until settlement
            orderbooks: Optional live orderbooks for market-aware shrinkage
        """
        if not self._setup_complete:
            raise RuntimeError("Bot not setup. Call setup() first.")

        # Identify dead bins (upper_bound < current_count)
        dead_bins = identify_dead_bins(self.bin_upper_bounds, current_count)
        self.portfolio.dead_bins = dead_bins

        # Only log when dead bins change (avoid spam)
        if dead_bins != self._last_logged_dead_bins:
            if dead_bins:
                logger.info(f"[{self.event_name}] Dead bins (count={current_count}): {dead_bins}")
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

        probabilities, blend_context = compute_market_consensus_blend(
            probabilities=probabilities,
            dead_bins=dead_bins,
            orderbooks=orderbooks,
            consensus_config=self.config.market_consensus,
            hours_remaining=hours_remaining,
        )
        if blend_context is not None:
            logger.debug(
                "[%s] Market consensus blend applied: alpha=%.3f coverage=%.2f avg_spread=%.3f gap=%.3f",
                self.event_name,
                blend_context["alpha"],
                blend_context["coverage_ratio"],
                blend_context["avg_spread"],
                blend_context["gap"],
            )

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
            if (
                self._fresh_start_started_at is None and
                getattr(self.config, "market_impact", None) is not None and
                self.config.market_impact.fresh_start_enabled
            ):
                self._fresh_start_started_at = time.time()

            # NOTE: Portfolio sync is now handled by the executor BEFORE EACH order decision
            # The executor calls sync_portfolio callback before each generate_candidates() call
            # This ensures we always use official API data, not calculated estimates

            # Fetch fresh orderbooks from REST API for all LIVE bins (skip dead bins).
            # Always fetch from API rather than relying solely on WebSocket cache,
            # which can become stale and cause trades to be systematically blocked
            # until restart (when fresh API data is fetched).
            dead_bins = set(identify_dead_bins(self.bin_upper_bounds, current_count))
            orderbooks = {}
            for bin_idx, token_id in self.bin_token_ids.items():
                if bin_idx in dead_bins:
                    continue  # Skip dead bins - no need to fetch orderbooks
                ob = await self.orderbook_manager.fetch_orderbook(token_id, bin_idx)
                if ob:
                    orderbooks[bin_idx] = ob

            # Update probabilities (this also identifies dead bins)
            self.update_probabilities(
                current_count,
                hours_elapsed,
                hours_to_settlement,
                orderbooks=orderbooks,
            )

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
                fresh_start_started_at=self._fresh_start_started_at,
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
        logger.info(f"[{self.event_name}] Starting continuous Kelly optimization")

        while self._running:
            try:
                # Get current market state
                state = get_market_state()

                current_count = state.get("current_count", 0)
                hours_elapsed = state.get("hours_elapsed", 0)
                hours_to_settlement = state.get("hours_to_settlement", 0)

                # Check T_stop
                if hours_to_settlement <= self.config.t_stop_hours:
                    logger.info(f"[{self.event_name}] Reached T_stop. Holding positions to settlement.")
                    break

                # Run tick
                result = await self.run_tick(
                    current_count,
                    hours_elapsed,
                    hours_to_settlement,
                )

                logger.info(
                    f"[{self.event_name}] Tick: candidates={result.num_candidates}, "
                    f"executed={result.num_executed}, "
                    f"utility_gain={result.total_utility_gain:.6f}"
                )

                # Wait for next tick
                await asyncio.sleep(tick_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.event_name}] Error in tick: {e}")
                await asyncio.sleep(tick_interval_seconds)

        self._running = False
        logger.info(f"[{self.event_name}] Continuous optimization stopped")

    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary."""
        if not self.portfolio:
            return {}
        return self.portfolio.to_summary()

    async def enforce_integrity_deadline(self) -> bool:
        """
        Enforce a pending integrity deadline outside the normal trading loop.

        Returns True when a frozen event was recovered or cleared.
        """
        if not self._setup_complete or not self.kelly_executor:
            return False

        if self._tick_lock.locked():
            return False

        async with self._tick_lock:
            return await self.kelly_executor.enforce_integrity_deadline()

    def get_position_sync_summary(self) -> Dict:
        """Summarize side-local positions API ambiguities currently being retained."""
        summary = {
            "warning_count": 0,
            "warnings": [],
        }
        if not self.portfolio:
            return summary

        now_ts = time.time()
        for bin_idx, pos in sorted(self.portfolio.positions.items()):
            if pos.yes_api_missing_unverified and pos.yes_shares > 0.01:
                summary["warnings"].append(
                    {
                        "bin_index": bin_idx,
                        "side": "YES",
                        "shares": pos.yes_shares,
                        "missing_since": (
                            datetime.fromtimestamp(pos.yes_api_missing_since, timezone.utc).isoformat()
                            if pos.yes_api_missing_since > 0 else None
                        ),
                        "age_seconds": (
                            max(0.0, now_ts - pos.yes_api_missing_since)
                            if pos.yes_api_missing_since > 0 else None
                        ),
                        "missing_count": pos.yes_api_missing_count,
                    }
                )
            if pos.no_api_missing_unverified and pos.no_shares > 0.01:
                summary["warnings"].append(
                    {
                        "bin_index": bin_idx,
                        "side": "NO",
                        "shares": pos.no_shares,
                        "missing_since": (
                            datetime.fromtimestamp(pos.no_api_missing_since, timezone.utc).isoformat()
                            if pos.no_api_missing_since > 0 else None
                        ),
                        "age_seconds": (
                            max(0.0, now_ts - pos.no_api_missing_since)
                            if pos.no_api_missing_since > 0 else None
                        ),
                        "missing_count": pos.no_api_missing_count,
                    }
                )

        summary["warning_count"] = len(summary["warnings"])
        return summary

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
            "integrity": {
                "frozen": False,
                "reason": None,
                "frozen_at": None,
                "deadline_at": None,
                "overlay_entries": 0,
                "oldest_overlay_age_seconds": None,
                "last_forced_api_recovery_at": None,
                "unmatched_api_delta_count": 0,
            },
            "pending_orders": {
                "count": 0,
                "collateral": 0.0,
            },
            "position_sync": {
                "warning_count": 0,
                "warnings": [],
            },
            "user_stream": {
                "connected": False,
                "pending_tracked": 0,
            },
        }

        if self.kelly_executor:
            status["pending_orders"]["count"] = self.kelly_executor.get_pending_count()
            status["pending_orders"]["collateral"] = self.kelly_executor.get_pending_collateral()
            status["integrity"] = self.kelly_executor.get_integrity_summary()
        status["position_sync"] = self.get_position_sync_summary()

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
            logger.warning(
                f"[{self.event_name}] Failed to fetch positions from API: {e}",
                exc_info=True,
            )
            raise

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
            logger.warning(f"[{self.event_name}] Failed to fetch USDC balance: {e}")
            return 0.0

    @staticmethod
    def _parse_conditional_balance_shares(balance_info: dict) -> Optional[float]:
        """Parse conditional token balance from get_balance_allowance response."""
        if not isinstance(balance_info, dict):
            return None
        raw_balance = balance_info.get("balance")
        try:
            return float(raw_balance) / 1e6
        except (TypeError, ValueError):
            return None

    def _fetch_conditional_balance_shares(self, token_id: str, refresh: bool = True) -> Optional[float]:
        """Fetch conditional token balance for a specific YES/NO token."""
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

            params = BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL,
                token_id=token_id,
            )
            if refresh:
                try:
                    self.clob_client.update_balance_allowance(params)
                except Exception as e:
                    logger.debug(
                        f"[{self.event_name}] Conditional balance refresh failed for token "
                        f"{token_id[:16]}...: {e}"
                    )
            balance_info = self.clob_client.get_balance_allowance(params)
            return self._parse_conditional_balance_shares(balance_info)
        except Exception as e:
            logger.warning(
                f"[{self.event_name}] Failed to fetch conditional balance for token "
                f"{token_id[:16]}...: {e}"
            )
            return None

    def _side_attr_names(self, is_no: bool) -> Tuple[str, str, str, str, str]:
        """Return attribute names and log label for a position side."""
        if is_no:
            return (
                "no_shares",
                "no_avg_cost",
                "no_unpriced_shares",
                "no_unpriced_reserve",
                "NO",
            )
        return (
            "yes_shares",
            "yes_avg_cost",
            "yes_unpriced_shares",
            "yes_unpriced_reserve",
            "YES",
        )

    def _side_missing_attr_names(self, is_no: bool) -> Tuple[str, str, str]:
        """Return api-missing verification attribute names for one side."""
        if is_no:
            return (
                "no_api_missing_unverified",
                "no_api_missing_since",
                "no_api_missing_count",
            )
        return (
            "yes_api_missing_unverified",
            "yes_api_missing_since",
            "yes_api_missing_count",
        )

    def _get_side_missing_state(self, pos, is_no: bool) -> Tuple[bool, float, int]:
        """Read api-missing verification state for one side."""
        active_attr, since_attr, count_attr = self._side_missing_attr_names(is_no)
        return (
            bool(getattr(pos, active_attr)),
            float(getattr(pos, since_attr)),
            int(getattr(pos, count_attr)),
        )

    def _set_side_missing_state(
        self,
        pos,
        *,
        is_no: bool,
        active: bool,
        now_ts: Optional[float] = None,
    ) -> Tuple[bool, float, int]:
        """Write api-missing verification state for one side."""
        active_attr, since_attr, count_attr = self._side_missing_attr_names(is_no)
        previous_active = bool(getattr(pos, active_attr))
        previous_since = float(getattr(pos, since_attr))
        previous_count = int(getattr(pos, count_attr))
        if active:
            if now_ts is None:
                now_ts = time.time()
            since = previous_since if previous_active and previous_since > 0 else now_ts
            count = previous_count + 1 if previous_active else 1
            setattr(pos, active_attr, True)
            setattr(pos, since_attr, since)
            setattr(pos, count_attr, count)
            return True, since, count

        setattr(pos, active_attr, False)
        setattr(pos, since_attr, 0.0)
        setattr(pos, count_attr, 0)
        return False, 0.0, 0

    def _get_side_state(self, pos, is_no: bool) -> Tuple[float, float, float, float]:
        """Read total shares, avg cost, unresolved shares, and unresolved reserve."""
        shares_attr, avg_attr, unpriced_attr, reserve_attr, _ = self._side_attr_names(is_no)
        return (
            getattr(pos, shares_attr),
            getattr(pos, avg_attr),
            getattr(pos, unpriced_attr),
            getattr(pos, reserve_attr),
        )

    def _set_side_state(
        self,
        pos,
        *,
        is_no: bool,
        shares: float,
        avg_cost: float,
        unpriced_shares: float = 0.0,
        unpriced_reserve: float = 0.0,
    ) -> None:
        """Write side-local pricing state and refresh collateral."""
        shares_attr, avg_attr, unpriced_attr, reserve_attr, _ = self._side_attr_names(is_no)

        shares = max(0.0, shares)
        unpriced_shares = min(max(0.0, unpriced_shares), shares)
        unpriced_reserve = max(0.0, unpriced_reserve)
        if unpriced_shares <= 0.01:
            unpriced_shares = 0.0
            unpriced_reserve = 0.0
        priced_shares = max(0.0, shares - unpriced_shares)
        if priced_shares <= 0.01:
            avg_cost = 0.0

        setattr(pos, shares_attr, shares)
        setattr(pos, avg_attr, avg_cost)
        setattr(pos, unpriced_attr, unpriced_shares)
        setattr(pos, reserve_attr, unpriced_reserve)
        if shares <= 0.01:
            self._set_side_missing_state(pos, is_no=is_no, active=False)
        pos.recompute_collateral_used()

    def _preserve_local_side_state(
        self,
        pos,
        *,
        is_no: bool,
        api_shares: float,
    ) -> str:
        """
        Preserve local side pricing when API shares are unchanged or reduced.

        Unresolved shares are consumed first so same-side add blocks clear as
        soon as the ambiguous increment shrinks away.
        """
        current_shares, current_avg_cost, current_unpriced, current_reserve = self._get_side_state(pos, is_no)
        epsilon = 0.01
        api_shares = max(0.0, api_shares)

        if api_shares <= epsilon:
            self._set_side_state(pos, is_no=is_no, shares=0.0, avg_cost=0.0)
            return "position_closed"

        if api_shares >= current_shares - epsilon:
            self._set_side_state(
                pos,
                is_no=is_no,
                shares=api_shares,
                avg_cost=current_avg_cost,
                unpriced_shares=current_unpriced,
                unpriced_reserve=current_reserve,
            )
            return "local_preserve"

        reduction = current_shares - api_shares
        unpriced_removed = min(reduction, current_unpriced)
        reserve_removed = (
            current_reserve * (unpriced_removed / current_unpriced)
            if current_unpriced > epsilon
            else 0.0
        )
        remaining_unpriced = max(0.0, current_unpriced - unpriced_removed)
        remaining_reserve = max(0.0, current_reserve - reserve_removed)
        priced_shares_before = max(0.0, current_shares - current_unpriced)
        priced_removed = max(0.0, reduction - unpriced_removed)
        priced_shares_after = max(0.0, priced_shares_before - priced_removed)
        avg_cost = current_avg_cost if priced_shares_after > epsilon else 0.0

        self._set_side_state(
            pos,
            is_no=is_no,
            shares=api_shares,
            avg_cost=avg_cost,
            unpriced_shares=remaining_unpriced,
            unpriced_reserve=remaining_reserve,
        )
        return "local_preserve_reduction"

    def _sync_position_side_from_balance(
        self,
        *,
        pos,
        is_no: bool,
        verified_shares: float,
    ) -> str:
        """
        Apply a conditional-balance verified share count while preserving local pricing.

        Positive balance means the side still exists even if the positions API omitted it.
        We preserve existing cost basis and only mark extra shares as unresolved if the
        verified balance is somehow larger than our local position.
        """
        current_shares, current_avg_cost, current_unpriced, current_reserve = self._get_side_state(pos, is_no)
        epsilon = 0.01
        verified_shares = max(0.0, verified_shares)

        if verified_shares <= epsilon:
            self._set_side_state(pos, is_no=is_no, shares=0.0, avg_cost=0.0)
            return "verified_zero"

        if verified_shares <= current_shares + epsilon:
            return self._preserve_local_side_state(pos, is_no=is_no, api_shares=verified_shares)

        delta = verified_shares - current_shares
        self._set_side_state(
            pos,
            is_no=is_no,
            shares=verified_shares,
            avg_cost=current_avg_cost,
            unpriced_shares=current_unpriced + delta,
            unpriced_reserve=current_reserve + delta,
        )
        return "balance_increase_unpriced"

    def _handle_missing_api_side(
        self,
        *,
        pos,
        bin_idx: int,
        is_no: bool,
        token_id: Optional[str],
    ) -> str:
        """
        Handle a side that is absent from the positions API but still exists locally.

        We do not clear immediately. First verify with conditional token balance; only
        a verified zero clears the side. Otherwise keep the side locally so conflicting
        opposite-side trades stay blocked while still permitting risk-reducing sells.
        """
        shares_attr, avg_attr, unpriced_attr, reserve_attr, side_label = self._side_attr_names(is_no)
        local_shares = getattr(pos, shares_attr)
        local_avg_cost = getattr(pos, avg_attr)
        local_unpriced = getattr(pos, unpriced_attr)
        local_reserve = getattr(pos, reserve_attr)
        was_missing, missing_since, missing_count = self._get_side_missing_state(pos, is_no)
        now_ts = time.time()

        verified_balance = None
        if token_id:
            verified_balance = self._fetch_conditional_balance_shares(token_id=token_id, refresh=True)

        if verified_balance is not None and verified_balance <= 0.01:
            logger.warning(
                f"[{self.event_name}] API sync: bin {bin_idx} {side_label} missing from positions API "
                f"and conditional balance verified zero; clearing {local_shares:.2f} -> 0"
            )
            self._set_side_state(pos, is_no=is_no, shares=0.0, avg_cost=0.0)
            if was_missing:
                logger.info(
                    f"[{self.event_name}] API sync: bin {bin_idx} {side_label} missing-side ambiguity cleared "
                    f"(verified zero balance after {missing_count} missing snapshots)"
                )
            self._set_side_missing_state(pos, is_no=is_no, active=False)
            return "verified_zero"

        if verified_balance is not None and verified_balance > 0.01:
            resolve_source = self._sync_position_side_from_balance(
                pos=pos,
                is_no=is_no,
                verified_shares=verified_balance,
            )
            _, missing_since, missing_count = self._set_side_missing_state(
                pos,
                is_no=is_no,
                active=True,
                now_ts=now_ts,
            )
            logger.warning(
                f"[{self.event_name}] API sync: bin {bin_idx} {side_label} missing from positions API "
                f"but conditional balance still shows {verified_balance:.2f} shares; preserving local side "
                f"(source={resolve_source}, avg=${getattr(pos, avg_attr):.4f}, unpriced={getattr(pos, unpriced_attr):.2f}sh/${getattr(pos, reserve_attr):.2f})"
            )
            if self.on_position_sync_warning and not was_missing:
                try:
                    self.on_position_sync_warning(
                        PositionSyncWarningContext(
                            bin_index=bin_idx,
                            side=side_label,
                            token_id=token_id or "",
                            local_shares=getattr(pos, shares_attr),
                            verified_balance_shares=verified_balance,
                            missing_since=missing_since,
                            missing_count=missing_count,
                            reason="positions_api_omitted_side_but_balance_positive",
                        )
                    )
                except Exception:
                    logger.exception(
                        f"[{self.event_name}] Failed to emit position sync warning callback for bin {bin_idx} {side_label}"
                    )
            return "verified_positive"

        _, missing_since, missing_count = self._set_side_missing_state(
            pos,
            is_no=is_no,
            active=True,
            now_ts=now_ts,
        )
        logger.warning(
            f"[{self.event_name}] API sync: bin {bin_idx} {side_label} missing from positions API "
            f"but conditional balance could not be verified; retaining local side {local_shares:.2f} @ ${local_avg_cost:.4f} "
            f"(unpriced={local_unpriced:.2f}sh/${local_reserve:.2f}, missing_count={missing_count})"
        )
        if self.on_position_sync_warning and not was_missing:
            try:
                self.on_position_sync_warning(
                    PositionSyncWarningContext(
                        bin_index=bin_idx,
                        side=side_label,
                        token_id=token_id or "",
                        local_shares=local_shares,
                        verified_balance_shares=None,
                        missing_since=missing_since,
                        missing_count=missing_count,
                        reason="positions_api_omitted_side_balance_unavailable",
                    )
                )
            except Exception:
                logger.exception(
                    f"[{self.event_name}] Failed to emit position sync warning callback for bin {bin_idx} {side_label}"
                )
        return "balance_unknown"

    def _sync_position_side_from_api(
        self,
        *,
        pos,
        bin_idx: int,
        is_no: bool,
        api_shares: float,
        api_avg_price: float,
        api_value: float,
    ) -> str:
        """Apply one API-reported side to local state and return the pricing source."""
        current_shares, current_avg_cost, current_unpriced, current_reserve = self._get_side_state(pos, is_no)
        epsilon = 0.01
        api_shares = max(0.0, api_shares)

        if api_avg_price > 0:
            self._set_side_state(pos, is_no=is_no, shares=api_shares, avg_cost=api_avg_price)
            return "avgPrice"

        if api_value > 0 and api_shares > epsilon:
            self._set_side_state(pos, is_no=is_no, shares=api_shares, avg_cost=api_value / api_shares)
            return "initialValue"

        delta = api_shares - current_shares
        if delta <= epsilon:
            return self._preserve_local_side_state(pos, is_no=is_no, api_shares=api_shares)

        priced_delta_shares = 0.0
        priced_delta_avg = 0.0
        if self.kelly_executor is not None:
            priced_delta_shares, priced_delta_avg = (
                self.kelly_executor.get_overlay_price_hint_for_api_increase(
                    bin_index=bin_idx,
                    is_no=is_no,
                    share_increase=delta,
                )
            )

        priced_base_shares = max(0.0, current_shares - current_unpriced)
        priced_base_notional = priced_base_shares * current_avg_cost
        priced_delta_notional = priced_delta_shares * priced_delta_avg
        total_priced_shares = priced_base_shares + priced_delta_shares
        total_avg_cost = (
            (priced_base_notional + priced_delta_notional) / total_priced_shares
            if total_priced_shares > epsilon
            else 0.0
        )

        unresolved_delta = max(0.0, delta - priced_delta_shares)
        total_unpriced_shares = current_unpriced + unresolved_delta
        total_unpriced_reserve = current_reserve + unresolved_delta

        self._set_side_state(
            pos,
            is_no=is_no,
            shares=api_shares,
            avg_cost=total_avg_cost,
            unpriced_shares=total_unpriced_shares,
            unpriced_reserve=total_unpriced_reserve,
        )

        if priced_delta_shares > epsilon and unresolved_delta <= epsilon:
            return "overlay"
        if priced_delta_shares > epsilon:
            return "overlay+unpriced_reserve"
        return "unpriced_reserve"

    async def sync_positions_from_api(self, wallet_address: str) -> Tuple[float, Dict[int, float]]:
        """
        Sync portfolio state from Polymarket API.

        The API is the authoritative source of truth for positions.
        This sync unconditionally overwrites local state with API data:
        - Updates positions to match API (both increases and decreases)
        - Clears positions the API no longer reports (sold/closed)
        - Recalculates collateral_used from synced positions

        Note: The API has some latency (seconds to minutes). Confirmed
        WebSocket fills are tracked separately in the executor as a temporary
        planning overlay, but this API snapshot remains the authoritative base.

        Args:
            wallet_address: The wallet address to sync positions for

        Returns:
            Tuple of (usdc_balance, {bin_index: shares})
        """
        if not self._setup_complete:
            raise RuntimeError("Bot must be setup before syncing positions")

        # Fetch current state from API. If the positions endpoint fails, abort
        # the sync entirely and preserve the last authoritative local snapshot.
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
            api_value = pos_info["value"]

            yes_token = self.bin_token_ids.get(bin_idx)
            pos = self.portfolio.ensure_position(bin_idx, yes_token or "")

            shares_attr, avg_attr, unpriced_attr, reserve_attr, side_label = self._side_attr_names(is_no)
            local_shares = getattr(pos, shares_attr)
            local_avg_cost = getattr(pos, avg_attr)
            local_unpriced = getattr(pos, unpriced_attr)
            local_reserve = getattr(pos, reserve_attr)
            was_uncertain = local_unpriced > 0.01
            previous_resolve_source = None
            was_missing_unverified, _, _ = self._get_side_missing_state(pos, is_no)

            # Update to API value if different
            if abs(api_shares - local_shares) > 0.01 or api_avg_price > 0 or api_value > 0:
                previous_resolve_source = self._sync_position_side_from_api(
                    pos=pos,
                    bin_idx=bin_idx,
                    is_no=is_no,
                    api_shares=api_shares,
                    api_avg_price=api_avg_price,
                    api_value=api_value,
                )

                resolved_avg = getattr(pos, avg_attr)
                current_unpriced = getattr(pos, unpriced_attr)
                current_reserve = getattr(pos, reserve_attr)
                if current_unpriced > 0.01:
                    logger.warning(
                        f"[{self.event_name}] API sync: bin {bin_idx} {side_label} updated "
                        f"{local_shares:.2f} -> {api_shares:.2f} with unresolved increment "
                        f"{current_unpriced:.2f}sh reserve=${current_reserve:.2f} "
                        f"(source={previous_resolve_source}, local_avg=${local_avg_cost:.4f})"
                    )
                else:
                    logger.info(
                        f"[{self.event_name}] API sync: bin {bin_idx} {side_label} updated "
                        f"{local_shares:.2f} -> {api_shares:.2f} @ ${resolved_avg:.4f} "
                        f"(source={previous_resolve_source})"
                    )

                if was_uncertain and current_unpriced <= 0.01:
                    logger.info(
                        f"[{self.event_name}] API sync: bin {bin_idx} {side_label} cost basis uncertainty cleared "
                        f"(source={previous_resolve_source})"
                    )
                elif not was_uncertain and current_unpriced > 0.01:
                    logger.warning(
                        f"[{self.event_name}] API sync: bin {bin_idx} {side_label} cost basis unresolved; "
                        f"blocking same-side adds until a priced sync arrives"
                    )
                elif was_uncertain and current_unpriced > 0.01 and (
                    abs(current_unpriced - local_unpriced) > 0.01
                    or abs(current_reserve - local_reserve) > 0.01
                ):
                    logger.info(
                        f"[{self.event_name}] API sync: bin {bin_idx} {side_label} unresolved increment adjusted "
                        f"{local_unpriced:.2f}sh/${local_reserve:.2f} -> {current_unpriced:.2f}sh/${current_reserve:.2f}"
                    )
            if was_missing_unverified:
                self._set_side_missing_state(pos, is_no=is_no, active=False)
                logger.info(
                    f"[{self.event_name}] API sync: bin {bin_idx} {side_label} missing-side ambiguity cleared "
                    "(positions API reports the side again)"
                )

            synced_positions[bin_idx] = api_shares

        # Second, verify positions that API doesn't report before clearing them.
        for bin_idx in tracked_bins:
            pos = self.portfolio.positions.get(bin_idx)
            if not pos:
                continue

            # Check YES position - if API doesn't have it and we do, verify before clearing
            if pos.yes_shares > 0.01 and (bin_idx, False) not in api_bin_positions:
                self._handle_missing_api_side(
                    pos=pos,
                    bin_idx=bin_idx,
                    is_no=False,
                    token_id=self.bin_token_ids.get(bin_idx),
                )

            # Check NO position - if API doesn't have it and we do, verify before clearing
            if pos.no_shares > 0.01 and (bin_idx, True) not in api_bin_positions:
                self._handle_missing_api_side(
                    pos=pos,
                    bin_idx=bin_idx,
                    is_no=True,
                    token_id=self.bin_no_token_ids.get(bin_idx),
                )

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
