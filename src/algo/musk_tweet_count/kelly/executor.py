"""
Greedy executor for Kelly trading strategy.

Implements the greedy execution loop:
1. Generate trade candidates
2. Pick best candidate by utility gain
3. Execute trade (place order)
4. Wait for fill confirmation via WebSocket
5. Update portfolio on confirmed fill
6. Repeat until no profitable trades or max iterations

Order execution via py_clob_client.
Portfolio updates happen ONLY on WebSocket fill confirmation, NOT on order placement.
This ensures portfolio state matches actual on-chain positions.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, TYPE_CHECKING

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType

from .config import KellyConfig
from .orderbook import UnifiedOrderbook
from .portfolio import Portfolio
from .candidates import (
    TradeCandidate,
    TradeAction,
    generate_candidates,
)
from .websocket_client import OrderbookManager

if TYPE_CHECKING:
    from .user_stream import UserStreamClient, FillEvent, PendingOrder, OrderStatus

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of a single trade execution (order placement, not fill)."""

    success: bool  # True if order was placed successfully (not yet filled)
    candidate: TradeCandidate
    order_id: Optional[str] = None
    # Note: filled_size and filled_price are 0 at order placement
    # They are updated when fill confirmation arrives via WebSocket
    filled_size: float = 0.0
    filled_price: float = 0.0
    error: Optional[str] = None
    is_pending: bool = True  # True until fill confirmed


@dataclass
class TickResult:
    """Result of a single optimization tick."""

    num_candidates: int
    num_executed: int
    total_utility_gain: float
    executions: List[ExecutionResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class OrderExecutor:
    """
    Handles order execution via Polymarket CLOB.

    Supports both YES and NO token trading via the unified orderbook.
    """

    def __init__(
        self,
        clob_client: ClobClient,
        dry_run: bool = False,
    ):
        """
        Initialize order executor.

        Args:
            clob_client: Authenticated ClobClient
            dry_run: If True, simulate trades without execution
        """
        self.client = clob_client
        self.dry_run = dry_run

    def place_limit_order(
        self,
        token_id: str,
        side: str,  # "BUY" or "SELL"
        price: float,
        size: float,
    ) -> Optional[dict]:
        """
        Place a limit order.

        Args:
            token_id: Token ID (YES token for YES trades, or YES token for equivalent NO trades)
            side: "BUY" or "SELL"
            price: Limit price
            size: Order size in shares

        Returns:
            Order response dict or None on failure
        """
        if self.dry_run:
            logger.info(
                f"[DRY RUN] Would place {side} order: "
                f"token={token_id[:16]}..., price={price:.4f}, size={size:.2f}"
            )
            return {"order_id": "dry_run_order", "status": "simulated"}

        try:
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=side,
                order_type=OrderType.FAK,  # Fill and Kill (IOC) - allows partial fills, no stale orders
            )

            signed_order = self.client.create_order(order_args)
            response = self.client.post_order(signed_order)

            logger.info(
                f"Order placed: {side} {size:.2f} @ {price:.4f}, "
                f"order_id={response.get('orderID', 'unknown')}"
            )
            return response

        except Exception as e:
            logger.error(f"Order failed: {e}")
            return None

    def place_market_order(
        self,
        token_id: str,
        side: str,
        amount: float,
    ) -> Optional[dict]:
        """
        Place a market order.

        Args:
            token_id: Token ID
            side: "BUY" or "SELL"
            amount: Amount in USDC for buys, shares for sells

        Returns:
            Order response or None on failure
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would place market {side}: amount={amount:.2f}")
            return {"order_id": "dry_run_market", "status": "simulated"}

        try:
            response = self.client.create_market_order(
                token_id=token_id,
                side=side,
                amount=amount,
            )
            logger.info(f"Market order placed: {side} {amount:.2f}")
            return response

        except Exception as e:
            logger.error(f"Market order failed: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would cancel order: {order_id}")
            return True

        try:
            self.client.cancel(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    def execute_candidate(
        self,
        candidate: TradeCandidate,
        token_id: str,
    ) -> ExecutionResult:
        """
        Execute a trade candidate.

        Translates Kelly action to CLOB order.
        """
        action = candidate.action
        price = candidate.price
        size = candidate.size

        # Determine order side and token
        # For NO trades, we use the equivalent YES trade:
        # BUY NO @ X = SELL YES @ (1-X)
        # SELL NO @ X = BUY YES @ (1-X)

        if action == TradeAction.BUY_YES:
            side = "BUY"
            order_price = price
        elif action == TradeAction.SELL_YES:
            side = "SELL"
            order_price = price
        elif action == TradeAction.BUY_NO:
            # BUY NO @ X = SELL YES @ (1-X)
            side = "SELL"
            order_price = 1.0 - price
        elif action == TradeAction.SELL_NO:
            # SELL NO @ X = BUY YES @ (1-X)
            side = "BUY"
            order_price = 1.0 - price
        else:
            return ExecutionResult(
                success=False,
                candidate=candidate,
                error=f"Unknown action: {action}",
            )

        # Place order
        response = self.place_limit_order(
            token_id=token_id,
            side=side,
            price=order_price,
            size=size,
        )

        if response:
            return ExecutionResult(
                success=True,
                candidate=candidate,
                order_id=response.get("orderID"),
                # Don't assume fill - filled_size/price remain 0 until WebSocket confirmation
                filled_size=0.0,
                filled_price=0.0,
                is_pending=True,
            )
        else:
            return ExecutionResult(
                success=False,
                candidate=candidate,
                error="Order placement failed",
                is_pending=False,
            )


class KellyExecutor:
    """
    Main Kelly optimizer executor.

    Runs the greedy optimization loop:
    1. Generate candidates
    2. Execute best candidate (place order)
    3. Track as pending order (DO NOT update portfolio yet)
    4. Portfolio updates happen via fill callback when WebSocket confirms fill
    5. Repeat

    IMPORTANT: Portfolio is NOT updated on order placement. Updates happen only
    when fills are confirmed via the UserStreamClient WebSocket connection.

    Includes rate limiting to prevent runaway execution and respect API limits.
    """

    def __init__(
        self,
        config: KellyConfig,
        portfolio: Portfolio,
        orderbook_manager: OrderbookManager,
        order_executor: OrderExecutor,
        token_ids: Dict[int, str],  # bin_index -> YES token_id
        on_trade: Optional[Callable[[ExecutionResult], None]] = None,
        user_stream: Optional["UserStreamClient"] = None,
    ):
        """
        Initialize Kelly executor.

        Args:
            config: Kelly configuration
            portfolio: Portfolio state
            orderbook_manager: Orderbook manager (WebSocket or REST)
            order_executor: Order executor
            token_ids: Map of bin_index -> YES token_id
            on_trade: Optional callback for trade notifications
            user_stream: Optional UserStreamClient for fill confirmations
        """
        self.config = config
        self.portfolio = portfolio
        self.orderbook_manager = orderbook_manager
        self.order_executor = order_executor
        self.token_ids = token_ids
        self.on_trade = on_trade
        self.user_stream = user_stream

        # Reverse mapping: token_id -> bin_index (for fill callbacks)
        self.token_to_bin: Dict[str, int] = {v: k for k, v in token_ids.items()}

        # Execution state
        self._running = False
        self._last_tick_time = 0.0

        # Rate limiting state: track recent order timestamps
        self._order_timestamps: List[float] = []

        # Pending orders awaiting fill confirmation
        # Maps order_id -> (candidate, token_id)
        self._pending_orders: Dict[str, tuple[TradeCandidate, str]] = {}

        # Trade counter for logging
        self._trade_count: int = 0

        # Context for trade logging (set by caller before run_tick)
        self._log_context: Dict = {}

    def _check_rate_limit(self) -> bool:
        """
        Check if we're within rate limits.

        Returns:
            True if we can place another order, False if rate limited
        """
        now = time.time()
        rate_config = self.config.rate_limit

        # Clean up old timestamps (older than 1 minute)
        cutoff = now - 60.0
        self._order_timestamps = [ts for ts in self._order_timestamps if ts > cutoff]

        # Check orders per minute
        if len(self._order_timestamps) >= rate_config.max_orders_per_minute:
            logger.warning(
                f"Rate limit hit: {len(self._order_timestamps)} orders in last minute "
                f"(max: {rate_config.max_orders_per_minute})"
            )
            return False

        return True

    def _record_order(self) -> None:
        """Record an order timestamp for rate limiting."""
        self._order_timestamps.append(time.time())

    async def run_tick(
        self,
        hours_to_settlement: float,
    ) -> TickResult:
        """
        Run a single optimization tick.

        Generates candidates and executes trades until no more
        profitable trades or max iterations reached.

        Respects rate limits:
        - max_orders_per_tick: Stop after this many orders
        - min_order_delay_seconds: Wait between orders
        - max_orders_per_minute: Hard cap across ticks

        Args:
            hours_to_settlement: Hours until market settlement

        Returns:
            TickResult with execution summary
        """
        start_time = time.time()
        tick_result = TickResult(
            num_candidates=0,
            num_executed=0,
            total_utility_gain=0.0,
        )

        # Check T_stop
        if hours_to_settlement <= self.config.t_stop_hours:
            logger.info(
                f"Past T_stop ({self.config.t_stop_hours}h before settlement). "
                "Holding positions to settlement."
            )
            return tick_result

        # Get current orderbooks
        orderbooks = self._get_orderbooks()

        rate_config = self.config.rate_limit
        orders_this_tick = 0

        for iteration in range(self.config.max_iters_per_tick):
            # Check per-tick order limit
            if orders_this_tick >= rate_config.max_orders_per_tick:
                logger.info(
                    f"Reached max orders per tick ({rate_config.max_orders_per_tick})"
                )
                break

            # Check global rate limit
            if not self._check_rate_limit():
                logger.info("Rate limit reached, stopping tick early")
                break

            # Generate candidates
            candidates = generate_candidates(
                portfolio=self.portfolio,
                orderbooks=orderbooks,
                config=self.config,
                hours_to_settlement=hours_to_settlement,
            )

            if iteration == 0:
                tick_result.num_candidates = len(candidates)

            if not candidates:
                logger.debug(f"No candidates at iteration {iteration}")
                break

            # Get best candidate
            # Candidates are ordered: [sells..., buys sorted by utility]
            best = candidates[0]

            # Sells execute unconditionally (they meet fair value threshold by being generated)
            # Only check utility threshold for buys
            is_sell = best.action in (TradeAction.SELL_YES, TradeAction.SELL_NO)
            if not is_sell and best.utility_gain < self.config.tau:
                logger.debug(
                    f"Best buy candidate utility {best.utility_gain:.6f} "
                    f"< tau {self.config.tau}, stopping"
                )
                break

            # Execute trade
            token_id = self.token_ids.get(best.bin_index)
            if not token_id:
                logger.warning(f"No token_id for bin {best.bin_index}")
                continue

            result = self.order_executor.execute_candidate(best, token_id)
            tick_result.executions.append(result)

            if result.success:
                tick_result.num_executed += 1
                tick_result.total_utility_gain += best.utility_gain
                orders_this_tick += 1

                # Record for rate limiting
                self._record_order()

                # Track as pending order - DO NOT update portfolio yet
                # Portfolio update happens in handle_fill() when WebSocket confirms fill
                if result.order_id:
                    self._pending_orders[result.order_id] = (best, token_id)

                    # If user_stream is available, register the pending order
                    if self.user_stream:
                        from .user_stream import PendingOrder
                        pending = PendingOrder(
                            order_id=result.order_id,
                            token_id=token_id,
                            side="BUY" if best.action in (TradeAction.BUY_YES, TradeAction.SELL_NO) else "SELL",
                            price=best.price,
                            size=best.size,
                            bin_index=best.bin_index,
                        )
                        asyncio.create_task(self.user_stream.add_pending_order(pending))

                # Log detailed trade info
                self._log_trade_placed(best, token_id, result.order_id or "unknown")

                # Callback
                if self.on_trade:
                    self.on_trade(result)

                # Delay between orders (if more iterations expected)
                if iteration < self.config.max_iters_per_tick - 1:
                    await asyncio.sleep(rate_config.min_order_delay_seconds)
            else:
                logger.warning(f"Execution failed: {result.error}")
                break

            # Refresh orderbooks for next iteration
            orderbooks = self._get_orderbooks()

        tick_result.elapsed_seconds = time.time() - start_time
        self._last_tick_time = time.time()

        return tick_result

    def _get_orderbooks(self) -> Dict[int, UnifiedOrderbook]:
        """Get current orderbooks for all bins."""
        orderbooks = {}
        for bin_index, token_id in self.token_ids.items():
            ob = self.orderbook_manager.get_orderbook(token_id)
            if ob:
                orderbooks[bin_index] = ob
        return orderbooks

    def _update_portfolio(self, candidate: TradeCandidate, token_id: str, filled_size: float, filled_price: float) -> None:
        """
        Update portfolio after confirmed fill.

        Called from handle_fill() when WebSocket confirms trade execution.
        """
        action = candidate.action
        bin_index = candidate.bin_index
        # Use actual filled size/price, not the original candidate values
        size = filled_size
        price = filled_price

        if action == TradeAction.BUY_YES:
            self.portfolio.execute_buy_yes(bin_index, size, price, token_id)
        elif action == TradeAction.SELL_YES:
            self.portfolio.execute_sell_yes(bin_index, size, price)
        elif action == TradeAction.BUY_NO:
            self.portfolio.execute_buy_no(bin_index, size, price, token_id)
        elif action == TradeAction.SELL_NO:
            self.portfolio.execute_sell_no(bin_index, size, price)

    def handle_fill(self, fill_event: "FillEvent") -> None:
        """
        Handle fill confirmation from WebSocket.

        Called by UserStreamClient when a trade fill is confirmed.
        Updates portfolio with actual fill details.

        Args:
            fill_event: FillEvent from WebSocket
        """
        order_id = fill_event.order_id
        if not order_id:
            logger.warning("Received fill event with no order_id")
            return

        # Look up the pending order
        pending_info = self._pending_orders.get(order_id)
        if not pending_info:
            # This fill might be from a previous session or unknown order
            logger.info(f"Fill for unknown order {order_id[:16]}... - may be from previous session")
            return

        candidate, token_id = pending_info

        # Update portfolio with actual fill details
        self._update_portfolio(
            candidate=candidate,
            token_id=token_id,
            filled_size=fill_event.size,
            filled_price=fill_event.price,
        )

        # Log detailed fill info
        self._log_fill_confirmed(
            candidate=candidate,
            token_id=token_id,
            filled_size=fill_event.size,
            filled_price=fill_event.price,
            order_id=order_id,
        )

        # Check if order is fully filled
        from .user_stream import OrderStatus
        if fill_event.status in (OrderStatus.CONFIRMED, OrderStatus.MINED):
            # Order complete, remove from pending
            del self._pending_orders[order_id]
            logger.debug(f"Order {order_id[:16]}... completed and removed from pending")

    async def handle_stale_order(self, pending: "PendingOrder") -> None:
        """
        Handle stale order cancellation.

        Called by UserStreamClient when an order hasn't filled within timeout.
        Cancels the order to free up collateral.

        Args:
            pending: PendingOrder that is stale
        """
        order_id = pending.order_id

        logger.warning(
            f"Cancelling stale order: {order_id[:16]}..., "
            f"bin={pending.bin_index}, size={pending.size:.2f} @ {pending.price:.4f}"
        )

        # Cancel via order executor
        success = self.order_executor.cancel_order(order_id)

        if success:
            # Remove from our pending tracking
            if order_id in self._pending_orders:
                del self._pending_orders[order_id]
            logger.info(f"Stale order {order_id[:16]}... cancelled successfully")
        else:
            logger.error(f"Failed to cancel stale order {order_id[:16]}...")

    def get_pending_count(self) -> int:
        """Get count of pending orders awaiting fill."""
        return len(self._pending_orders)

    def get_pending_collateral(self) -> float:
        """
        Estimate collateral locked in pending orders.

        This is collateral that's committed but not yet reflected in portfolio.
        """
        total = 0.0
        for candidate, _ in self._pending_orders.values():
            if candidate.action in (TradeAction.BUY_YES, TradeAction.BUY_NO):
                total += candidate.price * candidate.size
        return total


    def set_log_context(
        self,
        probabilities: List[float] = None,
        orderbooks: Dict[int, "UnifiedOrderbook"] = None,
        bin_ranges: Dict[int, str] = None,
        current_count: int = 0,
        hours_to_settlement: float = 0,
        forecast_mean: float = 0,
        forecast_std: float = 0,
    ) -> None:
        """
        Set context for trade logging.

        Call this before run_tick() to provide context for detailed trade logs.
        """
        self._log_context = {
            "probabilities": probabilities or [],
            "orderbooks": orderbooks or {},
            "bin_ranges": bin_ranges or {},
            "current_count": current_count,
            "hours_to_settlement": hours_to_settlement,
            "forecast_mean": forecast_mean,
            "forecast_std": forecast_std,
        }

    def _log_trade_placed(self, candidate: TradeCandidate, token_id: str, order_id: str) -> None:
        """
        Log detailed trade info when order is placed (pending fill).

        Format: #N ACTION bin=X (range) | size @ price = $collateral | model=X% mkt=Y% edge=Z% | portfolio: $capital
        """
        from datetime import datetime

        self._trade_count += 1
        ctx = self._log_context

        # Get bin info
        bin_idx = candidate.bin_index
        bin_range = ctx.get("bin_ranges", {}).get(bin_idx, f"bin{bin_idx}")

        # Get probabilities
        probs = ctx.get("probabilities", [])
        model_prob = probs[bin_idx] if bin_idx < len(probs) else 0.0

        # Get market probability from orderbook
        orderbooks = ctx.get("orderbooks", {})
        ob = orderbooks.get(bin_idx)
        if candidate.action in (TradeAction.BUY_YES, TradeAction.SELL_YES):
            mkt_prob = ob.best_yes_ask if ob and ob.best_yes_ask else candidate.price
        else:
            mkt_prob = ob.best_no_ask if ob and ob.best_no_ask else candidate.price

        # Calculate collateral
        price = candidate.price
        size = candidate.size
        if candidate.action in (TradeAction.BUY_YES, TradeAction.BUY_NO):
            collateral = price * size
        else:
            collateral = (1 - price) * size  # Selling releases collateral

        # Calculate odds (potential return if correct)
        if candidate.action in (TradeAction.BUY_YES, TradeAction.BUY_NO):
            odds = (1.0 - price) / price if price > 0 else 0
        else:
            odds = price / (1.0 - price) if price < 1 else 0

        # Format action
        action = candidate.action.value

        # Portfolio state
        capital = self.portfolio.capital if self.portfolio else 0
        total_collateral = self.portfolio.total_collateral_used if self.portfolio else 0

        # Time info
        now = datetime.now().strftime("%H:%M:%S")
        hours_left = ctx.get("hours_to_settlement", 0)

        # Log the trade
        logger.info(
            f"[TRADE #{self._trade_count}] {now} | T-{hours_left:.1f}h | {action} bin={bin_idx} ({bin_range})"
        )
        logger.info(
            f"  → {size:.1f} shares @ {price:.3f} = ${collateral:.2f} | "
            f"model={model_prob:.1%} mkt={mkt_prob:.1%} edge={candidate.edge:+.1%} odds={odds:.1f}x"
        )
        logger.info(
            f"  → order_id={order_id[:16]}... | portfolio: ${capital:.2f} capital, ${total_collateral:.2f} collateral"
        )

    def _log_fill_confirmed(
        self,
        candidate: TradeCandidate,
        token_id: str,
        filled_size: float,
        filled_price: float,
        order_id: str,
    ) -> None:
        """
        Log when fill is confirmed via WebSocket.

        Shows actual fill details vs requested.
        """
        from datetime import datetime
        now = datetime.now().strftime("%H:%M:%S")

        # Calculate actual collateral
        if candidate.action in (TradeAction.BUY_YES, TradeAction.BUY_NO):
            collateral = filled_price * filled_size
        else:
            collateral = (1 - filled_price) * filled_size

        # Slippage
        price_diff = filled_price - candidate.price
        slippage_bps = abs(price_diff) * 10000

        # Portfolio state after fill
        capital = self.portfolio.capital if self.portfolio else 0
        total_collateral = self.portfolio.total_collateral_used if self.portfolio else 0

        logger.info(
            f"[FILL ✓] {now} | {candidate.action.value} bin={candidate.bin_index} | "
            f"{filled_size:.1f} @ {filled_price:.3f} (req: {candidate.size:.1f} @ {candidate.price:.3f}) | "
            f"slip={slippage_bps:.0f}bps"
        )
        logger.info(
            f"  → portfolio: ${capital:.2f} capital, ${total_collateral:.2f} collateral | "
            f"order={order_id[:16]}..."
        )

    async def run_continuous(
        self,
        get_hours_to_settlement: Callable[[], float],
        tick_interval_seconds: float = 60.0,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """
        Run continuous optimization loop.

        Args:
            get_hours_to_settlement: Callable returning hours to settlement
            tick_interval_seconds: Seconds between ticks
            stop_event: Optional event to signal stop
        """
        self._running = True
        logger.info("Starting continuous Kelly optimization")

        while self._running:
            if stop_event and stop_event.is_set():
                break

            try:
                hours = get_hours_to_settlement()

                # Stop if past T_stop
                if hours <= self.config.t_stop_hours:
                    logger.info("Reached T_stop. Stopping optimization.")
                    break

                # Run tick
                result = await self.run_tick(hours)

                logger.info(
                    f"Tick complete: candidates={result.num_candidates}, "
                    f"executed={result.num_executed}, "
                    f"utility_gain={result.total_utility_gain:.6f}, "
                    f"elapsed={result.elapsed_seconds:.2f}s"
                )

                # Wait for next tick
                await asyncio.sleep(tick_interval_seconds)

            except Exception as e:
                logger.error(f"Error in tick: {e}")
                await asyncio.sleep(tick_interval_seconds)

        self._running = False
        logger.info("Kelly optimization stopped")

    def stop(self) -> None:
        """Signal executor to stop."""
        self._running = False

    def get_portfolio_summary(self) -> dict:
        """Get current portfolio summary."""
        return self.portfolio.to_summary()


class UnifiedKellyExecutor:
    """
    Backend-agnostic Kelly executor that works with abstract interfaces.

    This executor can be used for both live trading and backtesting by
    swapping out the backend implementations:
    - Live: Use LiveOrderbookProvider, LiveTradeExecutor
    - Backtest: Use BacktestOrderbookProvider, BacktestTradeExecutor

    The trading logic (candidate generation, utility calculation) is
    identical regardless of backend.
    """

    def __init__(
        self,
        config: KellyConfig,
        portfolio: Portfolio,
        orderbook_provider,  # OrderbookProvider (from backend.py)
        trade_executor,  # TradeExecutor (from backend.py)
        token_ids: Dict[int, str],
        on_trade: Optional[Callable[["ExecutionResult"], None]] = None,
    ):
        """
        Initialize unified Kelly executor.

        Args:
            config: Kelly configuration
            portfolio: Portfolio state
            orderbook_provider: Abstract orderbook provider
            trade_executor: Abstract trade executor
            token_ids: Map of bin_index -> YES token_id
            on_trade: Optional callback for trade notifications
        """
        self.config = config
        self.portfolio = portfolio
        self.orderbook_provider = orderbook_provider
        self.trade_executor = trade_executor
        self.token_ids = token_ids
        self.on_trade = on_trade

        # Rate limiting state (used for live trading, no-op for backtest)
        self._order_timestamps: List[float] = []

    def run_tick_sync(
        self,
        hours_to_settlement: float,
    ) -> TickResult:
        """
        Run a single optimization tick (synchronous version).

        This is the main entry point for backtesting where we don't need
        async delays between orders.

        Args:
            hours_to_settlement: Hours until market settlement

        Returns:
            TickResult with execution summary
        """
        start_time = time.time()
        tick_result = TickResult(
            num_candidates=0,
            num_executed=0,
            total_utility_gain=0.0,
        )

        # Check T_stop
        if hours_to_settlement <= self.config.t_stop_hours:
            logger.debug(
                f"Past T_stop ({self.config.t_stop_hours}h before settlement). "
                "Holding positions to settlement."
            )
            return tick_result

        # Get current orderbooks from provider
        orderbooks = self.orderbook_provider.get_all_orderbooks()

        if not orderbooks:
            logger.debug("No orderbooks available")
            return tick_result

        rate_config = self.config.rate_limit
        orders_this_tick = 0

        for iteration in range(self.config.max_iters_per_tick):
            # Check per-tick order limit
            if orders_this_tick >= rate_config.max_orders_per_tick:
                logger.debug(
                    f"Reached max orders per tick ({rate_config.max_orders_per_tick})"
                )
                break

            # Generate candidates using production Kelly logic
            candidates = generate_candidates(
                portfolio=self.portfolio,
                orderbooks=orderbooks,
                config=self.config,
                hours_to_settlement=hours_to_settlement,
            )

            if iteration == 0:
                tick_result.num_candidates = len(candidates)

            if not candidates:
                logger.debug(f"No candidates at iteration {iteration}")
                break

            # Get best candidate
            # Candidates are ordered: [sells..., buys sorted by utility]
            best = candidates[0]

            # Sells execute unconditionally, only check utility for buys
            is_sell = best.action in (TradeAction.SELL_YES, TradeAction.SELL_NO)
            if not is_sell and best.utility_gain < self.config.tau:
                logger.debug(
                    f"Best buy candidate utility {best.utility_gain:.6f} "
                    f"< tau {self.config.tau}, stopping"
                )
                break

            # Execute trade through abstract executor
            token_id = self.token_ids.get(best.bin_index, f"token_{best.bin_index}")

            # Import here to avoid circular imports
            from .backend import ExecutionResult as BackendExecutionResult

            backend_result = self.trade_executor.execute(best, token_id)

            # Convert to TickResult format
            result = ExecutionResult(
                success=backend_result.success,
                candidate=best,
                order_id=backend_result.order_id,
                filled_size=backend_result.filled_size,
                filled_price=backend_result.filled_price,
                error=backend_result.error,
            )

            tick_result.executions.append(result)

            if result.success:
                tick_result.num_executed += 1
                tick_result.total_utility_gain += best.utility_gain
                orders_this_tick += 1

                # Record for rate limiting (live trading)
                self._order_timestamps.append(time.time())

                # Callback
                if self.on_trade:
                    self.on_trade(result)

                logger.debug(
                    f"Executed: {best.action.value} bin={best.bin_index} "
                    f"size={best.size:.2f} @ {best.price:.4f} "
                    f"utility_gain={best.utility_gain:.6f} edge={best.edge:.2%}"
                )
            else:
                logger.warning(f"Execution failed: {result.error}")
                break

            # Refresh orderbooks for next iteration
            self.orderbook_provider.refresh()
            orderbooks = self.orderbook_provider.get_all_orderbooks()

        tick_result.elapsed_seconds = time.time() - start_time
        return tick_result

    async def run_tick(
        self,
        hours_to_settlement: float,
    ) -> TickResult:
        """
        Run a single optimization tick (async version for live trading).

        Includes rate limiting delays between orders.
        """
        start_time = time.time()
        tick_result = TickResult(
            num_candidates=0,
            num_executed=0,
            total_utility_gain=0.0,
        )

        # Check T_stop
        if hours_to_settlement <= self.config.t_stop_hours:
            logger.info(
                f"Past T_stop ({self.config.t_stop_hours}h before settlement). "
                "Holding positions to settlement."
            )
            return tick_result

        orderbooks = self.orderbook_provider.get_all_orderbooks()

        if not orderbooks:
            return tick_result

        rate_config = self.config.rate_limit
        orders_this_tick = 0

        for iteration in range(self.config.max_iters_per_tick):
            if orders_this_tick >= rate_config.max_orders_per_tick:
                logger.info(
                    f"Reached max orders per tick ({rate_config.max_orders_per_tick})"
                )
                break

            # Check rate limit
            if not self._check_rate_limit(rate_config):
                logger.info("Rate limit reached, stopping tick early")
                break

            candidates = generate_candidates(
                portfolio=self.portfolio,
                orderbooks=orderbooks,
                config=self.config,
                hours_to_settlement=hours_to_settlement,
            )

            if iteration == 0:
                tick_result.num_candidates = len(candidates)

            if not candidates:
                break

            # Candidates are ordered: [sells..., buys sorted by utility]
            best = candidates[0]

            # Sells execute unconditionally, only check utility for buys
            is_sell = best.action in (TradeAction.SELL_YES, TradeAction.SELL_NO)
            if not is_sell and best.utility_gain < self.config.tau:
                break

            token_id = self.token_ids.get(best.bin_index, f"token_{best.bin_index}")

            from .backend import ExecutionResult as BackendExecutionResult

            backend_result = self.trade_executor.execute(best, token_id)

            result = ExecutionResult(
                success=backend_result.success,
                candidate=best,
                order_id=backend_result.order_id,
                filled_size=backend_result.filled_size,
                filled_price=backend_result.filled_price,
                error=backend_result.error,
            )

            tick_result.executions.append(result)

            if result.success:
                tick_result.num_executed += 1
                tick_result.total_utility_gain += best.utility_gain
                orders_this_tick += 1

                self._order_timestamps.append(time.time())

                if self.on_trade:
                    self.on_trade(result)

                logger.info(
                    f"Executed: {best.action.value} bin={best.bin_index} "
                    f"size={best.size:.2f} @ {best.price:.4f} "
                    f"utility_gain={best.utility_gain:.6f} edge={best.edge:.2%}"
                )

                # Delay between orders for live trading
                if iteration < self.config.max_iters_per_tick - 1:
                    await asyncio.sleep(rate_config.min_order_delay_seconds)
            else:
                logger.warning(f"Execution failed: {result.error}")
                break

            self.orderbook_provider.refresh()
            orderbooks = self.orderbook_provider.get_all_orderbooks()

        tick_result.elapsed_seconds = time.time() - start_time
        return tick_result

    def _check_rate_limit(self, rate_config) -> bool:
        """Check if we're within rate limits."""
        now = time.time()
        cutoff = now - 60.0
        self._order_timestamps = [ts for ts in self._order_timestamps if ts > cutoff]

        if len(self._order_timestamps) >= rate_config.max_orders_per_minute:
            return False

        return True

    def get_portfolio_summary(self) -> dict:
        """Get current portfolio summary."""
        return self.portfolio.to_summary()
