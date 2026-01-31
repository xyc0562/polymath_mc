"""
Greedy executor for Kelly trading strategy.

Implements the greedy execution loop:
1. Generate trade candidates
2. Pick best candidate by utility gain
3. Execute trade
4. Update portfolio
5. Repeat until no profitable trades or max iterations

Order execution via py_clob_client.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

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

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of a single trade execution."""

    success: bool
    candidate: TradeCandidate
    order_id: Optional[str] = None
    filled_size: float = 0.0
    filled_price: float = 0.0
    error: Optional[str] = None


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
                order_type=OrderType.GTC,  # Good-til-cancelled
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
                filled_size=size,  # Assume fill for now
                filled_price=price,
            )
        else:
            return ExecutionResult(
                success=False,
                candidate=candidate,
                error="Order placement failed",
            )


class KellyExecutor:
    """
    Main Kelly optimizer executor.

    Runs the greedy optimization loop:
    1. Generate candidates
    2. Execute best candidate
    3. Update portfolio
    4. Repeat

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
        """
        self.config = config
        self.portfolio = portfolio
        self.orderbook_manager = orderbook_manager
        self.order_executor = order_executor
        self.token_ids = token_ids
        self.on_trade = on_trade

        # Execution state
        self._running = False
        self._last_tick_time = 0.0

        # Rate limiting state: track recent order timestamps
        self._order_timestamps: List[float] = []

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
            best = candidates[0]

            if best.utility_gain < self.config.tau:
                logger.debug(
                    f"Best candidate utility {best.utility_gain:.6f} "
                    f"< tau {self.config.tau}"
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

                # Update portfolio
                self._update_portfolio(best, token_id)

                # Callback
                if self.on_trade:
                    self.on_trade(result)

                logger.info(
                    f"Executed: {best.action.value} bin={best.bin_index} "
                    f"size={best.size:.2f} @ {best.price:.4f} "
                    f"utility_gain={best.utility_gain:.6f} edge={best.edge:.2%}"
                )

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

    def _update_portfolio(self, candidate: TradeCandidate, token_id: str) -> None:
        """Update portfolio after successful trade."""
        action = candidate.action
        bin_index = candidate.bin_index
        size = candidate.size
        price = candidate.price

        if action == TradeAction.BUY_YES:
            self.portfolio.execute_buy_yes(bin_index, size, price, token_id)
        elif action == TradeAction.SELL_YES:
            self.portfolio.execute_sell_yes(bin_index, size, price)
        elif action == TradeAction.BUY_NO:
            self.portfolio.execute_buy_no(bin_index, size, price, token_id)
        elif action == TradeAction.SELL_NO:
            self.portfolio.execute_sell_no(bin_index, size, price)

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
