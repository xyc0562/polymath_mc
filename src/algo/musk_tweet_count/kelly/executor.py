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
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, TYPE_CHECKING

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType, PostOrdersArgs

from .config import KellyConfig
from .orderbook import UnifiedOrderbook
from .portfolio import Portfolio
from .candidates import (
    TradeCandidate,
    TradeAction,
    generate_candidates,
    MIN_ORDER_SIZE,
    MIN_ORDER_VALUE_USD,
)
from .websocket_client import OrderbookManager

if TYPE_CHECKING:
    from .user_stream import UserStreamClient, FillEvent, PendingOrder, OrderStatus

logger = logging.getLogger(__name__)


def _has_more_than_2dp(value: float) -> bool:
    """Check if a float has more than 2 decimal places."""
    return abs(value - round(value, 2)) > 1e-9


def _fak_size_step(price: float) -> int:
    """
    Compute the minimum size step for FAK orders at a given price.

    FAK orders require maker_amount (size * price) to have <= 2 decimal places.
    In USDC microdollars (6 decimals), this means size * price_micros must be
    divisible by 10000 (since $0.01 = 10000 microdollars).

    Returns the minimum size increment that satisfies this constraint.
    """
    price_micros = round(price * 1_000_000)
    g = math.gcd(price_micros, 10_000)
    return 10_000 // g


def _round_to_fak_size(size: int, price: float, round_up: bool = False) -> int:
    """
    Round a size to the nearest valid FAK size (where size * price has <= 2dp).

    If round_up=True, rounds up; otherwise rounds down.
    """
    step = _fak_size_step(price)
    if step <= 1:
        return size
    if round_up:
        return math.ceil(size / step) * step
    else:
        return (size // step) * step


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
        event_name: Optional[str] = None,
    ):
        """
        Initialize order executor.

        Args:
            clob_client: Authenticated ClobClient
            dry_run: If True, simulate trades without execution
            event_name: Optional event name for logging context
        """
        self.client = clob_client
        self.dry_run = dry_run
        self.event_name = event_name or "unknown"
        self._last_error: Optional[str] = None  # Store last error message for cooldown logic
        # Bin ranges for logging (set by KellyExecutor before each tick)
        self.bin_ranges: Dict[int, str] = {}

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
            # Let py-clob-client handle price rounding based on market tick size.
            # The library fetches tick_size per token (0.01, 0.001, or 0.0001)
            # and rounds price/amounts accordingly in create_order().
            rounded_size = math.floor(size)  # Integer shares

            # Basic price validation (library also validates against tick_size range)
            if price <= 0 or price >= 1:
                logger.warning(f"Invalid price: {price:.4f}")
                return None
            if side == "SELL":
                if rounded_size < 1:
                    logger.warning(f"Sell size {rounded_size} below minimum 1 share")
                    return None
            else:
                if rounded_size < MIN_ORDER_SIZE and rounded_size * price < MIN_ORDER_VALUE_USD:
                    logger.warning(f"Buy size {rounded_size} below {MIN_ORDER_SIZE} shares and value ${rounded_size * price:.2f} below ${MIN_ORDER_VALUE_USD}")
                    return None

            logger.info(
                f"[{self.event_name}][ORDER PARAMS] {side} size={rounded_size:.0f} @ {price:.4f} "
                f"maker_amt={rounded_size * price:.4f} (raw_size: {size:.4f})"
            )

            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=rounded_size,
                side=side,
            )

            signed_order = self.client.create_order(order_args)
            logger.debug(f"Signed order created: {type(signed_order)}")

            # FAK = Fill and Kill (IOC) - allows partial fills, cancels unfilled remainder
            response = self.client.post_order(signed_order, orderType=OrderType.FAK)

            logger.info(
                f"Order placed: {side} {rounded_size:.0f} @ {price:.4f}, "
                f"order_id={response.get('orderID', 'unknown')}"
            )
            return response

        except Exception as e:
            logger.error(f"Order failed: {e}")
            self._last_error = str(e)
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

    def place_batch_orders(
        self,
        orders: List[dict],
    ) -> List[dict]:
        """
        Place multiple orders via batch API.

        Each order dict has keys: token_id, side, price, size, plus optional metadata.

        Args:
            orders: List of dicts with {token_id, side, price, size}.

        Returns:
            Response from batch API (list of order results), or
            simulated responses in dry-run mode.
        """
        if self.dry_run:
            results = []
            for i, order in enumerate(orders):
                rounded_size = math.floor(order["size"])
                logger.info(
                    f"[{self.event_name}][DRY RUN] Would place {order['side']} order: "
                    f"token={order['token_id'][:16]}..., price={order['price']:.4f}, "
                    f"size={rounded_size}"
                )
                results.append({"orderID": f"dry_run_{i}", "status": "simulated"})
            return results

        # Validate and sign each order
        signed_args = []
        order_map = []  # Track which original orders mapped to signed orders
        for i, order in enumerate(orders):
            rounded_size = math.floor(order["size"])
            price = order["price"]

            # Validation
            if price <= 0 or price >= 1:
                logger.warning(f"[BATCH] Invalid price: {price:.4f}, skipping order {i}")
                continue
            if order["side"] == "SELL":
                if rounded_size < 1:
                    logger.warning(f"[BATCH] Sell size {rounded_size} below 1 share, skipping order {i}")
                    continue
            else:
                # FAK orders require maker_amount >= $1.00
                maker_amount = rounded_size * price
                if maker_amount < MIN_ORDER_VALUE_USD:
                    logger.warning(
                        f"[BATCH] Buy maker_amount ${maker_amount:.4f} below "
                        f"${MIN_ORDER_VALUE_USD}, skipping order {i}"
                    )
                    continue

            # Ensure maker_amount has <= 2 decimal places (FAK requirement)
            # Only round DOWN to avoid overshooting utility-optimal size
            maker_amount = rounded_size * price
            if _has_more_than_2dp(maker_amount):
                adjusted_size = _round_to_fak_size(rounded_size, price, round_up=False)
                if adjusted_size < 1 or (order["side"] == "BUY" and adjusted_size * price < MIN_ORDER_VALUE_USD):
                    logger.warning(
                        f"[BATCH] No valid FAK size at or below {rounded_size} "
                        f"(price={price:.4f}, step={_fak_size_step(price)}), skipping order {i}"
                    )
                    continue
                logger.info(
                    f"[BATCH] Adjusted size {rounded_size} -> {adjusted_size} "
                    f"for 2dp maker_amount (${adjusted_size * price:.4f})"
                )
                rounded_size = adjusted_size

            logger.info(
                f"[{self.event_name}][BATCH ORDER {i}] {order['side']} "
                f"size={rounded_size} @ {price:.4f} maker_amt={rounded_size * price:.4f}"
            )

            try:
                order_args = OrderArgs(
                    token_id=order["token_id"],
                    price=price,
                    size=rounded_size,
                    side=order["side"],
                )
                signed = self.client.create_order(order_args)
                signed_args.append(PostOrdersArgs(order=signed, orderType=OrderType.FAK))
                order_map.append(i)
            except Exception as e:
                logger.error(f"[BATCH] Failed to sign order {i}: {e}")
                continue

        if not signed_args:
            logger.warning(f"[{self.event_name}][BATCH] No valid orders to submit")
            return []

        try:
            logger.info(
                f"[{self.event_name}][BATCH] Submitting {len(signed_args)} orders in one API call"
            )
            response = self.client.post_orders(signed_args)
            logger.info(f"[{self.event_name}][BATCH] Response: {response}")
            return response if isinstance(response, list) else [response]
        except Exception as e:
            logger.error(f"[{self.event_name}][BATCH] Batch order submission failed: {e}")
            self._last_error = str(e)
            return []

    def execute_candidate(
        self,
        candidate: TradeCandidate,
        token_id: str,
    ) -> ExecutionResult:
        """
        Execute a trade candidate.

        Translates Kelly action to CLOB order.
        Now uses separate YES/NO token IDs - no conversion needed.
        """
        action = candidate.action
        price = candidate.price
        size = candidate.size

        # Determine order side
        # token_id is already the correct token (YES for YES actions, NO for NO actions)
        if action == TradeAction.BUY_YES:
            side = "BUY"
            order_price = price
        elif action == TradeAction.SELL_YES:
            side = "SELL"
            order_price = price
        elif action == TradeAction.BUY_NO:
            # Buy on NO token directly
            side = "BUY"
            order_price = price
        elif action == TradeAction.SELL_NO:
            # Sell on NO token directly
            side = "SELL"
            order_price = price
        else:
            return ExecutionResult(
                success=False,
                candidate=candidate,
                error=f"Unknown action: {action}",
            )

        # Log order details before placement (note: size will be floored to 2 decimals in place_limit_order)
        bin_range = self.bin_ranges.get(candidate.bin_index, "")
        bin_info = f"bin={candidate.bin_index} ({bin_range})" if bin_range else f"bin={candidate.bin_index}"
        logger.info(
            f"[{self.event_name}] Placing order: {action.value} {bin_info} | "
            f"{side} {size:.2f} @ {order_price:.2f} | "
            f"fair={candidate.reservation_price:.4f} edge={candidate.edge:+.2%} util={candidate.utility_gain:.4f} | "
            f"token={token_id[:16]}..."
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
                error=self._last_error or "Order placement failed",
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
        no_token_ids: Optional[Dict[int, str]] = None,  # bin_index -> NO token_id
        on_trade: Optional[Callable[[ExecutionResult], None]] = None,
        user_stream: Optional["UserStreamClient"] = None,
        sync_portfolio: Optional[Callable[[], None]] = None,
        event_name: Optional[str] = None,
    ):
        """
        Initialize Kelly executor.

        Args:
            config: Kelly configuration
            portfolio: Portfolio state
            orderbook_manager: Orderbook manager (WebSocket or REST)
            order_executor: Order executor
            token_ids: Map of bin_index -> YES token_id
            no_token_ids: Map of bin_index -> NO token_id (for BUY_NO/SELL_NO)
            on_trade: Optional callback for trade notifications
            user_stream: Optional UserStreamClient for fill confirmations
            sync_portfolio: Callback to sync portfolio from API before each decision
            event_name: Optional event name for logging (e.g., "Feb 03 - Feb 10")
        """
        self.config = config
        self.portfolio = portfolio
        self.orderbook_manager = orderbook_manager
        self.order_executor = order_executor
        self.token_ids = token_ids  # YES token IDs
        self.no_token_ids = no_token_ids or {}  # NO token IDs
        self.on_trade = on_trade
        self.user_stream = user_stream
        self.sync_portfolio = sync_portfolio  # Callback to sync from API before each decision
        self.event_name = event_name or "unknown"

        # Reverse mapping: token_id -> bin_index (for fill callbacks)
        self.token_to_bin: Dict[str, int] = {v: k for k, v in token_ids.items()}
        # Also add NO token mappings
        for bin_idx, no_token_id in self.no_token_ids.items():
            if no_token_id:
                self.token_to_bin[no_token_id] = bin_idx

        # Execution state
        self._running = False
        self._last_tick_time = 0.0

        # Rate limiting state: track recent order timestamps
        self._order_timestamps: List[float] = []

        # Pending orders awaiting fill confirmation
        # Maps order_id -> (candidate, token_id)
        self._pending_orders: Dict[str, tuple[TradeCandidate, str]] = {}

        # Confirmation events for waiting on block confirmation
        # Maps order_id -> asyncio.Event (set when MINED/CONFIRMED)
        self._confirmation_events: Dict[str, asyncio.Event] = {}

        # Trade counter for logging
        self._trade_count: int = 0

        # Context for trade logging (set by caller before run_tick)
        self._log_context: Dict = {}

        # FAK failure cooldown tracking: bin_index -> timestamp of last FAK failure
        # Used to avoid spamming failed orders when liquidity dries up
        self._fak_failure_times: Dict[int, float] = {}

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

    def _is_bin_in_fak_cooldown(self, bin_index: int) -> bool:
        """
        Check if a bin is in FAK failure cooldown.

        Returns True if the bin recently had a FAK failure and should not be retried yet.
        """
        if bin_index not in self._fak_failure_times:
            return False

        cooldown = self.config.rate_limit.fak_failure_cooldown_seconds
        elapsed = time.time() - self._fak_failure_times[bin_index]

        if elapsed < cooldown:
            return True

        # Cooldown expired, remove from tracking
        del self._fak_failure_times[bin_index]
        return False

    def _record_fak_failure(self, bin_index: int) -> None:
        """Record a FAK order failure for cooldown tracking."""
        self._fak_failure_times[bin_index] = time.time()
        cooldown = self.config.rate_limit.fak_failure_cooldown_seconds
        logger.info(
            f"FAK order failed for bin {bin_index}, cooldown for {cooldown:.0f}s"
        )

    async def run_tick(
        self,
        hours_to_settlement: float,
        verbose: bool = False,
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
            verbose: If True, log detailed rejection reasons for candidates

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

        # Check global rate limit
        if not self._check_rate_limit():
            logger.info(f"[{self.event_name}] Rate limit reached, skipping tick")
            return tick_result

        # Get current orderbooks
        orderbooks = self._get_orderbooks()

        rate_config = self.config.rate_limit

        # Step 1: Sync portfolio from API (single sync per tick)
        if self.sync_portfolio:
            try:
                await self.sync_portfolio()
                logger.info(
                    f"[{self.event_name}][KELLY] Synced from API: "
                    f"capital=${self.portfolio.capital:.2f}, "
                    f"invested=${self.portfolio.total_collateral_used:.2f}"
                )
            except Exception as e:
                logger.warning(f"[{self.event_name}] Failed to sync portfolio: {e}")
                # Continue with existing state if sync fails

        # Step 2: Compute ALL optimal trades in simulation
        # This runs entirely on a COPY of self.portfolio — never mutates it.
        planned_trades = self._compute_optimal_trades(
            orderbooks, hours_to_settlement, verbose
        )

        if not planned_trades:
            tick_result.elapsed_seconds = time.time() - start_time
            self._last_tick_time = time.time()
            return tick_result

        # Respect per-tick order limit
        max_orders = rate_config.max_orders_per_tick
        if len(planned_trades) > max_orders:
            logger.info(
                f"[{self.event_name}] Capping {len(planned_trades)} planned trades "
                f"to max_orders_per_tick={max_orders}"
            )
            planned_trades = planned_trades[:max_orders]

        tick_result.num_candidates = len(planned_trades)

        # Step 3: Execute orders
        if self.order_executor.dry_run:
            # Dry run: simulate all fills optimistically on LIVE portfolio
            for trade in planned_trades:
                token_id = self._get_token_id_for_action(trade)
                if not token_id:
                    continue

                # Update live portfolio optimistically (OK in dry-run)
                self._update_portfolio(
                    candidate=trade,
                    token_id=token_id,
                    filled_size=trade.size,
                    filled_price=trade.price,
                )

                br = self._bin_range(trade.bin_index)
                bin_info = f"bin={trade.bin_index} ({br})" if br else f"bin={trade.bin_index}"
                logger.info(
                    f"[{self.event_name}][ORDER PLACED - DRY RUN] {trade.action.value} {bin_info} | "
                    f"size={trade.size:.1f} @ {trade.price:.3f}"
                )

                self._log_trade_placed(trade, token_id, f"dry_run_{self._trade_count}")
                self._record_order()

                result = ExecutionResult(
                    success=True,
                    candidate=trade,
                    order_id=f"dry_run_{self._trade_count}",
                    filled_size=trade.size,
                    filled_price=trade.price,
                    is_pending=False,
                )
                tick_result.executions.append(result)
                tick_result.num_executed += 1
                tick_result.total_utility_gain += trade.utility_gain

                if self.on_trade:
                    self.on_trade(result)
        else:
            # Live mode: batch submit all orders
            order_specs = []
            trade_token_pairs = []  # Parallel list for tracking
            for trade in planned_trades:
                token_id = self._get_token_id_for_action(trade)
                if not token_id:
                    continue

                side = "BUY" if trade.action in (TradeAction.BUY_YES, TradeAction.BUY_NO) else "SELL"
                order_specs.append({
                    "token_id": token_id,
                    "side": side,
                    "price": trade.price,
                    "size": trade.size,
                })
                trade_token_pairs.append((trade, token_id))

                # Log each planned trade
                br = self._bin_range(trade.bin_index)
                bin_info = f"bin={trade.bin_index} ({br})" if br else f"bin={trade.bin_index}"
                logger.info(
                    f"[{self.event_name}] Placing order: {trade.action.value} {bin_info} | "
                    f"{side} {trade.size:.2f} @ {trade.price:.4f} | "
                    f"fair={trade.reservation_price:.4f} edge={trade.edge:+.2%} "
                    f"util={trade.utility_gain:.4f} | token={token_id[:16]}..."
                )

            if not order_specs:
                tick_result.elapsed_seconds = time.time() - start_time
                self._last_tick_time = time.time()
                return tick_result

            # Merge orders for same (token_id, side, price)
            merged_specs = []
            merged_trade_pairs = []  # Each entry is a list of (trade, token_id)
            merge_key_to_idx = {}
            for spec, (trade, token_id) in zip(order_specs, trade_token_pairs):
                key = (spec["token_id"], spec["side"], spec["price"])
                if key in merge_key_to_idx:
                    idx = merge_key_to_idx[key]
                    merged_specs[idx]["size"] += spec["size"]
                    merged_trade_pairs[idx].append((trade, token_id))
                    logger.info(
                        f"[{self.event_name}] Merged order for bin={trade.bin_index}: "
                        f"+{spec['size']:.0f} -> total {merged_specs[idx]['size']:.0f} shares"
                    )
                else:
                    merge_key_to_idx[key] = len(merged_specs)
                    merged_specs.append(dict(spec))
                    merged_trade_pairs.append([(trade, token_id)])

            order_specs = merged_specs
            # Flatten trade_token_pairs for response tracking (one entry per merged order)
            # We'll use the first trade from each group as representative
            trade_token_pairs = [pairs[0] for pairs in merged_trade_pairs]

            # Batch submit
            batch_response = self.order_executor.place_batch_orders(order_specs)

            # Process batch response — extract order IDs and track pending orders
            # Batch API returns list of dicts with orderID, errorMsg, success fields
            order_results = []  # List of (order_id_or_None, error_msg_or_None)
            if isinstance(batch_response, dict):
                # Single response object with orderIDs list
                oids = batch_response.get("orderIDs", [])
                error = batch_response.get("errorMsg", "")
                order_results = [(oid if oid else None, error) for oid in oids]
            elif isinstance(batch_response, list):
                for resp in batch_response:
                    if isinstance(resp, dict):
                        oid = resp.get("orderID")
                        if not oid:
                            oids = resp.get("orderIDs", [])
                            oid = oids[0] if oids else None
                        error = resp.get("errorMsg", "")
                        # Empty string orderID means failure despite success=True
                        if oid == "":
                            oid = None
                        order_results.append((oid, error))
                    else:
                        order_results.append((None, ""))

            # Track each order
            for i, (trade, token_id) in enumerate(trade_token_pairs):
                order_id = order_results[i][0] if i < len(order_results) else None
                error_msg = order_results[i][1] if i < len(order_results) else ""

                # If there's an error, record FAK failure to prevent infinite retry
                if not order_id and error_msg:
                    logger.warning(
                        f"[{self.event_name}] Batch order for bin={trade.bin_index} "
                        f"failed: {error_msg}"
                    )
                    self._record_fak_failure(trade.bin_index)

                if order_id:
                    # Track as pending
                    self._pending_orders[order_id] = (trade, token_id)

                    # Create confirmation event for fill waiting
                    confirm_event = asyncio.Event()
                    self._confirmation_events[order_id] = confirm_event

                    # Register with user_stream
                    if self.user_stream:
                        from .user_stream import PendingOrder
                        pending = PendingOrder(
                            order_id=order_id,
                            token_id=token_id,
                            side="BUY" if trade.action in (TradeAction.BUY_YES, TradeAction.BUY_NO) else "SELL",
                            price=trade.price,
                            size=trade.size,
                            bin_index=trade.bin_index,
                        )
                        asyncio.create_task(self.user_stream.add_pending_order(pending))

                    self._log_trade_placed(trade, token_id, order_id)
                    self._record_order()

                    result = ExecutionResult(
                        success=True,
                        candidate=trade,
                        order_id=order_id,
                        is_pending=True,
                    )
                else:
                    result = ExecutionResult(
                        success=False,
                        candidate=trade,
                        error="No order_id in batch response",
                        is_pending=False,
                    )

                tick_result.executions.append(result)
                if result.success:
                    tick_result.num_executed += 1
                    tick_result.total_utility_gain += trade.utility_gain

                if self.on_trade:
                    self.on_trade(result)

            # Step 4: Wait for all fills (or tick timeout)
            remaining_timeout = rate_config.tick_timeout_seconds - (time.time() - start_time)
            if remaining_timeout > 0 and self._confirmation_events:
                logger.info(
                    f"[{self.event_name}] Waiting for {len(self._confirmation_events)} "
                    f"fill confirmations (timeout: {remaining_timeout:.0f}s)..."
                )
                try:
                    await asyncio.wait_for(
                        self._wait_all_confirmations(),
                        timeout=remaining_timeout,
                    )
                    logger.info(
                        f"[{self.event_name}] All fill confirmations received"
                    )
                except asyncio.TimeoutError:
                    pending_count = len(self._confirmation_events)
                    logger.warning(
                        f"[{self.event_name}] Tick timeout waiting for "
                        f"{pending_count} fill confirmation(s)"
                    )
                    # Clean up unresolved confirmation events
                    self._confirmation_events.clear()

        # Log tick summary
        logger.info(
            f"Tick result: candidates={tick_result.num_candidates}, "
            f"executed={tick_result.num_executed}, "
            f"utility_gain={tick_result.total_utility_gain:.6f}, "
            f"elapsed={time.time() - start_time:.2f}s"
        )
        for result in tick_result.executions:
            if result.success:
                c = result.candidate
                status = "pending fill" if result.is_pending else "filled"
                logger.info(
                    f"  Order: {c.action.value} bin={c.bin_index} "
                    f"size={c.size:.1f} @ {c.price:.4f} ({status})"
                )

        tick_result.elapsed_seconds = time.time() - start_time
        self._last_tick_time = time.time()

        return tick_result

    def _find_optimal_size(
        self,
        candidate: TradeCandidate,
        orderbooks: Dict[int, UnifiedOrderbook],
        hours_to_settlement: float,
        max_iterations: int = 10,
    ) -> float:
        """
        Binary search for optimal chunk size to avoid overshooting Kelly-optimal.

        Tests whether the full chunk overshoots by simulating the fill and checking
        if the trade is still optimal. If it overshoots, binary searches for the
        largest size that doesn't.

        For BUY: overshoots if the same (bin, action) is no longer the best buy.
        For SELL: overshoots if an opposing BUY for the same bin appears (cycling).

        Args:
            candidate: Best candidate with full chunk size
            orderbooks: Current orderbooks
            hours_to_settlement: Hours until settlement
            max_iterations: Number of binary search iterations

        Returns:
            Optimal chunk size in shares
        """
        full_size = candidate.size
        price = candidate.price
        is_buy = candidate.action in (TradeAction.BUY_YES, TradeAction.BUY_NO)

        # Clamp full_size to what's actually available
        if is_buy:
            if price > 0:
                max_buy_shares = self.portfolio.available_capital / price
                full_size = min(full_size, max_buy_shares)
        else:
            position = self.portfolio.get_position(candidate.bin_index)
            if position:
                if candidate.action == TradeAction.SELL_YES:
                    full_size = min(full_size, position.yes_shares)
                elif candidate.action == TradeAction.SELL_NO:
                    full_size = min(full_size, position.no_shares)

        # Compute minimum tradeable size
        if is_buy:
            # FAK orders require maker_amount (size * price) >= $1.00
            if price > 0:
                min_size = max(1.0, math.ceil(MIN_ORDER_VALUE_USD / price))
            else:
                return 0.0
        else:
            # For sells/exits, minimum is 1 share
            min_size = 1.0

        # If full_size is at or below minimum, just use it
        if full_size <= min_size:
            return full_size

        # Quick check: does full chunk overshoot?
        if not self._check_overshoots(candidate, full_size, orderbooks, hours_to_settlement):
            return full_size

        br = self._bin_range(candidate.bin_index)
        bin_info = f"bin={candidate.bin_index} ({br})" if br else f"bin={candidate.bin_index}"
        logger.info(
            f"[{self.event_name}] Full chunk ({full_size:.0f} shares) overshoots for "
            f"{candidate.action.value} {bin_info}, binary searching..."
        )

        # Binary search: find largest size that doesn't overshoot
        lo = min_size
        hi = full_size
        best_valid = min_size  # Fallback: use minimum even if it overshoots

        for i in range(max_iterations):
            mid = (lo + hi) / 2.0

            # Converged (less than 1 share difference)
            if hi - lo < 1.0:
                break

            if self._check_overshoots(candidate, mid, orderbooks, hours_to_settlement):
                hi = mid
            else:
                best_valid = mid
                lo = mid

        # Floor to integer shares
        optimal = max(1.0, math.floor(best_valid))

        logger.info(
            f"[{self.event_name}] Optimal size: {optimal:.0f} shares "
            f"(full was {full_size:.0f}, {optimal / full_size:.0%} of chunk)"
        )

        return optimal

    def _check_overshoots(
        self,
        candidate: TradeCandidate,
        test_size: float,
        orderbooks: Dict[int, UnifiedOrderbook],
        hours_to_settlement: float,
    ) -> bool:
        """
        Check if filling test_size shares would overshoot Kelly-optimal.

        For BUY: Returns True if the same (bin, action) is no longer the best
        buy candidate after the simulated fill.

        For SELL: Returns True if an opposing BUY for the same bin appears
        after the simulated fill (would cause immediate re-entry cycling).
        """
        action = candidate.action
        bin_index = candidate.bin_index
        price = candidate.price

        # Simulate the fill
        if action == TradeAction.BUY_YES:
            hyp = self.portfolio.simulate_buy_yes(bin_index, test_size, price)
        elif action == TradeAction.BUY_NO:
            hyp = self.portfolio.simulate_buy_no(bin_index, test_size, price)
        elif action == TradeAction.SELL_YES:
            hyp = self.portfolio.simulate_sell_yes(bin_index, test_size, price)
        elif action == TradeAction.SELL_NO:
            hyp = self.portfolio.simulate_sell_no(bin_index, test_size, price)
        else:
            return False

        # Preserve external capital limit
        hyp.external_capital_limit = self.portfolio.external_capital_limit

        # Regenerate candidates on hypothetical portfolio
        new_candidates = generate_candidates(
            portfolio=hyp,
            orderbooks=orderbooks,
            config=self.config,
            hours_to_settlement=hours_to_settlement,
            verbose=False,
        )

        is_buy = action in (TradeAction.BUY_YES, TradeAction.BUY_NO)

        if not new_candidates:
            # For buys: no candidates = used up all utility, overshot
            # For sells: no candidates = fully exited, fine
            return is_buy

        if is_buy:
            # Check if same (bin, action) is still the best BUY candidate
            buy_candidates = [
                c for c in new_candidates
                if c.action in (TradeAction.BUY_YES, TradeAction.BUY_NO)
            ]

            if not buy_candidates:
                return True  # No buy candidates left = overshot

            best_buy = buy_candidates[0]  # Already sorted by utility descending
            return not (best_buy.bin_index == bin_index and best_buy.action == action)
        else:
            # For sells: check if opposing BUY for the same bin appeared
            # This means we over-exited and would immediately re-enter (cycling)
            opposing = (
                TradeAction.BUY_YES if action == TradeAction.SELL_YES
                else TradeAction.BUY_NO
            )
            for c in new_candidates:
                if c.bin_index == bin_index and c.action == opposing:
                    return True  # Opposing buy appeared = over-exited
            return False

    def _check_overshoots_on(
        self,
        portfolio: Portfolio,
        candidate: TradeCandidate,
        test_size: float,
        orderbooks: Dict[int, UnifiedOrderbook],
        hours_to_settlement: float,
    ) -> bool:
        """
        Check if filling test_size shares would overshoot Kelly-optimal.

        Same logic as _check_overshoots but operates on an arbitrary portfolio
        instead of self.portfolio. Used by the simulation loop to avoid
        contaminating the live portfolio.
        """
        action = candidate.action
        bin_index = candidate.bin_index
        price = candidate.price

        # Simulate the fill on the given portfolio (returns a new copy)
        if action == TradeAction.BUY_YES:
            hyp = portfolio.simulate_buy_yes(bin_index, test_size, price)
        elif action == TradeAction.BUY_NO:
            hyp = portfolio.simulate_buy_no(bin_index, test_size, price)
        elif action == TradeAction.SELL_YES:
            hyp = portfolio.simulate_sell_yes(bin_index, test_size, price)
        elif action == TradeAction.SELL_NO:
            hyp = portfolio.simulate_sell_no(bin_index, test_size, price)
        else:
            return False

        # Preserve external capital limit
        hyp.external_capital_limit = portfolio.external_capital_limit

        # Regenerate candidates on hypothetical portfolio
        new_candidates = generate_candidates(
            portfolio=hyp,
            orderbooks=orderbooks,
            config=self.config,
            hours_to_settlement=hours_to_settlement,
            verbose=False,
        )

        is_buy = action in (TradeAction.BUY_YES, TradeAction.BUY_NO)

        if not new_candidates:
            return is_buy  # Buys: overshot; Sells: fully exited, fine

        if is_buy:
            # Overshoot = this bin is no longer the best buy after the fill.
            # The simulation loop handles alternating between bins.
            buy_candidates = [
                c for c in new_candidates
                if c.action in (TradeAction.BUY_YES, TradeAction.BUY_NO)
            ]
            if not buy_candidates:
                return True
            best_buy = buy_candidates[0]
            return not (best_buy.bin_index == bin_index and best_buy.action == action)
        else:
            opposing = (
                TradeAction.BUY_YES if action == TradeAction.SELL_YES
                else TradeAction.BUY_NO
            )
            for c in new_candidates:
                if c.bin_index == bin_index and c.action == opposing:
                    return True
            return False

    def _find_optimal_size_on(
        self,
        portfolio: Portfolio,
        candidate: TradeCandidate,
        orderbooks: Dict[int, UnifiedOrderbook],
        hours_to_settlement: float,
        max_iterations: int = 10,
    ) -> float:
        """
        Binary search for optimal chunk size against a given portfolio.

        Like _find_optimal_size but:
        - Operates on an arbitrary portfolio (not self.portfolio) to avoid
          contaminating the live portfolio during simulation.
        - Uses c_bin_max as the upper bound for buys (not delta_ratio chunk).
          This allows reaching optimal position in fewer iterations.

        For SELL: upper bound is the full position in the bin.
        """
        from .orderbook import get_available_depth

        price = candidate.price
        is_buy = candidate.action in (TradeAction.BUY_YES, TradeAction.BUY_NO)

        if is_buy:
            # Upper bound: c_bin_max worth of shares, minus existing position in this bin
            position = portfolio.get_position(candidate.bin_index)
            existing_collateral = position.collateral_used if position else 0.0
            remaining_bin_budget = max(0.0, self.config.collateral.c_bin_max - existing_collateral)

            if price > 0:
                full_size = remaining_bin_budget / price
                # Also cap by available capital
                full_size = min(full_size, portfolio.available_capital / price)
            else:
                return 0.0
        else:
            # For sells: full position
            position = portfolio.get_position(candidate.bin_index)
            if not position:
                return 0.0
            if candidate.action == TradeAction.SELL_YES:
                full_size = position.yes_shares
            else:
                full_size = position.no_shares

        # Cap by available orderbook depth
        ob = orderbooks.get(candidate.bin_index)
        if ob:
            depth = get_available_depth(ob, candidate.action.value)
            if depth > 0:
                full_size = min(full_size, depth)

        # Minimum tradeable size
        if is_buy:
            if price > 0:
                min_size = max(1.0, math.ceil(MIN_ORDER_VALUE_USD / price))
            else:
                return 0.0
        else:
            min_size = 1.0

        if full_size <= min_size:
            return full_size if full_size >= 1.0 else 0.0

        # Quick check: does full chunk overshoot?
        if not self._check_overshoots_on(portfolio, candidate, full_size, orderbooks, hours_to_settlement):
            return full_size

        br = self._bin_range(candidate.bin_index)
        bin_info = f"bin={candidate.bin_index} ({br})" if br else f"bin={candidate.bin_index}"
        logger.debug(
            f"[{self.event_name}] Full chunk ({full_size:.0f} shares) overshoots for "
            f"{candidate.action.value} {bin_info}, binary searching..."
        )

        # Binary search: find largest size that doesn't overshoot
        lo = min_size
        hi = full_size
        best_valid = min_size  # Fallback

        for _ in range(max_iterations):
            if hi - lo < 1.0:
                break
            mid = (lo + hi) / 2.0
            if self._check_overshoots_on(portfolio, candidate, mid, orderbooks, hours_to_settlement):
                hi = mid
            else:
                best_valid = mid
                lo = mid

        optimal = max(1.0, math.floor(best_valid))

        logger.info(
            f"[{self.event_name}] Optimal size for {candidate.action.value} {bin_info}: "
            f"{optimal:.0f} shares (max was {full_size:.0f})"
        )

        return optimal

    def _compute_optimal_trades(
        self,
        orderbooks: Dict[int, UnifiedOrderbook],
        hours_to_settlement: float,
        verbose: bool = False,
    ) -> List[TradeCandidate]:
        """
        Compute all optimal trades via greedy simulation on a HYPOTHETICAL portfolio.

        Creates a deep copy of self.portfolio and runs a simulation loop:
        1. Generate candidates on hypothetical portfolio
        2. Pick best (sells first, then buys by utility)
        3. Binary search for optimal size (upper bound = c_bin_max for buys)
        4. Simulate trade on hypothetical portfolio (never touches self.portfolio)
        5. Accumulate in planned_trades
        6. Repeat until no positive-utility trades remain

        IMPORTANT: self.portfolio is NEVER modified. All simulation happens on copies.

        Returns:
            List of TradeCandidate with optimized sizes, ready for batch execution.
        """
        # Deep copy the portfolio for simulation — self.portfolio stays untouched
        hyp = self.portfolio._copy()
        hyp.external_capital_limit = self.portfolio.external_capital_limit
        planned_trades = []

        for sim_iter in range(self.config.max_iters_per_tick):
            # Generate candidates on hypothetical portfolio
            candidates = generate_candidates(
                portfolio=hyp,
                orderbooks=orderbooks,
                config=self.config,
                hours_to_settlement=hours_to_settlement,
                verbose=(verbose and sim_iter == 0),
            )

            if not candidates:
                logger.debug(f"[{self.event_name}][SIM iter={sim_iter}] No candidates")
                break

            # Pick best candidate (sells come first, then buys by utility)
            best = candidates[0]

            is_sell = best.action in (TradeAction.SELL_YES, TradeAction.SELL_NO)
            if not is_sell and best.utility_gain < self.config.min_utility:
                logger.debug(
                    f"[{self.event_name}][SIM iter={sim_iter}] Best buy utility "
                    f"{best.utility_gain:.6f} < tau {self.config.min_utility}, stopping"
                )
                break

            # Skip bins in FAK cooldown
            if self._is_bin_in_fak_cooldown(best.bin_index):
                candidates = [c for c in candidates if c.bin_index != best.bin_index]
                if not candidates:
                    break
                best = candidates[0]
                is_sell = best.action in (TradeAction.SELL_YES, TradeAction.SELL_NO)
                if not is_sell and best.utility_gain < self.config.min_utility:
                    break
                if self._is_bin_in_fak_cooldown(best.bin_index):
                    break

            # Binary search for optimal size on the HYPOTHETICAL portfolio
            optimal_size = self._find_optimal_size_on(
                hyp, best, orderbooks, hours_to_settlement
            )

            if optimal_size < 1.0:
                logger.debug(
                    f"[{self.event_name}][SIM iter={sim_iter}] Optimal size < 1 for "
                    f"{best.action.value} bin={best.bin_index}, stopping"
                )
                break

            best.size = optimal_size

            # Simulate trade on hypothetical portfolio (returns NEW copy)
            hyp = self._simulate_trade(hyp, best)

            planned_trades.append(best)

            # Log
            br = self._bin_range(best.bin_index)
            bin_info = f"bin={best.bin_index} ({br})" if br else f"bin={best.bin_index}"
            logger.info(
                f"[{self.event_name}][SIM iter={sim_iter}] {best.action.value} {bin_info} "
                f"size={optimal_size:.0f} @ {best.price:.4f} util={best.utility_gain:.6f}"
            )

        if planned_trades:
            logger.info(
                f"[{self.event_name}] Simulation complete: {len(planned_trades)} trades planned"
            )
        else:
            logger.debug(f"[{self.event_name}] Simulation complete: no trades needed")

        return planned_trades

    def _simulate_trade(self, portfolio: Portfolio, candidate: TradeCandidate) -> Portfolio:
        """
        Apply a simulated trade to a portfolio copy and return the new state.

        IMPORTANT: This never mutates the input portfolio. All simulate_*
        methods on Portfolio call _copy() internally and return a new object.
        """
        action = candidate.action
        bi = candidate.bin_index
        size = candidate.size
        price = candidate.price

        if action == TradeAction.BUY_YES:
            new_p = portfolio.simulate_buy_yes(bi, size, price)
        elif action == TradeAction.SELL_YES:
            new_p = portfolio.simulate_sell_yes(bi, size, price)
        elif action == TradeAction.BUY_NO:
            new_p = portfolio.simulate_buy_no(bi, size, price)
        elif action == TradeAction.SELL_NO:
            new_p = portfolio.simulate_sell_no(bi, size, price)
        else:
            return portfolio

        # Preserve external capital limit on the copy
        new_p.external_capital_limit = portfolio.external_capital_limit
        return new_p

    def _get_token_id_for_action(self, candidate: TradeCandidate) -> Optional[str]:
        """Get the token_id for a candidate's action (YES or NO token)."""
        if candidate.action in (TradeAction.BUY_NO, TradeAction.SELL_NO):
            token_id = self.no_token_ids.get(candidate.bin_index)
            if not token_id:
                logger.warning(f"No NO token_id for bin {candidate.bin_index}")
            return token_id
        else:
            token_id = self.token_ids.get(candidate.bin_index)
            if not token_id:
                logger.warning(f"No YES token_id for bin {candidate.bin_index}")
            return token_id

    async def _wait_all_confirmations(self) -> None:
        """Wait for all pending confirmation events to fire."""
        if not self._confirmation_events:
            return
        events = list(self._confirmation_events.values())
        await asyncio.gather(*[e.wait() for e in events])

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
        This is the ONLY place portfolio is updated during active trading.
        """
        action = candidate.action
        bin_index = candidate.bin_index
        # Use actual filled size/price, not the original candidate values
        size = filled_size
        price = filled_price

        # Get collateral before update for logging
        collateral_before = self.portfolio.total_collateral_used

        if action == TradeAction.BUY_YES:
            self.portfolio.execute_buy_yes(bin_index, size, price, token_id)
        elif action == TradeAction.SELL_YES:
            self.portfolio.execute_sell_yes(bin_index, size, price)
        elif action == TradeAction.BUY_NO:
            self.portfolio.execute_buy_no(bin_index, size, price, token_id)
        elif action == TradeAction.SELL_NO:
            self.portfolio.execute_sell_no(bin_index, size, price)

        # Log the update
        collateral_after = self.portfolio.total_collateral_used
        pos = self.portfolio.get_position(bin_index)
        shares_str = f"YES:{pos.yes_shares:.1f} NO:{pos.no_shares:.1f}" if pos else "none"
        logger.info(
            f"[PORTFOLIO UPDATE] {action.value} bin={bin_index} | "
            f"{size:.1f} @ {price:.4f} = ${size * price:.2f} | "
            f"invested: ${collateral_before:.2f} -> ${collateral_after:.2f} | "
            f"position: {shares_str}"
        )

    def handle_fill(self, fill_event: "FillEvent") -> None:
        """
        Handle fill confirmation from WebSocket.

        Called by UserStreamClient when a trade fill is confirmed.

        NOTE: We do NOT update portfolio here. Portfolio updates come from
        API sync before each Kelly decision. This avoids:
        - Double-counting from multiple callbacks (MATCHED, MINED, CONFIRMED)
        - Race conditions with manual trades
        - Stale local state vs authoritative API state

        This handler is only for:
        - Logging fill events for awareness
        - Removing completed orders from pending tracking

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
            # This fill might be from a previous session or manual order
            logger.info(f"Fill for unknown order {order_id[:16]}... - may be from previous session or manual trade")
            return

        candidate, token_id = pending_info

        # Log the fill (but don't update portfolio - that happens via API sync)
        from .user_stream import OrderStatus
        status_str = fill_event.status.name if hasattr(fill_event.status, 'name') else str(fill_event.status)
        logger.info(
            f"[FILL {status_str}] {candidate.action.value} bin={candidate.bin_index} | "
            f"{fill_event.size:.1f} @ {fill_event.price:.4f} = ${fill_event.size * fill_event.price:.2f}"
        )

        # Signal confirmation event on MINED so waiting code can proceed
        # (This allows the next order to be placed without waiting for full confirmation)
        if fill_event.status == OrderStatus.MINED:
            if order_id in self._confirmation_events:
                self._confirmation_events[order_id].set()
                del self._confirmation_events[order_id]
            logger.debug(f"Order {order_id[:16]}... mined, confirmation event signaled")

        # Remove from pending tracking only on CONFIRMED (not MINED)
        # This prevents "fill for unknown order" warnings when CONFIRMED arrives after MINED
        if fill_event.status == OrderStatus.CONFIRMED:
            if order_id in self._pending_orders:
                del self._pending_orders[order_id]
                logger.debug(f"Order {order_id[:16]}... confirmed and removed from pending")
            # Also signal confirmation event if not already done (in case MINED was missed)
            if order_id in self._confirmation_events:
                self._confirmation_events[order_id].set()
                del self._confirmation_events[order_id]

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

        # Cancel via order executor (may fail for FAK orders already killed by exchange)
        success = self.order_executor.cancel_order(order_id)

        if success:
            logger.info(f"Stale order {order_id[:16]}... cancelled successfully")
        else:
            logger.warning(f"Cancel failed for stale order {order_id[:16]}... (likely already dead FAK)")

        # Always clean up local tracking — for FAK orders the unfilled remainder
        # is already killed by the exchange, keeping it just causes stale loops
        if order_id in self._pending_orders:
            del self._pending_orders[order_id]

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
        # Propagate bin_ranges to order executor for log context
        if hasattr(self.order_executor, 'bin_ranges'):
            self.order_executor.bin_ranges = bin_ranges or {}

    def _bin_range(self, bin_index: int) -> str:
        """Get bin range string for logging (e.g., '340-359')."""
        return self._log_context.get("bin_ranges", {}).get(bin_index, "")

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
            f"[{self.event_name}][TRADE #{self._trade_count}] {now} | T-{hours_left:.1f}h | {action} bin={bin_idx} ({bin_range})"
        )
        logger.info(
            f"  → {size:.1f} shares @ {price:.3f} = ${collateral:.2f} | "
            f"model={model_prob:.1%} mkt={mkt_prob:.1%} edge={candidate.edge:+.1%} odds={odds:.1f}x"
        )
        logger.info(
            f"  → order_id={order_id[:16]}... | portfolio: ${capital:.2f} capital, ${total_collateral:.2f} invested"
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
            if not is_sell and best.utility_gain < self.config.min_utility:
                logger.debug(
                    f"Best buy candidate utility {best.utility_gain:.6f} "
                    f"< tau {self.config.min_utility}, stopping"
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
        verbose: bool = False,
    ) -> TickResult:
        """
        Run a single optimization tick (async version for live trading).

        Includes rate limiting delays between orders.

        Args:
            hours_to_settlement: Hours until market settlement
            verbose: If True, log detailed rejection reasons for candidates
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

            # Generate candidates (verbose on first iteration)
            candidates = generate_candidates(
                portfolio=self.portfolio,
                orderbooks=orderbooks,
                config=self.config,
                hours_to_settlement=hours_to_settlement,
                verbose=(verbose and iteration == 0),
            )

            if iteration == 0:
                tick_result.num_candidates = len(candidates)

            if not candidates:
                break

            # Candidates are ordered: [sells..., buys sorted by utility]
            best = candidates[0]

            # Sells execute unconditionally, only check utility for buys
            is_sell = best.action in (TradeAction.SELL_YES, TradeAction.SELL_NO)
            if not is_sell and best.utility_gain < self.config.min_utility:
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
