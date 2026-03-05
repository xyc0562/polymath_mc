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


def _round_price_to_tick(price: float, tick_size: str) -> float:
    """Round price the same way py-clob-client does for a given tick_size."""
    dp = len(tick_size.rstrip("0").split(".")[-1]) if "." in tick_size else 0
    return round(price * (10**dp)) / (10**dp)


def _fak_size_step(price: float, tick_size: str = None) -> int:
    """
    Compute the minimum size step for FAK orders at a given price.

    FAK orders require maker_amount (size * price) to have <= 2 decimal places.
    We work in tick units: price_ticks = price / 0.0001. Then size * price_ticks
    must be divisible by 100 (since $0.01 = 100 ticks).

    If tick_size is provided, the price is first rounded to match what
    py-clob-client will submit (e.g., tick_size="0.001" rounds to 3dp).
    Without this, the step may be computed for a price that the API never sees.
    """
    if tick_size:
        price = _round_price_to_tick(price, tick_size)
    price_ticks = round(price * 10_000)
    g = math.gcd(price_ticks, 100)
    return 100 // g


def _best_fak_price(
    price: float, size: int, side: str, max_tick_bump: int = 15, tick_size: str = None,
) -> tuple[float, int]:
    """
    Find the best FAK-compatible (price, adjusted_size) near the target price.

    For BUY: tries bumping price UP by 1-N ticks to find a step that wastes
    fewer shares. FAK fills at best available price, so overpaying by a few
    ticks is negligible.

    For SELL: tries bumping price DOWN by 1-N ticks.

    tick_size: market tick size (e.g., "0.01", "0.001"). When provided, prices
    are rounded to tick precision before computing the step, matching the
    rounding that py-clob-client applies in create_order().

    Returns (adjusted_price, adjusted_size). If no improvement found, returns
    the original price with size rounded to the original step.
    """
    base_step = _fak_size_step(price, tick_size)
    if base_step <= 1:
        return price, size

    best_size = (size // base_step) * base_step
    best_price = price

    # Use market tick size as bump increment (fall back to finest tick)
    tick = float(tick_size) if tick_size else 0.0001
    dp = len(tick_size.rstrip("0").split(".")[-1]) if tick_size and "." in tick_size else 4

    for bump in range(1, max_tick_bump + 1):
        if side == "BUY":
            candidate_price = round(price + bump * tick, dp)
        else:
            candidate_price = round(price - bump * tick, dp)
            if candidate_price <= 0:
                continue

        step = _fak_size_step(candidate_price, tick_size)
        adjusted = (size // step) * step if step > 1 else size
        if adjusted > best_size:
            best_size = adjusted
            best_price = candidate_price

    return best_price, best_size



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

    @staticmethod
    def _parse_conditional_balance_shares(balance_info: dict) -> Optional[float]:
        """Parse conditional token balance from get_balance_allowance response."""
        if not isinstance(balance_info, dict):
            return None
        raw_balance = balance_info.get("balance")
        try:
            # CLOB returns 6-decimal scaled quantity for conditional balances.
            return float(raw_balance) / 1e6
        except (TypeError, ValueError):
            return None

    def _fetch_conditional_balance_allowance(
        self,
        token_id: str,
        refresh: bool = True,
    ) -> Optional[dict]:
        """Fetch conditional token balance/allowance for a specific token_id."""
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

            params = BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL,
                token_id=token_id,
            )
            if refresh:
                try:
                    # Refresh exchange-side view first to reduce stale balance errors.
                    self.client.update_balance_allowance(params)
                except Exception as e:
                    logger.debug(
                        f"[{self.event_name}] Conditional allowance refresh failed "
                        f"for token {token_id[:16]}...: {e}"
                    )
            return self.client.get_balance_allowance(params)
        except Exception as e:
            logger.warning(
                f"[{self.event_name}] Failed to fetch conditional balance/allowance "
                f"for token {token_id[:16]}...: {e}"
            )
            return None

    def log_sell_balance_diagnostics(self, token_id: str, requested_size: float) -> None:
        """
        Log CLOB conditional balance/allowance snapshot after sell rejection.

        This helps distinguish local-position mismatch from exchange available-balance issues.
        """
        balance_info = self._fetch_conditional_balance_allowance(token_id=token_id, refresh=True)
        if not balance_info:
            return

        available_shares = self._parse_conditional_balance_shares(balance_info)
        allowances = balance_info.get("allowances", {}) if isinstance(balance_info, dict) else {}
        nonzero_spenders = 0
        if isinstance(allowances, dict):
            for value in allowances.values():
                try:
                    if int(value) > 0:
                        nonzero_spenders += 1
                except Exception:
                    continue

        if available_shares is None:
            logger.warning(
                f"[{self.event_name}] Sell rejection diagnostics: token={token_id[:16]}..., "
                f"requested={requested_size:.2f}, clob_balance=<unknown>, "
                f"nonzero_allowances={nonzero_spenders}"
            )
        else:
            logger.warning(
                f"[{self.event_name}] Sell rejection diagnostics: token={token_id[:16]}..., "
                f"requested={requested_size:.2f}, clob_balance={available_shares:.2f}, "
                f"nonzero_allowances={nonzero_spenders}"
            )

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
                logger.warning(f"[{self.event_name}] Invalid price: {price:.4f}")
                return None
            if side == "SELL":
                if rounded_size < 1:
                    logger.warning(f"[{self.event_name}] Sell size {rounded_size} below minimum 1 share")
                    return None
            else:
                if rounded_size < MIN_ORDER_SIZE and rounded_size * price < MIN_ORDER_VALUE_USD:
                    logger.warning(f"[{self.event_name}] Buy size {rounded_size} below {MIN_ORDER_SIZE} shares and value ${rounded_size * price:.2f} below ${MIN_ORDER_VALUE_USD}")
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
                f"[{self.event_name}] Order placed: {side} {rounded_size:.0f} @ {price:.4f}, "
                f"order_id={response.get('orderID', 'unknown')}"
            )
            return response

        except Exception as e:
            logger.error(f"[{self.event_name}] Order failed: {e}")
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
            logger.info(f"[{self.event_name}][DRY RUN] Would place market {side}: amount={amount:.2f}")
            return {"order_id": "dry_run_market", "status": "simulated"}

        try:
            response = self.client.create_market_order(
                token_id=token_id,
                side=side,
                amount=amount,
            )
            logger.info(f"[{self.event_name}] Market order placed: {side} {amount:.2f}")
            return response

        except Exception as e:
            logger.error(f"[{self.event_name}] Market order failed: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if self.dry_run:
            logger.info(f"[{self.event_name}][DRY RUN] Would cancel order: {order_id}")
            return True

        try:
            self.client.cancel(order_id)
            logger.info(f"[{self.event_name}] Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"[{self.event_name}] Cancel failed: {e}")
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
        # Returns one result per input order (aligned 1:1 with input list)
        SKIP_RESULT = {"orderID": None, "errorMsg": "", "_skipped": True}
        results = [dict(SKIP_RESULT) for _ in orders]  # Default: all skipped
        signed_args = []
        signed_indices = []  # Which input indices have signed orders
        # Track remaining exchange-visible sellable balance per token within this batch.
        # Prevents submitting multiple SELLs that collectively exceed CLOB available balance.
        sell_available_cache: Dict[str, Optional[float]] = {}

        # Resolve tick_size per token (cached after first call)
        tick_sizes: Dict[str, str] = {}
        for order in orders:
            tid = order["token_id"]
            if tid not in tick_sizes:
                try:
                    tick_sizes[tid] = self.client.get_tick_size(tid)
                except Exception:
                    tick_sizes[tid] = "0.01"  # safe default

        for i, order in enumerate(orders):
            rounded_size = math.floor(order["size"])
            price = order["price"]
            ts = tick_sizes.get(order["token_id"])

            # Validation
            if price <= 0 or price >= 1:
                logger.warning(f"[{self.event_name}][BATCH] Invalid price: {price:.4f}, skipping order {i}")
                continue
            if order["side"] == "SELL":
                if rounded_size < 1:
                    logger.warning(f"[{self.event_name}][BATCH] Sell size {rounded_size} below 1 share, skipping order {i}")
                    continue

                token_id = order["token_id"]
                if token_id not in sell_available_cache:
                    balance_info = self._fetch_conditional_balance_allowance(
                        token_id=token_id,
                        refresh=True,
                    )
                    available_shares = self._parse_conditional_balance_shares(balance_info or {})
                    sell_available_cache[token_id] = available_shares

                available_shares = sell_available_cache.get(token_id)
                if available_shares is not None:
                    max_sellable = math.floor(max(0.0, available_shares))
                    if max_sellable < 1:
                        logger.warning(
                            f"[{self.event_name}][BATCH] SELL token={token_id[:16]}... skipped: "
                            f"CLOB available balance is {available_shares:.2f} shares"
                        )
                        continue
                    if rounded_size > max_sellable:
                        logger.warning(
                            f"[{self.event_name}][BATCH] SELL token={token_id[:16]}... "
                            f"size capped {rounded_size} -> {max_sellable} "
                            f"(CLOB available={available_shares:.2f})"
                        )
                        rounded_size = max_sellable
                    sell_available_cache[token_id] = max(0.0, available_shares - rounded_size)
            else:
                # FAK orders require maker_amount >= $1.00
                maker_amount = rounded_size * price
                if maker_amount < MIN_ORDER_VALUE_USD:
                    logger.warning(
                        f"[{self.event_name}][BATCH] Buy maker_amount ${maker_amount:.4f} below "
                        f"${MIN_ORDER_VALUE_USD}, skipping order {i}"
                    )
                    continue

            # Ensure maker_amount has <= 2 decimal places (FAK requirement)
            # Must check using the tick-rounded price (what py-clob-client actually submits)
            # Only applies to BUY orders — SELL maker_amount is the share count (integer)
            if order["side"] == "BUY":
                rounded_price = _round_price_to_tick(price, ts) if ts else price
                maker_amount = rounded_size * rounded_price
                if _has_more_than_2dp(maker_amount):
                    # Try nearby prices for a better step (FAK fills at best price anyway)
                    adj_price, adjusted_size = _best_fak_price(price, rounded_size, "BUY", tick_size=ts)
                    if adjusted_size < 1 or adjusted_size * adj_price < MIN_ORDER_VALUE_USD:
                        logger.warning(
                            f"[{self.event_name}][BATCH] No valid FAK size at or below {rounded_size} "
                            f"(price={price:.4f}, step={_fak_size_step(price, ts)}), skipping order {i}"
                        )
                        continue
                    if adj_price != price:
                        logger.info(
                            f"[{self.event_name}][BATCH] Adjusted price {price:.4f} -> {adj_price:.4f} "
                            f"(+{round((adj_price - price) * 10000):.0f} ticks) "
                            f"and size {rounded_size} -> {adjusted_size} "
                            f"for 2dp maker_amount (${adjusted_size * adj_price:.4f})"
                        )
                    else:
                        logger.info(
                            f"[{self.event_name}][BATCH] Adjusted size {rounded_size} -> {adjusted_size} "
                            f"for 2dp maker_amount (${adjusted_size * price:.4f})"
                        )
                    rounded_size = adjusted_size
                    price = adj_price

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
                signed_indices.append(i)
            except Exception as e:
                logger.error(f"[{self.event_name}][BATCH] Failed to sign order {i}: {e}")
                continue

        if not signed_args:
            logger.warning(f"[{self.event_name}][BATCH] No valid orders to submit")
            return results

        try:
            logger.info(
                f"[{self.event_name}][BATCH] Submitting {len(signed_args)} orders in one API call"
            )
            response = self.client.post_orders(signed_args)
            logger.info(f"[{self.event_name}][BATCH] Response: {response}")

            # Map API responses back to original order indices
            api_results = response if isinstance(response, list) else [response]
            for j, idx in enumerate(signed_indices):
                if j < len(api_results):
                    results[idx] = api_results[j] if isinstance(api_results[j], dict) else {}
                else:
                    results[idx] = {"orderID": None, "errorMsg": "No response from API"}

            return results
        except Exception as e:
            logger.error(f"[{self.event_name}][BATCH] Batch order submission failed: {e}")
            self._last_error = str(e)
            return results

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
        orderbook_manager: Optional[OrderbookManager] = None,
        order_executor: Optional[OrderExecutor] = None,
        token_ids: Dict[int, str] = None,
        no_token_ids: Optional[Dict[int, str]] = None,
        on_trade: Optional[Callable[[ExecutionResult], None]] = None,
        user_stream: Optional["UserStreamClient"] = None,
        sync_portfolio: Optional[Callable[[], None]] = None,
        event_name: Optional[str] = None,
        # Abstract backend params for backtest use
        orderbook_provider=None,   # OrderbookProvider from backend.py
        trade_executor=None,       # TradeExecutor from backend.py
    ):
        """
        Initialize Kelly executor.

        Production callers pass orderbook_manager + order_executor.
        Backtest callers pass orderbook_provider + trade_executor.

        Args:
            config: Kelly configuration
            portfolio: Portfolio state
            orderbook_manager: Orderbook manager (WebSocket or REST) - production
            order_executor: Order executor - production
            token_ids: Map of bin_index -> YES token_id
            no_token_ids: Map of bin_index -> NO token_id (for BUY_NO/SELL_NO)
            on_trade: Optional callback for trade notifications
            user_stream: Optional UserStreamClient for fill confirmations
            sync_portfolio: Callback to sync portfolio from API before each decision
            event_name: Optional event name for logging (e.g., "Feb 03 - Feb 10")
            orderbook_provider: Abstract orderbook provider - backtest
            trade_executor: Abstract trade executor - backtest
        """
        self.config = config
        self.portfolio = portfolio
        self.orderbook_manager = orderbook_manager
        self.order_executor = order_executor
        self.token_ids = token_ids or {}
        self.no_token_ids = no_token_ids or {}  # NO token IDs
        self.on_trade = on_trade
        self.user_stream = user_stream
        self.sync_portfolio = sync_portfolio  # Callback to sync from API before each decision
        self.event_name = event_name or "unknown"
        self.orderbook_provider = orderbook_provider
        self.trade_executor = trade_executor

        # Reverse mapping: token_id -> bin_index (for fill callbacks)
        self.token_to_bin: Dict[str, int] = {v: k for k, v in self.token_ids.items()}
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

        # Sell-side suppression after "not enough balance / allowance".
        # Keyed by (TradeAction, bin_index) so only the failing sell is cooled down.
        self._sell_balance_error_times: Dict[tuple[TradeAction, int], float] = {}
        self._sell_balance_error_sizes: Dict[tuple[TradeAction, int], float] = {}

        # Delay after CONFIRMED before proceeding to next iteration,
        # giving the API time to propagate the fill to positions endpoint.
        self._post_confirm_delay: float = 5.0

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
                f"[{self.event_name}] Rate limit hit: {len(self._order_timestamps)} orders in last minute "
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
        br = self._bin_range(bin_index)
        bin_info = f"bin={bin_index} ({br})" if br else f"bin={bin_index}"
        logger.info(
            f"[{self.event_name}] FAK order failed for {bin_info}, cooldown for {cooldown:.0f}s"
        )

    def _sell_suppression_key(self, candidate: TradeCandidate) -> Optional[tuple[TradeAction, int]]:
        if candidate.action in (TradeAction.SELL_YES, TradeAction.SELL_NO):
            return (candidate.action, candidate.bin_index)
        return None

    def _get_position_shares_for_action(self, action: TradeAction, bin_index: int) -> float:
        pos = self.portfolio.get_position(bin_index)
        if not pos:
            return 0.0
        if action == TradeAction.SELL_YES:
            return pos.yes_shares
        if action == TradeAction.SELL_NO:
            return pos.no_shares
        return 0.0

    def _is_sell_balance_error(self, error_msg: str) -> bool:
        msg = (error_msg or "").lower()
        return "not enough balance" in msg or "allowance" in msg

    def _clear_sell_balance_error_suppressions_on_position_change(self) -> None:
        """Clear sell suppressions once synced position size changes or cooldown expires."""
        if not self._sell_balance_error_times:
            return

        now = time.time()
        cooldown = self.config.rate_limit.sell_balance_error_cooldown_seconds
        cleared: list[tuple[TradeAction, int]] = []

        for key, recorded_at in list(self._sell_balance_error_times.items()):
            action, bin_index = key
            recorded_shares = self._sell_balance_error_sizes.get(key, 0.0)
            current_shares = self._get_position_shares_for_action(action, bin_index)

            if abs(current_shares - recorded_shares) > 0.01 or now - recorded_at >= cooldown:
                cleared.append(key)

        for action, bin_index in cleared:
            key = (action, bin_index)
            self._sell_balance_error_times.pop(key, None)
            self._sell_balance_error_sizes.pop(key, None)
            br = self._bin_range(bin_index)
            bin_info = f"bin={bin_index} ({br})" if br else f"bin={bin_index}"
            logger.info(
                f"[{self.event_name}] Cleared sell suppression for {action.value} {bin_info}"
            )

    def _is_sell_in_balance_error_cooldown(self, candidate: TradeCandidate) -> bool:
        key = self._sell_suppression_key(candidate)
        if key is None:
            return False

        recorded_at = self._sell_balance_error_times.get(key)
        if recorded_at is None:
            return False

        cooldown = self.config.rate_limit.sell_balance_error_cooldown_seconds
        elapsed = time.time() - recorded_at
        if elapsed >= cooldown:
            self._sell_balance_error_times.pop(key, None)
            self._sell_balance_error_sizes.pop(key, None)
            return False
        return True

    def _record_sell_balance_error(self, candidate: TradeCandidate) -> None:
        key = self._sell_suppression_key(candidate)
        if key is None:
            return

        self._sell_balance_error_times[key] = time.time()
        self._sell_balance_error_sizes[key] = self._get_position_shares_for_action(
            candidate.action, candidate.bin_index
        )

        cooldown = self.config.rate_limit.sell_balance_error_cooldown_seconds
        br = self._bin_range(candidate.bin_index)
        bin_info = f"bin={candidate.bin_index} ({br})" if br else f"bin={candidate.bin_index}"
        logger.warning(
            f"[{self.event_name}] Sell rejected for {candidate.action.value} {bin_info}; "
            f"suppressing retries for {cooldown:.0f}s until position sync changes"
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
                f"[{self.event_name}] Past T_stop ({self.config.t_stop_hours}h before settlement). "
                "Holding positions to settlement."
            )
            return tick_result

        # Check global rate limit
        if not self._check_rate_limit():
            logger.info(f"[{self.event_name}] Rate limit reached, skipping tick")
            return tick_result

        rate_config = self.config.rate_limit

        # Iterative loop: sync → compute → execute → wait CONFIRMED + delay → repeat
        # Each iteration syncs portfolio from API for authoritative state,
        # computes optimal trades, executes a batch, waits for CONFIRMED,
        # then pauses briefly for API propagation before the next iteration.
        max_orders = rate_config.max_orders_per_tick
        iteration = 0

        while True:
            iteration += 1
            remaining_timeout = rate_config.tick_timeout_seconds - (time.time() - start_time)
            if remaining_timeout <= 0:
                logger.info(f"[{self.event_name}] Tick timeout reached after {iteration - 1} iterations")
                break

            if tick_result.num_executed >= max_orders:
                logger.info(f"[{self.event_name}] Reached max_orders_per_tick={max_orders}")
                break

            # Sync portfolio from API for up-to-date state
            if self.sync_portfolio:
                try:
                    await self.sync_portfolio()
                    self._clear_sell_balance_error_suppressions_on_position_change()
                    logger.info(
                        f"[{self.event_name}][KELLY] iter={iteration} Synced from API: "
                        f"capital=${self.portfolio.capital:.2f}, "
                        f"invested=${self.portfolio.total_collateral_used:.2f}"
                    )
                except Exception as e:
                    logger.warning(f"[{self.event_name}] Failed to sync portfolio: {e}")

            # Refresh orderbooks
            orderbooks = self._get_orderbooks()

            # Compute optimal trades on current (freshly synced) portfolio
            planned_trades = self._compute_optimal_trades(
                orderbooks, hours_to_settlement, verbose=(verbose and iteration == 1)
            )

            if not planned_trades:
                logger.info(f"[{self.event_name}] iter={iteration}: no trades to execute")
                break

            # Cap by remaining order budget for this tick
            orders_remaining = max_orders - tick_result.num_executed
            if len(planned_trades) > orders_remaining:
                planned_trades = planned_trades[:orders_remaining]

            tick_result.num_candidates += len(planned_trades)

            # Collect results for this iteration's summary
            iter_results: list[tuple[TradeCandidate, str]] = []  # (trade, status)
            num_submitted_this_iter = 0

            if self.order_executor.dry_run:
                # Dry run: simulate all fills optimistically on LIVE portfolio
                for trade in planned_trades:
                    token_id = self._get_token_id_for_action(trade)
                    if not token_id:
                        continue

                    self._update_portfolio(
                        candidate=trade,
                        token_id=token_id,
                        filled_size=trade.size,
                        filled_price=trade.price,
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
                    iter_results.append((trade, "OK (dry)"))

                    if self.on_trade:
                        self.on_trade(result)
            else:
                # Live mode: drop BUY trades whose bin also has a SELL
                # (SELL failing + BUY succeeding = dual-position deadlock)
                sell_bins = {
                    t.bin_index for t in planned_trades
                    if t.action in (TradeAction.SELL_YES, TradeAction.SELL_NO)
                }
                if sell_bins:
                    before = len(planned_trades)
                    planned_trades = [
                        t for t in planned_trades
                        if t.action not in (TradeAction.BUY_YES, TradeAction.BUY_NO)
                        or t.bin_index not in sell_bins
                    ]
                    dropped = before - len(planned_trades)
                    if dropped:
                        logger.info(
                            "Dropped %d BUY trade(s) on bins with pending SELLs: %s",
                            dropped, sell_bins,
                        )

                # Build and submit batch
                order_specs = []
                trade_token_pairs = []
                for trade in planned_trades:
                    token_id = self._get_token_id_for_action(trade)
                    if not token_id:
                        continue

                    side = "BUY" if trade.action in (TradeAction.BUY_YES, TradeAction.BUY_NO) else "SELL"
                    fak_price = trade.limit_price if trade.limit_price > 0 else trade.price
                    order_specs.append({
                        "token_id": token_id,
                        "side": side,
                        "price": fak_price,
                        "size": trade.size,
                    })
                    trade_token_pairs.append((trade, token_id))

                if not order_specs:
                    break

                # Merge orders for same (token_id, side, price)
                merged_specs = []
                merged_trade_pairs = []
                merge_key_to_idx = {}
                for spec, (trade, token_id) in zip(order_specs, trade_token_pairs):
                    key = (spec["token_id"], spec["side"], spec["price"])
                    if key in merge_key_to_idx:
                        idx = merge_key_to_idx[key]
                        merged_specs[idx]["size"] += spec["size"]
                        merged_trade_pairs[idx].append((trade, token_id))
                    else:
                        merge_key_to_idx[key] = len(merged_specs)
                        merged_specs.append(dict(spec))
                        merged_trade_pairs.append([(trade, token_id)])

                order_specs = merged_specs
                trade_token_pairs = []
                for spec, pairs in zip(merged_specs, merged_trade_pairs):
                    trade, token_id = pairs[0]
                    trade.size = spec["size"]
                    trade_token_pairs.append((trade, token_id))

                # Batch submit
                batch_response = self.order_executor.place_batch_orders(order_specs)

                # Process batch response
                saw_sell_balance_error = False
                for i, (trade, token_id) in enumerate(trade_token_pairs):
                    resp = batch_response[i] if i < len(batch_response) else {}
                    if resp.get("_skipped"):
                        iter_results.append((trade, "SKIPPED"))
                        continue

                    oid = resp.get("orderID")
                    if oid == "":
                        oid = None
                    order_id = oid
                    error_msg = resp.get("errorMsg", "")

                    if not order_id and error_msg:
                        self._record_fak_failure(trade.bin_index)
                        if self._is_sell_balance_error(error_msg) and trade.action in (
                            TradeAction.SELL_YES,
                            TradeAction.SELL_NO,
                        ):
                            self._record_sell_balance_error(trade)
                            self.order_executor.log_sell_balance_diagnostics(
                                token_id=token_id,
                                requested_size=trade.size,
                            )
                            saw_sell_balance_error = True

                    if order_id:
                        self._pending_orders[order_id] = (trade, token_id)
                        confirm_event = asyncio.Event()
                        self._confirmation_events[order_id] = confirm_event

                        if self.user_stream:
                            from .user_stream import PendingOrder
                            pending = PendingOrder(
                                order_id=order_id,
                                token_id=token_id,
                                side="BUY" if trade.action in (TradeAction.BUY_YES, TradeAction.BUY_NO) else "SELL",
                                price=trade.price,
                                size=trade.size,
                                bin_index=trade.bin_index,
                                condition_id=token_id,  # Token uniquely identifies the market condition
                            )
                            asyncio.create_task(self.user_stream.add_pending_order(pending))

                        self._log_trade_placed(trade, token_id, order_id)
                        self._record_order()
                        num_submitted_this_iter += 1
                        iter_results.append((trade, "SUBMITTED"))

                        result = ExecutionResult(
                            success=True,
                            candidate=trade,
                            order_id=order_id,
                            is_pending=True,
                        )
                    else:
                        iter_results.append((trade, f"FAILED: {error_msg}"))
                        result = ExecutionResult(
                            success=False,
                            candidate=trade,
                            error=error_msg or "No order_id in batch response",
                            is_pending=False,
                        )

                    tick_result.executions.append(result)
                    if result.success:
                        tick_result.num_executed += 1
                        tick_result.total_utility_gain += trade.utility_gain

                    if self.on_trade:
                        self.on_trade(result)

                if saw_sell_balance_error and self.sync_portfolio:
                    try:
                        await self.sync_portfolio()
                        self._clear_sell_balance_error_suppressions_on_position_change()
                        logger.info(
                            f"[{self.event_name}] Re-synced portfolio after sell balance/allowance rejection"
                        )
                    except Exception as e:
                        logger.warning(
                            f"[{self.event_name}] Failed to re-sync after sell balance/allowance rejection: {e}"
                        )

                # Wait for CONFIRMED on all orders from this iteration (or tick timeout)
                if self._confirmation_events:
                    remaining_timeout = rate_config.tick_timeout_seconds - (time.time() - start_time)
                    if remaining_timeout > 0:
                        try:
                            await asyncio.wait_for(
                                self._wait_all_confirmations(),
                                timeout=remaining_timeout,
                            )
                            # Update statuses for confirmed orders
                            for j, (t, s) in enumerate(iter_results):
                                if s == "SUBMITTED":
                                    iter_results[j] = (t, "CONFIRMED")
                        except asyncio.TimeoutError:
                            for j, (t, s) in enumerate(iter_results):
                                if s == "SUBMITTED":
                                    iter_results[j] = (t, "TIMEOUT")
                            self._confirmation_events.clear()
                            # Log summary then stop — tick timeout
                            iter_elapsed = time.time() - start_time
                            self._log_iter_summary(iteration, iter_elapsed, iter_results)
                            break


            # Log iteration summary
            self._log_iter_summary(iteration, time.time() - start_time, iter_results)

            # Post-confirmation delay for API propagation before next iteration
            if not self.order_executor.dry_run and num_submitted_this_iter > 0:
                remaining_timeout = rate_config.tick_timeout_seconds - (time.time() - start_time)
                if remaining_timeout > self._post_confirm_delay:
                    await asyncio.sleep(self._post_confirm_delay)
                else:
                    break

        # Tick summary
        logger.info(
            f"[{self.event_name}] TICK COMPLETE: {iteration} iterations, "
            f"{tick_result.num_executed} executed, "
            f"{tick_result.num_candidates} candidates | "
            f"elapsed={time.time() - start_time:.1f}s"
        )

        # Tick summary logged by caller (_log_tick_result in trading_bot.py)

        tick_result.elapsed_seconds = time.time() - start_time
        self._last_tick_time = time.time()

        return tick_result

    def run_tick_sync(
        self,
        hours_to_settlement: float,
        verbose: bool = False,
    ) -> TickResult:
        """
        Run a single optimization tick synchronously (for backtest use).

        Uses the same _compute_optimal_trades pipeline as production (including
        _find_optimal_size_on binary search), then executes via the abstract
        trade_executor backend.

        Flow mirrors production run_tick: execute batch → refresh orderbooks → recompute → repeat.
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

        rate_config = self.config.rate_limit
        max_orders = rate_config.max_orders_per_tick
        iteration = 0

        while True:
            iteration += 1

            if tick_result.num_executed >= max_orders:
                logger.debug(f"Reached max_orders_per_tick={max_orders}")
                break

            # Get orderbooks via abstract backend
            orderbooks = self._get_orderbooks()
            if not orderbooks:
                logger.debug("No orderbooks available")
                break

            # Compute optimal trades (greedy simulation on portfolio copy with binary search)
            planned_trades = self._compute_optimal_trades(
                orderbooks, hours_to_settlement, verbose=(verbose and iteration == 1)
            )

            if not planned_trades:
                logger.debug(f"iter={iteration}: no trades to execute")
                break

            # Cap by remaining order budget
            orders_remaining = max_orders - tick_result.num_executed
            if len(planned_trades) > orders_remaining:
                planned_trades = planned_trades[:orders_remaining]

            tick_result.num_candidates += len(planned_trades)

            # Execute each planned trade via abstract backend
            executed_any = False
            for trade in planned_trades:
                token_id = self._get_token_id_for_action(trade)
                if not token_id:
                    # Fallback for backtest (no NO token IDs)
                    token_id = self.token_ids.get(trade.bin_index, f"token_{trade.bin_index}")

                from .backend import ExecutionResult as BackendExecutionResult
                backend_result = self.trade_executor.execute(trade, token_id)

                result = ExecutionResult(
                    success=backend_result.success,
                    candidate=trade,
                    order_id=backend_result.order_id,
                    filled_size=backend_result.filled_size,
                    filled_price=backend_result.filled_price,
                    error=backend_result.error,
                    is_pending=False,
                )
                tick_result.executions.append(result)

                if result.success:
                    tick_result.num_executed += 1
                    tick_result.total_utility_gain += trade.utility_gain
                    executed_any = True

                    if self.on_trade:
                        self.on_trade(result)

                    logger.debug(
                        f"Executed: {trade.action.value} bin={trade.bin_index} "
                        f"size={trade.size:.2f} @ {trade.price:.4f} "
                        f"utility_gain={trade.utility_gain:.6f} edge={trade.edge:.2%}"
                    )
                else:
                    logger.warning(f"Execution failed: {result.error}")

            if not executed_any:
                break

            # Refresh orderbooks for next iteration (re-sync via abstract backend)
            if self.orderbook_provider is not None:
                self.orderbook_provider.refresh()

        tick_result.elapsed_seconds = time.time() - start_time
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

        logger.debug(
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

            candidates = [
                c for c in candidates
                if not self._is_sell_in_balance_error_cooldown(c)
            ]

            if not candidates:
                logger.debug(f"[{self.event_name}][SIM iter={sim_iter}] No candidates")
                break

            # Pick best candidate (sells come first, then buys by utility/$)
            # Screening uses $2 chunk so utility_gain is tiny — just require positive.
            # The real min_utility check happens after optimal sizing below.
            best = candidates[0]

            if best.utility_gain <= 0:
                logger.debug(
                    f"[{self.event_name}][SIM iter={sim_iter}] Best candidate utility "
                    f"{best.utility_gain:.6f} <= 0, stopping"
                )
                break

            # Skip bins in FAK cooldown
            if self._is_bin_in_fak_cooldown(best.bin_index):
                candidates = [c for c in candidates if c.bin_index != best.bin_index]
                if not candidates:
                    break
                best = candidates[0]
                if best.utility_gain <= 0:
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

            # Compute actual utility gain for the optimally-sized trade
            hyp_after = self._simulate_trade(hyp, best)
            from .candidates import _compute_portfolio_utility_gain
            actual_utility = _compute_portfolio_utility_gain(hyp, hyp_after, self.config)

            is_sell = best.action in (TradeAction.SELL_YES, TradeAction.SELL_NO)
            min_util = self.config.min_sell_utility if is_sell else self.config.min_buy_utility
            if actual_utility < min_util:
                logger.debug(
                    f"[{self.event_name}][SIM iter={sim_iter}] Optimal-size utility "
                    f"{actual_utility:.6f} < {min_util} for {best.action.value} "
                    f"bin={best.bin_index}, stopping"
                )
                break

            best.utility_gain = actual_utility
            hyp = hyp_after

            planned_trades.append(best)

            # Log
            br = self._bin_range(best.bin_index)
            bin_info = f"bin={best.bin_index} ({br})" if br else f"bin={best.bin_index}"
            logger.info(
                f"[{self.event_name}][SIM iter={sim_iter}] {best.action.value} {bin_info} "
                f"size={optimal_size:.0f} @ {best.price:.4f} util={best.utility_gain:.6f}"
            )

        if planned_trades:
            # Merge trades for the same (bin_index, action) into single orders.
            # The greedy loop may produce many small trades for the same bin
            # (e.g., 25 × 2-share BUY_NO for bin 14). Merge them so execution
            # is one order per (bin, action).
            merged = self._merge_planned_trades(planned_trades)
            logger.info(
                f"[{self.event_name}] Simulation complete: {len(planned_trades)} trades planned, "
                f"{len(merged)} after merging"
            )
            return merged
        else:
            logger.debug(f"[{self.event_name}] Simulation complete: no trades needed")
            return planned_trades

    def _merge_planned_trades(self, trades: List[TradeCandidate]) -> List[TradeCandidate]:
        """
        Merge planned trades for the same (bin_index, action) into single orders.

        The greedy simulation may produce many small trades for the same bin
        (e.g., 25 × 2-share BUY_NO). Merging them into one order per (bin, action)
        is more efficient for execution.

        Uses VWAP for price and sums sizes and utility gains.
        """
        from collections import OrderedDict
        merged: OrderedDict[tuple, TradeCandidate] = OrderedDict()

        for t in trades:
            key = (t.bin_index, t.action)
            if key in merged:
                existing = merged[key]
                # VWAP: weighted average price
                total_size = existing.size + t.size
                if total_size > 0:
                    existing.price = (existing.price * existing.size + t.price * t.size) / total_size
                existing.size = total_size
                existing.utility_gain += t.utility_gain
                # Keep the worst limit_price (highest for buys, lowest for sells)
                if t.action in (TradeAction.BUY_YES, TradeAction.BUY_NO):
                    existing.limit_price = max(existing.limit_price, t.limit_price)
                else:
                    existing.limit_price = min(existing.limit_price, t.limit_price) if existing.limit_price > 0 else t.limit_price
            else:
                # Clone to avoid mutating the original
                merged[key] = TradeCandidate(
                    bin_index=t.bin_index,
                    action=t.action,
                    size=t.size,
                    price=t.price,
                    utility_gain=t.utility_gain,
                    reservation_price=t.reservation_price,
                    edge=t.edge,
                    limit_price=t.limit_price,
                )

        return list(merged.values())

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
        """Get the token_id for a candidate's action (YES or NO token).

        For backtest mode (no NO token IDs), returns None so the caller can
        fall back to the YES token_id (which is just a placeholder string).
        """
        if candidate.action in (TradeAction.BUY_NO, TradeAction.SELL_NO):
            token_id = self.no_token_ids.get(candidate.bin_index)
            if not token_id and self.order_executor is not None:
                # Only warn in production mode (backtest uses fallback)
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
        if self.orderbook_provider is not None:
            return self.orderbook_provider.get_all_orderbooks()
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
        br = self._bin_range(bin_index)
        bin_label = f"bin={bin_index} ({br})" if br else f"bin={bin_index}"
        logger.info(
            f"[{self.event_name}][PORTFOLIO UPDATE] {action.value} {bin_label} | "
            f"{size:.1f} @ {price:.4f} = ${size * price:.2f} | "
            f"invested: ${collateral_before:.2f} -> ${collateral_after:.2f} | "
            f"position: {shares_str}"
        )

    def handle_fill(self, fill_event: "FillEvent") -> None:
        """
        Handle fill confirmation from WebSocket.

        Called by UserStreamClient when a trade fill is confirmed.

        Logs fill events and signals confirmation on CONFIRMED status.
        Portfolio state is NOT updated here — we rely on API sync before
        each iteration for authoritative state.

        Args:
            fill_event: FillEvent from WebSocket
        """
        order_id = fill_event.order_id
        if not order_id:
            logger.warning(f"[{self.event_name}] Received fill event with no order_id")
            return

        # Look up the pending order
        pending_info = self._pending_orders.get(order_id)
        if not pending_info:
            # This fill might be from a previous session or manual order
            logger.info(f"[{self.event_name}] Fill for unknown order {order_id[:16]}... - may be from previous session or manual trade")
            return

        candidate, token_id = pending_info

        from .user_stream import OrderStatus
        status_str = fill_event.status.name if hasattr(fill_event.status, 'name') else str(fill_event.status)
        br = self._bin_range(candidate.bin_index)
        bin_label = f"bin={candidate.bin_index} ({br})" if br else f"bin={candidate.bin_index}"
        logger.info(
            f"[{self.event_name}][FILL {status_str}] {candidate.action.value} {bin_label} | "
            f"{fill_event.size:.1f} @ {fill_event.price:.4f} = ${fill_event.size * fill_event.price:.2f}"
        )

        # Signal confirmation on CONFIRMED (final status).
        # We wait for CONFIRMED (not MINED) so the API has more time
        # to propagate the fill before we sync portfolio state.
        if fill_event.status == OrderStatus.CONFIRMED:
            if order_id in self._confirmation_events:
                self._confirmation_events[order_id].set()
                del self._confirmation_events[order_id]
            if order_id in self._pending_orders:
                del self._pending_orders[order_id]
            logger.debug(f"Order {order_id[:16]}... confirmed, signaling and removing from pending")

    async def handle_stale_order(self, pending: "PendingOrder") -> None:
        """
        Handle stale order cancellation.

        Called by UserStreamClient when an order hasn't filled within timeout.
        Cancels the order to free up collateral.

        Args:
            pending: PendingOrder that is stale
        """
        order_id = pending.order_id

        br = self._bin_range(pending.bin_index)
        bin_label = f"bin={pending.bin_index} ({br})" if br else f"bin={pending.bin_index}"
        logger.warning(
            f"[{self.event_name}] Cancelling stale order: {order_id[:16]}..., "
            f"{bin_label}, size={pending.size:.2f} @ {pending.price:.4f}"
        )

        # Cancel via order executor (may fail for FAK orders already killed by exchange)
        success = self.order_executor.cancel_order(order_id)

        if success:
            logger.info(f"[{self.event_name}] Stale order {order_id[:16]}... cancelled successfully")
        else:
            logger.warning(f"[{self.event_name}] Cancel failed for stale order {order_id[:16]}... (likely already dead FAK)")

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

    def _log_iter_summary(
        self,
        iteration: int,
        elapsed: float,
        results: list,  # list of (TradeCandidate, status_str)
    ) -> None:
        """Log a clear summary at the end of each iteration."""
        logger.info(
            f"[{self.event_name}] --- Iteration {iteration} ({elapsed:.1f}s) ---"
        )
        for trade, status in results:
            br = self._bin_range(trade.bin_index)
            bin_label = f"bin {trade.bin_index} ({br})" if br else f"bin {trade.bin_index}"
            side = "BUY" if trade.action in (TradeAction.BUY_YES, TradeAction.BUY_NO) else "SELL"
            token_type = "YES" if trade.action in (TradeAction.BUY_YES, TradeAction.SELL_YES) else "NO"
            logger.info(
                f"  {side} {token_type} {bin_label} | "
                f"{trade.size:.0f} @ {trade.price:.4f} = ${trade.size * trade.price:.2f} | "
                f"edge={trade.edge:+.1%} | {status}"
            )

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

        br = self._bin_range(candidate.bin_index)
        bin_label = f"bin={candidate.bin_index} ({br})" if br else f"bin={candidate.bin_index}"
        logger.info(
            f"[{self.event_name}][FILL] {now} | {candidate.action.value} {bin_label} | "
            f"{filled_size:.1f} @ {filled_price:.3f} (req: {candidate.size:.1f} @ {candidate.price:.3f}) | "
            f"slip={slippage_bps:.0f}bps"
        )
        logger.info(
            f"[{self.event_name}]  -> portfolio: ${capital:.2f} capital, ${total_collateral:.2f} collateral | "
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
        logger.info(f"[{self.event_name}] Starting continuous Kelly optimization")

        while self._running:
            if stop_event and stop_event.is_set():
                break

            try:
                hours = get_hours_to_settlement()

                # Stop if past T_stop
                if hours <= self.config.t_stop_hours:
                    logger.info(f"[{self.event_name}] Reached T_stop. Stopping optimization.")
                    break

                # Run tick
                result = await self.run_tick(hours)

                logger.info(
                    f"[{self.event_name}] Tick complete: candidates={result.num_candidates}, "
                    f"executed={result.num_executed}, "
                    f"utility_gain={result.total_utility_gain:.6f}, "
                    f"elapsed={result.elapsed_seconds:.2f}s"
                )

                # Wait for next tick
                await asyncio.sleep(tick_interval_seconds)

            except Exception as e:
                logger.error(f"[{self.event_name}] Error in tick: {e}")
                await asyncio.sleep(tick_interval_seconds)

        self._running = False
        logger.info(f"[{self.event_name}] Kelly optimization stopped")

    def stop(self) -> None:
        """Signal executor to stop."""
        self._running = False

    def get_portfolio_summary(self) -> dict:
        """Get current portfolio summary."""
        return self.portfolio.to_summary()
