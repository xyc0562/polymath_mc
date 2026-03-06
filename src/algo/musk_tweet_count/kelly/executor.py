"""
Greedy executor for Kelly trading strategy.

Implements the greedy execution loop:
1. Generate trade candidates
2. Pick best candidate by utility gain
3. Execute trade (place order)
4. Wait for fill confirmation via WebSocket
5. Reconcile confirmed fills against the positions API
6. Repeat until no profitable trades or max iterations

Order execution via py_clob_client.
The positions API remains the only authoritative portfolio state. Confirmed
WebSocket fills are stored in a temporary overlay for planning until the API
catches up, which prevents duplicate rebuys when API propagation lags.
"""

import asyncio
import logging
import math
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType, PostOrdersArgs

from .config import KellyConfig
from .orderbook import UnifiedOrderbook
from .orderbook import (
    compute_vwap_buy_no,
    compute_vwap_buy_yes,
    compute_vwap_sell_no,
    compute_vwap_sell_yes,
)
from .portfolio import Portfolio
from .candidates import (
    TradeCandidate,
    TradeAction,
    generate_candidates,
    MIN_ORDER_SIZE,
    MIN_ORDER_VALUE_USD,
)
from .websocket_client import OrderbookManager
from .market_aware import compute_robust_kelly_fraction

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


@dataclass
class OverlayFillFragment:
    """Confirmed fill awaiting reconciliation with the positions API."""

    fill_key: str
    order_id: str
    token_id: str
    action: TradeAction
    bin_index: int
    price: float
    original_size: float
    remaining_size: float
    confirmed_at: float

    @property
    def is_buy(self) -> bool:
        return self.action in (TradeAction.BUY_YES, TradeAction.BUY_NO)

    @property
    def position_kind(self) -> str:
        return "YES" if self.action in (TradeAction.BUY_YES, TradeAction.SELL_YES) else "NO"


@dataclass
class RecentConfirmedFillRecord:
    """Recent confirmed fill used for duplicate detection."""

    fill_key: str
    order_id: str
    token_id: str
    action: TradeAction
    price: float
    size: float
    recorded_at: float


@dataclass
class RecentOrderContext:
    """Cached order context so late duplicate CONFIRMED fills stay attributable."""

    order_id: str
    candidate: TradeCandidate
    token_id: str
    created_at: float


@dataclass
class IntegrityState:
    """Per-event integrity state for overlay reconciliation."""

    frozen: bool = False
    reason: Optional[str] = None
    frozen_at: Optional[float] = None
    deadline_at: Optional[float] = None
    last_forced_api_recovery_at: Optional[float] = None
    unmatched_api_delta_count: int = 0


@dataclass
class BalanceAllowanceErrorContext:
    """Structured context for balance / allowance order rejections."""

    event_name: str
    action: str
    side: str
    token_type: str
    bin_index: int
    bin_range: str
    token_id: str
    requested_size: float
    requested_price: float
    requested_limit_price: float
    requested_notional: float
    reservation_price: float
    edge: float
    utility_gain: float
    error: str
    portfolio_available_capital: Optional[float] = None
    portfolio_total_collateral: Optional[float] = None
    pending_orders_count: int = 0
    local_yes_shares: float = 0.0
    local_no_shares: float = 0.0
    local_yes_avg_cost: float = 0.0
    local_no_avg_cost: float = 0.0
    clob_available_shares: Optional[float] = None
    nonzero_allowances: Optional[int] = None
    raw_balance: Optional[str] = None
    allowances: Optional[Dict[str, Any]] = None


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

    def get_sell_balance_diagnostics(
        self,
        token_id: str,
        requested_size: float,
    ) -> Dict[str, Any]:
        """Return exchange-side diagnostics for sell balance / allowance failures."""
        diagnostics: Dict[str, Any] = {
            "token_id": token_id,
            "requested_size": requested_size,
            "clob_available_shares": None,
            "nonzero_allowances": None,
            "raw_balance": None,
            "allowances": None,
        }
        balance_info = self._fetch_conditional_balance_allowance(token_id=token_id, refresh=True)
        if not isinstance(balance_info, dict):
            return diagnostics

        available_shares = self._parse_conditional_balance_shares(balance_info)
        allowances = balance_info.get("allowances", {})
        nonzero_spenders = 0
        if isinstance(allowances, dict):
            for value in allowances.values():
                try:
                    if int(value) > 0:
                        nonzero_spenders += 1
                except Exception:
                    continue

        diagnostics.update(
            {
                "clob_available_shares": available_shares,
                "nonzero_allowances": nonzero_spenders,
                "raw_balance": balance_info.get("balance"),
                "allowances": allowances if isinstance(allowances, dict) else None,
            }
        )
        return diagnostics

    def log_sell_balance_diagnostics(self, token_id: str, requested_size: float) -> Dict[str, Any]:
        """
        Log CLOB conditional balance/allowance snapshot after sell rejection.

        This helps distinguish local-position mismatch from exchange available-balance issues.
        """
        diagnostics = self.get_sell_balance_diagnostics(
            token_id=token_id,
            requested_size=requested_size,
        )
        available_shares = diagnostics.get("clob_available_shares")
        nonzero_spenders = diagnostics.get("nonzero_allowances")
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
        return diagnostics

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
    3. Track as pending order
    4. Record confirmed fills in an overlay ledger
    5. Reconcile the overlay against API position deltas
    6. Repeat

    IMPORTANT:
    - `self.portfolio` is the authoritative API-synced base portfolio.
    - Confirmed WebSocket fills never mutate the base portfolio directly.
    - Planning uses an effective portfolio derived from `API base + overlay`.

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
        on_balance_allowance_error: Optional[Callable[[BalanceAllowanceErrorContext], None]] = None,
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
            on_balance_allowance_error: Optional callback for balance / allowance rejections
            user_stream: Optional UserStreamClient for fill confirmations
            sync_portfolio: Callback to sync portfolio from API before each decision
            event_name: Optional event name for logging (e.g., "Feb 03 - Feb 10")
            orderbook_provider: Abstract orderbook provider - backtest
            trade_executor: Abstract trade executor - backtest
        """
        self.config = config
        self.portfolio = portfolio
        self.api_base_portfolio = portfolio
        self.orderbook_manager = orderbook_manager
        self.order_executor = order_executor
        self.token_ids = token_ids or {}
        self.no_token_ids = no_token_ids or {}  # NO token IDs
        self.on_trade = on_trade
        self.on_balance_allowance_error = on_balance_allowance_error
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

        # Delay after CONFIRMED before proceeding to next iteration,
        # giving the API time to propagate the fill to positions endpoint.
        self._post_confirm_delay: float = 5.0

        # Authoritative API snapshot from the previous sync. Used to compute
        # observed API deltas and reconcile confirmed local fills.
        self._last_api_snapshot: Optional[Portfolio] = None

        # Confirmed-but-unreconciled fills. These are replayed on top of the
        # API base portfolio when sizing the next iteration.
        self._overlay_ledger: List[OverlayFillFragment] = []

        # Duplicate confirmed fill protection survives beyond pending-order
        # lifetime because the user stream can resend late CONFIRMED messages.
        self._recent_confirmed_fill_keys: Dict[str, RecentConfirmedFillRecord] = {}

        # Order context also outlives pending-order lifetime so late duplicate
        # CONFIRMED fills can be attributed and deduped instead of being noisy.
        self._recent_order_context: Dict[str, RecentOrderContext] = {}

        # Integrity state is event-local and runtime-only.
        self._integrity_state = IntegrityState()

        self._recent_tracking_ttl_seconds = 30.0 * 60.0
        self._recent_tracking_cap = 10_000
        self._overlay_size_epsilon = 1e-6

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

    def _get_fak_cooldown_remaining(self, bin_index: int) -> Optional[float]:
        """Get remaining FAK cooldown seconds for a bin, or None if inactive."""
        if bin_index not in self._fak_failure_times:
            return None

        cooldown = self.config.rate_limit.fak_failure_cooldown_seconds
        elapsed = time.time() - self._fak_failure_times[bin_index]
        remaining = cooldown - elapsed

        if remaining > 0:
            return remaining

        del self._fak_failure_times[bin_index]
        return None

    def _record_fak_failure(self, bin_index: int) -> None:
        """Record a FAK order failure for cooldown tracking."""
        self._fak_failure_times[bin_index] = time.time()
        cooldown = self.config.rate_limit.fak_failure_cooldown_seconds
        br = self._bin_range(bin_index)
        bin_info = f"bin={bin_index} ({br})" if br else f"bin={bin_index}"
        logger.info(
            f"[{self.event_name}] FAK order failed for {bin_info}, cooldown for {cooldown:.0f}s"
        )

    def _is_sell_balance_error(self, error_msg: str) -> bool:
        msg = (error_msg or "").lower()
        return "not enough balance" in msg or "allowance" in msg

    def _emit_balance_allowance_error(
        self,
        candidate: TradeCandidate,
        token_id: str,
        error_msg: str,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit structured context for balance / allowance rejections."""
        if not self.on_balance_allowance_error:
            return

        position = self.portfolio.get_position(candidate.bin_index) if self.portfolio else None
        side = "BUY" if candidate.action in (TradeAction.BUY_YES, TradeAction.BUY_NO) else "SELL"
        token_type = "YES" if candidate.action in (TradeAction.BUY_YES, TradeAction.SELL_YES) else "NO"
        context = BalanceAllowanceErrorContext(
            event_name=self.event_name,
            action=candidate.action.value,
            side=side,
            token_type=token_type,
            bin_index=candidate.bin_index,
            bin_range=self._bin_range(candidate.bin_index),
            token_id=token_id,
            requested_size=candidate.size,
            requested_price=candidate.price,
            requested_limit_price=candidate.limit_price if candidate.limit_price > 0 else candidate.price,
            requested_notional=candidate.size * candidate.price,
            reservation_price=candidate.reservation_price,
            edge=candidate.edge,
            utility_gain=candidate.utility_gain,
            error=error_msg,
            portfolio_available_capital=self.portfolio.available_capital if self.portfolio else None,
            portfolio_total_collateral=self.portfolio.total_collateral_used if self.portfolio else None,
            pending_orders_count=len(self._pending_orders),
            local_yes_shares=position.yes_shares if position else 0.0,
            local_no_shares=position.no_shares if position else 0.0,
            local_yes_avg_cost=position.yes_avg_cost if position else 0.0,
            local_no_avg_cost=position.no_avg_cost if position else 0.0,
            clob_available_shares=(diagnostics or {}).get("clob_available_shares"),
            nonzero_allowances=(diagnostics or {}).get("nonzero_allowances"),
            raw_balance=(diagnostics or {}).get("raw_balance"),
            allowances=(diagnostics or {}).get("allowances"),
        )
        try:
            self.on_balance_allowance_error(context)
        except Exception as e:
            logger.warning(
                f"[{self.event_name}] Balance/allowance error callback failed: {e}"
            )

    def _warm_balance_cache(self, token_id: str) -> None:
        """Proactively refresh CLOB balance cache for a token after a BUY fill."""
        if not self.order_executor or not token_id:
            return
        try:
            self.order_executor._fetch_conditional_balance_allowance(
                token_id=token_id, refresh=True,
            )
        except Exception as e:
            logger.debug(
                f"[{self.event_name}] Balance cache warm failed for "
                f"token {token_id[:16]}...: {e}"
            )

    @staticmethod
    def _clone_candidate(candidate: TradeCandidate) -> TradeCandidate:
        """Clone a trade candidate for durable order/fill bookkeeping."""
        return TradeCandidate(
            bin_index=candidate.bin_index,
            action=candidate.action,
            size=candidate.size,
            price=candidate.price,
            utility_gain=candidate.utility_gain,
            reservation_price=candidate.reservation_price,
            edge=candidate.edge,
            limit_price=candidate.limit_price,
            threshold_price=candidate.threshold_price,
        )

    def _fresh_start_age_seconds(self, now: Optional[float] = None) -> Optional[float]:
        """Return throttle age in seconds from the first trading tick, if known."""
        raw = self._log_context.get("fresh_start_started_at")
        if raw is None:
            return None
        try:
            started_at = float(raw)
        except (TypeError, ValueError):
            return None

        current_time = now if now is not None else time.time()
        return max(0.0, current_time - started_at)

    def _fresh_start_throttle_active(self, now: Optional[float] = None) -> bool:
        """Return True when the fresh-start wall-clock throttle window is active."""
        market_impact = getattr(self.config, "market_impact", None)
        if market_impact is None or not market_impact.fresh_start_enabled:
            return False

        age_seconds = self._fresh_start_age_seconds(now)
        if age_seconds is None:
            return False

        return age_seconds <= market_impact.fresh_start_minutes * 60.0

    @staticmethod
    def _best_execution_price(
        orderbook: UnifiedOrderbook,
        action: TradeAction,
    ) -> Optional[float]:
        """Return top-of-book price for the given trade action."""
        if action == TradeAction.BUY_YES:
            return orderbook.best_yes_ask
        if action == TradeAction.SELL_YES:
            return orderbook.best_yes_bid
        if action == TradeAction.BUY_NO:
            return orderbook.best_no_ask
        if action == TradeAction.SELL_NO:
            return orderbook.best_no_bid
        return None

    @staticmethod
    def _compute_trade_path_metrics(
        orderbook: UnifiedOrderbook,
        action: TradeAction,
        size: float,
    ) -> tuple[float, float, float]:
        """Return (vwap, filled_size, worst_price) for executing size shares."""
        if action == TradeAction.BUY_YES:
            return compute_vwap_buy_yes(orderbook, size)
        if action == TradeAction.SELL_YES:
            return compute_vwap_sell_yes(orderbook, size)
        if action == TradeAction.BUY_NO:
            return compute_vwap_buy_no(orderbook, size)
        if action == TradeAction.SELL_NO:
            return compute_vwap_sell_no(orderbook, size)
        return 0.0, 0.0, 0.0

    @staticmethod
    def _within_market_impact_budget(
        action: TradeAction,
        worst_price: float,
        allowed_worst_price: float,
    ) -> bool:
        """Check whether the deepest consumed level stays inside the price budget."""
        eps = 1e-9
        if action in (TradeAction.BUY_YES, TradeAction.BUY_NO):
            return worst_price <= allowed_worst_price + eps
        return worst_price + eps >= allowed_worst_price

    def _apply_fresh_start_market_impact(
        self,
        trades: List[TradeCandidate],
        orderbooks: Dict[int, UnifiedOrderbook],
    ) -> List[TradeCandidate]:
        """Clip executable size during the opening wall-clock window of an event."""
        now = time.time()
        if not self._fresh_start_throttle_active(now):
            return trades

        market_impact = self.config.market_impact
        alpha = min(1.0, max(0.0, market_impact.fresh_start_edge_fraction))
        age_seconds = self._fresh_start_age_seconds(now) or 0.0
        throttled: List[TradeCandidate] = []

        for trade in trades:
            orderbook = orderbooks.get(trade.bin_index)
            if orderbook is None:
                throttled.append(trade)
                continue

            best_price = self._best_execution_price(orderbook, trade.action)
            if best_price is None:
                throttled.append(trade)
                continue

            threshold_price = trade.threshold_price
            if trade.action in (TradeAction.BUY_YES, TradeAction.BUY_NO):
                allowed_worst_price = best_price + alpha * (threshold_price - best_price)
            else:
                allowed_worst_price = best_price - alpha * (best_price - threshold_price)

            requested_size = max(0, math.floor(trade.size))
            if requested_size < 1:
                logger.info(
                    f"[{self.event_name}] Fresh-start throttle skipped {trade.action.value} "
                    f"bin={trade.bin_index}: requested size {trade.size:.2f} < 1 share"
                )
                continue

            def evaluate(size: int) -> Optional[tuple[float, float, float]]:
                if size < 1:
                    return None
                vwap, filled, worst_price = self._compute_trade_path_metrics(
                    orderbook, trade.action, float(size)
                )
                if filled + 1e-9 < size or worst_price <= 0:
                    return None
                return vwap, filled, worst_price

            best_valid_size = 0
            best_valid_metrics: Optional[tuple[float, float, float]] = None

            full_metrics = evaluate(requested_size)
            if full_metrics and self._within_market_impact_budget(
                trade.action,
                full_metrics[2],
                allowed_worst_price,
            ):
                best_valid_size = requested_size
                best_valid_metrics = full_metrics
            else:
                lo = 1
                hi = requested_size
                while lo <= hi:
                    mid = (lo + hi) // 2
                    metrics = evaluate(mid)
                    if metrics and self._within_market_impact_budget(
                        trade.action,
                        metrics[2],
                        allowed_worst_price,
                    ):
                        best_valid_size = mid
                        best_valid_metrics = metrics
                        lo = mid + 1
                    else:
                        hi = mid - 1

            if best_valid_size < 1 or best_valid_metrics is None:
                logger.info(
                    f"[{self.event_name}] Fresh-start throttle skipped {trade.action.value} "
                    f"bin={trade.bin_index}: top={best_price:.4f} threshold={threshold_price:.4f} "
                    f"allowed={allowed_worst_price:.4f} age={age_seconds:.0f}s "
                    f"reason=fresh_start_throttle"
                )
                continue

            clipped_trade = self._clone_candidate(trade)
            original_size = trade.size
            clipped_trade.size = float(best_valid_size)
            clipped_trade.price = best_valid_metrics[0]
            clipped_trade.limit_price = best_valid_metrics[2]

            if clipped_trade.action in (TradeAction.BUY_YES, TradeAction.BUY_NO):
                maker_amount = clipped_trade.size * clipped_trade.limit_price
                if maker_amount + 1e-9 < MIN_ORDER_VALUE_USD:
                    logger.info(
                        f"[{self.event_name}] Fresh-start throttle skipped {trade.action.value} "
                        f"bin={trade.bin_index}: clipped maker_amount=${maker_amount:.4f} "
                        f"< ${MIN_ORDER_VALUE_USD:.2f} age={age_seconds:.0f}s "
                        f"reason=fresh_start_throttle"
                    )
                    continue

            if clipped_trade.size + 1e-9 < original_size:
                if original_size > 0:
                    clipped_trade.utility_gain *= clipped_trade.size / original_size
                logger.info(
                    f"[{self.event_name}] Fresh-start throttle clipped {trade.action.value} "
                    f"bin={trade.bin_index}: size {original_size:.2f} -> {clipped_trade.size:.2f} | "
                    f"top={best_price:.4f} threshold={threshold_price:.4f} "
                    f"allowed={allowed_worst_price:.4f} vwap={clipped_trade.price:.4f} "
                    f"worst={clipped_trade.limit_price:.4f} age={age_seconds:.0f}s "
                    f"reason=fresh_start_throttle"
                )

            throttled.append(clipped_trade)

        return throttled

    @staticmethod
    def _format_timestamp(ts: Optional[float]) -> Optional[str]:
        """Render a UNIX timestamp as an ISO-8601 UTC string."""
        if ts is None:
            return None
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    def _prune_recent_tracking(self, now: Optional[float] = None) -> None:
        """Drop expired entries from dedup/context caches and trim to size cap."""
        now = now or time.time()
        cutoff = now - self._recent_tracking_ttl_seconds

        self._recent_confirmed_fill_keys = {
            key: record
            for key, record in self._recent_confirmed_fill_keys.items()
            if record.recorded_at >= cutoff
        }
        self._recent_order_context = {
            order_id: ctx
            for order_id, ctx in self._recent_order_context.items()
            if ctx.created_at >= cutoff
        }

        if len(self._recent_confirmed_fill_keys) > self._recent_tracking_cap:
            trimmed = sorted(
                self._recent_confirmed_fill_keys.items(),
                key=lambda item: item[1].recorded_at,
            )[-self._recent_tracking_cap:]
            self._recent_confirmed_fill_keys = dict(trimmed)

        if len(self._recent_order_context) > self._recent_tracking_cap:
            trimmed = sorted(
                self._recent_order_context.items(),
                key=lambda item: item[1].created_at,
            )[-self._recent_tracking_cap:]
            self._recent_order_context = dict(trimmed)

    def _remember_order_context(
        self,
        order_id: str,
        candidate: TradeCandidate,
        token_id: str,
        now: Optional[float] = None,
    ) -> None:
        """Cache order context past pending-order lifetime for late duplicate fills."""
        now = now or time.time()
        self._recent_order_context[order_id] = RecentOrderContext(
            order_id=order_id,
            candidate=self._clone_candidate(candidate),
            token_id=token_id,
            created_at=now,
        )
        self._prune_recent_tracking(now)

    def _confirmed_fill_key(
        self,
        fill_event: "FillEvent",
        token_id: str,
    ) -> str:
        """Build a stable idempotency key for confirmed fills."""
        if fill_event.match_id:
            return fill_event.match_id
        token = fill_event.token_id or token_id
        return (
            f"{fill_event.order_id}|{token}|{fill_event.size:.8f}|{fill_event.price:.8f}"
        )

    @staticmethod
    def _float_close(left: float, right: float, tol: float = 1e-6) -> bool:
        """Floating-point comparison helper."""
        return abs(left - right) <= tol

    def _freeze_integrity(self, reason: str, now: Optional[float] = None) -> None:
        """Freeze trading for this event until overlay reconciles or deadline hits."""
        now = now or time.time()
        was_frozen = self._integrity_state.frozen
        self._integrity_state.frozen = True
        self._integrity_state.reason = reason
        if self._integrity_state.frozen_at is None:
            self._integrity_state.frozen_at = now
        if self._integrity_state.deadline_at is None:
            self._integrity_state.deadline_at = (
                self._integrity_state.frozen_at
                + self.config.rate_limit.integrity_freeze_max_seconds
            )
        if was_frozen:
            logger.error(f"[{self.event_name}][INTEGRITY] Still frozen: {reason}")
        else:
            logger.error(
                f"[{self.event_name}][INTEGRITY] Event frozen: {reason} | "
                f"deadline={self._format_timestamp(self._integrity_state.deadline_at)}"
            )

    def _clear_integrity_freeze(self, reason: str, now: Optional[float] = None) -> None:
        """Clear an event-local freeze after reconciliation or forced recovery."""
        now = now or time.time()
        if self._integrity_state.frozen:
            logger.warning(f"[{self.event_name}][INTEGRITY] Event unfrozen: {reason}")
        self._integrity_state.frozen = False
        self._integrity_state.reason = None
        self._integrity_state.frozen_at = None
        self._integrity_state.deadline_at = None
        self._prune_recent_tracking(now)

    def _force_api_recovery(self, now: Optional[float] = None) -> None:
        """Drop residual overlay after the hard deadline and trust the API base."""
        now = now or time.time()
        if not self._integrity_state.frozen:
            return
        dropped_entries = len(self._overlay_ledger)
        dropped_size = sum(fragment.remaining_size for fragment in self._overlay_ledger)
        self._overlay_ledger.clear()
        self._integrity_state.last_forced_api_recovery_at = now
        logger.critical(
            f"[{self.event_name}][INTEGRITY] Forced API recovery after "
            f"{self.config.rate_limit.integrity_freeze_max_seconds:.0f}s: "
            f"dropped {dropped_entries} overlay fragment(s), {dropped_size:.2f} residual shares"
        )
        self._clear_integrity_freeze("forced_api_recovery", now=now)

    def _maybe_force_api_recovery(self, now: Optional[float] = None) -> None:
        """Trigger hard recovery once the freeze deadline has elapsed."""
        now = now or time.time()
        deadline = self._integrity_state.deadline_at
        if self._integrity_state.frozen and deadline is not None and now >= deadline:
            self._force_api_recovery(now)

    async def enforce_integrity_deadline(self, now: Optional[float] = None) -> bool:
        """
        Enforce the hard wall-clock integrity deadline outside the trading loop.

        This is used by higher-level runtime loops so frozen events can recover
        even when no sync-driven trading tick is triggered.
        """
        now = now or time.time()
        deadline = self._integrity_state.deadline_at
        if not self._integrity_state.frozen or deadline is None or now < deadline:
            return False

        self._prune_recent_tracking(now)

        # Refresh the authoritative API base once at deadline if possible.
        if self.sync_portfolio:
            try:
                await self.sync_portfolio()
            except Exception as e:
                logger.warning(
                    f"[{self.event_name}][INTEGRITY] Deadline recovery API sync failed: {e}"
                )

        current_snapshot = self.api_base_portfolio._copy()
        current_snapshot.external_capital_limit = self.api_base_portfolio.external_capital_limit
        self._reconcile_overlay_against_api(self._last_api_snapshot, current_snapshot)
        self._last_api_snapshot = current_snapshot._copy()

        now = now or time.time()
        if not self._overlay_ledger:
            self._clear_integrity_freeze("overlay_reconciled_at_deadline", now=now)
            return True

        self._force_api_recovery(now)
        self._last_api_snapshot = self.api_base_portfolio._copy()
        return True

    def _oldest_overlay_age_seconds(self, now: Optional[float] = None) -> Optional[float]:
        """Get age in seconds of the oldest residual overlay fragment."""
        if not self._overlay_ledger:
            return None
        now = now or time.time()
        oldest = min(fragment.confirmed_at for fragment in self._overlay_ledger)
        return max(0.0, now - oldest)

    def _prune_overlay_ledger(self) -> None:
        """Remove fully reconciled overlay fragments."""
        self._overlay_ledger = [
            fragment
            for fragment in self._overlay_ledger
            if fragment.remaining_size > self._overlay_size_epsilon
        ]

    def _maybe_freeze_for_stale_overlay(self, now: Optional[float] = None) -> None:
        """Freeze the event if residual overlay has gone stale."""
        if not self._overlay_ledger:
            if self._integrity_state.frozen:
                self._clear_integrity_freeze("overlay_reconciled", now=now)
            return

        now = now or time.time()
        oldest_age = self._oldest_overlay_age_seconds(now)
        grace = self.config.rate_limit.overlay_reconciliation_grace_seconds
        if oldest_age is not None and oldest_age >= grace:
            self._freeze_integrity(
                f"overlay unreconciled for {oldest_age:.1f}s (grace {grace:.0f}s)",
                now=now,
            )

    def _compute_api_deltas(
        self,
        previous_snapshot: Optional[Portfolio],
        current_snapshot: Portfolio,
    ) -> List[tuple[int, str, float]]:
        """Compute per-bin YES/NO share deltas between API snapshots."""
        if previous_snapshot is None:
            return []

        deltas: List[tuple[int, str, float]] = []
        tracked_bins = set(previous_snapshot.positions) | set(current_snapshot.positions)
        for bin_index in tracked_bins:
            prev_pos = previous_snapshot.get_position(bin_index)
            curr_pos = current_snapshot.get_position(bin_index)

            prev_yes = prev_pos.yes_shares if prev_pos else 0.0
            curr_yes = curr_pos.yes_shares if curr_pos else 0.0
            if not self._float_close(prev_yes, curr_yes, tol=0.01):
                deltas.append((bin_index, "YES", curr_yes - prev_yes))

            prev_no = prev_pos.no_shares if prev_pos else 0.0
            curr_no = curr_pos.no_shares if curr_pos else 0.0
            if not self._float_close(prev_no, curr_no, tol=0.01):
                deltas.append((bin_index, "NO", curr_no - prev_no))
        return deltas

    def _consume_overlay_delta(
        self,
        *,
        bin_index: int,
        position_kind: str,
        is_buy_delta: bool,
        amount: float,
    ) -> float:
        """Consume matching overlay fragments FIFO and return consumed share count."""
        consumed = 0.0
        remaining = amount
        for fragment in self._overlay_ledger:
            if fragment.remaining_size <= self._overlay_size_epsilon:
                continue
            if fragment.bin_index != bin_index:
                continue
            if fragment.position_kind != position_kind:
                continue
            if fragment.is_buy != is_buy_delta:
                continue

            take = min(fragment.remaining_size, remaining)
            if take <= self._overlay_size_epsilon:
                continue
            fragment.remaining_size -= take
            remaining -= take
            consumed += take
            if remaining <= self._overlay_size_epsilon:
                break

        self._prune_overlay_ledger()
        return consumed

    def _reconcile_overlay_against_api(
        self,
        previous_snapshot: Optional[Portfolio],
        current_snapshot: Portfolio,
    ) -> None:
        """Consume overlay entries that the latest API snapshot has absorbed."""
        deltas = self._compute_api_deltas(previous_snapshot, current_snapshot)
        for bin_index, position_kind, delta in deltas:
            if abs(delta) <= self._overlay_size_epsilon:
                continue

            is_buy_delta = delta > 0
            magnitude = abs(delta)
            matching_total = sum(
                fragment.remaining_size
                for fragment in self._overlay_ledger
                if fragment.bin_index == bin_index
                and fragment.position_kind == position_kind
                and fragment.is_buy == is_buy_delta
            )
            conflicting_total = sum(
                fragment.remaining_size
                for fragment in self._overlay_ledger
                if fragment.bin_index == bin_index
                and fragment.position_kind == position_kind
                and fragment.is_buy != is_buy_delta
            )

            if matching_total <= self._overlay_size_epsilon:
                if conflicting_total > self._overlay_size_epsilon:
                    self._freeze_integrity(
                        f"API delta {delta:+.2f} {position_kind} shares on bin {bin_index} "
                        f"conflicts with residual overlay",
                    )
                    return

                self._integrity_state.unmatched_api_delta_count += 1
                logger.warning(
                    f"[{self.event_name}][INTEGRITY] Unmatched API delta accepted: "
                    f"bin={bin_index} {position_kind} {delta:+.2f} shares "
                    f"(possible missed WS fill or manual trade)"
                )
                continue

            consumed = self._consume_overlay_delta(
                bin_index=bin_index,
                position_kind=position_kind,
                is_buy_delta=is_buy_delta,
                amount=magnitude,
            )

            if consumed + self._overlay_size_epsilon < magnitude:
                unmatched = magnitude - consumed
                if conflicting_total > self._overlay_size_epsilon:
                    self._freeze_integrity(
                        f"API delta {delta:+.2f} {position_kind} shares on bin {bin_index} "
                        f"exceeded matching overlay while opposite residual overlay exists",
                    )
                    return

                self._integrity_state.unmatched_api_delta_count += 1
                logger.warning(
                    f"[{self.event_name}][INTEGRITY] API delta exceeded overlay by "
                    f"{unmatched:.2f} shares on bin={bin_index} {position_kind}; "
                    f"accepting remainder as authoritative API move"
                )

        self._prune_overlay_ledger()
        if not self._overlay_ledger and self._integrity_state.frozen:
            self._clear_integrity_freeze("overlay_reconciled")

    def _build_effective_portfolio(self) -> Portfolio:
        """Build the planning portfolio as API base plus residual overlay."""
        effective = self.api_base_portfolio._copy()
        effective.external_capital_limit = self.api_base_portfolio.external_capital_limit

        for fragment in self._overlay_ledger:
            if fragment.remaining_size <= self._overlay_size_epsilon:
                continue

            size = fragment.remaining_size
            bin_index = fragment.bin_index
            price = fragment.price
            yes_token_id = self.token_ids.get(bin_index, fragment.token_id)

            if fragment.action == TradeAction.BUY_YES:
                effective.execute_buy_yes(bin_index, size, price, yes_token_id)
            elif fragment.action == TradeAction.BUY_NO:
                effective.execute_buy_no(bin_index, size, price, yes_token_id)
            elif fragment.action == TradeAction.SELL_YES:
                position = effective.get_position(bin_index)
                held = position.yes_shares if position else 0.0
                if held + self._overlay_size_epsilon < size:
                    self._freeze_integrity(
                        f"overlay SELL_YES exceeds effective YES shares on bin {bin_index}: "
                        f"held={held:.2f}, sell={size:.2f}"
                    )
                    return self.api_base_portfolio._copy()
                effective.execute_sell_yes(bin_index, size, price)
            elif fragment.action == TradeAction.SELL_NO:
                position = effective.get_position(bin_index)
                held = position.no_shares if position else 0.0
                if held + self._overlay_size_epsilon < size:
                    self._freeze_integrity(
                        f"overlay SELL_NO exceeds effective NO shares on bin {bin_index}: "
                        f"held={held:.2f}, sell={size:.2f}"
                    )
                    return self.api_base_portfolio._copy()
                effective.execute_sell_no(bin_index, size, price)

            if effective.capital < -self._overlay_size_epsilon:
                self._freeze_integrity(
                    f"overlay replay produced negative capital ${effective.capital:.2f}"
                )
                return self.api_base_portfolio._copy()

        return effective

    def _integrate_api_sync(self) -> Portfolio:
        """Reconcile overlay against the latest API sync and derive planning state."""
        now = time.time()
        self._prune_recent_tracking(now)
        current_snapshot = self.api_base_portfolio._copy()
        current_snapshot.external_capital_limit = self.api_base_portfolio.external_capital_limit

        self._reconcile_overlay_against_api(self._last_api_snapshot, current_snapshot)
        self._last_api_snapshot = current_snapshot._copy()

        self._maybe_freeze_for_stale_overlay(now)
        self._maybe_force_api_recovery(now)

        effective = self._build_effective_portfolio()
        self._maybe_force_api_recovery(now)
        if self._integrity_state.frozen:
            return self.api_base_portfolio._copy()
        return effective

    def _record_confirmed_fill(
        self,
        *,
        fill_key: str,
        order_id: str,
        candidate: TradeCandidate,
        token_id: str,
        fill_event: "FillEvent",
        now: Optional[float] = None,
    ) -> bool:
        """Record a confirmed fill into the overlay ledger if it is new."""
        now = now or time.time()
        existing = self._recent_confirmed_fill_keys.get(fill_key)
        if existing:
            same_fill = (
                existing.order_id == order_id
                and existing.token_id == token_id
                and existing.action == candidate.action
                and self._float_close(existing.price, fill_event.price)
                and self._float_close(existing.size, fill_event.size)
            )
            if not same_fill:
                self._freeze_integrity(
                    f"duplicate confirmed fill key {fill_key} arrived with conflicting economics",
                    now=now,
                )
            else:
                logger.info(
                    f"[{self.event_name}][INTEGRITY] Duplicate CONFIRMED fill ignored: "
                    f"order={order_id[:16]}..., key={fill_key}"
                )
            return False

        self._recent_confirmed_fill_keys[fill_key] = RecentConfirmedFillRecord(
            fill_key=fill_key,
            order_id=order_id,
            token_id=token_id,
            action=candidate.action,
            price=fill_event.price,
            size=fill_event.size,
            recorded_at=now,
        )
        self._overlay_ledger.append(
            OverlayFillFragment(
                fill_key=fill_key,
                order_id=order_id,
                token_id=token_id,
                action=candidate.action,
                bin_index=candidate.bin_index,
                price=fill_event.price,
                original_size=fill_event.size,
                remaining_size=fill_event.size,
                confirmed_at=now,
            )
        )
        self._prune_recent_tracking(now)
        return True

    def _mark_order_confirmed(
        self,
        order_id: str,
        candidate: TradeCandidate,
        token_id: str,
    ) -> None:
        """Advance local confirmation bookkeeping for a confirmed order."""
        if order_id in self._confirmation_events:
            self._confirmation_events[order_id].set()
            del self._confirmation_events[order_id]
        if order_id in self._pending_orders:
            del self._pending_orders[order_id]
        self._remember_order_context(order_id, candidate, token_id)

        if candidate.action in (TradeAction.BUY_YES, TradeAction.BUY_NO):
            self._warm_balance_cache(token_id)

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

        # Iterative loop: sync API base → reconcile overlay → compute on
        # effective state → execute → wait CONFIRMED + delay → repeat.
        # Each iteration starts from authoritative API data, augments it with
        # any unreconciled confirmed fills, then sizes trades from that
        # effective portfolio.
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

            effective_portfolio = self.api_base_portfolio._copy()

            # Sync API base portfolio and derive the effective planning state
            if self.sync_portfolio:
                try:
                    await self.sync_portfolio()
                    effective_portfolio = self._integrate_api_sync()
                    logger.info(
                        f"[{self.event_name}][KELLY] iter={iteration} Synced from API: "
                        f"base_capital=${self.api_base_portfolio.capital:.2f}, "
                        f"base_invested=${self.api_base_portfolio.total_collateral_used:.2f}, "
                        f"effective_capital=${effective_portfolio.capital:.2f}, "
                        f"overlay_entries={len(self._overlay_ledger)}"
                    )
                except Exception as e:
                    logger.warning(f"[{self.event_name}] Failed to sync portfolio: {e}")
                    effective_portfolio = self.api_base_portfolio._copy()

            if self._integrity_state.frozen:
                logger.warning(
                    f"[{self.event_name}][INTEGRITY] iter={iteration}: trading paused while frozen | "
                    f"reason={self._integrity_state.reason} | "
                    f"deadline={self._format_timestamp(self._integrity_state.deadline_at)}"
                )
                break

            # Refresh orderbooks
            orderbooks = self._get_orderbooks()

            # Compute optimal trades on the current effective portfolio
            planned_trades = self._compute_optimal_trades(
                portfolio=effective_portfolio,
                orderbooks=orderbooks,
                hours_to_settlement=hours_to_settlement,
                verbose=(verbose and iteration == 1),
            )

            if not planned_trades:
                logger.info(f"[{self.event_name}] iter={iteration}: no trades to execute")
                break

            # Collect results for this iteration's summary
            iter_results: list[tuple[TradeCandidate, str]] = []  # (trade, status)
            num_submitted_this_iter = 0

            if not self.order_executor.dry_run:
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

            planned_trades = self._apply_fresh_start_market_impact(planned_trades, orderbooks)
            if not planned_trades:
                logger.info(f"[{self.event_name}] iter={iteration}: no executable trades after throttling")
                break

            # Cap by remaining order budget for this tick
            orders_remaining = max_orders - tick_result.num_executed
            if len(planned_trades) > orders_remaining:
                planned_trades = planned_trades[:orders_remaining]

            tick_result.num_candidates += len(planned_trades)

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

                    self._log_trade_placed(
                        trade,
                        token_id,
                        f"dry_run_{self._trade_count}",
                        portfolio=self.api_base_portfolio,
                    )
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
                        if self._is_sell_balance_error(error_msg):
                            diagnostics = None
                            if trade.action in (
                                TradeAction.SELL_YES,
                                TradeAction.SELL_NO,
                            ):
                                diagnostics = self.order_executor.log_sell_balance_diagnostics(
                                    token_id=token_id,
                                    requested_size=trade.size,
                                )
                                saw_sell_balance_error = True
                            self._emit_balance_allowance_error(
                                candidate=trade,
                                token_id=token_id,
                                error_msg=error_msg,
                                diagnostics=diagnostics,
                            )

                    if order_id:
                        tracked_trade = self._clone_candidate(trade)
                        self._pending_orders[order_id] = (tracked_trade, token_id)
                        self._remember_order_context(order_id, tracked_trade, token_id)
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

                        self._log_trade_placed(
                            tracked_trade,
                            token_id,
                            order_id,
                            portfolio=effective_portfolio,
                        )
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
                portfolio=self.portfolio,
                orderbooks=orderbooks,
                hours_to_settlement=hours_to_settlement,
                verbose=(verbose and iteration == 1),
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
        portfolio: Portfolio,
        candidate: TradeCandidate,
        orderbooks: Dict[int, UnifiedOrderbook],
        hours_to_settlement: float,
        tick_config: KellyConfig,
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
                max_buy_shares = portfolio.available_capital / price
                full_size = min(full_size, max_buy_shares)
        else:
            position = portfolio.get_position(candidate.bin_index)
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
        if not self._check_overshoots(
            portfolio,
            candidate,
            full_size,
            orderbooks,
            hours_to_settlement,
            tick_config
        ):
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

            if self._check_overshoots(
                portfolio,
                candidate,
                mid,
                orderbooks,
                hours_to_settlement,
                tick_config,
            ):
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
        portfolio: Portfolio,
        candidate: TradeCandidate,
        test_size: float,
        orderbooks: Dict[int, UnifiedOrderbook],
        hours_to_settlement: float,
        tick_config: KellyConfig,
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
            config=tick_config,
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
        tick_config: KellyConfig,
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
            config=tick_config,
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
        tick_config: KellyConfig,
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
        if not self._check_overshoots_on(
            portfolio, candidate, full_size, orderbooks, hours_to_settlement, tick_config
        ):
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
            if self._check_overshoots_on(
                portfolio, candidate, mid, orderbooks, hours_to_settlement, tick_config
            ):
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

    def _build_tick_config(
        self,
        orderbooks: Dict[int, UnifiedOrderbook],
    ) -> tuple[KellyConfig, Optional[dict]]:
        """Build a per-tick Kelly config with optional robust-Kelly haircuting."""
        effective_fraction, context = compute_robust_kelly_fraction(
            base_kelly_fraction=self.config.kelly_fraction,
            probabilities=self.portfolio.probabilities,
            dead_bins=self.portfolio.dead_bins,
            orderbooks=orderbooks,
            robust_config=self.config.robust_kelly,
        )
        if abs(effective_fraction - self.config.kelly_fraction) < 1e-12:
            return self.config, context
        return replace(self.config, kelly_fraction=effective_fraction), context

    def _compute_optimal_trades(
        self,
        portfolio: Portfolio,
        orderbooks: Dict[int, UnifiedOrderbook],
        hours_to_settlement: float,
        verbose: bool = False,
    ) -> List[TradeCandidate]:
        """
        Compute all optimal trades via greedy simulation on a HYPOTHETICAL portfolio.

        Creates a deep copy of the provided portfolio and runs a simulation loop:
        1. Generate candidates on hypothetical portfolio
        2. Pick best (sells first, then buys by utility)
        3. Binary search for optimal size (upper bound = c_bin_max for buys)
        4. Simulate trade on hypothetical portfolio (never touches self.portfolio)
        5. Accumulate in planned_trades
        6. Repeat until no positive-utility trades remain

        IMPORTANT: the input portfolio is NEVER modified. All simulation happens on copies.

        Returns:
            List of TradeCandidate with optimized sizes, ready for batch execution.
        """
        # Deep copy the portfolio for simulation — the caller's portfolio stays untouched
        hyp = portfolio._copy()
        hyp.external_capital_limit = portfolio.external_capital_limit
        planned_trades = []
        tick_config, robust_context = self._build_tick_config(orderbooks)

        if robust_context is not None:
            logger.debug(
                "[%s] Robust Kelly haircut: fraction=%.3f (base=%.3f, mult=%.3f, coverage=%.2f, avg_spread=%.3f, disagreement=%.3f)",
                self.event_name,
                robust_context["effective_fraction"],
                self.config.kelly_fraction,
                robust_context["fraction_multiplier"],
                robust_context["coverage_ratio"],
                robust_context["avg_spread"],
                robust_context["disagreement"],
            )

        for sim_iter in range(self.config.max_iters_per_tick):
            # Generate candidates on hypothetical portfolio
            candidates = generate_candidates(
                portfolio=hyp,
                orderbooks=orderbooks,
                config=tick_config,
                hours_to_settlement=hours_to_settlement,
                verbose=(verbose and sim_iter == 0),
            )

            if not candidates:
                logger.debug(f"[{self.event_name}][SIM iter={sim_iter}] No candidates")
                break

            selected: Optional[TradeCandidate] = None
            selected_after: Optional[Portfolio] = None
            rejected: List[str] = []

            # Scan candidates in returned priority order. Invalid earlier candidates
            # should not block later candidates from the same simulation iteration.
            for candidate in candidates:
                if candidate.utility_gain <= 0:
                    rejected.append(
                        self._log_optimizer_rejection(
                            sim_iter,
                            candidate,
                            "screen_utility<=0",
                            (
                                f"screen_util={candidate.utility_gain:.6f} "
                                f"price={candidate.price:.4f} size={candidate.size:.1f}"
                            ),
                            verbose=verbose,
                        )
                    )
                    continue

                cooldown_remaining = self._get_fak_cooldown_remaining(candidate.bin_index)
                if cooldown_remaining is not None:
                    rejected.append(
                        self._log_optimizer_rejection(
                            sim_iter,
                            candidate,
                            "fak_cooldown",
                            (
                                f"remaining={cooldown_remaining:.1f}s "
                                f"screen_util={candidate.utility_gain:.6f} "
                                f"price={candidate.price:.4f}"
                            ),
                            verbose=verbose,
                        )
                    )
                    continue

                optimal_size = self._find_optimal_size_on(
                    hyp, candidate, orderbooks, hours_to_settlement
                )

                if optimal_size < 1.0:
                    rejected.append(
                        self._log_optimizer_rejection(
                            sim_iter,
                            candidate,
                            "optimal_size<1",
                            (
                                f"optimal_size={optimal_size:.2f} "
                                f"screen_util={candidate.utility_gain:.6f} "
                                f"price={candidate.price:.4f}"
                            ),
                            verbose=verbose,
                        )
                    )
                    continue

                sized_candidate = replace(candidate, size=optimal_size)
                hyp_after = self._simulate_trade(hyp, sized_candidate)
                from .candidates import _compute_portfolio_utility_gain

                actual_utility = _compute_portfolio_utility_gain(hyp, hyp_after, tick_config)

                is_sell = sized_candidate.action in (TradeAction.SELL_YES, TradeAction.SELL_NO)
                min_util = tick_config.min_sell_utility if is_sell else tick_config.min_buy_utility
                if actual_utility < min_util:
                    rejected.append(
                        self._log_optimizer_rejection(
                            sim_iter,
                            sized_candidate,
                            "sized_utility_below_min",
                            (
                                f"screen_util={candidate.utility_gain:.6f} "
                                f"sized_util={actual_utility:.6f} "
                                f"min_util={min_util:.6f} "
                                f"size={optimal_size:.0f} price={candidate.price:.4f}"
                            ),
                            verbose=verbose,
                        )
                    )
                    continue

                sized_candidate.utility_gain = actual_utility
                selected = sized_candidate
                selected_after = hyp_after
                break

            if selected is None or selected_after is None:
                if rejected:
                    self._log_optimizer_exhausted(sim_iter, rejected, verbose=verbose)
                break

            hyp = selected_after
            planned_trades.append(selected)

            logger.info(
                f"[{self.event_name}][SIM iter={sim_iter}] {self._candidate_label(selected)} "
                f"size={selected.size:.0f} @ {selected.price:.4f} util={selected.utility_gain:.6f}"
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
                    if existing.threshold_price > 0 and t.threshold_price > 0:
                        existing.threshold_price = min(existing.threshold_price, t.threshold_price)
                    else:
                        existing.threshold_price = max(existing.threshold_price, t.threshold_price)
                else:
                    existing.limit_price = min(existing.limit_price, t.limit_price) if existing.limit_price > 0 else t.limit_price
                    existing.threshold_price = max(existing.threshold_price, t.threshold_price)
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
                    threshold_price=t.threshold_price,
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
        Update the authoritative base portfolio.

        This is used only in dry-run / simulated execution paths. Live trading
        keeps the API snapshot authoritative and applies confirmed fills via the
        overlay ledger until the API catches up.
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

        Logs fill events and records CONFIRMED fills in the overlay ledger.
        The authoritative base portfolio is not updated here; it is refreshed
        from the positions API before each iteration.

        Args:
            fill_event: FillEvent from WebSocket
        """
        order_id = fill_event.order_id
        if not order_id:
            logger.warning(f"[{self.event_name}] Received fill event with no order_id")
            return

        now = time.time()
        self._prune_recent_tracking(now)

        pending_info = self._pending_orders.get(order_id)
        if pending_info:
            candidate, token_id = pending_info
        else:
            cached = self._recent_order_context.get(order_id)
            if not cached:
                logger.info(
                    f"[{self.event_name}] Fill for unknown order {order_id[:16]}... "
                    f"- may be from previous session or manual trade"
                )
                return
            candidate, token_id = cached.candidate, cached.token_id

        if not candidate:
            return

        from .user_stream import OrderStatus
        status_str = fill_event.status.name if hasattr(fill_event.status, 'name') else str(fill_event.status)
        br = self._bin_range(candidate.bin_index)
        bin_label = f"bin={candidate.bin_index} ({br})" if br else f"bin={candidate.bin_index}"
        logger.info(
            f"[{self.event_name}][FILL {status_str}] {candidate.action.value} {bin_label} | "
            f"{fill_event.size:.1f} @ {fill_event.price:.4f} = ${fill_event.size * fill_event.price:.2f}"
        )

        if fill_event.status == OrderStatus.CONFIRMED:
            fill_key = self._confirmed_fill_key(fill_event, token_id)
            recorded = self._record_confirmed_fill(
                fill_key=fill_key,
                order_id=order_id,
                candidate=candidate,
                token_id=token_id,
                fill_event=fill_event,
                now=now,
            )
            self._mark_order_confirmed(order_id, candidate, token_id)
            if recorded:
                logger.debug(
                    f"[{self.event_name}] Recorded CONFIRMED fill overlay for "
                    f"order {order_id[:16]}..., key={fill_key}"
                )

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
        fresh_start_started_at: Optional[float] = None,
    ) -> None:
        """
        Set context for trade logging.

        Call this before run_tick() to provide context for detailed trade logs.
        `fresh_start_started_at` is the UNIX timestamp of the event's first
        trading tick and is used for fresh-start market impact throttling.
        """
        self._log_context = {
            "probabilities": probabilities or [],
            "orderbooks": orderbooks or {},
            "bin_ranges": bin_ranges or {},
            "current_count": current_count,
            "hours_to_settlement": hours_to_settlement,
            "forecast_mean": forecast_mean,
            "forecast_std": forecast_std,
            "fresh_start_started_at": fresh_start_started_at,
        }
        # Propagate bin_ranges to order executor for log context
        if hasattr(self.order_executor, 'bin_ranges'):
            self.order_executor.bin_ranges = bin_ranges or {}

    def _bin_range(self, bin_index: int) -> str:
        """Get bin range string for logging (e.g., '340-359')."""
        return self._log_context.get("bin_ranges", {}).get(bin_index, "")

    def _candidate_label(self, candidate: TradeCandidate) -> str:
        """Format a candidate label with action and bin range for logs."""
        br = self._bin_range(candidate.bin_index)
        bin_info = f"bin={candidate.bin_index} ({br})" if br else f"bin={candidate.bin_index}"
        return f"{candidate.action.value} {bin_info}"

    def _log_optimizer_rejection(
        self,
        sim_iter: int,
        candidate: TradeCandidate,
        reason: str,
        details: str,
        *,
        verbose: bool,
    ) -> str:
        """Log and return a normalized optimizer rejection summary."""
        message = (
            f"[{self.event_name}][SIM iter={sim_iter}] Reject {self._candidate_label(candidate)}: "
            f"{reason} {details}".rstrip()
        )
        if verbose:
            logger.info(message)
        else:
            logger.debug(message)
        return f"{self._candidate_label(candidate)} -> {reason}"

    def _log_optimizer_exhausted(
        self,
        sim_iter: int,
        rejected: List[str],
        *,
        verbose: bool,
    ) -> None:
        """Log that all candidates were exhausted for a simulation iteration."""
        message = (
            f"[{self.event_name}][SIM iter={sim_iter}] All candidates exhausted: "
            + "; ".join(rejected)
        )
        if verbose:
            logger.info(message)
        else:
            logger.debug(message)

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

    def _log_trade_placed(
        self,
        candidate: TradeCandidate,
        token_id: str,
        order_id: str,
        portfolio: Optional[Portfolio] = None,
    ) -> None:
        """
        Log detailed trade info when order is placed (pending fill).

        Format: #N ACTION bin=X (range) | size @ price = $collateral | model=X% mkt=Y% edge=Z% | portfolio: $capital
        """
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
        portfolio = portfolio or self.api_base_portfolio
        capital = portfolio.capital if portfolio else 0
        total_collateral = portfolio.total_collateral_used if portfolio else 0

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
        portfolio: Optional[Portfolio] = None,
    ) -> None:
        """
        Log when fill is confirmed via WebSocket.

        Shows actual fill details vs requested.
        """
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
        portfolio = portfolio or self.api_base_portfolio
        capital = portfolio.capital if portfolio else 0
        total_collateral = portfolio.total_collateral_used if portfolio else 0

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

    def get_integrity_summary(self) -> dict:
        """Get event-local overlay/integrity status for monitoring."""
        now = time.time()
        self._prune_recent_tracking(now)
        return {
            "frozen": self._integrity_state.frozen,
            "reason": self._integrity_state.reason,
            "frozen_at": self._format_timestamp(self._integrity_state.frozen_at),
            "deadline_at": self._format_timestamp(self._integrity_state.deadline_at),
            "overlay_entries": len(self._overlay_ledger),
            "oldest_overlay_age_seconds": self._oldest_overlay_age_seconds(now),
            "last_forced_api_recovery_at": self._format_timestamp(
                self._integrity_state.last_forced_api_recovery_at
            ),
            "unmatched_api_delta_count": self._integrity_state.unmatched_api_delta_count,
        }

    def get_portfolio_summary(self) -> dict:
        """Get current portfolio summary."""
        return self.api_base_portfolio.to_summary()
