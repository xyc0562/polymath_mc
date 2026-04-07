"""
Order manager for hybrid maker/taker execution.

- Maker orders: GTC + postOnly=True (server rejects if would cross spread)
- Taker orders: FAK with musk tick-size/2dp mechanics
- Batch operations via py_clob_client post_orders / cancel_orders
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType

from .fak_utils import (
    _best_fak_price,
    _round_price_to_tick,
    MIN_ORDER_SIZE,
    MIN_ORDER_VALUE_USD,
)

from .config import OrderConfig
from .position_manager import OrderAction, DesiredAction

logger = logging.getLogger(__name__)


@dataclass
class ExecutionReport:
    """Best-effort local summary of cancel/post effects."""

    canceled_order_ids: List[str]
    posted_actions: List[OrderAction]
    raw_results: List[dict]


class OrderManager:
    """
    Executes order actions: batch cancel, then batch post.

    Maker orders use GTC + postOnly=True.
    Taker orders use FAK with musk tick-size/2dp mechanics.
    """

    def __init__(
        self,
        clob_client: ClobClient,
        order_config: OrderConfig,
        dry_run: bool = True,
    ):
        self._client = clob_client
        self._config = order_config
        self._dry_run = dry_run
        self._tick_size_cache: dict[str, str] = {}

    def execute_actions(self, actions: List[OrderAction]) -> ExecutionReport:
        """
        Execute a list of order actions.

        Order of operations:
        1. Collect all cancel_order_ids across actions
        2. Batch cancel
        3. Build and batch post new orders

        Returns list of order responses.
        """
        # 1. Collect cancels
        cancel_ids = []
        for a in actions:
            cancel_ids.extend(a.cancel_order_ids)

        # 2. Batch cancel
        canceled_order_ids: List[str] = []
        if cancel_ids:
            canceled_order_ids = self._batch_cancel(cancel_ids)

        # 3. Build and post orders
        results: List[dict] = []
        posted_actions: List[OrderAction] = []
        post_actions = [a for a in actions if a.action != DesiredAction.HOLD and a.size > 0]

        for action in post_actions:
            if action.is_maker:
                result = self._execute_maker(action)
            else:
                result = self._execute_taker(action)
            if result is not None:
                results.append(result)
                posted_actions.append(action)

        return ExecutionReport(
            canceled_order_ids=canceled_order_ids,
            posted_actions=posted_actions,
            raw_results=results,
        )

    def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        if self._dry_run:
            logger.info("[DRY RUN] Would cancel all orders")
            return
        try:
            self._client.cancel_all()
            logger.info("Cancelled all open orders")
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")

    def _batch_cancel(self, order_ids: List[str]) -> List[str]:
        """Cancel a batch of orders by ID."""
        if not order_ids:
            return []

        unique_ids = list(set(order_ids))

        if self._dry_run:
            logger.info(f"[DRY RUN] Would cancel {len(unique_ids)} orders")
            return unique_ids

        try:
            self._client.cancel_orders(unique_ids)
            logger.info(f"Cancelled {len(unique_ids)} orders")
            return unique_ids
        except Exception as e:
            logger.error(f"Batch cancel failed: {e}")
            # Fallback: cancel individually
            canceled: List[str] = []
            for oid in unique_ids:
                try:
                    self._client.cancel(oid)
                    canceled.append(oid)
                except Exception as e2:
                    logger.error(f"Individual cancel {oid[:8]}... failed: {e2}")
            return canceled

    def _execute_maker(self, action: OrderAction) -> Optional[dict]:
        """
        Place a maker order: GTC + postOnly=True.

        postOnly ensures the order is rejected if it would cross the spread,
        preventing unintended taker fills.
        """
        price = action.price
        size = action.size
        tick_size = self._get_tick_size(action.token_id)
        if tick_size:
            price = _round_price_to_tick(price, tick_size)

        if price <= 0 or price >= 1:
            logger.warning(f"Invalid maker price: {price:.4f}")
            return None

        if size < MIN_ORDER_SIZE:
            logger.debug(f"Maker order too small: {size} shares")
            return None

        # Determine side for CLOB
        side = "BUY" if action.action in (DesiredAction.BUY_YES, DesiredAction.BUY_NO) else "SELL"

        logger.info(
            f"[MAKER] {action.action.value} {action.bin_key.expiry_date}/{action.bin_key.strike:.0f} "
            f"| {size} @ {price:.4f} | edge={action.edge:.4f}"
        )

        if self._dry_run:
            logger.info(f"[DRY RUN] Would post GTC+postOnly: {side} {size} @ {price:.4f}")
            return {
                "orderID": f"dry_run_{action.action.value}_{time.time_ns()}",
                "status": "simulated",
                "type": "maker",
            }

        try:
            order_args = OrderArgs(
                token_id=action.token_id,
                price=price,
                size=size,
                side=side,
            )
            signed_order = self._client.create_order(order_args)

            # Post with GTC + postOnly=True
            response = self._client.post_order(
                signed_order,
                orderType=OrderType.GTC,
                postOnly=True,
            )

            order_id = response.get("orderID", "unknown")
            logger.info(f"[ORDER] Maker posted: {order_id}")
            return response

        except Exception as e:
            logger.error(f"[ORDER FAILED] Maker: {e}")
            return None

    def _execute_taker(self, action: OrderAction) -> Optional[dict]:
        """
        Place a taker FAK order using musk tick-size/2dp mechanics.

        Reuses _fak_size_step, _best_fak_price from musk executor
        to ensure maker_amount has <=2 decimal places.
        """
        price = action.price
        size = action.size
        tick_size = self._get_tick_size(action.token_id)
        if tick_size:
            price = _round_price_to_tick(price, tick_size)

        if price <= 0 or price >= 1:
            logger.warning(f"Invalid taker price: {price:.4f}")
            return None

        # Determine side
        side = "BUY" if action.action in (DesiredAction.BUY_YES, DesiredAction.BUY_NO) else "SELL"

        # Apply FAK size/price mechanics for 2dp compliance
        fak_price, fak_size = _best_fak_price(price, size, side, tick_size=tick_size)

        if fak_size < MIN_ORDER_SIZE:
            logger.debug(f"FAK order too small after adjustment: {fak_size} shares")
            return None

        if fak_size * fak_price < MIN_ORDER_VALUE_USD:
            logger.debug(f"FAK order below min value: ${fak_size * fak_price:.2f}")
            return None

        logger.info(
            f"[TAKER] {action.action.value} {action.bin_key.expiry_date}/{action.bin_key.strike:.0f} "
            f"| {fak_size} @ {fak_price:.4f} (orig: {size} @ {price:.4f}) "
            f"| edge={action.edge:.4f}"
        )

        if self._dry_run:
            logger.info(f"[DRY RUN] Would post FAK: {side} {fak_size} @ {fak_price:.4f}")
            return {
                "orderID": f"dry_run_{action.action.value}_{time.time_ns()}",
                "status": "simulated",
                "type": "taker",
            }

        try:
            order_args = OrderArgs(
                token_id=action.token_id,
                price=fak_price,
                size=fak_size,
                side=side,
            )
            signed_order = self._client.create_order(order_args)
            response = self._client.post_order(signed_order, orderType=OrderType.FAK)

            order_id = response.get("orderID", "unknown")
            logger.info(f"[ORDER] Taker posted: {order_id}")
            return response

        except Exception as e:
            logger.error(f"[ORDER FAILED] Taker: {e}")
            return None

    def _get_tick_size(self, token_id: str) -> Optional[str]:
        if token_id in self._tick_size_cache:
            return self._tick_size_cache[token_id]
        try:
            tick_size = self._client.get_tick_size(token_id)
        except Exception as exc:
            logger.warning(f"Failed to fetch tick size for {token_id[:12]}...: {exc}")
            return None
        if tick_size:
            self._tick_size_cache[token_id] = tick_size
        return tick_size
