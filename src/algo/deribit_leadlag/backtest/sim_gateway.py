"""
Simulated order gateway for the lead-lag backtest.

Drop-in replacement for `OrderManager`: same `execute_actions` /
`cancel_all_orders` surface, same `ExecutionReport` shape so
`PositionManager.apply_execution_report` consumes it without modification.

v1 scope:
  - Taker fills cross the recorded best ask (BUY) / `1 - best_yes_bid`
    (SELL or NO BUY). Size is capped to a depth proxy derived from
    `max_order_size_usd / fill_price`.
  - Maker fill modeling is policy-driven (`maker_policy`):
      * `"discard"` — drop maker actions entirely. PnL is a strict
        lower bound. Useful for proving the trading core wires up.
      * `"instant_optimistic"` (default) — treat the maker post as
        instantly filled at the post price (action.price). This is an
        upper bound: it assumes the queued bid gets hit within the
        snapshot interval. Together with the discard-mode lower bound
        these give the realistic PnL envelope. Documented as optimistic
        in plan section "Out of scope (defer to v2)".
      The recorded books in 2026-04 are level-1 only and usually have
      degenerate asks at 0.999; the algo overwhelmingly chooses maker,
      so `discard` produces 0 fills on this window.
  - The gateway calls `position_mgr.handle_fill(...)` synchronously for
    every filled order, mirroring what the live WebSocket fill path
    produces. Inventory moves in `PositionManager` exactly as
    `_on_fill` would.
  - Fees (Polymarket's `0.072 * p * (1-p)`) are tracked in the fill log
    via the optional `fill_callback`. They do NOT thread through
    `handle_fill` — `PositionManager` stays inventory-only, matching
    production where fees live in trade reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

from ..fak_utils import MIN_ORDER_SIZE, MIN_ORDER_VALUE_USD, _best_fak_price
from ..order_manager import ExecutionReport
from ..position_manager import (
    BinKey,
    DesiredAction,
    OrderAction,
    PositionManager,
    compute_polymarket_fee,
)
from .data_provider import SimulatedOrderbook

logger = logging.getLogger(__name__)


@dataclass
class SimFill:
    """One filled order, captured by the gateway and forwarded to the ledger."""

    timestamp: datetime
    bin_key: BinKey
    condition_id: str
    yes_token_id: str
    no_token_id: str
    action: DesiredAction
    side: str  # "YES" or "NO"
    direction: str  # "BUY" or "SELL"
    token_id: str
    fill_price: float
    fill_size: int
    fee: float
    edge_at_post: float
    order_id: str


FillCallback = Callable[[SimFill], None]


def _condition_id_lookup(position_mgr: PositionManager, bin_key: BinKey):
    bs = position_mgr.get_bins().get(bin_key)
    if bs is None:
        return None
    return bs.poly_market


MAKER_POLICY_DISCARD = "discard"
MAKER_POLICY_INSTANT = "instant_optimistic"
_VALID_MAKER_POLICIES = (MAKER_POLICY_DISCARD, MAKER_POLICY_INSTANT)


class SimulatedOrderGateway:
    """Backtest order gateway. Public surface mirrors `OrderManager`."""

    def __init__(
        self,
        position_mgr: PositionManager,
        fee_rate: float,
        max_order_size_usd: float,
        fill_callback: Optional[FillCallback] = None,
        maker_policy: str = MAKER_POLICY_INSTANT,
    ):
        if maker_policy not in _VALID_MAKER_POLICIES:
            raise ValueError(
                f"maker_policy must be one of {_VALID_MAKER_POLICIES}, got {maker_policy!r}"
            )
        self._position_mgr = position_mgr
        self._fee_rate = fee_rate
        self._max_order_size_usd = max_order_size_usd
        self._fill_callback = fill_callback
        self._maker_policy = maker_policy
        self._order_seq = 0
        self._now: Optional[datetime] = None
        self._orderbooks: Dict[str, SimulatedOrderbook] = {}

    # --- per-tick wiring (called by runner before each strategy tick) ---

    def set_clock(self, now: datetime) -> None:
        self._now = now

    def set_orderbooks(self, orderbooks: Dict[str, SimulatedOrderbook]) -> None:
        self._orderbooks = orderbooks

    # --- public surface (matches OrderManager) ---

    def execute_actions(self, actions: List[OrderAction]) -> ExecutionReport:
        cancel_ids: List[str] = []
        for a in actions:
            cancel_ids.extend(a.cancel_order_ids)

        # No resting maker queue in v1 — cancels just echo back.
        canceled_order_ids = list(dict.fromkeys(cancel_ids))

        posted_actions: List[OrderAction] = []
        raw_results: List[dict] = []

        for action in actions:
            if action.action == DesiredAction.HOLD:
                continue
            if action.size <= 0:
                continue
            if action.is_maker:
                if self._maker_policy == MAKER_POLICY_DISCARD:
                    logger.debug(
                        "Sim: discarding maker %s %s/%.0f size=%d (policy=discard)",
                        action.action.value,
                        action.bin_key.expiry_date,
                        action.bin_key.strike,
                        action.size,
                    )
                    continue
                # MAKER_POLICY_INSTANT — fill at the algo's chosen post price.
                result = self._execute_at_price(action, fill_price=float(action.price))
            else:
                result = self._execute_taker(action)

            if result is not None:
                posted_actions.append(action)
                raw_results.append(result)

        return ExecutionReport(
            canceled_order_ids=canceled_order_ids,
            posted_actions=posted_actions,
            raw_results=raw_results,
        )

    def cancel_all_orders(self) -> None:
        """No-op in v1 (no resting maker queue)."""
        return None

    # --- internals ---

    def _execute_taker(self, action: OrderAction) -> Optional[dict]:
        if self._now is None:
            raise RuntimeError("SimulatedOrderGateway.set_clock() not called this tick")

        market = _condition_id_lookup(self._position_mgr, action.bin_key)
        if market is None:
            return None

        # Pull execution price from current snapshot.
        ob = self._orderbooks.get(market.yes_token_id)
        if ob is None:
            return None

        cross_price = self._cross_price_for_action(action, ob)
        if cross_price is None or cross_price <= 0 or cross_price >= 1:
            return None

        return self._fill_at_price(action, fill_price=float(cross_price), order_type="taker")

    def _execute_at_price(self, action: OrderAction, fill_price: float) -> Optional[dict]:
        """Optimistic maker fill at the algo's chosen post price.

        Sanity checks that the post price is in (0, 1). Used by the
        `instant_optimistic` maker policy.
        """
        if self._now is None:
            raise RuntimeError("SimulatedOrderGateway.set_clock() not called this tick")
        if fill_price <= 0 or fill_price >= 1:
            return None
        return self._fill_at_price(action, fill_price=fill_price, order_type="maker_optimistic")

    def _fill_at_price(
        self,
        action: OrderAction,
        fill_price: float,
        order_type: str,
    ) -> Optional[dict]:
        market = _condition_id_lookup(self._position_mgr, action.bin_key)
        if market is None:
            return None

        side, _, _ = self._derive_sides(action)

        # Apply FAK 2dp mechanics. Maker posts also need 2dp compliance for
        # the simulated fill path (mirrors what the bot would have signed).
        fak_price, fak_size = _best_fak_price(fill_price, int(action.size), side, tick_size=None)

        if fak_size < MIN_ORDER_SIZE:
            return None
        if fak_size * fak_price < MIN_ORDER_VALUE_USD:
            return None

        # Cap by configured max order size.
        max_size_by_usd = int(self._max_order_size_usd / max(fak_price, 0.01))
        fak_size = min(fak_size, max_size_by_usd)
        if fak_size < MIN_ORDER_SIZE:
            return None

        order_id = self._next_order_id()
        # Optimistic maker policy assumes zero exchange fees (post-only orders
        # rebate or pay nothing). Taker pays the standard fee.
        fee = (
            compute_polymarket_fee(fak_price, self._fee_rate) * fak_size
            if order_type == "taker"
            else 0.0
        )

        # Mutate inventory through PositionManager — the same path the live
        # WebSocket _on_fill callback uses.
        self._position_mgr.handle_fill(
            order_id=order_id,
            token_id=action.token_id,
            side=side,
            filled_size=float(fak_size),
        )

        # Emit to ledger.
        if self._fill_callback is not None:
            yes_or_no = "YES" if action.token_id == market.yes_token_id else "NO"
            self._fill_callback(
                SimFill(
                    timestamp=self._now,
                    bin_key=action.bin_key,
                    condition_id=market.condition_id,
                    yes_token_id=market.yes_token_id,
                    no_token_id=market.no_token_id,
                    action=action.action,
                    side=yes_or_no,
                    direction=side,
                    token_id=action.token_id,
                    fill_price=float(fak_price),
                    fill_size=int(fak_size),
                    fee=float(fee),
                    edge_at_post=float(action.edge),
                    order_id=order_id,
                )
            )

        return {
            "orderID": order_id,
            "status": "FILLED",
            "type": order_type,
            "filled_size": fak_size,
            "filled_price": fak_price,
        }

    @staticmethod
    def _derive_sides(action: OrderAction) -> tuple[str, str, str]:
        """Return (BUY/SELL, YES/NO, friendly action label)."""
        if action.action == DesiredAction.BUY_YES:
            return "BUY", "YES", "BUY_YES"
        if action.action == DesiredAction.BUY_NO:
            return "BUY", "NO", "BUY_NO"
        if action.action == DesiredAction.SELL_YES:
            return "SELL", "YES", "SELL_YES"
        if action.action == DesiredAction.SELL_NO:
            return "SELL", "NO", "SELL_NO"
        return "BUY", "YES", "HOLD"

    @staticmethod
    def _cross_price_for_action(
        action: OrderAction,
        ob: SimulatedOrderbook,
    ) -> Optional[float]:
        """Pick the level we'd cross when posting this taker action.

        Mirrors how `position_manager._get_execution_prices` builds the
        execution prices: BUY_YES crosses `best_yes_ask`, BUY_NO crosses
        `1 - best_yes_bid`, SELL_YES crosses `best_yes_bid`, SELL_NO
        crosses `1 - best_yes_ask`. If the requested side has no quote, we
        do not fill.
        """
        if action.action == DesiredAction.BUY_YES:
            return ob.best_yes_ask
        if action.action == DesiredAction.SELL_YES:
            return ob.best_yes_bid
        if action.action == DesiredAction.BUY_NO:
            return ob.best_no_ask
        if action.action == DesiredAction.SELL_NO:
            return ob.best_no_bid
        return None

    def _next_order_id(self) -> str:
        self._order_seq += 1
        return f"sim-{self._order_seq:08d}"
