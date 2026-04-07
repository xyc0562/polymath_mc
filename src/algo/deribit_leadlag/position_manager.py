"""
Inventory-based position management for lead-lag trading.

Tracks positions, computes desired inventory targets from adjusted reference
probabilities plus live orderbook data, and generates buy/sell/cancel actions.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Tuple

import requests

from .config import AllocationConfig, OrderConfig, SignalConfig
from .implied_probs import ImpliedProb
from .polymarket_discovery import ThresholdMarket
from .settlement import CompatibilityClass

logger = logging.getLogger(__name__)

POLYMARKET_DATA_API = "https://data-api.polymarket.com"


class DesiredAction(Enum):
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    SELL_YES = "SELL_YES"
    SELL_NO = "SELL_NO"
    HOLD = "HOLD"


@dataclass(frozen=True)
class BinKey:
    expiry_date: date
    strike: float


@dataclass
class OpenOrder:
    order_id: str
    bin_key: BinKey
    side: str  # "BUY" or "SELL"
    token_id: str
    price: float
    size: float
    posted_at: float
    edge_at_post: float
    is_maker: bool


@dataclass
class OrderAction:
    """An action to take on a specific bin."""

    bin_key: BinKey
    action: DesiredAction
    token_id: str
    price: float
    size: int
    is_maker: bool
    edge: float
    cancel_order_ids: List[str] = field(default_factory=list)


@dataclass
class BinState:
    bin_key: BinKey
    poly_market: ThresholdMarket
    compatibility: CompatibilityClass
    yes_position: float = 0.0
    no_position: float = 0.0
    target_yes: float = 0.0
    target_no: float = 0.0
    entry_action: DesiredAction = DesiredAction.HOLD
    entry_price: float = 0.0
    entry_edge: float = 0.0
    entry_is_maker: bool = True
    yes_exit_price: float = 0.0
    yes_exit_edge: float = 0.0
    yes_exit_is_maker: bool = True
    no_exit_price: float = 0.0
    no_exit_edge: float = 0.0
    no_exit_is_maker: bool = True
    open_orders: List[OpenOrder] = field(default_factory=list)


def compute_polymarket_fee(price: float, fee_rate: float) -> float:
    """Fee formula used by Polymarket for taker flow."""
    return fee_rate * price * (1.0 - price)


def _clamp_price(price: float) -> float:
    return min(0.999, max(0.001, price))


class PositionManager:
    """
    Manages inventory-based position targets and generates order deltas.

    Lifecycle:
    1. update_markets: track eligible bins
    2. fetch_positions / fetch_open_orders: sync from exchange APIs
    3. compute_targets: derive desired inventory plus live execution plans
    4. compute_deltas: reconcile target vs actual vs pending open orders
    """

    def __init__(
        self,
        alloc_config: AllocationConfig,
        order_config: OrderConfig,
        signal_config: SignalConfig,
        wallet_address: str,
    ):
        self._alloc = alloc_config
        self._order = order_config
        self._signal = signal_config
        self._wallet = wallet_address.lower() if wallet_address else ""
        self._bins: Dict[BinKey, BinState] = {}

    def update_markets(
        self,
        markets: List[ThresholdMarket],
        compatibility_map: Dict[BinKey, CompatibilityClass],
    ) -> None:
        """Update tracked markets while preserving positions/open orders for existing bins."""
        new_bins: Dict[BinKey, BinState] = {}
        for market in markets:
            key = BinKey(expiry_date=market.expiry_date, strike=market.strike)
            compat = compatibility_map.get(key, CompatibilityClass.REJECT)
            if key in self._bins:
                existing = self._bins[key]
                existing.poly_market = market
                existing.compatibility = compat
                new_bins[key] = existing
            else:
                new_bins[key] = BinState(
                    bin_key=key,
                    poly_market=market,
                    compatibility=compat,
                )
        self._bins = new_bins

    def fetch_positions(self, condition_ids: Optional[List[str]] = None) -> None:
        """
        Fetch current positions from Polymarket Data API.

        Uses a single GET /positions snapshot keyed by token_id to avoid one
        network round-trip per market during startup and periodic sync.
        """
        if not self._wallet:
            logger.warning("Cannot fetch positions without wallet address")
            return

        for bs in self._bins.values():
            bs.yes_position = 0.0
            bs.no_position = 0.0

        token_to_bin = {
            bs.poly_market.yes_token_id: (bs, "yes")
            for bs in self._bins.values()
        }
        token_to_bin.update({
            bs.poly_market.no_token_id: (bs, "no")
            for bs in self._bins.values()
        })

        try:
            response = requests.get(
                f"{POLYMARKET_DATA_API}/positions",
                params={"user": self._wallet, "sizeThreshold": 0},
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning(f"Failed to fetch positions snapshot: {exc}")
            return

        if isinstance(payload, dict):
            positions = payload.get("positions", payload.get("data", []))
        else:
            positions = payload

        for pos in positions:
            asset = pos.get("asset")
            if isinstance(asset, str):
                token_id = asset
            elif isinstance(asset, dict):
                token_id = asset.get("id")
            else:
                token_id = pos.get("token_id") or pos.get("asset_id")
            shares = float(pos.get("shares") or pos.get("size") or 0)
            if shares <= 0:
                continue
            target = token_to_bin.get(token_id)
            if target is None:
                continue
            bs, side = target
            if side == "yes":
                bs.yes_position = shares
            else:
                bs.no_position = shares

        logger.info(
            "Fetched positions: %d bins, total YES=%.0f NO=%.0f",
            len(self._bins),
            sum(bs.yes_position for bs in self._bins.values()),
            sum(bs.no_position for bs in self._bins.values()),
        )

    def fetch_open_orders(self, clob_client) -> None:
        """Fetch current live orders from the CLOB client and map them into bins."""
        for bs in self._bins.values():
            bs.open_orders.clear()

        token_to_bin: Dict[str, BinState] = {}
        for bs in self._bins.values():
            token_to_bin[bs.poly_market.yes_token_id] = bs
            token_to_bin[bs.poly_market.no_token_id] = bs

        try:
            orders = clob_client.get_orders()
            live_orders = [order for order in orders if order.get("status") == "LIVE"]
        except Exception as exc:
            logger.warning(f"Failed to fetch open orders: {exc}")
            return

        for order in live_orders:
            token_id = order.get("asset_id", "")
            bs = token_to_bin.get(token_id)
            if bs is None:
                continue

            order_type = order.get("type") or order.get("order_type") or ""
            bs.open_orders.append(
                OpenOrder(
                    order_id=order.get("id", ""),
                    bin_key=bs.bin_key,
                    side=order.get("side", "BUY"),
                    token_id=token_id,
                    price=float(order.get("price", 0) or 0),
                    size=float(order.get("original_size", 0) or order.get("size", 0) or 0),
                    posted_at=time.time(),
                    edge_at_post=0.0,
                    is_maker=order_type in ("GTC", "GTD"),
                )
            )

        logger.info("Mapped %d live orders across bins", sum(len(bs.open_orders) for bs in self._bins.values()))

    def compute_targets(
        self,
        implied_probs: Dict[Tuple[date, float], ImpliedProb],
        orderbooks: Dict[str, object],
    ) -> None:
        """Compute desired inventory and execution plans for each tracked bin."""
        for bs in self._bins.values():
            self._reset_bin_plan(bs)

            if bs.compatibility not in (CompatibilityClass.TIME_ADJUSTED, CompatibilityClass.EXACT_MATCH):
                continue

            prob = implied_probs.get((bs.bin_key.expiry_date, bs.bin_key.strike))
            if prob is None:
                continue

            if prob.prob_mid < self._signal.min_prob or prob.prob_mid > self._signal.max_prob:
                continue

            (
                yes_buy_taker,
                yes_buy_maker,
                yes_sell_taker,
                yes_sell_maker,
                no_buy_taker,
                no_buy_maker,
                no_sell_taker,
                no_sell_maker,
            ) = self._get_execution_prices(bs, orderbooks.get(bs.poly_market.yes_token_id))

            haircut = 0.0
            if bs.compatibility == CompatibilityClass.TIME_ADJUSTED:
                haircut = self._signal.time_adjusted_basis_haircut
                if not getattr(prob, "has_next_day", True):
                    haircut += self._signal.no_next_day_extra_haircut

            yes_choice = self._choose_entry(
                fair_value=prob.prob_conservative,
                maker_price=yes_buy_maker,
                taker_price=yes_buy_taker,
                bounds_width=prob.bounds_width,
                haircut=haircut,
            )
            no_choice = self._choose_entry(
                fair_value=1.0 - prob.prob_aggressive,
                maker_price=no_buy_maker,
                taker_price=no_buy_taker,
                bounds_width=prob.bounds_width,
                haircut=haircut,
            )

            target_usd = self._compute_allocation(bs.bin_key)
            if yes_choice and target_usd > 0 and (not no_choice or yes_choice[2] >= no_choice[2]):
                bs.target_yes = target_usd / yes_choice[1]
                bs.entry_action = DesiredAction.BUY_YES
                bs.entry_is_maker = yes_choice[0]
                bs.entry_price = yes_choice[1]
                bs.entry_edge = yes_choice[2]
            elif no_choice and target_usd > 0:
                bs.target_no = target_usd / no_choice[1]
                bs.entry_action = DesiredAction.BUY_NO
                bs.entry_is_maker = no_choice[0]
                bs.entry_price = no_choice[1]
                bs.entry_edge = no_choice[2]

            yes_maker_edge = yes_sell_maker - prob.prob_aggressive - haircut
            yes_taker_edge = (
                yes_sell_taker
                - prob.prob_aggressive
                - compute_polymarket_fee(yes_sell_taker, self._order.polymarket_crypto_fee_rate)
                - haircut
            )
            if yes_taker_edge >= self._order.emergency_exit_edge:
                bs.yes_exit_price = yes_sell_taker
                bs.yes_exit_edge = yes_taker_edge
                bs.yes_exit_is_maker = False
            else:
                bs.yes_exit_price = yes_sell_maker
                bs.yes_exit_edge = yes_maker_edge
                bs.yes_exit_is_maker = True

            no_fair_high = 1.0 - prob.prob_conservative
            no_maker_edge = no_sell_maker - no_fair_high - haircut
            no_taker_edge = (
                no_sell_taker
                - no_fair_high
                - compute_polymarket_fee(no_sell_taker, self._order.polymarket_crypto_fee_rate)
                - haircut
            )
            if no_taker_edge >= self._order.emergency_exit_edge:
                bs.no_exit_price = no_sell_taker
                bs.no_exit_edge = no_taker_edge
                bs.no_exit_is_maker = False
            else:
                bs.no_exit_price = no_sell_maker
                bs.no_exit_edge = no_maker_edge
                bs.no_exit_is_maker = True

    def compute_deltas(self) -> List[OrderAction]:
        """Reconcile desired inventory against holdings and currently open orders."""
        actions: List[OrderAction] = []

        for bs in self._bins.values():
            if bs.compatibility not in (CompatibilityClass.TIME_ADJUSTED, CompatibilityClass.EXACT_MATCH):
                reject_cancel_ids = [oo.order_id for oo in bs.open_orders]
                if reject_cancel_ids:
                    actions.append(
                        OrderAction(
                            bin_key=bs.bin_key,
                            action=DesiredAction.HOLD,
                            token_id=bs.poly_market.yes_token_id,
                            price=0.0,
                            size=0,
                            is_maker=True,
                            edge=0.0,
                            cancel_order_ids=reject_cancel_ids,
                        )
                    )
                continue

            desired_specs: List[Tuple[str, str, float, bool]] = []
            if bs.target_yes > bs.yes_position + 1e-9 and bs.entry_action == DesiredAction.BUY_YES:
                desired_specs.append(("BUY", bs.poly_market.yes_token_id, bs.entry_price, bs.entry_is_maker))
            if bs.target_no > bs.no_position + 1e-9 and bs.entry_action == DesiredAction.BUY_NO:
                desired_specs.append(("BUY", bs.poly_market.no_token_id, bs.entry_price, bs.entry_is_maker))
            if bs.yes_position > bs.target_yes + 1e-9 and bs.yes_exit_price > 0:
                desired_specs.append(("SELL", bs.poly_market.yes_token_id, bs.yes_exit_price, bs.yes_exit_is_maker))
            if bs.no_position > bs.target_no + 1e-9 and bs.no_exit_price > 0:
                desired_specs.append(("SELL", bs.poly_market.no_token_id, bs.no_exit_price, bs.no_exit_is_maker))

            stale_ids = [
                order.order_id
                for order in bs.open_orders
                if not any(self._order_matches_spec(order, *spec) for spec in desired_specs)
            ]
            cancel_ids = stale_ids.copy()

            entry_is_yes = bs.entry_action == DesiredAction.BUY_YES
            entry_is_no = bs.entry_action == DesiredAction.BUY_NO

            pending_buy_yes = self._pending_size(
                bs.open_orders,
                "BUY",
                bs.poly_market.yes_token_id,
                bs.entry_price if entry_is_yes else None,
                bs.entry_is_maker if entry_is_yes else None,
            )
            remaining_buy_yes = max(0.0, bs.target_yes - bs.yes_position - pending_buy_yes)
            if remaining_buy_yes > 0 and entry_is_yes:
                size = int(min(remaining_buy_yes, self._alloc.max_order_size_usd / max(bs.entry_price, 0.01)))
                if size > 0:
                    actions.append(
                        OrderAction(
                            bin_key=bs.bin_key,
                            action=DesiredAction.BUY_YES,
                            token_id=bs.poly_market.yes_token_id,
                            price=bs.entry_price,
                            size=size,
                            is_maker=bs.entry_is_maker,
                            edge=bs.entry_edge,
                            cancel_order_ids=cancel_ids,
                        )
                    )
                    cancel_ids = []

            pending_sell_yes = self._pending_size(
                bs.open_orders,
                "SELL",
                bs.poly_market.yes_token_id,
                bs.yes_exit_price if bs.yes_exit_price > 0 else None,
                bs.yes_exit_is_maker if bs.yes_exit_price > 0 else None,
            )
            remaining_sell_yes = max(0.0, bs.yes_position - bs.target_yes - pending_sell_yes)
            if remaining_sell_yes > 0 and bs.yes_exit_price > 0:
                size = int(min(remaining_sell_yes, self._alloc.max_order_size_usd / max(bs.yes_exit_price, 0.01)))
                if size > 0:
                    actions.append(
                        OrderAction(
                            bin_key=bs.bin_key,
                            action=DesiredAction.SELL_YES,
                            token_id=bs.poly_market.yes_token_id,
                            price=bs.yes_exit_price,
                            size=size,
                            is_maker=bs.yes_exit_is_maker,
                            edge=bs.yes_exit_edge,
                            cancel_order_ids=cancel_ids,
                        )
                    )
                    cancel_ids = []

            pending_buy_no = self._pending_size(
                bs.open_orders,
                "BUY",
                bs.poly_market.no_token_id,
                bs.entry_price if entry_is_no else None,
                bs.entry_is_maker if entry_is_no else None,
            )
            remaining_buy_no = max(0.0, bs.target_no - bs.no_position - pending_buy_no)
            if remaining_buy_no > 0 and entry_is_no:
                size = int(min(remaining_buy_no, self._alloc.max_order_size_usd / max(bs.entry_price, 0.01)))
                if size > 0:
                    actions.append(
                        OrderAction(
                            bin_key=bs.bin_key,
                            action=DesiredAction.BUY_NO,
                            token_id=bs.poly_market.no_token_id,
                            price=bs.entry_price,
                            size=size,
                            is_maker=bs.entry_is_maker,
                            edge=bs.entry_edge,
                            cancel_order_ids=cancel_ids,
                        )
                    )
                    cancel_ids = []

            pending_sell_no = self._pending_size(
                bs.open_orders,
                "SELL",
                bs.poly_market.no_token_id,
                bs.no_exit_price if bs.no_exit_price > 0 else None,
                bs.no_exit_is_maker if bs.no_exit_price > 0 else None,
            )
            remaining_sell_no = max(0.0, bs.no_position - bs.target_no - pending_sell_no)
            if remaining_sell_no > 0 and bs.no_exit_price > 0:
                size = int(min(remaining_sell_no, self._alloc.max_order_size_usd / max(bs.no_exit_price, 0.01)))
                if size > 0:
                    actions.append(
                        OrderAction(
                            bin_key=bs.bin_key,
                            action=DesiredAction.SELL_NO,
                            token_id=bs.poly_market.no_token_id,
                            price=bs.no_exit_price,
                            size=size,
                            is_maker=bs.no_exit_is_maker,
                            edge=bs.no_exit_edge,
                            cancel_order_ids=cancel_ids,
                        )
                    )
                    cancel_ids = []

            if cancel_ids:
                actions.append(
                    OrderAction(
                        bin_key=bs.bin_key,
                        action=DesiredAction.HOLD,
                        token_id=bs.poly_market.yes_token_id,
                        price=0.0,
                        size=0,
                        is_maker=True,
                        edge=0.0,
                        cancel_order_ids=cancel_ids,
                    )
                )

        if actions:
            buy_actions = [a for a in actions if a.action in (DesiredAction.BUY_YES, DesiredAction.BUY_NO)]
            sell_actions = [a for a in actions if a.action in (DesiredAction.SELL_YES, DesiredAction.SELL_NO)]
            logger.info("Computed %d buy + %d sell actions", len(buy_actions), len(sell_actions))

        return actions

    def handle_fill(self, order_id: str, token_id: str, side: str, filled_size: float) -> None:
        """Apply a confirmed fill to local positions and optimistic open-order tracking."""
        for bs in self._bins.values():
            if token_id == bs.poly_market.yes_token_id:
                bs.yes_position = self._apply_fill_delta(bs.yes_position, side, filled_size)
                self._apply_fill_to_open_orders(bs, order_id, filled_size)
                return
            if token_id == bs.poly_market.no_token_id:
                bs.no_position = self._apply_fill_delta(bs.no_position, side, filled_size)
                self._apply_fill_to_open_orders(bs, order_id, filled_size)
                return

    def apply_execution_report(self, report) -> None:
        """Update local open-order view from recent cancels/posts so we do not repost every tick."""
        canceled = set(getattr(report, "canceled_order_ids", []))
        if canceled:
            for bs in self._bins.values():
                bs.open_orders = [oo for oo in bs.open_orders if oo.order_id not in canceled]

        raw_results = list(getattr(report, "raw_results", []))
        posted_actions = list(getattr(report, "posted_actions", []))
        for action, result in zip(posted_actions, raw_results):
            order_id = result.get("orderID") or result.get("order_id") or ""
            if not order_id:
                continue
            bs = self._bins.get(action.bin_key)
            if bs is None:
                continue
            if any(existing.order_id == order_id for existing in bs.open_orders):
                continue
            side = "BUY" if action.action in (DesiredAction.BUY_YES, DesiredAction.BUY_NO) else "SELL"
            bs.open_orders.append(
                OpenOrder(
                    order_id=order_id,
                    bin_key=action.bin_key,
                    side=side,
                    token_id=action.token_id,
                    price=action.price,
                    size=action.size,
                    posted_at=time.time(),
                    edge_at_post=action.edge,
                    is_maker=action.is_maker,
                )
            )

    def get_bins(self) -> Dict[BinKey, BinState]:
        return self._bins

    def get_total_exposure(self) -> float:
        return sum(
            bs.yes_position * bs.poly_market.yes_price
            + bs.no_position * bs.poly_market.no_price
            for bs in self._bins.values()
        )

    def _compute_allocation(self, bin_key: BinKey) -> float:
        """Compute max USD allocation for a bin while respecting all configured caps."""
        bs = self._bins.get(bin_key)
        if bs is None:
            return 0.0

        bin_exposure = bs.yes_position * bs.poly_market.yes_price + bs.no_position * bs.poly_market.no_price
        date_exposure = sum(
            b.yes_position * b.poly_market.yes_price + b.no_position * b.poly_market.no_price
            for b in self._bins.values()
            if b.bin_key.expiry_date == bin_key.expiry_date
        )
        total_exposure = sum(
            b.yes_position * b.poly_market.yes_price + b.no_position * b.poly_market.no_price
            for b in self._bins.values()
        )

        return max(
            0.0,
            min(
                self._alloc.per_bin_max_usd - bin_exposure,
                self._alloc.per_date_max_usd - date_exposure,
                self._alloc.total_max_usd - total_exposure,
                self._alloc.max_order_size_usd,
            ),
        )

    def _choose_entry(
        self,
        fair_value: float,
        maker_price: float,
        taker_price: float,
        bounds_width: float,
        haircut: float,
    ) -> Optional[Tuple[bool, float, float]]:
        """Return (is_maker, chosen_price, effective_edge) for the preferred entry path."""
        maker_edge = fair_value - maker_price - haircut
        taker_edge = (
            fair_value
            - taker_price
            - compute_polymarket_fee(taker_price, self._order.polymarket_crypto_fee_rate)
            - haircut
        )

        if bounds_width <= self._signal.max_bounds_width_taker and taker_edge >= self._order.taker_min_edge:
            return False, taker_price, taker_edge

        if bounds_width <= self._signal.max_bounds_width_maker and maker_edge >= self._order.maker_min_edge:
            return True, maker_price, maker_edge

        return None

    def _get_execution_prices(
        self,
        bs: BinState,
        orderbook: Optional[object],
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        """Return taker/maker buy and sell prices for YES and NO."""
        indicative_yes = _clamp_price(bs.poly_market.yes_price or 0.5)
        indicative_no = _clamp_price(bs.poly_market.no_price or 0.5)

        best_yes_bid = getattr(orderbook, "best_yes_bid", None) if orderbook is not None else None
        best_yes_ask = getattr(orderbook, "best_yes_ask", None) if orderbook is not None else None
        best_no_bid = getattr(orderbook, "best_no_bid", None) if orderbook is not None else None
        best_no_ask = getattr(orderbook, "best_no_ask", None) if orderbook is not None else None

        yes_buy_taker = _clamp_price(best_yes_ask if best_yes_ask is not None else indicative_yes)
        yes_buy_maker = self._passive_buy_price(best_yes_bid, best_yes_ask, indicative_yes)
        yes_sell_taker = _clamp_price(best_yes_bid if best_yes_bid is not None else indicative_yes)
        yes_sell_maker = self._passive_sell_price(best_yes_bid, best_yes_ask, indicative_yes)

        no_buy_taker = _clamp_price(best_no_ask if best_no_ask is not None else indicative_no)
        no_buy_maker = self._passive_buy_price(best_no_bid, best_no_ask, indicative_no)
        no_sell_taker = _clamp_price(best_no_bid if best_no_bid is not None else indicative_no)
        no_sell_maker = self._passive_sell_price(best_no_bid, best_no_ask, indicative_no)

        return (
            yes_buy_taker,
            yes_buy_maker,
            yes_sell_taker,
            yes_sell_maker,
            no_buy_taker,
            no_buy_maker,
            no_sell_taker,
            no_sell_maker,
        )

    @staticmethod
    def _passive_buy_price(best_bid: Optional[float], best_ask: Optional[float], fallback: float) -> float:
        if best_bid is not None and best_bid > 0:
            return _clamp_price(best_bid)
        if best_ask is not None and best_ask > 0:
            return _clamp_price(best_ask - 0.01)
        return _clamp_price(fallback)

    @staticmethod
    def _passive_sell_price(best_bid: Optional[float], best_ask: Optional[float], fallback: float) -> float:
        if best_ask is not None and best_ask > 0:
            return _clamp_price(best_ask)
        if best_bid is not None and best_bid > 0:
            return _clamp_price(best_bid + 0.01)
        return _clamp_price(fallback)

    @staticmethod
    def _order_matches_spec(order: OpenOrder, side: str, token_id: str, price: float, is_maker: bool) -> bool:
        return (
            order.side == side
            and order.token_id == token_id
            and order.is_maker == is_maker
            and abs(order.price - price) < 1e-6
        )

    def _pending_size(
        self,
        open_orders: List[OpenOrder],
        side: str,
        token_id: str,
        price: Optional[float],
        is_maker: Optional[bool],
    ) -> float:
        if price is None or is_maker is None:
            return 0.0
        return sum(
            order.size
            for order in open_orders
            if self._order_matches_spec(order, side, token_id, price, is_maker)
        )

    @staticmethod
    def _apply_fill_delta(position: float, side: str, filled_size: float) -> float:
        if side == "SELL":
            return max(0.0, position - filled_size)
        return position + filled_size

    @staticmethod
    def _apply_fill_to_open_orders(bs: BinState, order_id: str, filled_size: float) -> None:
        for order in list(bs.open_orders):
            if order.order_id != order_id:
                continue
            order.size = max(0.0, order.size - filled_size)
            if order.size <= 1e-9:
                bs.open_orders.remove(order)
            break

    @staticmethod
    def _reset_bin_plan(bs: BinState) -> None:
        bs.target_yes = 0.0
        bs.target_no = 0.0
        bs.entry_action = DesiredAction.HOLD
        bs.entry_price = 0.0
        bs.entry_edge = 0.0
        bs.entry_is_maker = True
        bs.yes_exit_price = 0.0
        bs.yes_exit_edge = 0.0
        bs.yes_exit_is_maker = True
        bs.no_exit_price = 0.0
        bs.no_exit_edge = 0.0
        bs.no_exit_is_maker = True
