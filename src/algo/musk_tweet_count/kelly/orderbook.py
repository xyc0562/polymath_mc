"""
Unified orderbook handling for Polymarket YES/NO tokens.

Key insight: YES and NO tokens share the same orderbook.
- BUY YES @ X = SELL NO @ (1-X)
- SELL YES @ X = BUY NO @ (1-X)

This module provides:
- UnifiedOrderbook: Symmetric view of YES/NO orderbooks
- VWAP calculation for chunk execution
- Depth analysis for adaptive sizing
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from decimal import Decimal


@dataclass
class OrderbookLevel:
    """A single price level in the orderbook."""

    price: float  # Price in range [0, 1]
    size: float  # Size in shares


@dataclass
class UnifiedOrderbook:
    """
    Unified view of YES/NO orderbook for a single bin.

    Since YES and NO share the same orderbook:
    - YES bids are derived from NO asks
    - YES asks are derived from NO bids
    - BUY YES @ X means someone SELL NO @ (1-X)
    """

    bin_index: int
    yes_token_id: str

    # Raw orderbook from API (YES perspective)
    yes_bids: List[OrderbookLevel] = field(default_factory=list)  # Sorted high to low
    yes_asks: List[OrderbookLevel] = field(default_factory=list)  # Sorted low to high

    # Timestamp of last update
    last_updated: Optional[float] = None

    @property
    def best_yes_bid(self) -> Optional[float]:
        """Best bid price for YES (highest someone will pay)."""
        return self.yes_bids[0].price if self.yes_bids else None

    @property
    def best_yes_ask(self) -> Optional[float]:
        """Best ask price for YES (lowest someone will sell)."""
        return self.yes_asks[0].price if self.yes_asks else None

    @property
    def best_no_bid(self) -> Optional[float]:
        """
        Best bid price for NO.
        BUY NO @ X = SELL YES @ (1-X), so NO bid = 1 - YES ask
        """
        if self.yes_asks:
            return 1.0 - self.yes_asks[0].price
        return None

    @property
    def best_no_ask(self) -> Optional[float]:
        """
        Best ask price for NO.
        SELL NO @ X = BUY YES @ (1-X), so NO ask = 1 - YES bid
        """
        if self.yes_bids:
            return 1.0 - self.yes_bids[0].price
        return None

    @property
    def yes_spread(self) -> Optional[float]:
        """Bid-ask spread for YES token."""
        if self.best_yes_bid and self.best_yes_ask:
            return self.best_yes_ask - self.best_yes_bid
        return None

    @property
    def mid_price_yes(self) -> Optional[float]:
        """Mid price for YES token."""
        if self.best_yes_bid and self.best_yes_ask:
            return (self.best_yes_bid + self.best_yes_ask) / 2.0
        return None

    def get_yes_bid_depth(self, max_levels: int = 10) -> float:
        """Total size available on YES bid side."""
        return sum(level.size for level in self.yes_bids[:max_levels])

    def get_yes_ask_depth(self, max_levels: int = 10) -> float:
        """Total size available on YES ask side."""
        return sum(level.size for level in self.yes_asks[:max_levels])

    def get_no_bid_depth(self, max_levels: int = 10) -> float:
        """
        Total size available on NO bid side.
        NO bids = YES asks (inverted).
        """
        return self.get_yes_ask_depth(max_levels)

    def get_no_ask_depth(self, max_levels: int = 10) -> float:
        """
        Total size available on NO ask side.
        NO asks = YES bids (inverted).
        """
        return self.get_yes_bid_depth(max_levels)

    @classmethod
    def from_api_response(
        cls,
        bin_index: int,
        yes_token_id: str,
        bids: List[Dict],
        asks: List[Dict],
        timestamp: Optional[float] = None,
    ) -> "UnifiedOrderbook":
        """
        Create from Polymarket API response.

        API format: [{"price": "0.45", "size": "100"}, ...]
        """
        yes_bids = [
            OrderbookLevel(price=float(b["price"]), size=float(b["size"]))
            for b in bids
        ]
        yes_asks = [
            OrderbookLevel(price=float(a["price"]), size=float(a["size"]))
            for a in asks
        ]

        # Sort bids high to low, asks low to high
        yes_bids.sort(key=lambda x: x.price, reverse=True)
        yes_asks.sort(key=lambda x: x.price)

        return cls(
            bin_index=bin_index,
            yes_token_id=yes_token_id,
            yes_bids=yes_bids,
            yes_asks=yes_asks,
            last_updated=timestamp,
        )


def compute_vwap(
    levels: List[OrderbookLevel],
    target_size: float,
) -> Tuple[float, float]:
    """
    Compute volume-weighted average price for a target size.

    Walks through orderbook levels until target size is filled.

    Args:
        levels: Orderbook levels (sorted appropriately for direction)
        target_size: Target number of shares to fill

    Returns:
        Tuple of (vwap_price, filled_size)
        If not enough liquidity, filled_size < target_size
    """
    if not levels or target_size <= 0:
        return 0.0, 0.0

    total_cost = 0.0
    filled = 0.0

    for level in levels:
        remaining = target_size - filled
        if remaining <= 0:
            break

        fill_at_level = min(level.size, remaining)
        total_cost += fill_at_level * level.price
        filled += fill_at_level

    if filled <= 0:
        return 0.0, 0.0

    return total_cost / filled, filled


def compute_vwap_buy_yes(
    orderbook: UnifiedOrderbook,
    target_size: float,
) -> Tuple[float, float]:
    """
    Compute VWAP for buying YES tokens.

    Walks through ask side (we lift asks to buy).

    Returns:
        Tuple of (vwap_price, filled_size)
    """
    return compute_vwap(orderbook.yes_asks, target_size)


def compute_vwap_sell_yes(
    orderbook: UnifiedOrderbook,
    target_size: float,
) -> Tuple[float, float]:
    """
    Compute VWAP for selling YES tokens.

    Walks through bid side (we hit bids to sell).

    Returns:
        Tuple of (vwap_price, filled_size)
    """
    return compute_vwap(orderbook.yes_bids, target_size)


def compute_vwap_buy_no(
    orderbook: UnifiedOrderbook,
    target_size: float,
) -> Tuple[float, float]:
    """
    Compute VWAP for buying NO tokens.

    BUY NO @ X = SELL YES @ (1-X)
    So we hit YES bids and invert prices.

    Returns:
        Tuple of (vwap_price, filled_size) in NO terms
    """
    # Walk YES bids, but invert prices for NO
    if not orderbook.yes_bids or target_size <= 0:
        return 0.0, 0.0

    total_cost = 0.0
    filled = 0.0

    for level in orderbook.yes_bids:
        remaining = target_size - filled
        if remaining <= 0:
            break

        fill_at_level = min(level.size, remaining)
        # NO price = 1 - YES price
        no_price = 1.0 - level.price
        total_cost += fill_at_level * no_price
        filled += fill_at_level

    if filled <= 0:
        return 0.0, 0.0

    return total_cost / filled, filled


def compute_vwap_sell_no(
    orderbook: UnifiedOrderbook,
    target_size: float,
) -> Tuple[float, float]:
    """
    Compute VWAP for selling NO tokens.

    SELL NO @ X = BUY YES @ (1-X)
    So we lift YES asks and invert prices.

    Returns:
        Tuple of (vwap_price, filled_size) in NO terms
    """
    # Walk YES asks, but invert prices for NO
    if not orderbook.yes_asks or target_size <= 0:
        return 0.0, 0.0

    total_cost = 0.0
    filled = 0.0

    for level in orderbook.yes_asks:
        remaining = target_size - filled
        if remaining <= 0:
            break

        fill_at_level = min(level.size, remaining)
        # NO price = 1 - YES price
        no_price = 1.0 - level.price
        total_cost += fill_at_level * no_price
        filled += fill_at_level

    if filled <= 0:
        return 0.0, 0.0

    return total_cost / filled, filled


def estimate_slippage(
    orderbook: UnifiedOrderbook,
    action: str,  # "BUY_YES", "SELL_YES", "BUY_NO", "SELL_NO"
    size: float,
) -> float:
    """
    Estimate slippage for a given trade size.

    Slippage = VWAP - best price

    Args:
        orderbook: The orderbook
        action: Trade action
        size: Trade size in shares

    Returns:
        Estimated slippage (positive = worse execution)
    """
    if action == "BUY_YES":
        vwap, _ = compute_vwap_buy_yes(orderbook, size)
        best = orderbook.best_yes_ask
        if vwap and best:
            return vwap - best
    elif action == "SELL_YES":
        vwap, _ = compute_vwap_sell_yes(orderbook, size)
        best = orderbook.best_yes_bid
        if vwap and best:
            return best - vwap  # Inverted because selling
    elif action == "BUY_NO":
        vwap, _ = compute_vwap_buy_no(orderbook, size)
        best = orderbook.best_no_ask
        if vwap and best:
            return vwap - best
    elif action == "SELL_NO":
        vwap, _ = compute_vwap_sell_no(orderbook, size)
        best = orderbook.best_no_bid
        if vwap and best:
            return best - vwap

    return 0.0


def get_available_depth(
    orderbook: UnifiedOrderbook,
    action: str,
    max_levels: int = 10,
) -> float:
    """
    Get available depth for a given action.

    Args:
        orderbook: The orderbook
        action: Trade action
        max_levels: Maximum levels to consider

    Returns:
        Available size in shares
    """
    if action == "BUY_YES":
        return orderbook.get_yes_ask_depth(max_levels)
    elif action == "SELL_YES":
        return orderbook.get_yes_bid_depth(max_levels)
    elif action == "BUY_NO":
        return orderbook.get_no_ask_depth(max_levels)
    elif action == "SELL_NO":
        return orderbook.get_no_bid_depth(max_levels)
    return 0.0
