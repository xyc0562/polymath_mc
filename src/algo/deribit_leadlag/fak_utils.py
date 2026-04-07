"""
FAK order size/price utilities for Polymarket.

Extracted from musk kelly executor to avoid importing its full dependency chain.
These are standalone pure functions for tick-size and 2-decimal-place compliance.
"""

import math

MIN_ORDER_SIZE = 15
MIN_ORDER_VALUE_USD = 1.0


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

    For BUY: tries bumping price UP by 1-N ticks.
    For SELL: tries bumping price DOWN by 1-N ticks.

    Returns (adjusted_price, adjusted_size).
    """
    base_step = _fak_size_step(price, tick_size)
    if base_step <= 1:
        return price, size

    best_size = (size // base_step) * base_step
    best_price = price

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
