"""
Trade candidate generation for Kelly trading.

Generates potential trades based on:
- Reservation prices vs market prices
- Edge buffer requirements
- Adaptive chunk sizing
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import logging

from .config import KellyConfig, EdgeBufferConfig, AdaptiveDeltaConfig
from .orderbook import (
    UnifiedOrderbook,
    compute_vwap_buy_yes,
    compute_vwap_sell_yes,
    compute_vwap_buy_no,
    compute_vwap_sell_no,
    get_available_depth,
)
from .portfolio import Portfolio
from .kelly_math import compute_utility_gain

logger = logging.getLogger(__name__)


class TradeAction(Enum):
    """Possible trade actions."""

    BUY_YES = "BUY_YES"
    SELL_YES = "SELL_YES"
    BUY_NO = "BUY_NO"
    SELL_NO = "SELL_NO"


@dataclass
class TradeCandidate:
    """A potential trade with expected utility gain."""

    bin_index: int
    action: TradeAction
    size: float  # Shares
    price: float  # VWAP price
    utility_gain: float  # Expected utility improvement
    reservation_price: float  # Kelly fair price
    edge: float  # Edge = |market - reservation| / reservation

    @property
    def cost(self) -> float:
        """Cost of the trade (positive for buys)."""
        if self.action in (TradeAction.BUY_YES, TradeAction.BUY_NO):
            return self.size * self.price
        else:
            return -self.size * self.price  # Negative = proceeds


def compute_required_edge(
    reservation_price: float,
    config: EdgeBufferConfig,
) -> float:
    """
    Compute required edge based on reservation price.

    Requires larger edge at extreme prices where model errors
    have larger relative impact.

    At p=0.50: require base_edge (5%)
    At p=0.10: require ~8% (1.6x)
    At p=0.05: require ~9% (1.8x)
    """
    # Distance from center (0 at p=0.5, 1 at p=0 or p=1)
    extremity = abs(reservation_price - 0.5) * 2

    # Linear scaling: 1x at center, extreme_multiplier at edges
    multiplier = 1.0 + extremity * (config.extreme_multiplier - 1.0)

    return config.base_edge_pct * multiplier


def should_trade(
    market_price: float,
    reservation_price: float,
    action: TradeAction,
    config: EdgeBufferConfig,
) -> tuple[bool, float]:
    """
    Determine if trade meets edge requirement.

    Args:
        market_price: Current market price (VWAP)
        reservation_price: Kelly reservation price
        action: Trade action
        config: Edge buffer configuration

    Returns:
        Tuple of (should_trade, actual_edge)
    """
    if reservation_price <= 0 or reservation_price >= 1:
        return False, 0.0

    required_edge = compute_required_edge(reservation_price, config)

    if action in (TradeAction.BUY_YES, TradeAction.BUY_NO):
        # Buy if market_price < reservation * (1 - edge)
        threshold = reservation_price * (1 - required_edge)
        actual_edge = (reservation_price - market_price) / reservation_price
        return market_price < threshold, actual_edge

    else:  # SELL
        # Sell if market_price > reservation * (1 + edge)
        threshold = reservation_price * (1 + required_edge)
        actual_edge = (market_price - reservation_price) / reservation_price
        return market_price > threshold, actual_edge


def compute_adaptive_delta(
    config: AdaptiveDeltaConfig,
    available_depth: float,
    hours_to_settlement: float,
    t_stop_hours: float,
) -> float:
    """
    Compute adaptive chunk size based on market conditions.

    Args:
        config: Adaptive delta configuration
        available_depth: Available depth in orderbook
        hours_to_settlement: Hours until market settlement
        t_stop_hours: T_stop cutoff hours

    Returns:
        Adaptive chunk size in shares
    """
    # 1. Liquidity constraint: never take >X% of visible depth
    liquidity_delta = available_depth * config.max_depth_fraction

    # 2. Time constraint: ramp down as we approach T_stop
    hours_until_stop = hours_to_settlement - t_stop_hours
    if hours_until_stop <= 0:
        time_delta = config.min_delta  # At or past T_stop
    elif hours_until_stop >= config.time_ramp_hours:
        time_delta = config.base_delta  # Full size
    else:
        # Linear ramp: 100% at ramp_hours, 50% at 0h before T_stop
        time_factor = 0.5 + 0.5 * (hours_until_stop / config.time_ramp_hours)
        time_delta = config.base_delta * time_factor

    return max(
        config.min_delta,
        min(config.base_delta, liquidity_delta, time_delta),
    )


def generate_candidates(
    portfolio: Portfolio,
    orderbooks: dict[int, UnifiedOrderbook],
    config: KellyConfig,
    hours_to_settlement: float,
) -> List[TradeCandidate]:
    """
    Generate all feasible trade candidates.

    Args:
        portfolio: Current portfolio state
        orderbooks: Orderbooks for each bin (keyed by bin_index)
        config: Kelly configuration
        hours_to_settlement: Hours until settlement

    Returns:
        List of trade candidates sorted by utility gain (descending)
    """
    candidates = []

    # Get current reservation prices
    yes_prices, no_prices = portfolio.get_reservation_prices(config.w_floor)

    for bin_index in range(portfolio.num_bins):
        # Skip dead bins
        if bin_index in portfolio.dead_bins:
            continue

        orderbook = orderbooks.get(bin_index)
        if not orderbook:
            continue

        reservation_yes = yes_prices[bin_index]
        reservation_no = no_prices[bin_index]

        # Generate BUY YES candidate
        candidate = _generate_buy_yes_candidate(
            bin_index=bin_index,
            orderbook=orderbook,
            portfolio=portfolio,
            reservation_price=reservation_yes,
            config=config,
            hours_to_settlement=hours_to_settlement,
        )
        if candidate:
            candidates.append(candidate)

        # Generate SELL YES candidate (if we have position)
        position = portfolio.get_position(bin_index)
        if position and position.has_yes_position:
            candidate = _generate_sell_yes_candidate(
                bin_index=bin_index,
                orderbook=orderbook,
                portfolio=portfolio,
                reservation_price=reservation_yes,
                config=config,
                hours_to_settlement=hours_to_settlement,
            )
            if candidate:
                candidates.append(candidate)

        # Generate BUY NO candidate
        candidate = _generate_buy_no_candidate(
            bin_index=bin_index,
            orderbook=orderbook,
            portfolio=portfolio,
            reservation_price=reservation_no,
            config=config,
            hours_to_settlement=hours_to_settlement,
        )
        if candidate:
            candidates.append(candidate)

        # Generate SELL NO candidate (if we have position)
        if position and position.has_no_position:
            candidate = _generate_sell_no_candidate(
                bin_index=bin_index,
                orderbook=orderbook,
                portfolio=portfolio,
                reservation_price=reservation_no,
                config=config,
                hours_to_settlement=hours_to_settlement,
            )
            if candidate:
                candidates.append(candidate)

    # Sort by utility gain (descending)
    candidates.sort(key=lambda c: c.utility_gain, reverse=True)

    return candidates


def _generate_buy_yes_candidate(
    bin_index: int,
    orderbook: UnifiedOrderbook,
    portfolio: Portfolio,
    reservation_price: float,
    config: KellyConfig,
    hours_to_settlement: float,
) -> Optional[TradeCandidate]:
    """Generate a BUY YES candidate if profitable."""
    # Get available depth
    depth = get_available_depth(orderbook, "BUY_YES")
    if depth <= 0:
        return None

    # Compute adaptive chunk size
    delta = compute_adaptive_delta(
        config.adaptive_delta,
        depth,
        hours_to_settlement,
        config.t_stop_hours,
    )

    # Apply fractional Kelly
    delta *= config.kappa

    # Check we have capital
    if portfolio.available_capital < delta * 0.01:  # Rough check
        return None

    # Get VWAP for this chunk
    vwap, filled = compute_vwap_buy_yes(orderbook, delta)
    if filled <= 0:
        return None

    # Check edge requirement
    trade_ok, actual_edge = should_trade(
        vwap, reservation_price, TradeAction.BUY_YES, config.edge_buffer
    )
    if not trade_ok:
        return None

    # Simulate trade and compute utility gain
    new_portfolio = portfolio.simulate_buy_yes(bin_index, filled, vwap)
    utility_gain = _compute_portfolio_utility_gain(portfolio, new_portfolio, config)

    if utility_gain < config.tau:
        return None

    return TradeCandidate(
        bin_index=bin_index,
        action=TradeAction.BUY_YES,
        size=filled,
        price=vwap,
        utility_gain=utility_gain,
        reservation_price=reservation_price,
        edge=actual_edge,
    )


def _generate_sell_yes_candidate(
    bin_index: int,
    orderbook: UnifiedOrderbook,
    portfolio: Portfolio,
    reservation_price: float,
    config: KellyConfig,
    hours_to_settlement: float,
) -> Optional[TradeCandidate]:
    """Generate a SELL YES candidate if profitable."""
    position = portfolio.get_position(bin_index)
    if not position or not position.has_yes_position:
        return None

    depth = get_available_depth(orderbook, "SELL_YES")
    if depth <= 0:
        return None

    delta = compute_adaptive_delta(
        config.adaptive_delta,
        depth,
        hours_to_settlement,
        config.t_stop_hours,
    )

    # Don't sell more than we have
    delta = min(delta, position.yes_shares)
    delta *= config.kappa

    if delta <= 0:
        return None

    vwap, filled = compute_vwap_sell_yes(orderbook, delta)
    if filled <= 0:
        return None

    trade_ok, actual_edge = should_trade(
        vwap, reservation_price, TradeAction.SELL_YES, config.edge_buffer
    )
    if not trade_ok:
        return None

    new_portfolio = portfolio.simulate_sell_yes(bin_index, filled, vwap)
    utility_gain = _compute_portfolio_utility_gain(portfolio, new_portfolio, config)

    if utility_gain < config.tau:
        return None

    return TradeCandidate(
        bin_index=bin_index,
        action=TradeAction.SELL_YES,
        size=filled,
        price=vwap,
        utility_gain=utility_gain,
        reservation_price=reservation_price,
        edge=actual_edge,
    )


def _generate_buy_no_candidate(
    bin_index: int,
    orderbook: UnifiedOrderbook,
    portfolio: Portfolio,
    reservation_price: float,
    config: KellyConfig,
    hours_to_settlement: float,
) -> Optional[TradeCandidate]:
    """Generate a BUY NO candidate if profitable."""
    depth = get_available_depth(orderbook, "BUY_NO")
    if depth <= 0:
        return None

    delta = compute_adaptive_delta(
        config.adaptive_delta,
        depth,
        hours_to_settlement,
        config.t_stop_hours,
    )
    delta *= config.kappa

    if portfolio.available_capital < delta * 0.01:
        return None

    vwap, filled = compute_vwap_buy_no(orderbook, delta)
    if filled <= 0:
        return None

    trade_ok, actual_edge = should_trade(
        vwap, reservation_price, TradeAction.BUY_NO, config.edge_buffer
    )
    if not trade_ok:
        return None

    new_portfolio = portfolio.simulate_buy_no(bin_index, filled, vwap)
    utility_gain = _compute_portfolio_utility_gain(portfolio, new_portfolio, config)

    if utility_gain < config.tau:
        return None

    return TradeCandidate(
        bin_index=bin_index,
        action=TradeAction.BUY_NO,
        size=filled,
        price=vwap,
        utility_gain=utility_gain,
        reservation_price=reservation_price,
        edge=actual_edge,
    )


def _generate_sell_no_candidate(
    bin_index: int,
    orderbook: UnifiedOrderbook,
    portfolio: Portfolio,
    reservation_price: float,
    config: KellyConfig,
    hours_to_settlement: float,
) -> Optional[TradeCandidate]:
    """Generate a SELL NO candidate if profitable."""
    position = portfolio.get_position(bin_index)
    if not position or not position.has_no_position:
        return None

    depth = get_available_depth(orderbook, "SELL_NO")
    if depth <= 0:
        return None

    delta = compute_adaptive_delta(
        config.adaptive_delta,
        depth,
        hours_to_settlement,
        config.t_stop_hours,
    )

    delta = min(delta, position.no_shares)
    delta *= config.kappa

    if delta <= 0:
        return None

    vwap, filled = compute_vwap_sell_no(orderbook, delta)
    if filled <= 0:
        return None

    trade_ok, actual_edge = should_trade(
        vwap, reservation_price, TradeAction.SELL_NO, config.edge_buffer
    )
    if not trade_ok:
        return None

    new_portfolio = portfolio.simulate_sell_no(bin_index, filled, vwap)
    utility_gain = _compute_portfolio_utility_gain(portfolio, new_portfolio, config)

    if utility_gain < config.tau:
        return None

    return TradeCandidate(
        bin_index=bin_index,
        action=TradeAction.SELL_NO,
        size=filled,
        price=vwap,
        utility_gain=utility_gain,
        reservation_price=reservation_price,
        edge=actual_edge,
    )


def _compute_portfolio_utility_gain(
    before: Portfolio,
    after: Portfolio,
    config: KellyConfig,
) -> float:
    """Compute utility gain from a trade."""
    terminal_before = before.get_terminal_wealths(config.w_floor)
    terminal_after = after.get_terminal_wealths(config.w_floor)

    return compute_utility_gain(
        before.probabilities,
        terminal_before,
        terminal_after,
        config.w_floor,
    )
