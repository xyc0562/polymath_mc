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


def get_friction(fair_value: float, config: EdgeBufferConfig) -> float:
    """
    Get friction (c) based on probability zone.

    Tails (p <= 10% or p >= 90%) have higher friction due to worse liquidity.
    """
    if fair_value <= config.tail_threshold or fair_value >= (1 - config.tail_threshold):
        return config.friction_tail
    return config.friction_mid


def compute_buy_yes_threshold(
    fair_value: float,
    config: EdgeBufferConfig,
) -> float:
    """
    Compute max YES price to buy (BUY_YES threshold).

    Formula: p_m <= (p_f - c) / (1 + r)

    Where:
    - p_f = fair value
    - c = friction
    - r = required ROI on stake

    If p_f <= c, returns 0 (don't buy - friction dominates).
    """
    c = get_friction(fair_value, config)
    r = config.required_roi

    if fair_value <= c:
        return 0.0  # Friction dominates, don't buy YES

    return (fair_value - c) / (1 + r)


def compute_buy_no_threshold(
    fair_value: float,
    config: EdgeBufferConfig,
) -> float:
    """
    Compute min YES price to buy NO (BUY_NO threshold).

    When YES price is above this threshold, buy NO.

    Formula: p_m >= (p_f + c + r) / (1 + r)

    Where:
    - p_f = YES fair value
    - c = friction (based on NO probability = 1 - p_f)
    - r = required ROI on stake

    If p_f >= 1 - c - r, returns 1.0 (don't buy NO).
    """
    no_fair_value = 1.0 - fair_value
    c = get_friction(no_fair_value, config)  # Use NO's probability for friction
    r = config.required_roi

    if fair_value >= 1 - c - r:
        return 1.0  # Don't buy NO

    return (fair_value + c + r) / (1 + r)


def compute_required_edge(
    reservation_price: float,
    config: EdgeBufferConfig,
) -> float:
    """
    Compute required edge for display/logging purposes.

    This returns the edge required for BUY_YES at this reservation price.
    """
    threshold = compute_buy_yes_threshold(reservation_price, config)
    return reservation_price - threshold


def compute_buy_threshold(
    reservation_price: float,
    config: EdgeBufferConfig,
) -> float:
    """
    Compute the buy threshold (max price to pay) for buying this asset.

    This is used for BUY_YES when reservation_price is YES fair value,
    and for BUY_NO when reservation_price is NO fair value.
    """
    return compute_buy_yes_threshold(reservation_price, config)


def compute_sell_threshold(
    reservation_price: float,
    config: EdgeBufferConfig,
) -> float:
    """
    Compute the sell threshold (min price to receive) for NEW sell candidates.

    SELL_YES is equivalent to counterparty buying YES from us.
    We want YES to be overpriced, which is the BUY_NO condition.

    sell_threshold_yes = buy_no_threshold (min YES price for buying NO)

    Note: This is for generating new sell opportunities with edge.
    For exiting existing positions, use compute_exit_threshold instead.
    """
    return compute_buy_no_threshold(reservation_price, config)


def compute_exit_threshold(
    reservation_price: float,
    config: EdgeBufferConfig,
) -> float:
    """
    Compute the exit threshold for closing existing positions.

    When exiting an existing position, we don't need additional edge -
    we already captured edge on entry. Just exit at fair value or better.

    Args:
        reservation_price: Fair value of the position we hold
        config: Edge buffer config (unused, but kept for API consistency)

    Returns:
        Minimum price to accept for exiting (= fair value)
    """
    return reservation_price


def should_trade(
    market_price: float,
    reservation_price: float,
    action: TradeAction,
    config: EdgeBufferConfig,
) -> tuple[bool, float]:
    """
    Determine if trade meets edge requirement using stake-based ROI model.

    For BUY_YES/BUY_NO: reservation_price is the fair value of the asset being bought.
    For SELL_YES/SELL_NO: reservation_price is the fair value of the asset being sold.

    Args:
        market_price: Current market price (VWAP) of the asset being traded
        reservation_price: Fair value of the asset being traded
        action: Trade action
        config: Edge buffer configuration

    Returns:
        Tuple of (should_trade, actual_edge_pct)
        actual_edge_pct is the ROI on stake if price reverts to fair value
    """
    if reservation_price <= 0 or reservation_price >= 1:
        return False, 0.0

    if action == TradeAction.BUY_YES:
        # Buy YES if market price <= (fair - c) / (1 + r)
        threshold = compute_buy_yes_threshold(reservation_price, config)
        # ROI = (fair - paid) / paid
        actual_edge = (reservation_price - market_price) / market_price if market_price > 0 else 0.0
        return market_price <= threshold and threshold > 0, actual_edge

    elif action == TradeAction.BUY_NO:
        # BUY_NO: reservation_price = NO fair value
        # Convert to YES fair value for the formula
        yes_fair_value = 1.0 - reservation_price
        # YES price must be >= (p_f + c + r) / (1 + r) for us to buy NO
        yes_threshold = compute_buy_no_threshold(yes_fair_value, config)
        # Convert market price from NO to YES: yes_market = 1 - no_market
        yes_market_price = 1.0 - market_price
        # ROI on stake: we risk (1 - p_m) = no_price to win p_m = yes_price
        actual_edge = (reservation_price - market_price) / market_price if market_price > 0 else 0.0
        return yes_market_price >= yes_threshold and yes_threshold < 1.0, actual_edge

    elif action == TradeAction.SELL_YES:
        # SELL_YES: we want YES to be overpriced (same condition as BUY_NO)
        yes_threshold = compute_buy_no_threshold(reservation_price, config)
        actual_edge = (market_price - reservation_price) / reservation_price if reservation_price > 0 else 0.0
        return market_price >= yes_threshold and yes_threshold < 1.0, actual_edge

    else:  # SELL_NO
        # SELL_NO: we want NO to be overpriced
        # reservation_price = NO fair value
        # Convert to YES fair value, use BUY_YES threshold logic inverted
        yes_fair_value = 1.0 - reservation_price
        yes_threshold = compute_buy_yes_threshold(yes_fair_value, config)
        # NO market price must be > 1 - yes_threshold
        no_threshold = 1.0 - yes_threshold if yes_threshold > 0 else 1.0
        actual_edge = (market_price - reservation_price) / reservation_price if reservation_price > 0 else 0.0
        return market_price >= no_threshold, actual_edge


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

        # Get current position for this bin
        position = portfolio.get_position(bin_index)
        has_yes = position and position.has_yes_position
        has_no = position and position.has_no_position

        # Generate BUY YES candidate (only if we don't have NO position)
        # If we have NO, we should SELL_NO first rather than buying YES
        if not has_no:
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
        # Use model probability (not reservation price) for exit threshold
        if has_yes:
            model_prob_yes = portfolio.probabilities[bin_index]
            candidate = _generate_sell_yes_candidate(
                bin_index=bin_index,
                orderbook=orderbook,
                portfolio=portfolio,
                model_probability=model_prob_yes,
                config=config,
                hours_to_settlement=hours_to_settlement,
            )
            if candidate:
                candidates.append(candidate)

        # Generate BUY NO candidate (only if we don't have YES position)
        # If we have YES, we should SELL_YES first rather than buying NO
        if not has_yes:
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
        # Use model probability (not reservation price) for exit threshold
        if has_no:
            model_prob_no = 1.0 - portfolio.probabilities[bin_index]
            candidate = _generate_sell_no_candidate(
                bin_index=bin_index,
                orderbook=orderbook,
                portfolio=portfolio,
                model_probability=model_prob_no,
                config=config,
                hours_to_settlement=hours_to_settlement,
            )
            if candidate:
                candidates.append(candidate)

    # Separate sells from buys
    # Sells are processed first (exit at fair value or better, not utility-based)
    # Then buys are sorted by utility gain
    sells = [c for c in candidates if c.action in (TradeAction.SELL_YES, TradeAction.SELL_NO)]
    buys = [c for c in candidates if c.action in (TradeAction.BUY_YES, TradeAction.BUY_NO)]

    # Sort buys by utility gain (descending)
    buys.sort(key=lambda c: c.utility_gain, reverse=True)

    # Sells come first, then buys
    return sells + buys


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
        logger.debug(f"BUY_YES bin {bin_index}: no depth")
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
        logger.debug(f"BUY_YES bin {bin_index}: insufficient capital")
        return None

    # Get VWAP for this chunk
    vwap, filled = compute_vwap_buy_yes(orderbook, delta)
    if filled <= 0:
        logger.debug(f"BUY_YES bin {bin_index}: no fill at delta={delta:.2f}")
        return None

    # Check minimum perceived probability (from our model)
    if config.edge_buffer.min_perceived_prob > 0 and reservation_price < config.edge_buffer.min_perceived_prob:
        logger.debug(f"BUY_YES bin {bin_index}: model prob too low ({reservation_price:.4f} < {config.edge_buffer.min_perceived_prob:.4f})")
        return None

    # Check minimum market price
    if config.edge_buffer.min_market_price > 0 and vwap < config.edge_buffer.min_market_price:
        logger.debug(f"BUY_YES bin {bin_index}: market price too low ({vwap:.4f} < {config.edge_buffer.min_market_price:.4f})")
        return None

    # Check edge requirement
    trade_ok, actual_edge = should_trade(
        vwap, reservation_price, TradeAction.BUY_YES, config.edge_buffer
    )
    if not trade_ok:
        threshold = compute_buy_threshold(reservation_price, config.edge_buffer)
        logger.debug(f"BUY_YES bin {bin_index}: edge check failed (vwap={vwap:.4f}, res={reservation_price:.4f}, thresh={threshold:.4f})")
        return None

    # Simulate trade and compute utility gain
    new_portfolio = portfolio.simulate_buy_yes(bin_index, filled, vwap)
    utility_gain = _compute_portfolio_utility_gain(portfolio, new_portfolio, config)

    if utility_gain < config.tau:
        logger.debug(f"BUY_YES bin {bin_index}: utility too low ({utility_gain:.6f} < {config.tau:.6f})")
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
    model_probability: float,
    config: KellyConfig,
    hours_to_settlement: float,
) -> Optional[TradeCandidate]:
    """
    Generate a SELL YES candidate if profitable.

    Since we only generate SELL_YES when we have a YES position, this is
    always an EXIT (closing existing long), not a new short.

    For exits, we use model probability as the exit threshold (not reservation
    price). Exit when edge disappears (market >= model fair value).
    """
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

    # For EXITS: use model probability as threshold (exit when edge disappears)
    # We exit when market >= model fair value, not based on Kelly reservation
    exit_threshold = model_probability

    # Only exit if we can get fair value or better
    if vwap < exit_threshold:
        logger.debug(
            f"SELL_YES bin {bin_index}: below exit threshold "
            f"(vwap={vwap:.4f}, fair={exit_threshold:.4f})"
        )
        return None

    # Calculate edge relative to model fair value
    actual_edge = (vwap - model_probability) / model_probability if model_probability > 0 else 0.0

    new_portfolio = portfolio.simulate_sell_yes(bin_index, filled, vwap)
    utility_gain = _compute_portfolio_utility_gain(portfolio, new_portfolio, config)

    # For exits, skip tau check - we already passed fair value threshold
    # Utility gain at fair value is ~0, which would block exits unnecessarily

    return TradeCandidate(
        bin_index=bin_index,
        action=TradeAction.SELL_YES,
        size=filled,
        price=vwap,
        utility_gain=utility_gain,
        reservation_price=model_probability,  # Use model prob as "fair price" for exits
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
        logger.debug(f"BUY_NO bin {bin_index}: no depth")
        return None

    delta = compute_adaptive_delta(
        config.adaptive_delta,
        depth,
        hours_to_settlement,
        config.t_stop_hours,
    )
    delta *= config.kappa

    if portfolio.available_capital < delta * 0.01:
        logger.debug(f"BUY_NO bin {bin_index}: insufficient capital (need {delta * 0.01:.2f}, have {portfolio.available_capital:.2f})")
        return None

    vwap, filled = compute_vwap_buy_no(orderbook, delta)
    if filled <= 0:
        logger.debug(f"BUY_NO bin {bin_index}: no fill at delta={delta:.2f}")
        return None

    # Check minimum perceived probability (from our model)
    if config.edge_buffer.min_perceived_prob > 0 and reservation_price < config.edge_buffer.min_perceived_prob:
        logger.debug(f"BUY_NO bin {bin_index}: model prob too low ({reservation_price:.4f} < {config.edge_buffer.min_perceived_prob:.4f})")
        return None

    # Check minimum market price
    if config.edge_buffer.min_market_price > 0 and vwap < config.edge_buffer.min_market_price:
        logger.debug(f"BUY_NO bin {bin_index}: market price too low ({vwap:.4f} < {config.edge_buffer.min_market_price:.4f})")
        return None

    trade_ok, actual_edge = should_trade(
        vwap, reservation_price, TradeAction.BUY_NO, config.edge_buffer
    )
    if not trade_ok:
        req_edge = compute_required_edge(reservation_price, config.edge_buffer)
        threshold = reservation_price * (1 - req_edge)
        logger.debug(f"BUY_NO bin {bin_index}: edge check failed (vwap={vwap:.4f}, res={reservation_price:.4f}, req_edge={req_edge:.2%}, thresh={threshold:.4f})")
        return None

    new_portfolio = portfolio.simulate_buy_no(bin_index, filled, vwap)
    utility_gain = _compute_portfolio_utility_gain(portfolio, new_portfolio, config)

    if utility_gain < config.tau:
        logger.debug(f"BUY_NO bin {bin_index}: utility too low ({utility_gain:.6f} < {config.tau:.6f})")
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
    model_probability: float,
    config: KellyConfig,
    hours_to_settlement: float,
) -> Optional[TradeCandidate]:
    """
    Generate a SELL NO candidate if profitable.

    Since we only generate SELL_NO when we have a NO position, this is
    always an EXIT (closing existing long), not a new short.

    For exits, we use model probability as the exit threshold (not reservation
    price). Exit when edge disappears (market >= model fair value).
    """
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

    # For EXITS: use model probability as threshold (exit when edge disappears)
    # We exit when market >= model fair value, not based on Kelly reservation
    exit_threshold = model_probability

    # Only exit if we can get fair value or better
    if vwap < exit_threshold:
        logger.debug(
            f"SELL_NO bin {bin_index}: below exit threshold "
            f"(vwap={vwap:.4f}, fair={exit_threshold:.4f})"
        )
        return None

    # Calculate edge relative to model fair value
    actual_edge = (vwap - model_probability) / model_probability if model_probability > 0 else 0.0

    new_portfolio = portfolio.simulate_sell_no(bin_index, filled, vwap)
    utility_gain = _compute_portfolio_utility_gain(portfolio, new_portfolio, config)

    # For exits, skip tau check - we already passed fair value threshold
    # Utility gain at fair value is ~0, which would block exits unnecessarily

    return TradeCandidate(
        bin_index=bin_index,
        action=TradeAction.SELL_NO,
        size=filled,
        price=vwap,
        utility_gain=utility_gain,
        reservation_price=model_probability,  # Use model prob as "fair price" for exits
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
