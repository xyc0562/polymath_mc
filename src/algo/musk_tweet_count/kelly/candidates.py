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
import math

from .config import KellyConfig, EdgeBufferConfig
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

# Polymarket minimum order value (USD)
# Orders below this will be rejected by the API
MIN_ORDER_VALUE_USD = 1.0

# Polymarket minimum order size (shares)
# Orders below this will be rejected by the API
MIN_ORDER_SIZE = 15

# Fixed screening chunk size in USD for candidate generation.
# Small enough that utility gain ≈ marginal utility (good for ranking by utility/$).
# Edge check (should_trade) filters garbage; actual sizing done by _find_optimal_size_on.
SCREENING_CHUNK_USD = 2.0


def check_orderbook_liquidity(
    orderbook: UnifiedOrderbook,
    config: EdgeBufferConfig,
    side: str,  # "YES" or "NO"
) -> tuple[bool, str]:
    """
    Check if orderbook has sufficient liquidity for trading.

    Args:
        orderbook: The orderbook to check
        config: Edge buffer configuration
        side: "YES" for YES token, "NO" for NO token

    Returns:
        Tuple of (is_valid, reason). If not valid, reason explains why.
    """
    if side == "YES":
        best_bid = orderbook.best_yes_bid
        best_ask = orderbook.best_yes_ask
    else:
        best_bid = orderbook.best_no_bid
        best_ask = orderbook.best_no_ask

    # Check two-sided liquidity requirement
    if config.require_two_sided_liquidity:
        if best_bid is None or best_ask is None:
            return False, f"{side} has one-sided liquidity (bid={best_bid}, ask={best_ask})"

    # Check spread requirement (only if both sides exist)
    if config.max_spread_ratio > 0 and best_bid is not None and best_ask is not None:
        if best_bid > 0:
            spread_ratio = (best_ask - best_bid) / best_bid
            if spread_ratio > config.max_spread_ratio:
                return False, f"{side} spread too wide (ratio {spread_ratio:.2f} > {config.max_spread_ratio:.1f})"

    return True, ""


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
    limit_price: float = 0.0  # Worst orderbook level consumed (actual tick price for FAK)

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


def generate_candidates(
    portfolio: Portfolio,
    orderbooks: dict[int, UnifiedOrderbook],
    config: KellyConfig,
    hours_to_settlement: float,
    verbose: bool = False,
) -> List[TradeCandidate]:
    """
    Generate all feasible trade candidates.

    Args:
        portfolio: Current portfolio state
        orderbooks: Orderbooks for each bin (keyed by bin_index)
        config: Kelly configuration
        hours_to_settlement: Hours until settlement
        verbose: If True, log rejection reasons at INFO level

    Returns:
        List of trade candidates sorted by utility gain (descending)
    """
    candidates = []
    rejection_reasons: dict[int, list[str]] = {}  # bin_index -> list of rejection reasons

    # Log portfolio state that Kelly is using (should be fresh from API sync)
    num_positions = len([p for p in portfolio.positions.values()
                        if p.yes_shares > 0 or p.no_shares > 0])
    logger.debug(
        f"[KELLY INPUT] capital=${portfolio.capital:.2f}, "
        f"invested=${portfolio.total_collateral_used:.2f}, "
        f"positions={num_positions}, "
        f"c_event_max=${config.collateral.c_event_max:.0f}"
    )

    # Get current reservation prices (multi-bin Kelly)
    yes_prices, no_prices = portfolio.get_reservation_prices(config.w_floor, config.kelly_fraction)

    # Log portfolio state and top reservation prices for debugging
    if verbose:
        terminal_wealths = portfolio.get_terminal_wealths(config.w_floor)
        logger.info("Multi-bin Kelly state:")
        logger.info(
            f"  Capital: ${portfolio.capital:.2f}, Invested: ${portfolio.total_collateral_used:.2f}"
        )
        # Show bins with positions
        for bin_idx, pos in portfolio.positions.items():
            if pos.yes_shares > 0 or pos.no_shares > 0:
                logger.info(
                    f"  Bin {bin_idx}: YES={pos.yes_shares:.1f} NO={pos.no_shares:.1f} "
                    f"cost=${pos.collateral_used:.2f} | "
                    f"W[{bin_idx}]=${terminal_wealths[bin_idx]:.2f} | "
                    f"c*_YES={yes_prices[bin_idx]:.3f} c*_NO={no_prices[bin_idx]:.3f} | "
                    f"model_p={portfolio.probabilities[bin_idx]:.3f}"
                )
        # Show a few top model probability bins (potential candidates)
        prob_bins = sorted(
            [(i, p) for i, p in enumerate(portfolio.probabilities) if p > 0.01],
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        logger.info("  Top model probability bins:")
        for bin_idx, prob in prob_bins:
            logger.info(
                f"    Bin {bin_idx}: model_p={prob:.3f} c*_YES={yes_prices[bin_idx]:.3f} "
                f"c*_NO={no_prices[bin_idx]:.3f} W=${terminal_wealths[bin_idx]:.2f}"
            )

    for bin_index in range(portfolio.num_bins):
        # Skip dead bins
        if bin_index in portfolio.dead_bins:
            continue

        orderbook = orderbooks.get(bin_index)
        if not orderbook:
            if verbose:
                rejection_reasons.setdefault(bin_index, []).append("no orderbook")
            continue

        reservation_yes = yes_prices[bin_index]
        reservation_no = no_prices[bin_index]

        # Get current position for this bin
        position = portfolio.get_position(bin_index)
        has_yes = position and position.has_yes_position
        has_no = position and position.has_no_position

        # Check liquidity for YES and NO tokens (only for buying, not selling)
        yes_liquidity_ok, yes_reason = check_orderbook_liquidity(
            orderbook, config.edge_buffer, "YES"
        )
        no_liquidity_ok, no_reason = check_orderbook_liquidity(
            orderbook, config.edge_buffer, "NO"
        )

        # Generate BUY YES candidate (only if we don't have NO position)
        # If we have NO, we should SELL_NO first rather than buying YES
        # EXCEPTION: If NO position is stranded (below minimum size OR value < $1), allow buying YES
        no_is_stranded = False
        if has_no:
            no_shares = position.no_shares
            # NO sell price = 1 - YES ask price
            no_sell_price = 1.0 - orderbook.yes_asks[0].price if orderbook.yes_asks else 0.0
            no_value = no_shares * no_sell_price
            no_is_stranded = no_shares < MIN_ORDER_SIZE or no_value < MIN_ORDER_VALUE_USD
        if not has_no or no_is_stranded:
            if yes_liquidity_ok:
                candidate = _generate_buy_yes_candidate(
                    bin_index=bin_index,
                    orderbook=orderbook,
                    portfolio=portfolio,
                    reservation_price=reservation_yes,
                    config=config,
                    hours_to_settlement=hours_to_settlement,
                    verbose=verbose,
                    rejection_reasons=rejection_reasons,
                )
                if candidate:
                    candidates.append(candidate)
            elif verbose:
                rejection_reasons.setdefault(bin_index, []).append(f"BUY_YES: {yes_reason}")
        elif verbose and has_no and not no_is_stranded:
            # Log why BUY_YES is blocked due to existing NO position
            rejection_reasons.setdefault(bin_index, []).append(
                f"BUY_YES: have {position.no_shares:.0f} NO shares (sell NO first)"
            )

        # Generate SELL YES candidate (if we have position)
        # Use min(model probability, Kelly reservation) for exit threshold
        # NOTE: Always allow selling even if liquidity is poor (need to exit positions)
        if has_yes:
            model_prob_yes = portfolio.probabilities[bin_index]
            candidate = _generate_sell_yes_candidate(
                bin_index=bin_index,
                orderbook=orderbook,
                portfolio=portfolio,
                model_probability=model_prob_yes,
                reservation_price=reservation_yes,
                config=config,
                hours_to_settlement=hours_to_settlement,
                kelly_only_exit=config.kelly_only_exit,
            )
            if candidate:
                candidates.append(candidate)

        # Generate BUY NO candidate (only if we don't have YES position)
        # If we have YES, we should SELL_YES first rather than buying NO
        # EXCEPTION: If YES position is stranded (below minimum size OR value < $1), allow buying NO
        yes_is_stranded = False
        if has_yes:
            yes_shares = position.yes_shares
            # YES sell price = best bid
            yes_sell_price = orderbook.yes_bids[0].price if orderbook.yes_bids else 0.0
            yes_value = yes_shares * yes_sell_price
            yes_is_stranded = yes_shares < MIN_ORDER_SIZE or yes_value < MIN_ORDER_VALUE_USD
        if not has_yes or yes_is_stranded:
            if no_liquidity_ok:
                candidate = _generate_buy_no_candidate(
                    bin_index=bin_index,
                    orderbook=orderbook,
                    portfolio=portfolio,
                    reservation_price=reservation_no,
                    config=config,
                    hours_to_settlement=hours_to_settlement,
                    verbose=verbose,
                    rejection_reasons=rejection_reasons,
                )
                if candidate:
                    candidates.append(candidate)
            elif verbose:
                rejection_reasons.setdefault(bin_index, []).append(f"BUY_NO: {no_reason}")
        elif verbose and has_yes and not yes_is_stranded:
            # Log why BUY_NO is blocked due to existing YES position
            rejection_reasons.setdefault(bin_index, []).append(
                f"BUY_NO: have {position.yes_shares:.0f} YES shares (sell YES first)"
            )

        # Generate SELL NO candidate (if we have position)
        # Use min(model probability, Kelly reservation) for exit threshold
        # NOTE: Always allow selling even if liquidity is poor (need to exit positions)
        if has_no:
            model_prob_no = 1.0 - portfolio.probabilities[bin_index]
            candidate = _generate_sell_no_candidate(
                bin_index=bin_index,
                orderbook=orderbook,
                portfolio=portfolio,
                model_probability=model_prob_no,
                reservation_price=reservation_no,
                config=config,
                hours_to_settlement=hours_to_settlement,
                kelly_only_exit=config.kelly_only_exit,
            )
            if candidate:
                candidates.append(candidate)

    # Log rejection reasons if verbose
    if verbose and rejection_reasons:
        logger.info("Candidate rejection reasons:")
        for bin_idx in sorted(rejection_reasons.keys()):
            reasons = rejection_reasons[bin_idx]
            logger.info(f"  Bin {bin_idx}: {'; '.join(reasons)}")

    # Log ALL buy candidates with utility gains (for debugging multi-bin Kelly)
    if verbose:
        buy_candidates = [c for c in candidates if c.action in (TradeAction.BUY_YES, TradeAction.BUY_NO)]
        if buy_candidates:
            logger.info("All BUY candidates (sorted by utility gain):")
            # Sort by utility for display
            buy_sorted = sorted(buy_candidates, key=lambda c: c.utility_gain, reverse=True)
            for c in buy_sorted[:10]:  # Top 10
                logger.info(
                    f"  Bin {c.bin_index} {c.action.value}: "
                    f"util={c.utility_gain:.6f} edge={c.edge:+.2%} "
                    f"fair={c.reservation_price:.3f} vwap={c.price:.3f} "
                    f"size={c.size:.1f}"
                )

    # Separate sells from buys
    # Sells are processed first (exit at fair value or better, not utility-based)
    # Then buys are sorted by utility gain
    sells = [c for c in candidates if c.action in (TradeAction.SELL_YES, TradeAction.SELL_NO)]
    buys = [c for c in candidates if c.action in (TradeAction.BUY_YES, TradeAction.BUY_NO)]

    # Sort buys by utility gain per dollar spent (descending).
    # With a small screening chunk ($2), this approximates marginal utility per dollar.
    buys.sort(key=lambda c: c.utility_gain / c.cost if c.cost > 0 else 0, reverse=True)

    # Sells come first, then buys
    return sells + buys


def _generate_buy_yes_candidate(
    bin_index: int,
    orderbook: UnifiedOrderbook,
    portfolio: Portfolio,
    reservation_price: float,
    config: KellyConfig,
    hours_to_settlement: float,
    verbose: bool = False,
    rejection_reasons: dict = None,
) -> Optional[TradeCandidate]:
    """Generate a BUY YES candidate if profitable."""
    def reject(reason: str) -> None:
        logger.debug(f"BUY_YES bin {bin_index}: {reason}")
        if verbose and rejection_reasons is not None:
            rejection_reasons.setdefault(bin_index, []).append(f"BUY_YES: {reason}")

    # Get available depth in shares
    depth_shares = get_available_depth(orderbook, "BUY_YES")
    if depth_shares <= 0:
        reject("no depth")
        return None

    # Get best ask price to convert depth to USD
    best_ask = orderbook.yes_asks[0].price if orderbook.yes_asks else None
    if not best_ask or best_ask <= 0:
        reject("no ask price")
        return None

    # Small screening chunk for marginal utility estimation.
    # Actual sizing is done by the binary search in _find_optimal_size_on.
    delta_usd = SCREENING_CHUNK_USD

    # Convert USD to shares for VWAP calculation
    delta = delta_usd / best_ask

    # Check we have at least some capital
    if portfolio.available_capital < MIN_ORDER_VALUE_USD:
        reject(f"insufficient capital (${portfolio.available_capital:.2f})")
        return None

    # Check collateral limits (rough check — exact sizing done later)
    position = portfolio.get_position(bin_index)
    current_bin_collateral = position.collateral_used if position else 0.0
    total_collateral = portfolio.total_collateral_used

    if config.collateral.c_bin_max > 0:
        if current_bin_collateral >= config.collateral.c_bin_max:
            reject(f"bin collateral limit (${current_bin_collateral:.0f} >= ${config.collateral.c_bin_max:.0f})")
            return None

    if config.collateral.virtual_c_event_max > 0:
        if total_collateral >= config.collateral.virtual_c_event_max:
            reject(f"event collateral limit (${total_collateral:.0f} >= ${config.collateral.virtual_c_event_max:.0f})")
            return None

    # Get VWAP for this chunk
    vwap, filled, worst_price = compute_vwap_buy_yes(orderbook, delta)
    if filled <= 0:
        reject(f"no fill at delta={delta:.2f}")
        return None

    # Check minimum perceived probability (from our model)
    # Use actual model probability, not Kelly reservation price
    model_prob_yes = portfolio.probabilities[bin_index]
    if config.edge_buffer.min_perceived_prob > 0 and model_prob_yes < config.edge_buffer.min_perceived_prob:
        reject(f"model prob too low ({model_prob_yes:.1%} < {config.edge_buffer.min_perceived_prob:.1%})")
        return None

    # Check minimum market price
    if config.edge_buffer.min_market_price > 0 and vwap < config.edge_buffer.min_market_price:
        reject(f"market price too low ({vwap:.1%} < {config.edge_buffer.min_market_price:.1%})")
        return None

    # Check edge requirement
    trade_ok, actual_edge = should_trade(
        vwap, reservation_price, TradeAction.BUY_YES, config.edge_buffer
    )
    if not trade_ok:
        threshold = compute_buy_threshold(reservation_price, config.edge_buffer)
        reject(f"edge failed (ask={vwap:.1%} > thresh={threshold:.1%}, fair={reservation_price:.1%})")
        return None

    # Simulate trade and compute utility gain (screening only — require positive)
    new_portfolio = portfolio.simulate_buy_yes(bin_index, filled, vwap)
    utility_gain = _compute_portfolio_utility_gain(portfolio, new_portfolio, config)

    if utility_gain <= 0:
        reject(f"non-positive utility ({utility_gain:.6f})")
        return None

    return TradeCandidate(
        bin_index=bin_index,
        action=TradeAction.BUY_YES,
        size=filled,
        price=vwap,
        utility_gain=utility_gain,
        reservation_price=reservation_price,
        edge=actual_edge,
        limit_price=worst_price,
    )


def _generate_sell_yes_candidate(
    bin_index: int,
    orderbook: UnifiedOrderbook,
    portfolio: Portfolio,
    model_probability: float,
    reservation_price: float,
    config: KellyConfig,
    hours_to_settlement: float,
    kelly_only_exit: bool = False,
) -> Optional[TradeCandidate]:
    """
    Generate a SELL YES candidate if profitable.

    Since we only generate SELL_YES when we have a YES position, this is
    always an EXIT (closing existing long), not a new short.

    For exits, we use the LOWER of model probability and Kelly reservation
    price as the threshold. This allows:
    1. Exit when market >= model fair value (edge disappeared)
    2. Exit when market >= Kelly reservation price (utility-based exit,
       e.g., when concentrated position makes hedging valuable)
    """
    position = portfolio.get_position(bin_index)
    if not position or not position.has_yes_position:
        return None

    # Get available depth in shares
    depth_shares = get_available_depth(orderbook, "SELL_YES")
    if depth_shares <= 0:
        return None

    # Get best bid price (selling YES at bid)
    best_bid = orderbook.yes_bids[0].price if orderbook.yes_bids else None
    if not best_bid or best_bid <= 0:
        return None

    # Small screening chunk for marginal utility estimation
    delta_usd = SCREENING_CHUNK_USD

    # Convert USD to shares
    delta = delta_usd / best_bid

    # Don't sell more than we have
    delta = min(delta, position.yes_shares)

    # Floor to 2 decimal places to avoid "not enough balance" errors
    # due to floating point precision (e.g., trying to sell 30.9428 when we have 30.9427)
    delta = math.floor(delta * 100) / 100

    # PREVENTION: Don't leave stranded positions
    # If partial sell would leave < 1 share (unsellable), sell all instead
    remaining_shares = position.yes_shares - delta
    if 0 < remaining_shares < 1:
        delta = position.yes_shares
        delta = math.floor(delta * 100) / 100

    # For exits, only enforce 1 share minimum (no USD value requirement)
    if delta < 1:
        return None

    vwap, filled, worst_price = compute_vwap_sell_yes(orderbook, delta)
    if filled <= 0:
        return None

    # Sell friction: require market price meaningfully above Kelly fair value
    # This applies even in kelly_only_exit mode to prevent cycling
    sell_friction = config.edge_buffer.sell_friction
    if sell_friction > 0 and vwap < reservation_price + sell_friction:
        logger.debug(
            f"SELL_YES bin {bin_index}: sell friction "
            f"(vwap={vwap:.1%} < fair+friction={reservation_price + sell_friction:.1%})"
        )
        return None

    # For EXITS: use the LOWER of model probability and Kelly reservation price
    # This allows exit when EITHER condition is met:
    # 1. Market >= model fair value (edge disappeared)
    # 2. Market >= Kelly reservation (utility-based exit for concentrated positions)
    exit_threshold = min(model_probability, reservation_price)

    # Only exit if we can get threshold or better (skip in kelly_only_exit mode)
    if not kelly_only_exit:
        if vwap < exit_threshold:
            logger.debug(
                f"SELL_YES bin {bin_index}: below exit threshold "
                f"(vwap={vwap:.4f}, model={model_probability:.4f}, kelly={reservation_price:.4f}, threshold={exit_threshold:.4f})"
            )
            return None

    # Calculate edge relative to the threshold used
    actual_edge = (vwap - exit_threshold) / exit_threshold if exit_threshold > 0 else 0.0

    new_portfolio = portfolio.simulate_sell_yes(bin_index, filled, vwap)
    utility_gain = _compute_portfolio_utility_gain(portfolio, new_portfolio, config)

    # Screening: only require positive utility (min_sell_utility applied in optimizer)
    if utility_gain <= 0:
        logger.debug(
            f"SELL_YES bin {bin_index}: non-positive utility ({utility_gain:.6f}), skipping"
        )
        return None

    return TradeCandidate(
        bin_index=bin_index,
        action=TradeAction.SELL_YES,
        size=filled,
        price=vwap,
        utility_gain=utility_gain,
        reservation_price=exit_threshold,  # Threshold used (min of model prob and Kelly)
        edge=actual_edge,
        limit_price=worst_price,
    )


def _generate_buy_no_candidate(
    bin_index: int,
    orderbook: UnifiedOrderbook,
    portfolio: Portfolio,
    reservation_price: float,
    config: KellyConfig,
    hours_to_settlement: float,
    verbose: bool = False,
    rejection_reasons: dict = None,
) -> Optional[TradeCandidate]:
    """Generate a BUY NO candidate if profitable."""
    def reject(reason: str) -> None:
        logger.debug(f"BUY_NO bin {bin_index}: {reason}")
        if verbose and rejection_reasons is not None:
            rejection_reasons.setdefault(bin_index, []).append(f"BUY_NO: {reason}")

    # Get available depth in shares
    depth_shares = get_available_depth(orderbook, "BUY_NO")
    if depth_shares <= 0:
        reject("no depth")
        return None

    # Get best bid price (buying NO = selling YES at bid)
    # NO price = 1 - YES price, so NO cost = 1 - yes_bid
    best_yes_bid = orderbook.yes_bids[0].price if orderbook.yes_bids else None
    if not best_yes_bid or best_yes_bid <= 0:
        reject("no bid price")
        return None
    best_no_price = 1.0 - best_yes_bid

    # Small screening chunk for marginal utility estimation.
    # Actual sizing is done by the binary search in _find_optimal_size_on.
    delta_usd = SCREENING_CHUNK_USD

    # Convert USD to shares for VWAP calculation
    delta = delta_usd / best_no_price

    # Check we have at least some capital
    if portfolio.available_capital < MIN_ORDER_VALUE_USD:
        reject(f"insufficient capital (${portfolio.available_capital:.2f})")
        return None

    # Check collateral limits (rough check — exact sizing done later)
    position = portfolio.get_position(bin_index)
    current_bin_collateral = position.collateral_used if position else 0.0
    total_collateral = portfolio.total_collateral_used

    if config.collateral.c_bin_max > 0:
        if current_bin_collateral >= config.collateral.c_bin_max:
            reject(f"bin collateral limit (${current_bin_collateral:.0f} >= ${config.collateral.c_bin_max:.0f})")
            return None

    if config.collateral.virtual_c_event_max > 0:
        if total_collateral >= config.collateral.virtual_c_event_max:
            reject(f"event collateral limit (${total_collateral:.0f} >= ${config.collateral.virtual_c_event_max:.0f})")
            return None

    vwap, filled, worst_price = compute_vwap_buy_no(orderbook, delta)
    if filled <= 0:
        reject(f"no fill at delta={delta:.2f}")
        return None

    # Check minimum perceived probability (from our model)
    # Use actual model probability for NO, not Kelly reservation price
    model_prob_no = 1.0 - portfolio.probabilities[bin_index]
    if config.edge_buffer.min_perceived_prob > 0 and model_prob_no < config.edge_buffer.min_perceived_prob:
        reject(f"model prob too low ({model_prob_no:.1%} < {config.edge_buffer.min_perceived_prob:.1%})")
        return None

    # Check minimum market price
    if config.edge_buffer.min_market_price > 0 and vwap < config.edge_buffer.min_market_price:
        reject(f"market price too low ({vwap:.1%} < {config.edge_buffer.min_market_price:.1%})")
        return None

    trade_ok, actual_edge = should_trade(
        vwap, reservation_price, TradeAction.BUY_NO, config.edge_buffer
    )
    if not trade_ok:
        req_edge = compute_required_edge(reservation_price, config.edge_buffer)
        threshold = reservation_price * (1 - req_edge)
        reject(f"edge failed (ask={vwap:.1%} > thresh={threshold:.1%}, fair={reservation_price:.1%})")
        return None

    # Simulate trade and compute utility gain (screening only — require positive)
    new_portfolio = portfolio.simulate_buy_no(bin_index, filled, vwap)
    utility_gain = _compute_portfolio_utility_gain(portfolio, new_portfolio, config)

    if utility_gain <= 0:
        reject(f"non-positive utility ({utility_gain:.6f})")
        return None

    return TradeCandidate(
        bin_index=bin_index,
        action=TradeAction.BUY_NO,
        size=filled,
        price=vwap,
        utility_gain=utility_gain,
        reservation_price=reservation_price,
        edge=actual_edge,
        limit_price=worst_price,
    )


def _generate_sell_no_candidate(
    bin_index: int,
    orderbook: UnifiedOrderbook,
    portfolio: Portfolio,
    model_probability: float,
    reservation_price: float,
    config: KellyConfig,
    hours_to_settlement: float,
    kelly_only_exit: bool = False,
) -> Optional[TradeCandidate]:
    """
    Generate a SELL NO candidate if profitable.

    Since we only generate SELL_NO when we have a NO position, this is
    always an EXIT (closing existing long), not a new short.

    For exits, we use the LOWER of model probability and Kelly reservation
    price as the threshold. This allows:
    1. Exit when market >= model fair value (edge disappeared)
    2. Exit when market >= Kelly reservation price (utility-based exit,
       e.g., when concentrated position makes hedging valuable)
    """
    position = portfolio.get_position(bin_index)
    if not position or not position.has_no_position:
        return None

    # Get available depth in shares
    depth_shares = get_available_depth(orderbook, "SELL_NO")
    if depth_shares <= 0:
        return None

    # Get best ask price (selling NO = buying YES at ask)
    # NO sell price = 1 - YES ask price
    best_yes_ask = orderbook.yes_asks[0].price if orderbook.yes_asks else None
    if not best_yes_ask or best_yes_ask <= 0:
        return None
    best_no_price = 1.0 - best_yes_ask

    # Small screening chunk for marginal utility estimation
    delta_usd = SCREENING_CHUNK_USD

    # Convert USD to shares
    delta = delta_usd / best_no_price

    # Don't sell more than we have
    delta = min(delta, position.no_shares)

    # Floor to 2 decimal places to avoid "not enough balance" errors
    # due to floating point precision (e.g., trying to sell 30.9428 when we have 30.9427)
    delta = math.floor(delta * 100) / 100

    # PREVENTION: Don't leave stranded positions
    # If partial sell would leave < 1 share (unsellable), sell all instead
    remaining_shares = position.no_shares - delta
    if 0 < remaining_shares < 1:
        delta = position.no_shares
        delta = math.floor(delta * 100) / 100

    # For exits, only enforce 1 share minimum (no USD value requirement)
    if delta < 1:
        return None

    vwap, filled, worst_price = compute_vwap_sell_no(orderbook, delta)
    if filled <= 0:
        return None

    # Sell friction: require market price meaningfully above Kelly fair value
    # This applies even in kelly_only_exit mode to prevent cycling
    sell_friction = config.edge_buffer.sell_friction
    if sell_friction > 0 and vwap < reservation_price + sell_friction:
        logger.debug(
            f"SELL_NO bin {bin_index}: sell friction "
            f"(vwap={vwap:.1%} < fair+friction={reservation_price + sell_friction:.1%})"
        )
        return None

    # For EXITS: use the LOWER of model probability and Kelly reservation price
    # This allows exit when EITHER condition is met:
    # 1. Market >= model fair value (edge disappeared)
    # 2. Market >= Kelly reservation (utility-based exit for concentrated positions)
    exit_threshold = min(model_probability, reservation_price)

    # Only exit if we can get threshold or better (skip in kelly_only_exit mode)
    if not kelly_only_exit:
        if vwap < exit_threshold:
            logger.debug(
                f"SELL_NO bin {bin_index}: below exit threshold "
                f"(vwap={vwap:.4f}, model={model_probability:.4f}, kelly={reservation_price:.4f}, threshold={exit_threshold:.4f})"
            )
            return None

    # Calculate edge relative to the threshold used
    actual_edge = (vwap - exit_threshold) / exit_threshold if exit_threshold > 0 else 0.0

    new_portfolio = portfolio.simulate_sell_no(bin_index, filled, vwap)
    utility_gain = _compute_portfolio_utility_gain(portfolio, new_portfolio, config)

    # Screening: only require positive utility (min_sell_utility applied in optimizer)
    if utility_gain <= 0:
        logger.debug(
            f"SELL_NO bin {bin_index}: non-positive utility ({utility_gain:.6f}), skipping"
        )
        return None

    return TradeCandidate(
        bin_index=bin_index,
        action=TradeAction.SELL_NO,
        size=filled,
        price=vwap,
        utility_gain=utility_gain,
        reservation_price=exit_threshold,  # Threshold used (min of model prob and Kelly)
        edge=actual_edge,
        limit_price=worst_price,
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
        config.kelly_fraction,
    )
