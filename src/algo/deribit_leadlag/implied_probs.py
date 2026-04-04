"""
Implied probability extraction from Deribit option data.

Uses call-spread digital extraction (model-free) exclusively.
Strikes without liquid adjacent options are skipped rather than
falling back to model-dependent BS N(d2).
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

from .deribit_client import DeribitOption

logger = logging.getLogger(__name__)


@dataclass
class CallPrice:
    """USD-denominated call price at a single strike."""

    strike: float
    bid_usd: Optional[float]
    ask_usd: Optional[float]
    mid_usd: Optional[float]
    mark_usd: float
    volume: float
    open_interest: float

    @property
    def has_two_sided_quotes(self) -> bool:
        return self.bid_usd is not None and self.ask_usd is not None


@dataclass
class ImpliedProb:
    """Implied probability with conservative/mid/aggressive bounds."""

    prob_conservative: float  # Worst-case for our trade
    prob_mid: float
    prob_aggressive: float  # Best-case for our trade
    method: str  # "call_spread"
    forward: float
    T_years: float


def build_call_price_curve(
    options: List[DeribitOption],
    expiry: date,
) -> List[CallPrice]:
    """
    Build a sorted USD-denominated call price curve for a given expiry.

    Filters to calls at the specified expiry, converts BTC prices to USD.
    Returns sorted by strike ascending.
    """
    calls = [
        opt for opt in options
        if opt.option_type == "C" and opt.expiry_date == expiry
    ]

    curve = []
    for c in calls:
        curve.append(
            CallPrice(
                strike=c.strike,
                bid_usd=c.bid_price_usd,
                ask_usd=c.ask_price_usd,
                mid_usd=c.mid_price_usd,
                mark_usd=c.mark_price_usd,
                volume=c.volume_24h,
                open_interest=c.open_interest,
            )
        )

    curve.sort(key=lambda x: x.strike)
    return curve


def _find_bracketing_calls(
    curve: List[CallPrice],
    target_strike: float,
) -> Optional[Tuple[CallPrice, CallPrice]]:
    """
    Find two calls that bracket the target strike: one at or below, one at or above.

    If the target exactly matches a strike, use the strikes immediately
    below and above it for the digital extraction.
    """
    below = None
    above = None

    for cp in curve:
        if cp.strike < target_strike:
            below = cp
        elif cp.strike > target_strike:
            above = cp
            break
        else:
            # Exact match — still need neighbors for the spread
            # Use this strike and the next one above
            pass

    # If exact match, try to find neighbors
    if below is None or above is None:
        # Try exact match + neighbor approach
        for i, cp in enumerate(curve):
            if cp.strike == target_strike:
                if i > 0 and i < len(curve) - 1:
                    below = curve[i - 1]
                    above = curve[i + 1]
                elif i > 0:
                    below = curve[i - 1]
                    above = cp
                elif i < len(curve) - 1:
                    below = cp
                    above = curve[i + 1]
                break

    if below is None or above is None:
        return None
    if below.strike >= above.strike:
        return None

    return below, above


def digital_prob_from_call_spread(
    curve: List[CallPrice],
    target_strike: float,
    min_spread_usd: float = 50.0,
) -> Optional[ImpliedProb]:
    """
    Extract digital probability P(S > K) from adjacent call prices.

    P(S > K) ≈ [C(K_low) - C(K_high)] / (K_high - K_low)

    Uses bid/ask prices for conservative/aggressive bounds.

    Returns None if adjacent strikes are missing, illiquid, or spread
    is too small for reliable extraction.
    """
    bracket = _find_bracketing_calls(curve, target_strike)
    if bracket is None:
        return None

    lower, upper = bracket
    dk = upper.strike - lower.strike

    if dk <= 0:
        return None

    # Mid estimate
    c_low_mid = lower.mid_usd if lower.mid_usd is not None else lower.mark_usd
    c_high_mid = upper.mid_usd if upper.mid_usd is not None else upper.mark_usd

    spread_usd = c_low_mid - c_high_mid
    if spread_usd < min_spread_usd:
        logger.debug(
            f"Call spread too small for K={target_strike}: "
            f"${spread_usd:.1f} < ${min_spread_usd:.1f}"
        )
        return None

    prob_mid = spread_usd / dk
    prob_mid = max(0.0, min(1.0, prob_mid))

    # Conservative: worst-case (lower probability)
    # Use bid for the lower-strike call (less value), ask for the higher-strike (more cost)
    prob_conservative = prob_mid  # Default if quotes missing
    if lower.has_two_sided_quotes and upper.has_two_sided_quotes:
        spread_conservative = lower.bid_usd - upper.ask_usd
        prob_conservative = max(0.0, min(1.0, spread_conservative / dk))

    # Aggressive: best-case (higher probability)
    prob_aggressive = prob_mid
    if lower.has_two_sided_quotes and upper.has_two_sided_quotes:
        spread_aggressive = lower.ask_usd - upper.bid_usd
        prob_aggressive = max(0.0, min(1.0, spread_aggressive / dk))

    # Ensure ordering: conservative <= mid <= aggressive
    prob_conservative = min(prob_conservative, prob_mid)
    prob_aggressive = max(prob_aggressive, prob_mid)

    # If bounds are too wide, the call-spread is unreliable (illiquid options)
    bound_width = prob_aggressive - prob_conservative
    if bound_width > 0.30:
        logger.debug(
            f"Call-spread bounds too wide for K={target_strike}: "
            f"[{prob_conservative:.3f}, {prob_aggressive:.3f}] width={bound_width:.3f}"
        )
        return None

    return ImpliedProb(
        prob_conservative=prob_conservative,
        prob_mid=prob_mid,
        prob_aggressive=prob_aggressive,
        method="call_spread",
        forward=0.0,  # Set by caller
        T_years=0.0,  # Set by caller
    )

def time_to_resolution_years(
    now: datetime,
    poly_resolution_utc: datetime,
) -> float:
    """Compute time from now to Polymarket resolution in fractional years."""
    delta = poly_resolution_utc - now
    seconds = delta.total_seconds()
    if seconds <= 0:
        return 0.0
    return seconds / (365.25 * 24 * 3600)


def build_implied_prob_map(
    options: List[DeribitOption],
    now: datetime,
    target_strikes: Dict[Tuple[date, float], datetime],
    min_T_hours: float = 4.0,
    min_call_spread_usd: float = 50.0,
) -> Dict[Tuple[date, float], ImpliedProb]:
    """
    Build implied probability map for all target (date, strike) pairs.

    Uses call-spread digital extraction only. Strikes without liquid
    adjacent Deribit options are skipped.

    Args:
        options: All Deribit BTC options.
        now: Current time (UTC).
        target_strikes: Dict mapping (expiry_date, strike) -> Polymarket resolution time (UTC).
        min_T_hours: Minimum time to resolution to consider.
        min_call_spread_usd: Minimum USD value between adjacent calls.

    Returns:
        Dict mapping (date, strike) -> ImpliedProb.
    """
    result = {}

    # Group options by expiry for building call curves
    expiry_dates = set(d for d, _ in target_strikes.keys())
    curves = {}
    for exp in expiry_dates:
        curves[exp] = build_call_price_curve(options, exp)

    # Get forward prices per expiry from the options
    forwards = {}
    for opt in options:
        if opt.option_type == "C" and opt.underlying_price > 0:
            forwards[opt.expiry_date] = opt.underlying_price

    skipped_no_expiry = 0
    skipped_too_soon = 0
    skipped_no_spread = 0

    for (exp_date, strike), poly_resolution in target_strikes.items():
        curve = curves.get(exp_date, [])
        if not curve:
            skipped_no_expiry += 1
            continue

        T = time_to_resolution_years(now, poly_resolution)
        T_hours = T * 365.25 * 24

        if T_hours < min_T_hours:
            skipped_too_soon += 1
            continue

        forward = forwards.get(exp_date, 0.0)
        prob = digital_prob_from_call_spread(curve, strike, min_call_spread_usd)

        if prob is None:
            skipped_no_spread += 1
            continue

        prob.forward = forward
        prob.T_years = T
        result[(exp_date, strike)] = prob

    logger.info(
        f"Built implied probs for {len(result)}/{len(target_strikes)} targets "
        f"(skipped: {skipped_no_expiry} no Deribit expiry, "
        f"{skipped_too_soon} too soon, {skipped_no_spread} illiquid call-spread)"
    )
    return result
