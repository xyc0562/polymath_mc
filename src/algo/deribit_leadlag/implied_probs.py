"""
Implied probability extraction from Deribit option data.

Primary: Call-spread digital extraction (model-free).
Fallback: Black-76 N(d2) from individual strike IV.
"""

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

from scipy.stats import norm

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
    method: str  # "call_spread" or "bs_fallback"
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


# --- Black-76 N(d2) fallback ---


def _compute_d2(
    forward: float,
    strike: float,
    vol: float,
    T: float,
) -> Optional[float]:
    """
    Compute d2 = [ln(F/K) - σ²T/2] / (σ√T) for Black-76.

    Returns None if inputs are invalid (T <= 0, vol <= 0, etc.).
    """
    if T <= 0 or vol <= 0 or forward <= 0 or strike <= 0:
        return None

    sqrt_T = math.sqrt(T)
    d2 = (math.log(forward / strike) - 0.5 * vol * vol * T) / (vol * sqrt_T)
    return d2


def prob_above_strike_bs(
    forward: float,
    strike: float,
    vol: float,
    T_years: float,
) -> Optional[float]:
    """
    Compute P(S > K at T) via Black-76 N(d2).

    Args:
        forward: Forward price (Deribit underlying_price).
        strike: Strike price.
        vol: Implied volatility as decimal (0.65 for 65%).
        T_years: Time to target (Polymarket resolution) in years.

    Returns:
        Risk-neutral probability or None if computation fails.
    """
    d2 = _compute_d2(forward, strike, vol, T_years)
    if d2 is None:
        return None
    return float(norm.cdf(d2))


def bs_implied_prob(
    forward: float,
    strike: float,
    mark_iv: float,
    bid_iv: Optional[float],
    ask_iv: Optional[float],
    T_years: float,
) -> Optional[ImpliedProb]:
    """
    Compute implied probability with bid/ask bounds via Black-76 N(d2).

    Returns None if mark_iv is zero or T is invalid.
    """
    if mark_iv <= 0 or T_years <= 0:
        return None

    # Mid estimate using mark_iv
    prob_mid = prob_above_strike_bs(forward, strike, mark_iv, T_years)
    if prob_mid is None:
        return None

    # Bid/ask bounds
    # Higher IV -> probability moves toward 0.5
    # For P > 0.5 (ITM): higher IV = lower prob -> bid_iv gives aggressive, ask_iv gives conservative
    # For P < 0.5 (OTM): higher IV = higher prob -> ask_iv gives aggressive, bid_iv gives conservative
    # Simplification: compute both and take min/max
    probs = [prob_mid]
    if bid_iv is not None and bid_iv > 0:
        p = prob_above_strike_bs(forward, strike, bid_iv, T_years)
        if p is not None:
            probs.append(p)
    if ask_iv is not None and ask_iv > 0:
        p = prob_above_strike_bs(forward, strike, ask_iv, T_years)
        if p is not None:
            probs.append(p)

    return ImpliedProb(
        prob_conservative=min(probs),
        prob_mid=prob_mid,
        prob_aggressive=max(probs),
        method="bs_fallback",
        forward=forward,
        T_years=T_years,
    )


# --- Orchestrator ---


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

    Tries call-spread digital extraction first, falls back to BS N(d2).

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

    for (exp_date, strike), poly_resolution in target_strikes.items():
        T = time_to_resolution_years(now, poly_resolution)
        T_hours = T * 365.25 * 24

        if T_hours < min_T_hours:
            logger.debug(
                f"Skipping ({exp_date}, {strike}): T={T_hours:.1f}h < {min_T_hours}h"
            )
            continue

        forward = forwards.get(exp_date, 0.0)

        # Try call-spread first
        curve = curves.get(exp_date, [])
        prob = digital_prob_from_call_spread(curve, strike, min_call_spread_usd)

        if prob is not None:
            prob.forward = forward
            prob.T_years = T
            result[(exp_date, strike)] = prob
            continue

        # Fallback: BS N(d2)
        # Find the option at this exact strike for IV data
        matching_opt = None
        for opt in options:
            if (
                opt.option_type == "C"
                and opt.expiry_date == exp_date
                and opt.strike == strike
            ):
                matching_opt = opt
                break

        if matching_opt is not None and matching_opt.mark_iv > 0:
            prob = bs_implied_prob(
                forward=forward,
                strike=strike,
                mark_iv=matching_opt.mark_iv,
                bid_iv=matching_opt.bid_iv,
                ask_iv=matching_opt.ask_iv,
                T_years=T,
            )
            if prob is not None:
                result[(exp_date, strike)] = prob
                continue

        logger.debug(
            f"No probability estimate for ({exp_date}, {strike}): "
            f"no call-spread bracket and no matching option IV"
        )

    logger.info(
        f"Built implied probs for {len(result)}/{len(target_strikes)} targets "
        f"({sum(1 for p in result.values() if p.method == 'call_spread')} call-spread, "
        f"{sum(1 for p in result.values() if p.method == 'bs_fallback')} BS fallback)"
    )
    return result
