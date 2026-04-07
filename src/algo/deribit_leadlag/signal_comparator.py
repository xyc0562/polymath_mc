"""
Compare Deribit implied probabilities against Polymarket prices.

Implements call-spread price interpolation for time-adjusted markets
(Deribit 08:00 UTC → Polymarket 16:00 UTC). Evaluates BOTH BUY_YES
and BUY_NO for every bin.

Output is named `adjusted_reference_prob` — never `fair_prob` or `truth_prob`.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from .config import OrderConfig, SignalConfig
from .implied_probs import (
    CallPrice,
    ImpliedProb,
    build_call_price_curve,
    _find_bracketing_calls,
)
from .polymarket_discovery import ThresholdMarket
from .settlement import CompatibilityClass

logger = logging.getLogger(__name__)

# Weight for interpolating from same-day (08:00 UTC) to Poly cutoff (16:00 UTC).
# 8 hours into a 24-hour interval between consecutive Deribit expiries.
TIME_INTERP_WEIGHT = 8.0 / 24.0  # 1/3


def compute_polymarket_fee(price: float, fee_rate: float = 0.072) -> float:
    """fee = feeRate * p * (1 - p). Taker-only; makers pay 0."""
    return fee_rate * price * (1.0 - price)


def _get_call_price_at_strike(
    curve: List[CallPrice],
    strike: float,
) -> Optional[CallPrice]:
    """Find the CallPrice at an exact strike, or None."""
    for cp in curve:
        if cp.strike == strike:
            return cp
    return None


def _interpolate_call_price(
    same_day: Optional[float],
    next_day: Optional[float],
    weight: float = TIME_INTERP_WEIGHT,
) -> Optional[float]:
    """Linearly interpolate a single call price between two expiries."""
    if same_day is None:
        return None
    if next_day is None:
        return same_day  # Fallback: use same-day only
    return (1.0 - weight) * same_day + weight * next_day


@dataclass
class AdjustedReferenceProb:
    """Result of the time-adjustment model for a single bin."""

    prob_conservative: float
    prob_mid: float
    prob_aggressive: float
    method: str  # "call_spread_interpolated" or "call_spread_same_day_fallback"
    has_next_day: bool
    dk_same: float  # Strike width used for same-day
    dk_next: float  # Strike width used for next-day (0 if no next-day)

    @property
    def bounds_width(self) -> float:
        return self.prob_aggressive - self.prob_conservative


def interpolate_call_spread_to_poly_time(
    curve_same_day: List[CallPrice],
    curve_next_day: Optional[List[CallPrice]],
    target_strike: float,
    min_spread_usd: float = 15.0,
    max_bracket_dk: float = 3000.0,
) -> Optional[AdjustedReferenceProb]:
    """
    Interpolate call prices across two Deribit expiries to Polymarket
    resolution time (16:00 UTC), then extract digital probability.

    Method:
    1. Find bracketing calls in same-day expiry → C_same(K_low), C_same(K_high)
    2. Find bracketing calls in next-day expiry → C_next(K_low), C_next(K_high)
    3. Interpolate each call price: C_interp = (1-w)*C_same + w*C_next
    4. Digital from interpolated spread: P = (C_interp(K_low) - C_interp(K_high)) / dK

    If next-day has different strikes, interpolates using each expiry's own brackets.
    If dK > max_bracket_dk, rejects the bin.
    """
    # Same-day brackets (required)
    bracket_same = _find_bracketing_calls(curve_same_day, target_strike)
    if bracket_same is None:
        return None

    lower_same, upper_same = bracket_same
    dk_same = upper_same.strike - lower_same.strike
    if dk_same <= 0 or dk_same > max_bracket_dk:
        return None

    # Same-day mid call prices
    c_low_same_mid = lower_same.mid_usd if lower_same.mid_usd is not None else lower_same.mark_usd
    c_high_same_mid = upper_same.mid_usd if upper_same.mid_usd is not None else upper_same.mark_usd

    spread_same = c_low_same_mid - c_high_same_mid
    if spread_same < min_spread_usd:
        return None

    # Try next-day brackets
    has_next_day = False
    dk_next = 0.0

    if curve_next_day:
        bracket_next = _find_bracketing_calls(curve_next_day, target_strike)
        if bracket_next is not None:
            lower_next, upper_next = bracket_next
            dk_next_candidate = upper_next.strike - lower_next.strike
            if dk_next_candidate > 0 and dk_next_candidate <= max_bracket_dk:
                has_next_day = True
                dk_next = dk_next_candidate

    if has_next_day:
        # Interpolate call prices
        c_low_next_mid = lower_next.mid_usd if lower_next.mid_usd is not None else lower_next.mark_usd
        c_high_next_mid = upper_next.mid_usd if upper_next.mid_usd is not None else upper_next.mark_usd

        c_low_interp = _interpolate_call_price(c_low_same_mid, c_low_next_mid)
        c_high_interp = _interpolate_call_price(c_high_same_mid, c_high_next_mid)

        # Use average dK if strikes differ between expiries
        dk_eff = (dk_same + dk_next) / 2.0
        spread_interp = c_low_interp - c_high_interp
        prob_mid = max(0.0, min(1.0, spread_interp / dk_eff))

        # Conservative/aggressive: interpolate bid/ask separately
        prob_conservative = prob_mid
        prob_aggressive = prob_mid

        if (lower_same.has_two_sided_quotes and upper_same.has_two_sided_quotes
                and lower_next.has_two_sided_quotes and upper_next.has_two_sided_quotes):
            # Conservative: lower bid, upper ask (worst-case spread)
            c_low_cons = _interpolate_call_price(lower_same.bid_usd, lower_next.bid_usd)
            c_high_cons = _interpolate_call_price(upper_same.ask_usd, upper_next.ask_usd)
            if c_low_cons is not None and c_high_cons is not None:
                prob_conservative = max(0.0, min(1.0, (c_low_cons - c_high_cons) / dk_eff))

            # Aggressive: lower ask, upper bid (best-case spread)
            c_low_agg = _interpolate_call_price(lower_same.ask_usd, lower_next.ask_usd)
            c_high_agg = _interpolate_call_price(upper_same.bid_usd, upper_next.bid_usd)
            if c_low_agg is not None and c_high_agg is not None:
                prob_aggressive = max(0.0, min(1.0, (c_low_agg - c_high_agg) / dk_eff))
        elif lower_same.has_two_sided_quotes and upper_same.has_two_sided_quotes:
            # Only same-day has two-sided quotes — use same-day bounds, widen slightly
            prob_conservative = max(0.0, min(1.0, (lower_same.bid_usd - upper_same.ask_usd) / dk_same))
            prob_aggressive = max(0.0, min(1.0, (lower_same.ask_usd - upper_same.bid_usd) / dk_same))

        # Ensure ordering
        prob_conservative = min(prob_conservative, prob_mid)
        prob_aggressive = max(prob_aggressive, prob_mid)

        method = "call_spread_interpolated"
    else:
        # Fallback: same-day only
        prob_mid = max(0.0, min(1.0, spread_same / dk_same))

        prob_conservative = prob_mid
        prob_aggressive = prob_mid
        if lower_same.has_two_sided_quotes and upper_same.has_two_sided_quotes:
            spread_cons = lower_same.bid_usd - upper_same.ask_usd
            prob_conservative = max(0.0, min(1.0, spread_cons / dk_same))
            spread_agg = lower_same.ask_usd - upper_same.bid_usd
            prob_aggressive = max(0.0, min(1.0, spread_agg / dk_same))

        prob_conservative = min(prob_conservative, prob_mid)
        prob_aggressive = max(prob_aggressive, prob_mid)

        method = "call_spread_same_day_fallback"

    return AdjustedReferenceProb(
        prob_conservative=prob_conservative,
        prob_mid=prob_mid,
        prob_aggressive=prob_aggressive,
        method=method,
        has_next_day=has_next_day,
        dk_same=dk_same,
        dk_next=dk_next,
    )


@dataclass
class BinSignal:
    """Signal result for a single bin after time adjustment and edge computation."""

    strike: float
    expiry_date: date
    adjusted_ref: AdjustedReferenceProb
    poly_market: ThresholdMarket
    compatibility: CompatibilityClass

    # Best side
    side: str  # "BUY_YES", "BUY_NO", or "NONE"
    entry_price: float
    gross_edge: float
    fee: float
    basis_haircut: float
    effective_edge: float  # gross - fee - haircut

    # For the other side (diagnostic)
    alt_side: str
    alt_effective_edge: float


def build_adjusted_prob_map(
    options: List,  # List[DeribitOption]
    now: datetime,
    markets: List[ThresholdMarket],
    signal_config: SignalConfig,
    compatibility_map: Dict,  # BinKey -> CompatibilityClass
) -> Dict[Tuple[date, float], AdjustedReferenceProb]:
    """
    Build adjusted reference probabilities for all eligible markets.

    For time_adjusted markets:
    - Interpolate call-spread prices between same-day and next-day expiries
    - Recompute digital from interpolated spread

    Returns dict mapping (expiry_date, strike) -> AdjustedReferenceProb.
    """
    from .position_manager import BinKey

    result = {}

    # Build call curves for all relevant expiries
    all_expiry_dates = set()
    for m in markets:
        all_expiry_dates.add(m.expiry_date)
        # Also need next-day expiry for interpolation
        all_expiry_dates.add(m.expiry_date + timedelta(days=1))

    curves = {}
    for exp in all_expiry_dates:
        curve = build_call_price_curve(options, exp)
        if curve:
            curves[exp] = curve

    skipped = {"reject": 0, "no_curve": 0, "no_spread": 0, "too_soon": 0}

    for m in markets:
        key = BinKey(expiry_date=m.expiry_date, strike=m.strike)
        compat = compatibility_map.get(key, CompatibilityClass.REJECT)

        if compat == CompatibilityClass.REJECT:
            skipped["reject"] += 1
            continue

        # Time check
        from .implied_probs import time_to_resolution_years
        T = time_to_resolution_years(now, m.resolution_time_utc)
        T_hours = T * 365.25 * 24
        if T_hours < signal_config.min_time_to_expiry_hours:
            skipped["too_soon"] += 1
            continue

        curve_same = curves.get(m.expiry_date)
        if not curve_same:
            skipped["no_curve"] += 1
            continue

        curve_next = curves.get(m.expiry_date + timedelta(days=1))

        adj = interpolate_call_spread_to_poly_time(
            curve_same_day=curve_same,
            curve_next_day=curve_next,
            target_strike=m.strike,
            min_spread_usd=signal_config.min_call_spread_usd,
            max_bracket_dk=signal_config.max_bracket_dk,
        )

        if adj is None:
            skipped["no_spread"] += 1
            continue

        result[(m.expiry_date, m.strike)] = adj

    logger.info(
        f"Built adjusted probs for {len(result)}/{len(markets)} markets "
        f"(skipped: {skipped})"
    )
    return result


def evaluate_bin_signals(
    markets: List[ThresholdMarket],
    adjusted_probs: Dict[Tuple[date, float], AdjustedReferenceProb],
    compatibility_map: Dict,  # BinKey -> CompatibilityClass
    signal_config: SignalConfig,
    order_config: OrderConfig,
) -> List[BinSignal]:
    """
    Evaluate both BUY_YES and BUY_NO for each bin.

    For time_adjusted markets, applies basis_haircut after edge computation.
    Returns list of BinSignals (only bins with positive effective edge on at least one side).
    """
    from .position_manager import BinKey

    signals = []

    for m in markets:
        key = BinKey(expiry_date=m.expiry_date, strike=m.strike)
        adj = adjusted_probs.get((m.expiry_date, m.strike))
        if adj is None:
            continue

        compat = compatibility_map.get(key, CompatibilityClass.REJECT)
        if compat == CompatibilityClass.REJECT:
            continue

        # Probability range filter
        if adj.prob_mid < signal_config.min_prob or adj.prob_mid > signal_config.max_prob:
            continue

        # Determine haircut
        haircut = 0.0
        if compat == CompatibilityClass.TIME_ADJUSTED:
            haircut = signal_config.time_adjusted_basis_haircut
            if not adj.has_next_day:
                haircut += signal_config.no_next_day_extra_haircut

        fee_rate = order_config.polymarket_crypto_fee_rate

        # Evaluate BUY_YES: edge = prob_conservative - yes_price - fee - haircut
        yes_price = m.yes_price
        yes_fee = compute_polymarket_fee(yes_price, fee_rate)
        yes_gross = adj.prob_conservative - yes_price
        yes_effective = yes_gross - yes_fee - haircut

        # Evaluate BUY_NO: edge = (1-prob_aggressive) - no_price - fee - haircut
        no_price = m.no_price
        no_fee = compute_polymarket_fee(no_price, fee_rate)
        no_gross = (1.0 - adj.prob_aggressive) - no_price
        no_effective = no_gross - no_fee - haircut

        # Pick best side
        if yes_effective >= no_effective and yes_effective > 0:
            signals.append(BinSignal(
                strike=m.strike,
                expiry_date=m.expiry_date,
                adjusted_ref=adj,
                poly_market=m,
                compatibility=compat,
                side="BUY_YES",
                entry_price=yes_price,
                gross_edge=yes_gross,
                fee=yes_fee,
                basis_haircut=haircut,
                effective_edge=yes_effective,
                alt_side="BUY_NO",
                alt_effective_edge=no_effective,
            ))
        elif no_effective > 0:
            signals.append(BinSignal(
                strike=m.strike,
                expiry_date=m.expiry_date,
                adjusted_ref=adj,
                poly_market=m,
                compatibility=compat,
                side="BUY_NO",
                entry_price=no_price,
                gross_edge=no_gross,
                fee=no_fee,
                basis_haircut=haircut,
                effective_edge=no_effective,
                alt_side="BUY_YES",
                alt_effective_edge=yes_effective,
            ))
        # Both negative → no signal for this bin (correct behavior)

    # Sort by effective edge descending
    signals.sort(key=lambda s: s.effective_edge, reverse=True)

    if signals:
        logger.info(
            f"Evaluated {len(signals)} bins with positive edge "
            f"(best: {signals[0].side} {signals[0].expiry_date}/{signals[0].strike:.0f} "
            f"edge={signals[0].effective_edge:.4f})"
        )
    else:
        logger.info("No bins with positive effective edge")

    return signals
