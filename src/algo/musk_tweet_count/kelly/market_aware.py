"""
Market-aware probability shrinkage helpers for robust Kelly trading.
"""

from typing import Any, Dict, List, Optional, Tuple

from .config import MarketAwareConfig, RobustKellyConfig, MarketBuyGuardConfig
from .orderbook import UnifiedOrderbook


def _extract_yes_quote(orderbook: Any) -> Tuple[Optional[float], Optional[float]]:
    """Read YES bid/ask from either a live UnifiedOrderbook or a simulated backtest book."""
    if hasattr(orderbook, "best_yes_bid") and hasattr(orderbook, "best_yes_ask"):
        return orderbook.best_yes_bid, orderbook.best_yes_ask
    if hasattr(orderbook, "yes_bid") and hasattr(orderbook, "yes_ask"):
        return orderbook.yes_bid, orderbook.yes_ask
    return None, None


def compute_market_implied_probabilities(
    num_bins: int,
    dead_bins: List[int],
    orderbooks: Dict[int, UnifiedOrderbook],
    market_config: Any,
) -> Optional[Tuple[List[float], float, float]]:
    """
    Build a normalized market-implied bin distribution from two-sided YES mids.

    Returns:
        Tuple of (market_probs, coverage_ratio, avg_spread) if quote quality is good
        enough, otherwise None.
    """
    dead_bin_set = set(dead_bins)
    live_bins = 0
    covered_bins = 0
    spreads: List[float] = []
    market_probabilities = [0.0] * num_bins

    for bin_index in range(num_bins):
        if bin_index in dead_bin_set:
            continue

        live_bins += 1
        orderbook = orderbooks.get(bin_index)
        if orderbook is None:
            continue

        best_bid, best_ask = _extract_yes_quote(orderbook)
        if best_bid is None or best_ask is None or best_ask < best_bid:
            continue

        mid_price = (best_bid + best_ask) / 2.0
        if mid_price <= 0:
            continue

        covered_bins += 1
        spreads.append(best_ask - best_bid)
        market_probabilities[bin_index] = mid_price

    if live_bins == 0 or covered_bins == 0:
        return None

    coverage_ratio = covered_bins / live_bins
    if coverage_ratio < market_config.min_coverage_ratio:
        return None

    avg_spread = sum(spreads) / len(spreads)
    if market_config.max_avg_spread > 0 and avg_spread > market_config.max_avg_spread:
        return None

    total = sum(market_probabilities)
    if total <= 0:
        return None

    market_probabilities = [p / total for p in market_probabilities]
    return market_probabilities, coverage_ratio, avg_spread


def _compute_market_disagreement_context(
    probabilities: List[float],
    dead_bins: List[int],
    orderbooks: Optional[Dict[int, UnifiedOrderbook]],
    market_config: Any,
) -> Optional[Dict[str, float]]:
    """Compute market disagreement diagnostics shared by robust Kelly layers."""
    if not orderbooks:
        return None

    market_summary = compute_market_implied_probabilities(
        num_bins=len(probabilities),
        dead_bins=dead_bins,
        orderbooks=orderbooks,
        market_config=market_config,
    )
    if market_summary is None:
        return None

    market_probabilities, coverage_ratio, avg_spread = market_summary
    disagreement = 0.5 * sum(
        abs(p_model - p_market)
        for p_model, p_market in zip(probabilities, market_probabilities)
    )
    if disagreement <= 0:
        return None

    if market_config.disagreement_scale > 0:
        disagreement_weight = min(1.0, disagreement / market_config.disagreement_scale)
    else:
        disagreement_weight = 1.0

    if market_config.max_avg_spread > 0:
        spread_quality = max(0.0, 1.0 - (avg_spread / market_config.max_avg_spread))
    else:
        spread_quality = 1.0

    return {
        "coverage_ratio": coverage_ratio,
        "avg_spread": avg_spread,
        "disagreement": disagreement,
        "disagreement_weight": disagreement_weight,
        "spread_quality": spread_quality,
        "market_probabilities": market_probabilities,
    }


def compute_market_aware_blend(
    probabilities: List[float],
    dead_bins: List[int],
    orderbooks: Optional[Dict[int, UnifiedOrderbook]],
    market_config: MarketAwareConfig,
) -> Tuple[List[float], Optional[Dict[str, float]]]:
    """
    Shrink model probabilities slightly toward market-implied probabilities.

    Returns:
        Tuple of (possibly blended probabilities, diagnostics context or None).
    """
    if not market_config.enabled or not orderbooks:
        return probabilities, None

    context = _compute_market_disagreement_context(
        probabilities=probabilities,
        dead_bins=dead_bins,
        orderbooks=orderbooks,
        market_config=market_config,
    )
    if context is None:
        return probabilities, None

    market_probabilities = context["market_probabilities"]
    coverage_ratio = context["coverage_ratio"]
    avg_spread = context["avg_spread"]
    disagreement = context["disagreement"]
    disagreement_weight = context["disagreement_weight"]
    spread_quality = context["spread_quality"]

    blend = market_config.max_blend * coverage_ratio * spread_quality * disagreement_weight
    if blend <= 0:
        return probabilities, None

    blended = [
        (1.0 - blend) * p_model + blend * p_market
        for p_model, p_market in zip(probabilities, market_probabilities)
    ]
    total = sum(blended)
    if total <= 0:
        return probabilities, None

    blended = [p / total for p in blended]
    return blended, {
        "blend": blend,
        "coverage_ratio": coverage_ratio,
        "avg_spread": avg_spread,
        "disagreement": disagreement,
    }


def compute_robust_kelly_fraction(
    base_kelly_fraction: float,
    probabilities: List[float],
    dead_bins: List[int],
    orderbooks: Optional[Dict[int, UnifiedOrderbook]],
    robust_config: RobustKellyConfig,
) -> Tuple[float, Optional[Dict[str, float]]]:
    """
    Compute a market-aware effective Kelly fraction.

    The model probabilities remain unchanged. Only the Kelly aggressiveness is
    reduced when the market disagrees and quote quality is good.
    """
    if not robust_config.enabled or not orderbooks or base_kelly_fraction <= 0:
        return base_kelly_fraction, None

    context = _compute_market_disagreement_context(
        probabilities=probabilities,
        dead_bins=dead_bins,
        orderbooks=orderbooks,
        market_config=robust_config,
    )
    if context is None:
        return base_kelly_fraction, None

    min_multiplier = min(1.0, max(0.0, robust_config.min_fraction_multiplier))
    haircut_weight = (
        context["coverage_ratio"]
        * context["spread_quality"]
        * context["disagreement_weight"]
    )
    fraction_multiplier = 1.0 - haircut_weight * (1.0 - min_multiplier)
    effective_fraction = base_kelly_fraction * fraction_multiplier
    if effective_fraction <= 0:
        return base_kelly_fraction, None

    return effective_fraction, {
        "effective_fraction": effective_fraction,
        "fraction_multiplier": fraction_multiplier,
        "coverage_ratio": context["coverage_ratio"],
        "avg_spread": context["avg_spread"],
        "disagreement": context["disagreement"],
    }


def compute_market_buy_guard(
    probabilities: List[float],
    dead_bins: List[int],
    orderbooks: Optional[Dict[int, UnifiedOrderbook]],
    guard_config: MarketBuyGuardConfig,
) -> Optional[Dict[str, float]]:
    """
    Compute a buy-threshold guardrail from market disagreement.

    Returns a context containing an extra probability-point widening to apply
    only to new BUY entries.
    """
    if not guard_config.enabled or not orderbooks:
        return None

    context = _compute_market_disagreement_context(
        probabilities=probabilities,
        dead_bins=dead_bins,
        orderbooks=orderbooks,
        market_config=guard_config,
    )
    if context is None:
        return None

    guard_weight = (
        context["coverage_ratio"]
        * context["spread_quality"]
        * context["disagreement_weight"]
    )
    widening = guard_config.max_threshold_widening * guard_weight
    if widening <= 0:
        return None

    return {
        "threshold_widening": widening,
        "coverage_ratio": context["coverage_ratio"],
        "avg_spread": context["avg_spread"],
        "disagreement": context["disagreement"],
        "guard_weight": guard_weight,
    }
