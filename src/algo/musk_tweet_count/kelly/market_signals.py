"""
Market quote quality and consensus helpers for Kelly trading.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .config import MarketConsensusConfig, RobustKellyConfig, MarketBuyGuardConfig
from .orderbook import UnifiedOrderbook


def _extract_yes_quote(orderbook: Any) -> Tuple[Optional[float], Optional[float]]:
    """Read YES bid/ask from either a live or simulated orderbook."""
    if hasattr(orderbook, "best_yes_bid") and hasattr(orderbook, "best_yes_ask"):
        return orderbook.best_yes_bid, orderbook.best_yes_ask
    if hasattr(orderbook, "yes_bid") and hasattr(orderbook, "yes_ask"):
        return orderbook.yes_bid, orderbook.yes_ask
    return None, None


def compute_market_quote_context(
    probabilities: List[float],
    dead_bins: List[int],
    orderbooks: Optional[Dict[int, UnifiedOrderbook]],
    market_config: Any,
) -> Optional[Dict[str, Any]]:
    """
    Build trusted-quote context from per-bin YES quotes.

    This always returns the trusted-bin subset when orderbooks are available,
    even if overall quote quality is too poor for blending or disagreement
    guards. That lets callers block fresh BUYs in untrusted bins without also
    requiring enough overall coverage to blend probabilities.
    """
    if not orderbooks:
        return None

    dead_bin_set = set(dead_bins)
    live_bins = 0
    spreads: List[float] = []
    trusted_bins: List[int] = []
    trusted_mid_prices: Dict[int, float] = {}
    max_bin_spread = getattr(market_config, "max_bin_spread", 0.0)

    for bin_index in range(len(probabilities)):
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
        spread = best_ask - best_bid
        if mid_price <= 0:
            continue
        if max_bin_spread > 0 and spread > max_bin_spread:
            continue

        trusted_bins.append(bin_index)
        trusted_mid_prices[bin_index] = mid_price
        spreads.append(spread)

    if live_bins == 0:
        return None

    trusted_market_total = sum(trusted_mid_prices.values())
    trusted_market_subdist = {
        bin_index: mid_price / trusted_market_total
        for bin_index, mid_price in trusted_mid_prices.items()
    } if trusted_market_total > 0 else {}

    trusted_model_mass = sum(probabilities[bin_index] for bin_index in trusted_bins)
    trusted_model_subdist = {
        bin_index: probabilities[bin_index] / trusted_model_mass
        for bin_index in trusted_bins
    } if trusted_model_mass > 0 else {}

    disagreement = 0.0
    if trusted_market_subdist and trusted_model_subdist:
        disagreement = 0.5 * sum(
            abs(
                trusted_model_subdist.get(bin_index, 0.0)
                - trusted_market_subdist.get(bin_index, 0.0)
            )
            for bin_index in trusted_bins
        )

    coverage_ratio = len(trusted_bins) / live_bins
    avg_spread = sum(spreads) / len(spreads) if spreads else 0.0
    max_avg_spread = getattr(market_config, "max_avg_spread", 0.0)
    blend_allowed = (
        len(trusted_bins) >= 2
        and coverage_ratio >= getattr(market_config, "min_coverage_ratio", 0.0)
        and (max_avg_spread <= 0 or avg_spread <= max_avg_spread)
        and trusted_market_total > 0
    )

    return {
        "trusted_bins": trusted_bins,
        "trusted_bin_set": set(trusted_bins),
        "coverage_ratio": coverage_ratio,
        "avg_spread": avg_spread,
        "trusted_market_subdist": trusted_market_subdist,
        "trusted_model_mass": trusted_model_mass,
        "disagreement": disagreement,
        "blend_allowed": blend_allowed,
    }


def _compute_market_disagreement_context(
    probabilities: List[float],
    dead_bins: List[int],
    orderbooks: Optional[Dict[int, UnifiedOrderbook]],
    market_config: Any,
    quote_context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Compute market-disagreement diagnostics for Kelly guardrails."""
    context = quote_context or compute_market_quote_context(
        probabilities=probabilities,
        dead_bins=dead_bins,
        orderbooks=orderbooks,
        market_config=market_config,
    )
    if context is None or not context["blend_allowed"]:
        return None

    disagreement = context["disagreement"]
    if disagreement <= 0:
        return None

    if market_config.disagreement_scale > 0:
        disagreement_weight = min(1.0, disagreement / market_config.disagreement_scale)
    else:
        disagreement_weight = 1.0

    if market_config.max_avg_spread > 0:
        spread_quality = max(0.0, 1.0 - (context["avg_spread"] / market_config.max_avg_spread))
    else:
        spread_quality = 1.0

    return {
        **context,
        "disagreement_weight": disagreement_weight,
        "spread_quality": spread_quality,
    }


def compute_market_consensus_blend(
    probabilities: List[float],
    dead_bins: List[int],
    orderbooks: Optional[Dict[int, UnifiedOrderbook]],
    consensus_config: MarketConsensusConfig,
    hours_remaining: float,
) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    """Blend model probabilities toward trusted market consensus."""
    if not consensus_config.enabled or not orderbooks:
        return probabilities, None

    context = compute_market_quote_context(
        probabilities=probabilities,
        dead_bins=dead_bins,
        orderbooks=orderbooks,
        market_config=consensus_config,
    )
    if context is None or not context["blend_allowed"] or context["trusted_model_mass"] <= 0:
        return probabilities, None

    alpha_t = 1.0
    if consensus_config.time_enabled:
        tau = max(consensus_config.time_tau, 1e-6)
        alpha_t = 1.0 - math.exp(-max(hours_remaining, 0.0) / tau)

    alpha_p = 1.0
    gap = context["disagreement"]
    if consensus_config.gap_enabled:
        scale = max(consensus_config.gap_scale, 1e-6)
        alpha_p = max(
            consensus_config.gap_floor,
            1.0 - (gap / scale) ** consensus_config.gap_gamma,
        )

    alpha = max(0.0, min(1.0, max(consensus_config.min_model_weight, alpha_t * alpha_p)))
    if alpha >= 1.0 - 1e-12:
        return probabilities, None

    model_mass = context["trusted_model_mass"]
    trusted_market_subdist = context["trusted_market_subdist"]
    blended = list(probabilities)
    for bin_index in context["trusted_bins"]:
        blended[bin_index] = (
            alpha * probabilities[bin_index]
            + (1.0 - alpha) * model_mass * trusted_market_subdist[bin_index]
        )

    for bin_index in dead_bins:
        blended[bin_index] = 0.0

    total = sum(blended)
    if context["trusted_bins"] and abs(total - 1.0) > 1e-9:
        anchor_bin = context["trusted_bins"][0]
        blended[anchor_bin] += 1.0 - total

    return blended, {
        "alpha": alpha,
        "alpha_t": alpha_t,
        "alpha_p": alpha_p,
        "hours_remaining": hours_remaining,
        "gap": gap,
        "coverage_ratio": context["coverage_ratio"],
        "avg_spread": context["avg_spread"],
        "trusted_bins": float(len(context["trusted_bins"])),
    }


def compute_robust_kelly_fraction(
    base_kelly_fraction: float,
    probabilities: List[float],
    dead_bins: List[int],
    orderbooks: Optional[Dict[int, UnifiedOrderbook]],
    robust_config: RobustKellyConfig,
) -> Tuple[float, Optional[Dict[str, float]]]:
    """Reduce Kelly aggressiveness when trusted market quotes disagree."""
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
    """Compute a buy-threshold widening guardrail from trusted quote disagreement."""
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
