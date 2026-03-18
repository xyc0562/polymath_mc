"""Late-boundary majority YES take-profit helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .config import LateBoundaryTakeProfitConfig
from .orderbook import UnifiedOrderbook, compute_vwap_sell_yes
from .portfolio import Portfolio


@dataclass(frozen=True)
class LateBoundaryTakeProfitContext:
    """Per-tick context for the late-boundary majority YES take-profit guard."""

    active: bool
    current_count: int
    current_bin_index: Optional[int] = None
    hours_remaining: float = 0.0
    silence_minutes: float = 0.0
    silence_threshold_minutes: float = 0.0
    distance_to_next_bin: Optional[int] = None
    yes_shares: float = 0.0
    reference_size: float = 0.0
    reference_vwap: float = 0.0
    strength: float = 0.0
    size_floor: float = 0.0
    size_cap: float = 0.0
    skipped_reason: Optional[str] = None


def compute_late_boundary_take_profit_threshold(
    config: LateBoundaryTakeProfitConfig,
    hours_remaining: float,
) -> float:
    """Return the adaptive silence threshold in minutes."""
    decay_hours = max(0.0, config.start_hours - hours_remaining)
    threshold = (
        config.silence_threshold_start_minutes
        - config.silence_threshold_step_per_hour * decay_hours
    )
    return float(max(config.silence_threshold_floor_minutes, threshold))


def build_late_boundary_take_profit_context(
    *,
    config: LateBoundaryTakeProfitConfig,
    current_count: int,
    bin_ranges: Sequence[Tuple[int, float]],
    hours_remaining: float,
    silence_minutes: Optional[float],
    portfolio: Portfolio,
    orderbooks: dict[int, UnifiedOrderbook],
) -> LateBoundaryTakeProfitContext:
    """Build the late-boundary take-profit context from live state."""
    if not config.enabled:
        return LateBoundaryTakeProfitContext(
            active=False,
            current_count=current_count,
            hours_remaining=hours_remaining,
            skipped_reason="disabled",
        )

    if hours_remaining > config.start_hours:
        return LateBoundaryTakeProfitContext(
            active=False,
            current_count=current_count,
            hours_remaining=hours_remaining,
            skipped_reason="too_early",
        )

    current_bin_index = _find_bin_index(current_count, bin_ranges)
    if current_bin_index is None:
        return LateBoundaryTakeProfitContext(
            active=False,
            current_count=current_count,
            hours_remaining=hours_remaining,
            skipped_reason="current_bin_not_found",
        )

    if current_bin_index + 1 >= len(bin_ranges):
        return LateBoundaryTakeProfitContext(
            active=False,
            current_count=current_count,
            current_bin_index=current_bin_index,
            hours_remaining=hours_remaining,
            skipped_reason="no_next_bin",
        )

    current_upper = bin_ranges[current_bin_index][1]
    if not math.isfinite(current_upper):
        return LateBoundaryTakeProfitContext(
            active=False,
            current_count=current_count,
            current_bin_index=current_bin_index,
            hours_remaining=hours_remaining,
            skipped_reason="open_ended_current_bin",
        )

    distance_to_next_bin = int(current_upper - current_count + 1)
    if distance_to_next_bin < 1 or distance_to_next_bin > config.max_distance_to_next_bin:
        return LateBoundaryTakeProfitContext(
            active=False,
            current_count=current_count,
            current_bin_index=current_bin_index,
            hours_remaining=hours_remaining,
            distance_to_next_bin=distance_to_next_bin,
            skipped_reason="distance_out_of_range",
        )

    position = portfolio.get_position(current_bin_index)
    if position is None or position.yes_shares < 1.0:
        return LateBoundaryTakeProfitContext(
            active=False,
            current_count=current_count,
            current_bin_index=current_bin_index,
            hours_remaining=hours_remaining,
            distance_to_next_bin=distance_to_next_bin,
            skipped_reason="no_yes_position",
        )

    silence_value = float(silence_minutes or 0.0)
    silence_threshold = compute_late_boundary_take_profit_threshold(config, hours_remaining)
    if silence_value < silence_threshold:
        return LateBoundaryTakeProfitContext(
            active=False,
            current_count=current_count,
            current_bin_index=current_bin_index,
            hours_remaining=hours_remaining,
            silence_minutes=silence_value,
            silence_threshold_minutes=silence_threshold,
            distance_to_next_bin=distance_to_next_bin,
            yes_shares=position.yes_shares,
            skipped_reason="insufficient_silence",
        )

    orderbook = orderbooks.get(current_bin_index)
    if orderbook is None:
        return LateBoundaryTakeProfitContext(
            active=False,
            current_count=current_count,
            current_bin_index=current_bin_index,
            hours_remaining=hours_remaining,
            silence_minutes=silence_value,
            silence_threshold_minutes=silence_threshold,
            distance_to_next_bin=distance_to_next_bin,
            yes_shares=position.yes_shares,
            skipped_reason="no_orderbook",
        )

    reference_size = float(max(1, math.ceil(config.min_sell_fraction * position.yes_shares)))
    reference_vwap, filled, _worst = compute_vwap_sell_yes(orderbook, reference_size)
    if filled + 1e-9 < reference_size or reference_vwap <= 0.0:
        return LateBoundaryTakeProfitContext(
            active=False,
            current_count=current_count,
            current_bin_index=current_bin_index,
            hours_remaining=hours_remaining,
            silence_minutes=silence_value,
            silence_threshold_minutes=silence_threshold,
            distance_to_next_bin=distance_to_next_bin,
            yes_shares=position.yes_shares,
            reference_size=reference_size,
            skipped_reason="insufficient_majority_depth",
        )

    if reference_vwap + 1e-9 < config.trigger_price:
        return LateBoundaryTakeProfitContext(
            active=False,
            current_count=current_count,
            current_bin_index=current_bin_index,
            hours_remaining=hours_remaining,
            silence_minutes=silence_value,
            silence_threshold_minutes=silence_threshold,
            distance_to_next_bin=distance_to_next_bin,
            yes_shares=position.yes_shares,
            reference_size=reference_size,
            reference_vwap=reference_vwap,
            skipped_reason="trigger_price_not_met",
        )

    a_time = _clip((config.start_hours - hours_remaining) / 2.0)
    a_silence = _clip((silence_value - silence_threshold) / 60.0)
    a_distance = _clip(
        ((config.max_distance_to_next_bin + 1) - distance_to_next_bin)
        / float(config.max_distance_to_next_bin)
    )
    a_price = _clip((reference_vwap - config.trigger_price) / 0.10)
    strength = min(a_time, a_silence, a_price) * a_distance

    size_floor = float(reference_size)
    desired_cap = math.ceil(
        (
            config.min_sell_fraction
            + strength * (config.max_sell_fraction - config.min_sell_fraction)
        )
        * position.yes_shares
    )
    size_cap = float(max(reference_size, desired_cap))
    size_cap = float(min(size_cap, math.floor(position.yes_shares + 1e-9)))

    return LateBoundaryTakeProfitContext(
        active=True,
        current_count=current_count,
        current_bin_index=current_bin_index,
        hours_remaining=hours_remaining,
        silence_minutes=silence_value,
        silence_threshold_minutes=silence_threshold,
        distance_to_next_bin=distance_to_next_bin,
        yes_shares=position.yes_shares,
        reference_size=reference_size,
        reference_vwap=reference_vwap,
        strength=strength,
        size_floor=size_floor,
        size_cap=size_cap,
    )


def _find_bin_index(current_count: int, bin_ranges: Sequence[Tuple[int, float]]) -> Optional[int]:
    for idx, (lower, upper) in enumerate(bin_ranges):
        if lower <= current_count <= upper:
            return idx
    return None


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
