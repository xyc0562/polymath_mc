"""
Core Kelly criterion mathematics for multi-bin trading.

This module implements the Kelly criterion optimization for mutually exclusive
bin outcomes, supporting both full and fractional Kelly via CRRA power utility.

Terminal Wealth:
    W_j = C + q_j^Y + Σ_{i≠j} q_i^N

where:
    - C = initial capital
    - q_j^Y = YES position in winning bin j
    - q_i^N = NO positions in losing bins (each pays $1)

Utility function (parameterized by kelly_fraction α):
    U(W) = log(W)                       when α = 1  (full Kelly)
    U(W) = W^(1-γ) / (1-γ)             when α < 1  (fractional Kelly, γ = 1/α)

Normalizer:
    S = Σ_j (p_j · W_j^(-γ))

Reservation Prices:
    c*_YES[i] = (p_i · W_i^(-γ)) / S
    c*_NO[i] = 1 - c*_YES[i]

When α = 1 (γ = 1), this reduces to the standard log-utility Kelly formulas.
When α < 1 (γ > 1), the investor is more risk-averse → smaller positions.
"""

import math
from typing import Dict, List, Optional, Tuple


def compute_terminal_wealth(
    bin_index: int,
    capital: float,
    yes_positions: Dict[int, float],
    no_positions: Dict[int, float],
    num_bins: int,
) -> float:
    """
    Compute terminal wealth if bin j wins.

    W_j = C + q_j^Y + Σ_{i≠j} q_i^N

    Args:
        bin_index: The winning bin index j
        capital: Current capital C (USDC)
        yes_positions: Dict mapping bin index -> YES shares held
        no_positions: Dict mapping bin index -> NO shares held
        num_bins: Total number of bins

    Returns:
        Terminal wealth W_j if bin j wins
    """
    # Start with capital
    w_j = capital

    # Add YES position in winning bin (pays $1 per share)
    w_j += yes_positions.get(bin_index, 0.0)

    # Add NO positions in ALL other bins (each pays $1 per share)
    for i in range(num_bins):
        if i != bin_index:
            w_j += no_positions.get(i, 0.0)

    return w_j


def compute_all_terminal_wealths(
    capital: float,
    yes_positions: Dict[int, float],
    no_positions: Dict[int, float],
    num_bins: int,
) -> List[float]:
    """
    Compute terminal wealth for all possible winning bins.

    Args:
        capital: Current capital C (USDC)
        yes_positions: Dict mapping bin index -> YES shares held
        no_positions: Dict mapping bin index -> NO shares held
        num_bins: Total number of bins

    Returns:
        List of terminal wealths [W_0, W_1, ..., W_{n-1}]
    """
    return [
        compute_terminal_wealth(j, capital, yes_positions, no_positions, num_bins)
        for j in range(num_bins)
    ]


def compute_normalizer_S(
    probabilities: List[float],
    terminal_wealths: List[float],
    w_floor: float = 1.0,
    kelly_fraction: float = 1.0,
) -> float:
    """
    Compute the normalizer S = Σ_j (p_j · W_j^(-γ)) where γ = 1/kelly_fraction.

    This normalizer ensures reservation prices sum to 1.

    Args:
        probabilities: Probability distribution [p_0, p_1, ..., p_{n-1}]
        terminal_wealths: Terminal wealths [W_0, W_1, ..., W_{n-1}]
        w_floor: Minimum wealth floor to prevent division issues
        kelly_fraction: Fractional Kelly parameter α ∈ (0, 1]. γ = 1/α.

    Returns:
        Normalizer S
    """
    gamma = 1.0 / kelly_fraction
    s = 0.0
    for p_j, w_j in zip(probabilities, terminal_wealths):
        if p_j > 0:
            w_j_floored = max(w_j, w_floor)
            s += p_j * w_j_floored ** (-gamma)
    return s


def compute_reservation_price_yes(
    bin_index: int,
    probabilities: List[float],
    terminal_wealths: List[float],
    normalizer_S: float,
    w_floor: float = 1.0,
    kelly_fraction: float = 1.0,
) -> float:
    """
    Compute Kelly reservation price for YES on bin i.

    c*_YES[i] = (p_i · W_i^(-γ)) / S  where γ = 1/kelly_fraction.

    This is the fair price at which the Kelly investor is indifferent
    to buying or not buying YES on bin i.

    Args:
        bin_index: Bin index i
        probabilities: Probability distribution
        terminal_wealths: Terminal wealths
        normalizer_S: Pre-computed normalizer S
        w_floor: Minimum wealth floor
        kelly_fraction: Fractional Kelly parameter α ∈ (0, 1]. γ = 1/α.

    Returns:
        Reservation price for YES on bin i (0 to 1)
    """
    gamma = 1.0 / kelly_fraction
    p_i = probabilities[bin_index]
    w_i = max(terminal_wealths[bin_index], w_floor)

    if normalizer_S <= 0 or p_i <= 0:
        return 0.0

    return (p_i * w_i ** (-gamma)) / normalizer_S


def compute_reservation_price_no(
    bin_index: int,
    probabilities: List[float],
    terminal_wealths: List[float],
    normalizer_S: float,
    w_floor: float = 1.0,
    kelly_fraction: float = 1.0,
) -> float:
    """
    Compute Kelly reservation price for NO on bin i.

    c*_NO[i] = 1 - c*_YES[i]

    Args:
        bin_index: Bin index i
        probabilities: Probability distribution
        terminal_wealths: Terminal wealths
        normalizer_S: Pre-computed normalizer S
        w_floor: Minimum wealth floor
        kelly_fraction: Fractional Kelly parameter α ∈ (0, 1]. γ = 1/α.

    Returns:
        Reservation price for NO on bin i (0 to 1)
    """
    c_yes = compute_reservation_price_yes(
        bin_index, probabilities, terminal_wealths, normalizer_S, w_floor, kelly_fraction
    )
    return 1.0 - c_yes


def compute_all_reservation_prices(
    probabilities: List[float],
    terminal_wealths: List[float],
    w_floor: float = 1.0,
    kelly_fraction: float = 1.0,
) -> Tuple[List[float], List[float]]:
    """
    Compute all Kelly reservation prices for YES and NO.

    Args:
        probabilities: Probability distribution
        terminal_wealths: Terminal wealths
        w_floor: Minimum wealth floor
        kelly_fraction: Fractional Kelly parameter α ∈ (0, 1]. γ = 1/α.

    Returns:
        Tuple of (yes_prices, no_prices) lists
    """
    normalizer_S = compute_normalizer_S(probabilities, terminal_wealths, w_floor, kelly_fraction)

    yes_prices = []
    no_prices = []

    for i in range(len(probabilities)):
        c_yes = compute_reservation_price_yes(
            i, probabilities, terminal_wealths, normalizer_S, w_floor, kelly_fraction
        )
        yes_prices.append(c_yes)
        no_prices.append(1.0 - c_yes)

    return yes_prices, no_prices


def compute_expected_log_utility(
    probabilities: List[float],
    terminal_wealths: List[float],
    w_floor: float = 1.0,
    kelly_fraction: float = 1.0,
) -> float:
    """
    Compute expected utility E[U(W)] using CRRA power utility.

    When kelly_fraction = 1 (γ = 1): U(W) = log(W)  (full Kelly)
    When kelly_fraction < 1 (γ > 1): U(W) = W^(1-γ) / (1-γ)  (more risk-averse)

    Args:
        probabilities: Probability distribution
        terminal_wealths: Terminal wealths
        w_floor: Minimum wealth floor
        kelly_fraction: Fractional Kelly parameter α ∈ (0, 1]. γ = 1/α.

    Returns:
        Expected utility
    """
    gamma = 1.0 / kelly_fraction
    utility = 0.0
    for p_j, w_j in zip(probabilities, terminal_wealths):
        if p_j > 0:
            w_j_floored = max(w_j, w_floor)
            if abs(gamma - 1.0) < 1e-9:
                utility += p_j * math.log(w_j_floored)
            else:
                utility += p_j * w_j_floored ** (1.0 - gamma) / (1.0 - gamma)
    return utility


def compute_utility_gain(
    probabilities: List[float],
    terminal_wealths_before: List[float],
    terminal_wealths_after: List[float],
    w_floor: float = 1.0,
    kelly_fraction: float = 1.0,
) -> float:
    """
    Compute utility gain from a trade.

    ΔU = E[U(W_after)] - E[U(W_before)]

    Args:
        probabilities: Probability distribution
        terminal_wealths_before: Terminal wealths before trade
        terminal_wealths_after: Terminal wealths after trade
        w_floor: Minimum wealth floor
        kelly_fraction: Fractional Kelly parameter α ∈ (0, 1]. γ = 1/α.

    Returns:
        Utility gain (positive = good trade)
    """
    u_before = compute_expected_log_utility(probabilities, terminal_wealths_before, w_floor, kelly_fraction)
    u_after = compute_expected_log_utility(probabilities, terminal_wealths_after, w_floor, kelly_fraction)
    return u_after - u_before


def renormalize_probabilities(
    probabilities: List[float],
    dead_bins: List[int],
) -> List[float]:
    """
    Renormalize probabilities after removing dead bins.

    Dead bins (where upper bound < current count) have p=0.
    Remaining probabilities are scaled to sum to 1.

    Args:
        probabilities: Original probability distribution
        dead_bins: List of dead bin indices

    Returns:
        Renormalized probability distribution
    """
    result = probabilities.copy()

    # Zero out dead bins
    for i in dead_bins:
        result[i] = 0.0

    # Compute sum of remaining probabilities
    total = sum(result)

    if total <= 0:
        # All bins dead - shouldn't happen in practice
        return result

    # Renormalize
    return [p / total for p in result]


def identify_dead_bins(
    bin_upper_bounds: List[int],
    current_count: int,
) -> List[int]:
    """
    Identify dead bins where upper bound < current count.

    Args:
        bin_upper_bounds: Upper bound for each bin
        current_count: Current tweet count

    Returns:
        List of dead bin indices
    """
    return [i for i, upper in enumerate(bin_upper_bounds) if upper < current_count]
