"""
Sampling distributions for tweet count forecasting.

Provides NegBin and COM-Poisson samplers with a unified interface.
"""

import math
from functools import lru_cache
from typing import Tuple

import numpy as np
from scipy.special import gammaln


def sample_negbin(mean: float, k: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample from Negative Binomial distribution.

    Args:
        mean: Expected count
        k: Dispersion parameter (lower = more variance)
        size: Number of samples
        rng: Random generator

    Returns:
        Array of samples
    """
    if mean <= 0:
        return np.zeros(size)

    p = k / (k + mean)
    if p <= 0 or p >= 1 or k <= 0:
        return np.full(size, int(round(mean)))

    return rng.negative_binomial(k, p, size=size).astype(float)


def sample_negbin_scalar(mean: float, k: float, rng: np.random.Generator) -> int:
    """Sample a single value from Negative Binomial."""
    if mean <= 0:
        return 0
    p = k / (k + mean)
    if p <= 0 or p >= 1 or k <= 0:
        return int(round(mean))
    return int(rng.negative_binomial(k, p))


def sample_negbin_reflected(mean: float, k: float, size: int, rng: np.random.Generator,
                            reflect_frac: float = 0.1) -> np.ndarray:
    """
    Sample from NegBin with left-tail reflection.

    Draws from NegBin, then for each sample below mean, with probability
    reflect_frac, reflects it to the other side of the mean:
        x -> 2*mean - x  (clamped to 0)

    This trims the fat left tail and redistributes that mass to the right,
    making the distribution more symmetric around the mean while keeping
    the overall variance similar.

    Args:
        mean: Expected count
        k: Dispersion parameter (lower = more variance)
        size: Number of samples
        rng: Random generator
        reflect_frac: Fraction of below-mean samples to reflect (0=no change, 1=full reflect)

    Returns:
        Array of samples
    """
    if mean <= 0:
        return np.zeros(size)

    p = k / (k + mean)
    if p <= 0 or p >= 1 or k <= 0:
        return np.full(size, int(round(mean)))

    samples = rng.negative_binomial(k, p, size=size).astype(float)

    # Identify samples below the mean
    below_mask = samples < mean
    n_below = below_mask.sum()

    if n_below > 0:
        # Randomly select which below-mean samples to reflect
        reflect_mask = below_mask.copy()
        reflect_mask[below_mask] &= (rng.uniform(size=n_below) < reflect_frac)

        # Reflect: x -> 2*mean - x
        samples[reflect_mask] = 2 * mean - samples[reflect_mask]
        # Clamp to non-negative
        np.maximum(samples, 0, out=samples)

    return samples


def sample_negbin_reflected_scalar(mean: float, k: float, rng: np.random.Generator,
                                   reflect_frac: float = 0.1) -> int:
    """Sample a single value from reflected NegBin."""
    result = sample_negbin_reflected(mean, k, 1, rng, reflect_frac)
    return int(result[0])


# ---------------------------------------------------------------------------
# COM-Poisson: cached CDF builder + fast sampling
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _cmp_cdf(mean_round: float, nu_round: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build and cache CDF for COM-Poisson with given (mean, ν).

    Parameters are rounded to 2 decimal places for cache efficiency.
    Returns (xs, cdf) arrays.
    """
    mean = mean_round
    nu = nu_round

    # Support: 6σ above mean
    target_std = math.sqrt(mean / nu) if nu > 0 else mean
    max_x = int(mean + 6 * target_std) + 1
    max_x = max(max_x, 20)

    xs = np.arange(max_x + 1, dtype=float)
    log_facts = gammaln(xs + 1)  # vectorized, no Python loop

    def _pmf(lam):
        log_lam = math.log(max(lam, 1e-300))
        log_p = xs * log_lam - nu * log_facts
        log_p -= log_p.max()
        p = np.exp(log_p)
        p /= p.sum()
        return p

    def _mean_for_lam(lam):
        return (xs * _pmf(lam)).sum()

    # Closed-form initial estimate (Shmueli et al. 2005):
    #   E[X] ≈ λ^(1/ν) - (ν-1)/(2ν)
    #   => λ ≈ (μ + (ν-1)/(2ν))^ν
    base = mean + (nu - 1) / (2 * nu)
    if base > 0:
        lam0 = base ** nu
    else:
        lam0 = 0.0
    lam0 = max(lam0, 1e-3)

    # Newton-Raphson refinement (typically converges in 3-5 iterations)
    lam = lam0
    for _ in range(15):
        m = _mean_for_lam(lam)
        err = m - mean
        if abs(err) < 1e-6:
            break
        # Numerical derivative
        dlam = max(lam * 1e-5, 1e-8)
        dm = _mean_for_lam(lam + dlam) - m
        deriv = dm / dlam
        if abs(deriv) < 1e-12:
            break
        lam -= err / deriv
        lam = max(lam, 1e-6)

    # Fall back to binary search if Newton didn't converge
    if abs(_mean_for_lam(lam) - mean) > 0.01:
        lo, hi = 1e-3, max(mean, 1.0) ** 3 + 10
        for _ in range(60):
            mid = (lo + hi) / 2
            if _mean_for_lam(mid) < mean:
                lo = mid
            else:
                hi = mid
        lam = (lo + hi) / 2

    pmf = _pmf(lam)
    cdf = np.cumsum(pmf)
    return xs, cdf


def _cmp_nu_from_k(mean: float, k: float, nu_scale: float = 1.0) -> float:
    """Map NegBin-style k to COM-Poisson ν, with optional scaling.

    Base mapping: ν = k/(k+μ)  (matches NegBin variance)
    nu_scale > 1 → higher ν → thinner tails (especially left)
    """
    nu = k / (k + mean) * nu_scale
    return max(0.05, min(nu, 5.0))



def sample_com_poisson(mean: float, k: float, size: int, rng: np.random.Generator,
                       nu_scale: float = 1.0) -> np.ndarray:
    """
    Sample from Conway-Maxwell-Poisson distribution.

    COM-Poisson PMF: P(X=x) = λ^x / (x!)^ν / Z(λ,ν)

    Uses cached CDF + closed-form λ initialization for speed.

    Args:
        mean: Expected count
        k: Dispersion parameter (same interface as NegBin — lower = more variance)
        size: Number of samples
        rng: Random generator
        nu_scale: Multiplier on ν (>1 = thinner left tail, <1 = fatter)

    Returns:
        Array of samples
    """
    if mean <= 0:
        return np.zeros(size)
    if k <= 0:
        return np.full(size, int(round(mean)))

    nu = _cmp_nu_from_k(mean, k, nu_scale)

    # Round for cache hits (2 decimal places)
    mean_r = round(mean, 2)
    nu_r = round(nu, 4)

    _, cdf = _cmp_cdf(mean_r, nu_r)
    u = rng.uniform(0, 1, size)
    return np.searchsorted(cdf, u).astype(float)


def sample_com_poisson_scalar(mean: float, k: float, rng: np.random.Generator,
                              nu_scale: float = 1.0) -> int:
    """Sample a single value from COM-Poisson."""
    if mean <= 0:
        return 0
    if k <= 0:
        return int(round(mean))

    nu = _cmp_nu_from_k(mean, k, nu_scale)
    mean_r = round(mean, 2)
    nu_r = round(nu, 4)

    _, cdf = _cmp_cdf(mean_r, nu_r)
    u = rng.random()
    return int(np.searchsorted(cdf, u))
