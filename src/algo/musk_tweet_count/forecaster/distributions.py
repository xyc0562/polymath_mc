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


# Bounds for the CMP ν solve. The lower bound is a numerical guard; the
# old code floored the CLOSED-FORM map k/(k+μ) at 0.05, which clipped the
# whole live parameter range (remaining mean 20-100, bucket k' = k/2.5 in
# 0.2-1.2 → ν 0.004-0.05), delivering roughly HALF the intended std on
# evening buckets and silently turning today_std_inflation_factor into a
# no-op (k'=0.4 and k=1.0 both floored to the same ν).
CMP_NU_MIN = 1e-4
CMP_NU_MAX = 5.0


def _cmp_variance(mean_round: float, nu_round: float) -> float:
    """Variance of the mean-matched CMP at (mean, ν) from its cached CDF."""
    xs, cdf = _cmp_cdf(mean_round, nu_round)
    pmf = np.diff(np.concatenate(([0.0], cdf)))
    m = float((xs * pmf).sum())
    return float((((xs - m) ** 2) * pmf).sum())


@lru_cache(maxsize=512)
def _cmp_nu_from_k(mean: float, k: float, nu_scale: float = 1.0) -> float:
    """Solve the CMP ν that delivers the NegBin target variance μ(1+μ/k).

    The closed-form map ν = k/(k+μ) relies on the approximation
    var ≈ μ/ν, which under-delivers the target std by 30-60% across this
    system's parameter range (measured 2026-07-09) — so solve ν
    numerically instead. Variance is monotone decreasing in ν for the
    mean-matched family, so a log-scale bisection converges quickly; the
    result is lru_cached per rounded (mean, k, nu_scale) and callers pass
    rounded args, so the solve runs once per parameter set.

    ATTAINABILITY: at fixed mean μ the CMP family's variance is capped by
    its ν→0 geometric-like limit, var ≈ μ(1+μ). NegBin targets with
    k < 1 exceed that ceiling and saturate at it (ν pinned at CMP_NU_MIN)
    — the sampler then delivers the widest distribution the family
    admits. Consequence: dispersion inflation (k/s) only widens the
    distribution while the inflated k stays ≥ ~1; beyond that the knob
    saturates. This is a property of COM-Poisson itself, not the solve.

    nu_scale > 1 → higher ν → thinner tails (applied to the solved ν).
    """
    target_var = mean * (1.0 + mean / k)

    lo, hi = CMP_NU_MIN, CMP_NU_MAX
    if _cmp_variance(mean, round(lo, 6)) <= target_var:
        # Even the widest admissible CMP cannot reach the target variance
        nu = lo
    elif _cmp_variance(mean, round(hi, 6)) >= target_var:
        nu = hi
    else:
        for _ in range(40):
            mid = math.sqrt(lo * hi)
            if _cmp_variance(mean, round(mid, 6)) > target_var:
                # Too wide → need thinner → raise ν
                lo = mid
            else:
                hi = mid
            if hi / lo < 1.01:
                break
        nu = math.sqrt(lo * hi)

    nu *= nu_scale
    return max(CMP_NU_MIN, min(nu, CMP_NU_MAX))



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

    # Round args so the cached ν-solve and CDF are reused across ticks
    mean_r = round(mean, 2)
    k_r = round(k, 4)
    if k_r <= 0:
        return np.full(size, int(round(mean)))
    nu = _cmp_nu_from_k(mean_r, k_r, round(nu_scale, 4))
    nu_r = round(nu, 6)

    _, cdf = _cmp_cdf(mean_r, nu_r)
    u = rng.uniform(0, 1, size)
    return np.searchsorted(cdf, u).astype(float)


def sample_com_poisson_scalar(mean: float, k: float, rng: np.random.Generator,
                              nu_scale: float = 1.0) -> int:
    """Sample a single value from COM-Poisson."""
    if mean <= 0:
        return 0

    mean_r = round(mean, 2)
    k_r = round(k, 4)
    if k_r <= 0:
        return int(round(mean))
    nu = _cmp_nu_from_k(mean_r, k_r, round(nu_scale, 4))
    nu_r = round(nu, 6)

    _, cdf = _cmp_cdf(mean_r, nu_r)
    u = rng.random()
    return int(np.searchsorted(cdf, u))
