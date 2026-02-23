"""
Sampling distributions for tweet count forecasting.

Provides NegBin and COM-Poisson samplers with a unified interface.
"""

import math
import numpy as np


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


def sample_com_poisson(mean: float, k: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample from Conway-Maxwell-Poisson distribution.

    COM-Poisson PMF: P(X=x) = λ^x / (x!)^ν / Z(λ,ν)

    Has lighter left tails than NegBin for the same mean and variance,
    because NegBin's Gamma-Poisson mixing creates extreme low draws
    that COM-Poisson avoids.

    Args:
        mean: Expected count
        k: Dispersion parameter (same interface as NegBin — lower = more variance)
        size: Number of samples
        rng: Random generator

    Returns:
        Array of samples
    """
    if mean <= 0:
        return np.zeros(size)
    if k <= 0:
        return np.full(size, int(round(mean)))

    # Map NegBin-style k to COM-Poisson ν
    # NegBin var = μ(1+μ/k), COM-Poisson var ≈ μ/ν
    # Match: ν = k/(k+μ)
    nu = k / (k + mean)
    nu = max(0.05, min(nu, 5.0))

    # Support: compute PMF up to reasonable max
    target_std = math.sqrt(mean / nu)
    max_x = int(mean + 6 * target_std) + 1
    max_x = max(max_x, 20)

    xs = np.arange(max_x + 1, dtype=float)
    log_factorials = np.array([math.lgamma(x + 1) for x in range(max_x + 1)])

    def compute_pmf(lam):
        log_lam = math.log(max(lam, 1e-300))
        log_pmf = xs * log_lam - nu * log_factorials
        log_pmf -= log_pmf.max()
        pmf = np.exp(log_pmf)
        pmf /= pmf.sum()
        return pmf

    def compute_mean_for_lambda(lam):
        pmf = compute_pmf(lam)
        return (xs * pmf).sum()

    # Binary search for λ that gives target mean
    lo, hi = 1e-3, max(mean, 1.0) ** 3 + 10
    for _ in range(60):
        mid = (lo + hi) / 2
        m = compute_mean_for_lambda(mid)
        if m < mean:
            lo = mid
        else:
            hi = mid

    pmf = compute_pmf((lo + hi) / 2)
    cdf = np.cumsum(pmf)

    # Vectorized inverse CDF sampling
    u = rng.uniform(0, 1, size)
    return np.searchsorted(cdf, u).astype(float)


def sample_com_poisson_scalar(mean: float, k: float, rng: np.random.Generator) -> int:
    """Sample a single value from COM-Poisson."""
    result = sample_com_poisson(mean, k, 1, rng)
    return int(result[0])
