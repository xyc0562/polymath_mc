"""
Poisson-Inverse Gaussian (PIG) GAS regime model.

Alternative to NB-GAS with heavier tails for occasional extreme days.

Model:
    f_t = log(μ_t)                              # log-intensity (state)
    f_{t+1} = ω + β·f_t + α·s_t                 # GAS recursion
    s_t = (y_t - μ_t) / μ_t                     # scaled score
    Y_t ~ PIG(μ_t = exp(f_t), σ)                # observation density

PIG Distribution:
    Y | Z ~ Poisson(μ·Z), Z ~ IG(1, 1/σ²)
    Mean: μ
    Variance: μ + σ²·μ²  (compare to NegBin: μ + μ²/k)

    The σ parameter controls overdispersion:
    - σ → 0: approaches Poisson
    - σ large: heavy overdispersion with fatter tails than NegBin
"""

import logging
from typing import List, Tuple

import numpy as np
from scipy import optimize
from scipy.special import kv, gammaln

from .config import GASConfig

logger = logging.getLogger(__name__)


def pig_logpmf(y: int, mu: float, sigma: float) -> float:
    """
    Log-PMF of Poisson-Inverse Gaussian distribution.

    Uses closed-form formula with modified Bessel function K.

    Model: Y | Z ~ Poisson(μZ), Z ~ IG(1, φ) where φ = 1/σ²
    Mean: μ
    Variance: μ + σ²μ²

    Derivation: The integral ∫z^(y-3/2) exp(-az - b/z) dz = 2(b/a)^(y/2-1/4) K_{y-1/2}(2√(ab))
    where a = μ + φ/2, b = φ/2.

    Args:
        y: count (non-negative integer)
        mu: mean parameter (> 0)
        sigma: dispersion parameter (> 0), where Var = μ + σ²μ²

    Returns:
        Log probability mass
    """
    if y < 0 or mu <= 0 or sigma <= 0:
        return -np.inf

    y = int(y)
    phi = 1 / (sigma ** 2)

    # Parameters for the Bessel function integral
    a = mu + phi / 2
    b = phi / 2

    # Bessel argument: 2√(ab)
    bessel_arg = 2 * np.sqrt(a * b)

    # Compute log K_{y-0.5}(bessel_arg)
    nu = y - 0.5
    if y == 0:
        # K_{-0.5}(x) = √(π/(2x)) * exp(-x)
        log_bessel = 0.5 * np.log(np.pi / (2 * bessel_arg)) - bessel_arg
    else:
        kv_val = kv(nu, bessel_arg)
        if kv_val <= 0 or not np.isfinite(kv_val):
            return -np.inf
        log_bessel = np.log(kv_val)

    # Full formula:
    # P(Y=y) = [μ^y √(φ/(2π)) / y!] * exp(φ) * 2 * (b/a)^((y-0.5)/2) * K_{y-0.5}(2√(ab))
    log_pmf = (
        y * np.log(mu)                          # μ^y
        + 0.5 * np.log(phi / (2 * np.pi))       # √(φ/(2π))
        - gammaln(y + 1)                        # 1/y!
        + phi                                   # exp(φ)
        + np.log(2)                             # factor of 2
        + (y - 0.5) / 2 * np.log(b / a)         # (b/a)^((y-0.5)/2)
        + log_bessel                            # K_{y-0.5}
    )

    return log_pmf if np.isfinite(log_pmf) else -np.inf


def pig_pmf(y: int, mu: float, sigma: float) -> float:
    """PMF of Poisson-Inverse Gaussian distribution."""
    return np.exp(pig_logpmf(y, mu, sigma))


def pig_mean_var(mu: float, sigma: float) -> Tuple[float, float]:
    """Return mean and variance of PIG(μ, σ)."""
    return mu, mu + sigma**2 * mu**2


def sigma_from_mean_var(mu: float, var: float) -> float:
    """Estimate σ from observed mean and variance."""
    # Var = μ + σ²μ² => σ² = (Var - μ) / μ²
    if var <= mu:
        return 0.1  # Minimum sigma for underdispersed data
    return np.sqrt((var - mu) / (mu ** 2))


class PIGGASRegimeModel:
    """PIG-GAS regime model for time-varying tweet intensity."""

    def __init__(self, config: GASConfig):
        self.config = config
        self.omega = config.omega_init
        self.alpha = config.alpha_init
        self.beta = config.beta_init
        self.sigma = 0.5  # Default, will be estimated
        self.f = None  # Current log-intensity state

    def fit(self, daily_counts: List[int], sigma: float = None) -> None:
        """
        Estimate (ω, α, β) via MLE on historical daily counts.

        Args:
            daily_counts: List of daily tweet counts (chronological order)
            sigma: Dispersion parameter. If None, estimated from data.
        """
        if len(daily_counts) < 10:
            logger.warning(
                f"Only {len(daily_counts)} observations for PIG-GAS fitting, "
                "using default parameters"
            )
            self.f = np.log(max(np.mean(daily_counts), 1.0))
            return

        counts = np.array(daily_counts, dtype=np.float64)

        # Estimate sigma from data if not provided
        if sigma is None:
            mu_hat = np.mean(counts)
            var_hat = np.var(counts, ddof=1)
            self.sigma = sigma_from_mean_var(mu_hat, var_hat)
            self.sigma = np.clip(self.sigma, 0.1, 2.0)  # Reasonable bounds
        else:
            self.sigma = sigma

        # Initial guesses
        x0 = np.array([self.config.omega_init, self.config.alpha_init, self.config.beta_init])

        # Bounds: ω ∈ (-2, 2), α ∈ (0.001, 0.5), β ∈ (0.5, 0.999)
        bounds = [
            (-2.0, 2.0),
            (self.config.alpha_min, self.config.alpha_max),
            (0.5, self.config.beta_max),
        ]

        result = optimize.minimize(
            self._negloglik,
            x0,
            args=(counts, self.sigma),
            method="L-BFGS-B",
            bounds=bounds,
        )

        if result.success:
            self.omega, self.alpha, self.beta = result.x
        else:
            logger.warning(f"PIG-GAS MLE optimization did not converge: {result.message}")
            if result.fun < self._negloglik(x0, counts, self.sigma):
                self.omega, self.alpha, self.beta = result.x

        # Run filter forward with fitted parameters to get final state
        f_values = self._run_filter(self.omega, self.alpha, self.beta, counts, self.sigma)
        self.f = f_values[-1]

        # Apply cap
        self.f = min(self.f, self.config.max_log_intensity)

        f_bar = self._unconditional_mean()
        logger.info(
            f"PIG-GAS fitted: ω={self.omega:.4f}, α={self.alpha:.4f}, β={self.beta:.4f}, "
            f"σ={self.sigma:.3f}, f_T={self.f:.3f} (μ={np.exp(self.f):.1f}), "
            f"f̄={f_bar:.3f} (μ̄={np.exp(f_bar):.1f})"
        )

    def _negloglik(self, params: np.ndarray, counts: np.ndarray, sigma: float) -> float:
        """Negative log-likelihood for scipy.optimize.minimize."""
        omega, alpha, beta = params

        f_values = self._run_filter(omega, alpha, beta, counts, sigma)

        # Sum of log-likelihoods
        loglik = 0.0
        for t, y in enumerate(counts):
            mu = np.exp(f_values[t])
            ll = pig_logpmf(int(y), mu, sigma)
            if np.isfinite(ll):
                loglik += ll
            else:
                loglik -= 1000  # Penalty for invalid

        return -loglik

    def _run_filter(
        self, omega: float, alpha: float, beta: float, counts: np.ndarray, sigma: float
    ) -> np.ndarray:
        """Run GAS filter forward to compute f_t sequence."""
        T = len(counts)
        f = np.zeros(T)

        # Initialize f_0 with log of first few observations mean
        init_mean = np.mean(counts[:min(5, T)])
        f[0] = np.log(max(init_mean, 1.0))

        for t in range(T - 1):
            mu_t = np.exp(f[t])
            mu_t = max(mu_t, 1e-6)
            y_t = counts[t]

            # Scaled Pearson score: (y - μ) / μ, bounded at -1 from below
            score = (y_t - mu_t) / mu_t

            # GAS update
            f_next = omega + beta * f[t] + alpha * score

            # Apply cap
            f[t + 1] = min(f_next, self.config.max_log_intensity)

        return f

    def _unconditional_mean(self) -> float:
        """Compute unconditional mean f̄ = ω / (1 - β)."""
        if self.beta >= 1:
            return self.f
        return self.omega / (1 - self.beta)

    def update(self, count: int) -> None:
        """Update state with new observation."""
        if self.f is None:
            self.f = np.log(max(count, 1.0))
            return

        mu = np.exp(self.f)
        mu = max(mu, 1e-6)
        # Scaled Pearson score: (y - μ) / μ, bounded at -1 from below
        score = (count - mu) / mu

        f_new = self.omega + self.beta * self.f + self.alpha * score
        self.f = min(f_new, self.config.max_log_intensity)

    def forecast(self, horizon: int = 1) -> float:
        """
        Forecast mean intensity h days ahead.

        Uses mean reversion: f_{t+h} = f̄ + β^h * (f_t - f̄)
        """
        if self.f is None:
            return 50.0

        f_bar = self._unconditional_mean()
        f_h = f_bar + (self.beta ** horizon) * (self.f - f_bar)
        f_h = min(f_h, self.config.max_log_intensity)

        return np.exp(f_h)

    def sample(self, mu: float, rng: np.random.Generator, sigma: float = None) -> int:
        """
        Sample from PIG(μ, σ) distribution.

        Uses composition: Y | Z ~ Poisson(μZ), Z ~ IG(1, 1/σ²)

        Args:
            mu: Mean parameter
            rng: Random number generator
            sigma: Dispersion parameter (default: self.sigma)
        """
        # Sample Z from Inverse Gaussian(mean=1, shape=1/σ²)
        # Using the standard IG sampling algorithm
        if sigma is None:
            sigma = self.sigma
        phi = 1 / (sigma ** 2)  # shape parameter

        # Sampling from IG(1, phi) using Michael et al. (1976) algorithm
        v = rng.standard_normal()
        y = v ** 2
        x = 1 + (y - np.sqrt(y * (4 * phi + y))) / (2 * phi)

        u = rng.random()
        if u <= 1 / (1 + x):
            z = x
        else:
            z = 1 / x

        # Sample Y from Poisson(μ * z)
        return int(rng.poisson(mu * z))

    @property
    def intensity(self) -> float:
        """Current intensity μ = exp(f)."""
        if self.f is None:
            return 50.0
        return np.exp(self.f)

    @property
    def current_intensity(self) -> float:
        """Alias for intensity (compatibility with GASRegimeModel interface)."""
        return self.intensity

    def forecast_intensity(self, horizon: int = 0) -> float:
        """Forecast log-intensity h steps ahead (for compatibility with GASRegimeModel)."""
        if self.f is None:
            return np.log(50.0)

        f_bar = self._unconditional_mean()
        f_h = f_bar + (self.beta ** horizon) * (self.f - f_bar)
        return min(f_h, self.config.max_log_intensity)
