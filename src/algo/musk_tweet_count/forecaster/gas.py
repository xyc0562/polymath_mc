"""
NB-GAS (Negative Binomial Generalized Autoregressive Score) regime model.

Replaces the EWMA-based regime model with a score-driven model where
parameters (ω, α, β) are estimated via MLE.

Model:
    f_t = log(μ_t)                              # log-intensity (state)
    f_{t+1} = ω + β·f_t + α·s_t                # GAS recursion
    s_t = (y_t - μ_t) / μ_t                     # scaled score
    Y_t ~ NegBin(μ_t = exp(f_t), k)             # observation density
"""

import logging
from typing import List, Tuple

import numpy as np
from scipy import optimize, stats

from .config import GASConfig

logger = logging.getLogger(__name__)


class GASRegimeModel:
    """NB-GAS regime model for time-varying tweet intensity."""

    def __init__(self, config: GASConfig):
        self.config = config
        self.omega = config.omega_init
        self.alpha = config.alpha_init
        self.beta = config.beta_init
        self.k = 2.0  # Set externally from DispersionEstimator
        self.f = None  # Current log-intensity state

    def fit(self, daily_counts: List[int], k: float) -> None:
        """Estimate (ω, α, β) via MLE on historical daily counts.

        Args:
            daily_counts: List of daily tweet counts (chronological order).
            k: Dispersion parameter from DispersionEstimator (fixed during fitting).
        """
        self.k = k

        if len(daily_counts) < 10:
            logger.warning(
                f"Only {len(daily_counts)} observations for GAS fitting, "
                "using default parameters"
            )
            self.f = np.log(max(np.mean(daily_counts), 1.0))
            return

        # Initial guesses
        x0 = np.array([self.config.omega_init, self.config.alpha_init, self.config.beta_init])

        # Bounds: ω ∈ (-2, 2), α ∈ (0.001, 0.5), β ∈ (0.5, 0.999)
        bounds = [
            (-2.0, 2.0),
            (self.config.alpha_min, self.config.alpha_max),
            (0.5, self.config.beta_max),
        ]

        counts = np.array(daily_counts, dtype=np.float64)

        result = optimize.minimize(
            self._negloglik,
            x0,
            args=(counts, k),
            method="L-BFGS-B",
            bounds=bounds,
        )

        if result.success:
            self.omega, self.alpha, self.beta = result.x
        else:
            logger.warning(f"GAS MLE optimization did not converge: {result.message}")
            # Use result anyway if it improved
            if result.fun < self._negloglik(x0, counts, k):
                self.omega, self.alpha, self.beta = result.x

        # Run filter forward with fitted parameters to get final state
        f_values = self._run_filter(self.omega, self.alpha, self.beta, counts, k)
        self.f = f_values[-1]

        # Apply cap
        self.f = min(self.f, self.config.max_log_intensity)

        f_bar = self._unconditional_mean()
        logger.info(
            f"GAS fitted: ω={self.omega:.4f}, α={self.alpha:.4f}, β={self.beta:.4f}, "
            f"k={k:.2f}, f_T={self.f:.3f} (μ={np.exp(self.f):.1f}), "
            f"f̄={f_bar:.3f} (μ̄={np.exp(f_bar):.1f})"
        )

    def _negloglik(self, params: np.ndarray, counts: np.ndarray, k: float) -> float:
        """Negative log-likelihood for scipy.optimize.minimize."""
        omega, alpha, beta = params

        f_values = self._run_filter(omega, alpha, beta, counts, k)

        # Compute log-likelihood: sum of NegBin log-pmf
        total_ll = 0.0
        for t in range(len(counts)):
            mu = np.exp(f_values[t])
            mu = max(mu, 1e-6)  # Prevent division by zero
            p = k / (k + mu)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            ll = stats.nbinom.logpmf(int(counts[t]), n=k, p=p)
            if np.isfinite(ll):
                total_ll += ll
            else:
                total_ll -= 100.0  # Penalty for bad values

        return -total_ll

    def _run_filter(
        self,
        omega: float,
        alpha: float,
        beta: float,
        counts: np.ndarray,
        k: float,
    ) -> np.ndarray:
        """Run GAS filter forward, return sequence of f_t values."""
        n = len(counts)
        f = np.empty(n)

        # Initialize f_0 from mean of first 7 observations
        init_window = min(7, n)
        f[0] = np.log(max(np.mean(counts[:init_window]), 1.0))

        for t in range(n - 1):
            mu = np.exp(f[t])
            mu = max(mu, 1e-6)

            # Scaled score: s_t = (y_t - μ_t) / μ_t
            score = (counts[t] - mu) / mu

            # GAS recursion
            f_next = omega + beta * f[t] + alpha * score

            # Apply cap
            f_next = min(f_next, self.config.max_log_intensity)

            f[t + 1] = f_next

        return f

    def update(self, count: int) -> None:
        """Update state with new observation (online, after fitting).

        Args:
            count: Observed daily tweet count.
        """
        if self.f is None:
            raise RuntimeError("GAS model not fitted")

        mu = np.exp(self.f)
        mu = max(mu, 1e-6)
        score = (count - mu) / mu
        self.f = self.omega + self.beta * self.f + self.alpha * score
        self.f = min(self.f, self.config.max_log_intensity)

    def forecast_intensity(self, horizon: int = 0) -> float:
        """Forecast log-intensity h steps ahead.

        For h=0: returns current f_t.
        For h>0: f̄·(1 - β^h) + β^h · f_t (mean-reverts toward unconditional mean).

        Args:
            horizon: Days ahead (0 = current).

        Returns:
            Forecasted log-intensity.
        """
        if self.f is None:
            raise RuntimeError("GAS model not fitted")

        if horizon == 0:
            return self.f

        # Multi-step forecast with mean reversion
        f_bar = self._unconditional_mean()
        beta_h = self.beta ** horizon
        forecast = f_bar * (1 - beta_h) + beta_h * self.f

        return min(forecast, self.config.max_log_intensity)

    @property
    def intensity(self) -> float:
        """Current intensity (exp scale)."""
        if self.f is None:
            raise RuntimeError("GAS model not fitted")
        return np.exp(self.f)

    @property
    def current_intensity(self) -> float:
        """Alias for intensity (compatibility with RegimeModel interface)."""
        return self.intensity

    def _unconditional_mean(self) -> float:
        """Unconditional mean log-intensity f̄ = ω/(1-β), capped for safety."""
        if self.beta >= 1:
            return self.f if self.f is not None else 0.0
        f_bar = self.omega / (1 - self.beta)
        return min(f_bar, self.config.max_log_intensity)

    @property
    def long_term_mean(self) -> float:
        """Unconditional mean intensity (exp scale)."""
        return np.exp(self._unconditional_mean())
