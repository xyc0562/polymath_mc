"""
NB-GAS (Negative Binomial Generalized Autoregressive Score) regime model.

Replaces the EWMA-based regime model with a score-driven model where
parameters (ω, α, β) are estimated via MLE.

Model:
    f_t = log(μ_t)                              # log-intensity (state)
    f_{t+1} = ω + β·f_t + α_t·s_t              # GAS recursion with adaptive α
    s_t = (y_t - μ_t) / μ_t                     # scaled Pearson score
    Y_t ~ NegBin(μ_t = exp(f_t), k)             # observation density

Change Point Detection:
    When |2-day avg - 7-day avg| / 7-day avg > threshold, we boost α temporarily
    to allow faster adaptation. This is symmetric for both drops and surges.
"""

import logging
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
from scipy import optimize, stats

from .config import GASConfig

logger = logging.getLogger(__name__)


class GASRegimeModel:
    """NB-GAS regime model for time-varying tweet intensity.

    Includes change point detection for faster adaptation to regime shifts.
    """

    def __init__(self, config: GASConfig):
        self.config = config
        self.omega = config.omega_init
        self.alpha = config.alpha_init
        self.beta = config.beta_init
        self.k = 2.0  # Set externally from DispersionEstimator
        self.f = None  # Current log-intensity state

        # Change point detection: track recent observations
        self._recent_counts: deque = deque(maxlen=7)

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

        # Bounds: ω ∈ (-2, 2), α ∈ (alpha_min, alpha_max), β ∈ (beta_min, beta_max)
        bounds = [
            (-2.0, 2.0),
            (self.config.alpha_min, self.config.alpha_max),
            (self.config.beta_min, self.config.beta_max),
        ]

        counts = np.array(daily_counts, dtype=np.float64)

        # Multi-start MLE. A single start can land on the max_log_intensity
        # plateau: at e.g. (ω=2, β=0.999) the filtered state pegs at the cap
        # for every day, the likelihood goes flat in all three parameters,
        # and L-BFGS-B reports success with a forecast of exp(cap)/day.
        # Moment-matched starts anchor ω to the sample mean at several
        # persistence levels so at least one start is in the sane basin.
        log_mean = np.log(max(np.mean(counts), 1.0))
        starts = [
            np.array([self.config.omega_init, self.config.alpha_init, self.config.beta_init]),
        ]
        for beta0 in (0.7, 0.9, 0.97):
            for alpha0 in (0.05, 0.2):
                starts.append(np.array([
                    np.clip((1.0 - beta0) * log_mean, bounds[0][0], bounds[0][1]),
                    np.clip(alpha0, self.config.alpha_min, self.config.alpha_max),
                    np.clip(beta0, self.config.beta_min, self.config.beta_max),
                ]))

        best_params = None
        best_nll = np.inf
        for x0 in starts:
            result = optimize.minimize(
                self._negloglik,
                x0,
                args=(counts, k),
                method="L-BFGS-B",
                bounds=bounds,
            )
            if not np.isfinite(result.fun):
                continue
            if self._is_degenerate_fit(result.x, counts, k):
                continue
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x

        if best_params is not None:
            self.omega, self.alpha, self.beta = best_params
        else:
            # Every start converged onto the degenerate plateau (or failed):
            # keep the configured defaults with ω anchored to the sample
            # mean so forecasts stay near observed intensity.
            logger.warning(
                "GAS MLE degenerate/failed from all starts; "
                "falling back to moment-anchored defaults"
            )
            self.alpha = self.config.alpha_init
            self.beta = self.config.beta_init
            self.omega = (1.0 - self.beta) * log_mean

        # Run filter forward with fitted parameters to get final state
        f_values = self._run_filter(self.omega, self.alpha, self.beta, counts, k)

        # Absorb the final observation: _run_filter's state sequence only
        # incorporates counts[0..n-2] (counts[n-1] enters the likelihood but
        # never the state). Advance one step so self.f is the filtered
        # intensity for the day AFTER the last training day.
        last_f = f_values[-1]
        mu_last = max(np.exp(min(last_f, self.config.max_log_intensity)), 1e-6)
        last_score = (counts[-1] - mu_last) / mu_last
        self.f = self.omega + self.beta * last_f + self.alpha * last_score

        # Apply cap
        self.f = min(self.f, self.config.max_log_intensity)

        # Initialize recent counts buffer with last 7 observations
        self._recent_counts.clear()
        for c in daily_counts[-7:]:
            self._recent_counts.append(c)

        f_bar = self._unconditional_mean()
        logger.info(
            f"GAS fitted: ω={self.omega:.4f}, α={self.alpha:.4f}, β={self.beta:.4f}, "
            f"k={k:.2f}, f_T={self.f:.3f} (μ={np.exp(self.f):.1f}), "
            f"f̄={f_bar:.3f} (μ̄={np.exp(f_bar):.1f})"
        )

    def _is_degenerate_fit(
        self,
        params: np.ndarray,
        counts: np.ndarray,
        k: float,
    ) -> bool:
        """Detect plateau/corner solutions that forecast the intensity cap.

        A fit is degenerate when its filtered state spends a large fraction
        of the sample pinned at max_log_intensity — there the likelihood is
        locally flat in (ω, α, β) and the optimizer stops on the plateau
        with success=True while forecasting exp(max_log_intensity)/day.
        """
        omega, alpha, beta = params
        f_values = self._run_filter(omega, alpha, beta, counts, k)
        capped_frac = np.mean(
            f_values >= self.config.max_log_intensity - 1e-9
        )
        return bool(capped_frac > 0.5)

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

            # Scaled Pearson score: s_t = (y_t - μ_t) / μ_t
            # Bounded at -1 from below, provides stability
            score = (counts[t] - mu) / mu

            # GAS recursion
            f_next = omega + beta * f[t] + alpha * score

            # Apply cap
            f_next = min(f_next, self.config.max_log_intensity)

            f[t + 1] = f_next

        return f

    def _detect_regime_shift(self) -> Optional[float]:
        """Detect regime shift by comparing 2-day avg to 7-day avg.

        Returns:
            Relative deviation if shift detected, None otherwise.
            Positive = surge, Negative = collapse.
        """
        if len(self._recent_counts) < 7:
            return None

        recent_list = list(self._recent_counts)
        avg_2d = np.mean(recent_list[-2:])  # Last 2 days
        avg_7d = np.mean(recent_list)        # Last 7 days

        if avg_7d < 1:
            return None

        deviation = (avg_2d - avg_7d) / avg_7d
        if abs(deviation) > self.config.cpd_threshold:
            return deviation
        return None

    def _get_adaptive_alpha(self) -> float:
        """Get α, boosted if regime shift detected."""
        shift = self._detect_regime_shift()
        if shift is not None:
            boosted = self.alpha * self.config.cpd_alpha_multiplier
            capped = min(boosted, self.config.cpd_alpha_cap)
            logger.debug(
                f"Regime shift detected ({shift:+.1%}), "
                f"boosting α: {self.alpha:.4f} → {capped:.4f}"
            )
            return capped
        return self.alpha

    def update(self, count: int) -> None:
        """Update state with new observation (online, after fitting).

        Args:
            count: Observed daily tweet count.
        """
        if self.f is None:
            raise RuntimeError("GAS model not fitted")

        # Add to recent counts buffer for change point detection
        self._recent_counts.append(count)

        mu = np.exp(self.f)
        mu = max(mu, 1e-6)

        # Scaled Pearson score: (y - μ) / μ, bounded at -1 from below
        score = (count - mu) / mu

        # Use adaptive α (boosted during regime shifts)
        alpha = self._get_adaptive_alpha()

        self.f = self.omega + self.beta * self.f + alpha * score
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
