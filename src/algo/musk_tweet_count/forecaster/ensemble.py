"""
Ensemble Monte Carlo forecaster combining multiple EWMA-based interday models.

Pools simulation samples from forecasters with different adaptation speeds
(slow and fast α values) to improve forecast accuracy and robustness.
"""

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional

import numpy as np

from .config import ForecasterConfig
from .data import ContractDayUtils, TweetEvent
from .intraday import BaseIntradayForecaster
from .interday import InterdayForecaster
from .monte_carlo import BinProbability, ForecastResult

logger = logging.getLogger(__name__)


class EnsembleMonteCarloForecaster:
    """
    Combines multiple Monte Carlo forecasters with different EWMA alphas.

    Uses sample-level combination (pooling raw simulation samples) rather than
    probability-level combination. This preserves full distributional information
    and allows for weighted combination based on forecaster performance.
    """

    def __init__(
        self,
        config: ForecasterConfig,
        nowcast: BaseIntradayForecaster,
        interday_forecasters: List[InterdayForecaster],
        contract_utils: ContractDayUtils,
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize ensemble Monte Carlo forecaster.

        Args:
            config: Full forecaster configuration
            nowcast: Shared intraday nowcast model (data-driven, same for all)
            interday_forecasters: List of interday forecasters [slow, fast]
            contract_utils: Contract-day utilities
            weights: Weights for each forecaster (default: equal)
        """
        self.config = config
        self.mc_config = config.monte_carlo
        self.nowcast = nowcast
        self.interday_forecasters = interday_forecasters
        self.contract_utils = contract_utils

        # Weights
        n_members = len(interday_forecasters)
        if weights is None:
            self.weights = [1.0 / n_members] * n_members
        else:
            if len(weights) != n_members:
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of forecasters ({n_members})"
                )
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]

        # Bins for probability calculation
        self.bins = config.bins

        logger.info(
            f"Initialized ensemble with {n_members} members, "
            f"weights={[f'{w:.2f}' for w in self.weights]}"
        )

    def _sample_today(
        self,
        mean: float,
        std: float,
        cum_so_far: int,
        rng: np.random.Generator,
    ) -> int:
        """Sample today's final count using log-normal distribution."""
        if std <= 0 or mean <= 0:
            return max(int(mean), cum_so_far)

        cv_squared = (std / mean) ** 2
        sigma_squared = np.log(1 + cv_squared)
        sigma = np.sqrt(sigma_squared)
        mu = np.log(mean) - sigma_squared / 2

        sample = rng.lognormal(mu, sigma)
        return max(int(round(sample)), cum_so_far)

    def _sample_future_days(
        self,
        interday: InterdayForecaster,
        base_date: date,
        horizons: List[int],
        rng: np.random.Generator,
        regime_adjustment: float = 1.0,
    ) -> List[int]:
        """Sample counts for future days using Negative Binomial."""
        samples = []

        # Apply dispersion inflation (lower k = higher variance)
        dispersion_inflation = self.mc_config.dispersion_inflation_factor

        for h in horizons:
            mean, k = interday.forecast_day(h, base_date)

            # Apply dispersion inflation: k_adjusted = k / inflation_factor
            k_adjusted = k / dispersion_inflation

            adjusted_mean = mean * regime_adjustment

            from .distributions import sample_negbin_scalar, sample_com_poisson_scalar, sample_negbin_reflected_scalar
            dist = self.mc_config.sampling_distribution
            if dist == "com_poisson":
                samples.append(sample_com_poisson_scalar(adjusted_mean, k_adjusted, rng))
            elif dist == "negbin_reflected":
                samples.append(sample_negbin_reflected_scalar(adjusted_mean, k_adjusted, rng))
            else:
                samples.append(sample_negbin_scalar(adjusted_mean, k_adjusted, rng))

        return samples

    def _compute_regime_adjustment(
        self,
        interday: InterdayForecaster,
        cum_so_far: int,
        tau: int,
        contract_date: date,
    ) -> float:
        """Compute adjustment factor using Bayesian/Kalman shrinkage."""
        F_min = self.mc_config.regime_adj_f_min
        sigma_prior = self.mc_config.regime_adj_sigma_prior
        sigma0 = self.mc_config.regime_adj_sigma0
        adj_min = self.mc_config.regime_adj_min
        adj_max = self.mc_config.regime_adj_max
        tau_gate = self.mc_config.regime_adj_tau_gate
        F_gate = self.mc_config.regime_adj_f_gate

        if tau < tau_gate:
            return 1.0

        is_weekend = self.contract_utils.is_weekend(contract_date)
        F_tau = self.nowcast.get_expected_progress(tau, is_weekend)

        if F_tau < F_gate:
            return 1.0

        F_tau = max(F_tau, F_min)

        implied_today = cum_so_far / F_tau

        lambda_prior = max(interday.regime.current_intensity, 1e-6)
        m_prior = np.log(lambda_prior + 1.0)

        y_obs = np.log(implied_today + 1.0)

        sigma_prior2 = sigma_prior ** 2
        sigma_obs2 = (sigma0 ** 2) / F_tau

        w = sigma_prior2 / (sigma_prior2 + sigma_obs2)

        m_post = (1.0 - w) * m_prior + w * y_obs
        lambda_post = np.exp(m_post) - 1.0
        lambda_post = max(lambda_post, 1e-6)

        adjustment = lambda_post / lambda_prior
        adjustment = max(adj_min, min(adj_max, adjustment))

        return adjustment

    def _run_member_simulation(
        self,
        interday: InterdayForecaster,
        events: List[TweetEvent],
        contract_date: date,
        now: datetime,
        horizon: int,
        n_simulations: int,
        seed: Optional[int],
        member_idx: int,
        today_samples: np.ndarray,
    ) -> np.ndarray:
        """Run simulation for a single ensemble member."""
        # Use different seed for each member to ensure diversity
        if seed is not None:
            member_seed = seed + member_idx * 1000
        else:
            member_seed = None
        rng = np.random.default_rng(member_seed)

        sums = today_samples.astype(float)

        if horizon == 1:
            # Today only - no interday component
            return sums

        # For horizon > 1, compute regime adjustment for this member
        cum_so_far = len([e for e in events if e.timestamp < now])
        tau = self.contract_utils.get_tau(now, contract_date)
        regime_adjustment = self._compute_regime_adjustment(
            interday, cum_so_far, tau, contract_date
        )

        future_horizons = list(range(1, horizon))

        for i in range(n_simulations):
            future_samples = self._sample_future_days(
                interday, contract_date, future_horizons, rng, regime_adjustment
            )
            sums[i] += sum(future_samples)

        return sums

    def _compute_bin_probabilities(self, sums: np.ndarray) -> List[BinProbability]:
        """Compute probability for each bin."""
        n = len(sums)
        probs = []

        for lower, upper in self.bins:
            count = np.sum((sums >= lower) & (sums <= upper))
            prob = count / n
            probs.append(BinProbability(lower=lower, upper=upper, probability=prob))

        return probs

    def simulate_horizon(
        self,
        events: List[TweetEvent],
        contract_date: date,
        now: datetime,
        horizon: int = 7,
        n_simulations: Optional[int] = None,
    ) -> ForecastResult:
        """
        Run ensemble Monte Carlo simulation for N-day sum.

        Pools samples from all ensemble members based on their weights.

        Args:
            events: Today's events so far
            contract_date: Today's contract date
            now: Current timestamp
            horizon: Forecast horizon in days (1-7)
            n_simulations: Total number of simulations (distributed by weight)

        Returns:
            ForecastResult with distribution and bin probabilities
        """
        start_time = time.time()

        if n_simulations is None:
            n_simulations = self.mc_config.n_simulations

        seed = self.mc_config.random_seed

        # Get today's nowcast (keep mean for metadata in ForecastResult)
        today_mean, today_std = self.nowcast.predict(events, contract_date, now)
        cum_so_far = len([e for e in events if e.timestamp < now])

        # Compute number of simulations per member based on weights
        n_per_member = [int(n_simulations * w) for w in self.weights]
        # Ensure we hit exactly n_simulations
        n_per_member[-1] = n_simulations - sum(n_per_member[:-1])

        # Run simulation for each ensemble member and collect samples
        all_samples = []
        regime_adjustments = []

        for idx, (interday, n_sims) in enumerate(
            zip(self.interday_forecasters, n_per_member)
        ):
            if n_sims <= 0:
                continue

            # Generate discrete today-samples for this member with its own rng
            member_seed_today = (seed + idx * 1000 + 500) if seed is not None else None
            member_rng_today = np.random.default_rng(member_seed_today)
            today_samples = self.nowcast.predict_samples(
                events, contract_date, now, n_sims, member_rng_today,
                std_inflation_factor=self.mc_config.today_std_inflation_factor,
            )

            member_sums = self._run_member_simulation(
                interday,
                events,
                contract_date,
                now,
                horizon,
                n_sims,
                seed,
                idx,
                today_samples,
            )
            all_samples.append(member_sums)

            # Track regime adjustment for this member
            if horizon > 1:
                tau = self.contract_utils.get_tau(now, contract_date)
                adj = self._compute_regime_adjustment(
                    interday, cum_so_far, tau, contract_date
                )
                regime_adjustments.append(adj)

        # Pool all samples
        sums = np.concatenate(all_samples)

        # Compute statistics on pooled samples
        mean = float(np.mean(sums))
        median = float(np.median(sums))
        std = float(np.std(sums))
        p5 = float(np.percentile(sums, 5))
        p25 = float(np.percentile(sums, 25))
        p75 = float(np.percentile(sums, 75))
        p95 = float(np.percentile(sums, 95))

        # Compute bin probabilities on pooled samples
        bin_probs = self._compute_bin_probabilities(sums)

        # Component breakdown (use weighted average of ensemble members)
        future_means_total = 0.0
        if horizon > 1:
            future_horizons = list(range(1, horizon))
            for idx, interday in enumerate(self.interday_forecasters):
                future_params = interday.get_forecast_params(future_horizons, contract_date)
                member_future_mean = sum(m for m, _ in future_params)
                if regime_adjustments:
                    member_future_mean *= regime_adjustments[idx] if idx < len(regime_adjustments) else 1.0
                future_means_total += self.weights[idx] * member_future_mean

        # Average regime adjustment
        avg_regime_adjustment = (
            float(np.mean(regime_adjustments)) if regime_adjustments else 1.0
        )

        elapsed_ms = (time.time() - start_time) * 1000

        result = ForecastResult(
            mean=mean,
            median=median,
            std=std,
            p5=p5,
            p25=p25,
            p75=p75,
            p95=p95,
            bin_probabilities=bin_probs,
            today_estimate=today_mean,
            future_days_estimate=future_means_total,
            regime_adjustment=avg_regime_adjustment,
            n_simulations=len(sums),
            simulation_time_ms=elapsed_ms,
        )

        logger.debug(
            f"Ensemble {horizon}-day simulation: mean={mean:.1f}, std={std:.1f}, "
            f"avg_regime_adj={avg_regime_adjustment:.2f}, time={elapsed_ms:.1f}ms, "
            f"n_members={len(self.interday_forecasters)}"
        )

        return result

    def simulate(
        self,
        events: List[TweetEvent],
        contract_date: date,
        now: datetime,
        n_simulations: Optional[int] = None,
    ) -> ForecastResult:
        """Run ensemble Monte Carlo simulation for 7-day sum."""
        return self.simulate_horizon(
            events, contract_date, now, horizon=7, n_simulations=n_simulations
        )
