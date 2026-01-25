"""
Monte Carlo simulation for 7-day tweet count forecasting.

Combines intraday nowcast (today) with interday forecasts (days 1-6)
to generate the full distribution of the 7-day sum.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import MonteCarloConfig, ForecasterConfig
from .data import ContractDayUtils, TweetEvent
from .intraday import IntradayNowcast
from .interday import InterdayForecaster

logger = logging.getLogger(__name__)


@dataclass
class BinProbability:
    """Probability for a single bin."""

    lower: int
    upper: int
    probability: float

    @property
    def label(self) -> str:
        """Human-readable bin label."""
        if self.upper >= 10000:
            return f"{self.lower}+"
        return f"{self.lower}-{self.upper}"


@dataclass
class ForecastResult:
    """Result of a 7-day forecast."""

    # Point estimates
    mean: float
    median: float
    std: float

    # Percentiles
    p5: float
    p25: float
    p75: float
    p95: float

    # Bin probabilities
    bin_probabilities: List[BinProbability]

    # Component breakdown
    today_estimate: float
    future_days_estimate: float

    # Simulation metadata
    n_simulations: int
    simulation_time_ms: float

    # Regime adjustment (how much we boosted/reduced future based on today)
    regime_adjustment: float = 1.0

    def get_bin_probability(self, count: int) -> float:
        """Get probability that final count falls in the bin containing 'count'."""
        for bp in self.bin_probabilities:
            if bp.lower <= count <= bp.upper:
                return bp.probability
        return 0.0

    def get_probability_above(self, threshold: int) -> float:
        """Get probability that final count is above threshold."""
        return sum(
            bp.probability for bp in self.bin_probabilities
            if bp.lower > threshold
        )

    def get_probability_below(self, threshold: int) -> float:
        """Get probability that final count is below threshold."""
        return sum(
            bp.probability for bp in self.bin_probabilities
            if bp.upper < threshold
        )


class MonteCarloForecaster:
    """
    Monte Carlo simulation for 7-day sum distribution.

    Combines:
    - Today's count: sampled from log-normal based on nowcast
    - Days 1-6: sampled from Negative Binomial based on interday model
    """

    def __init__(
        self,
        config: ForecasterConfig,
        nowcast: IntradayNowcast,
        interday: InterdayForecaster,
        contract_utils: ContractDayUtils,
    ):
        """
        Initialize Monte Carlo forecaster.

        Args:
            config: Full forecaster configuration
            nowcast: Fitted intraday nowcast model
            interday: Fitted interday forecaster
            contract_utils: Contract-day utilities
        """
        self.config = config
        self.mc_config = config.monte_carlo
        self.nowcast = nowcast
        self.interday = interday
        self.contract_utils = contract_utils

        # Bins for probability calculation
        self.bins = config.bins

    def _sample_today(
        self,
        mean: float,
        std: float,
        cum_so_far: int,
        rng: np.random.Generator,
    ) -> int:
        """
        Sample today's final count using log-normal distribution.

        Uses log-normal to avoid clipping bias from max(0, normal).

        Args:
            mean: Predicted mean from nowcast
            std: Predicted std from nowcast
            cum_so_far: Current cumulative count (floor)
            rng: Random number generator

        Returns:
            Sampled count (at least cum_so_far)
        """
        if std <= 0 or mean <= 0:
            return max(int(mean), cum_so_far)

        # Convert to log-normal parameters
        # If X ~ LogNormal(mu, sigma), then E[X] = exp(mu + sigma^2/2)
        # and Var[X] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
        #
        # Given desired mean and std, solve for mu and sigma:
        # sigma^2 = log(1 + (std/mean)^2)
        # mu = log(mean) - sigma^2/2

        cv_squared = (std / mean) ** 2
        sigma_squared = np.log(1 + cv_squared)
        sigma = np.sqrt(sigma_squared)
        mu = np.log(mean) - sigma_squared / 2

        # Sample
        sample = rng.lognormal(mu, sigma)

        # Ensure at least cum_so_far
        return max(int(round(sample)), cum_so_far)

    def _sample_future_days(
        self,
        base_date: date,
        horizons: List[int],
        rng: np.random.Generator,
        regime_adjustment: float = 1.0,
    ) -> List[int]:
        """
        Sample counts for future days using Negative Binomial.

        Args:
            base_date: Today's contract date
            horizons: List of horizons (typically [1, 2, 3, 4, 5, 6])
            rng: Random number generator
            regime_adjustment: Multiplier based on today's observed activity vs expected

        Returns:
            List of sampled counts
        """
        samples = []

        for h in horizons:
            mean, k = self.interday.forecast_day(h, base_date)

            # Apply regime adjustment from today's observation
            adjusted_mean = mean * regime_adjustment

            # Negative Binomial sampling
            # scipy uses (n, p) where n = k, p = k / (k + μ)
            p = k / (k + adjusted_mean)

            # Handle edge cases
            if p <= 0 or p >= 1 or k <= 0:
                samples.append(int(round(adjusted_mean)))
            else:
                sample = rng.negative_binomial(k, p)
                samples.append(int(sample))

        return samples

    def _compute_regime_adjustment(
        self,
        cum_so_far: int,
        tau: int,
        contract_date: date,
    ) -> float:
        """
        Compute adjustment factor using Bayesian/Kalman shrinkage.

        Treats the interday regime intensity as a prior and today's partial
        observation as noisy evidence. Computes a data-dependent weight that
        increases as more of the day is observed.

        This is effectively a 1D Kalman filter update on log daily intensity.

        Args:
            cum_so_far: Tweets observed so far today
            tau: Minutes since noon
            contract_date: Today's contract date

        Returns:
            Adjustment factor (1.0 = no adjustment, >1 = boost, <1 = reduce)
        """
        # Get config parameters
        F_min = self.mc_config.regime_adj_f_min
        sigma_prior = self.mc_config.regime_adj_sigma_prior
        sigma0 = self.mc_config.regime_adj_sigma0
        adj_min = self.mc_config.regime_adj_min
        adj_max = self.mc_config.regime_adj_max
        tau_gate = self.mc_config.regime_adj_tau_gate
        F_gate = self.mc_config.regime_adj_f_gate

        # Gate: need enough of day observed
        if tau < tau_gate:
            return 1.0

        # Get expected progress
        is_weekend = self.contract_utils.is_weekend(contract_date)
        F_tau = self.nowcast.progress_curve.get_expected_progress(tau, is_weekend)

        if F_tau < F_gate:
            return 1.0

        # Ensure F_tau is above minimum to avoid division issues
        F_tau = max(F_tau, F_min)

        # Compute implied full-day count from partial observation
        implied_today = cum_so_far / F_tau

        # Prior from interday regime (in log space)
        lambda_prior = max(self.interday.regime.current_intensity, 1e-6)
        m_prior = np.log(lambda_prior + 1.0)

        # Observation in log space
        y_obs = np.log(implied_today + 1.0)

        # Compute variances
        sigma_prior2 = sigma_prior ** 2
        sigma_obs2 = (sigma0 ** 2) / F_tau

        # Kalman gain (weight on observation)
        w = sigma_prior2 / (sigma_prior2 + sigma_obs2)

        # Posterior update
        m_post = (1.0 - w) * m_prior + w * y_obs
        lambda_post = np.exp(m_post) - 1.0
        lambda_post = max(lambda_post, 1e-6)

        # Adjustment factor
        adjustment = lambda_post / lambda_prior

        # Clamp to prevent extreme adjustments
        adjustment = max(adj_min, min(adj_max, adjustment))

        return adjustment

    def simulate_horizon(
        self,
        events: List[TweetEvent],
        contract_date: date,
        now: datetime,
        horizon: int = 7,
        n_simulations: Optional[int] = None,
    ) -> ForecastResult:
        """
        Run Monte Carlo simulation for N-day sum.

        Args:
            events: Today's events so far
            contract_date: Today's contract date
            now: Current timestamp
            horizon: Forecast horizon in days (1-7)
            n_simulations: Number of simulations (default from config)

        Returns:
            ForecastResult with distribution and bin probabilities
        """
        import time
        start_time = time.time()

        if n_simulations is None:
            n_simulations = self.mc_config.n_simulations

        # Initialize RNG
        seed = self.mc_config.random_seed
        rng = np.random.default_rng(seed)

        # Get today's nowcast
        today_mean, today_std = self.nowcast.predict(events, contract_date, now)
        cum_so_far = len([e for e in events if e.timestamp < now])

        # For horizon=1, just return today's forecast (no interday component)
        if horizon == 1:
            sums = np.zeros(n_simulations)
            for i in range(n_simulations):
                sums[i] = self._sample_today(today_mean, today_std, cum_so_far, rng)

            # Compute statistics
            mean = float(np.mean(sums))
            median = float(np.median(sums))
            std = float(np.std(sums))
            p5 = float(np.percentile(sums, 5))
            p25 = float(np.percentile(sums, 25))
            p75 = float(np.percentile(sums, 75))
            p95 = float(np.percentile(sums, 95))

            bin_probs = self._compute_bin_probabilities(sums)
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
                future_days_estimate=0.0,
                regime_adjustment=1.0,
                n_simulations=n_simulations,
                simulation_time_ms=elapsed_ms,
            )

            logger.debug(
                f"1-day simulation: mean={mean:.1f}, std={std:.1f}, time={elapsed_ms:.1f}ms"
            )

            return result

        # For horizon > 1, compute regime adjustment
        tau = self.contract_utils.get_tau(now, contract_date)
        regime_adjustment = self._compute_regime_adjustment(cum_so_far, tau, contract_date)

        # Future horizons: days 1 through (horizon-1)
        future_horizons = list(range(1, horizon))

        # Run simulations
        sums = np.zeros(n_simulations)

        for i in range(n_simulations):
            # Sample today
            today_sample = self._sample_today(today_mean, today_std, cum_so_far, rng)

            # Sample future days (with regime adjustment)
            future_samples = self._sample_future_days(contract_date, future_horizons, rng, regime_adjustment)

            # Sum
            sums[i] = today_sample + sum(future_samples)

        # Compute statistics
        mean = float(np.mean(sums))
        median = float(np.median(sums))
        std = float(np.std(sums))
        p5 = float(np.percentile(sums, 5))
        p25 = float(np.percentile(sums, 25))
        p75 = float(np.percentile(sums, 75))
        p95 = float(np.percentile(sums, 95))

        # Compute bin probabilities
        bin_probs = self._compute_bin_probabilities(sums)

        # Component breakdown
        future_params = self.interday.get_forecast_params(future_horizons, contract_date)
        future_means = [m for m, _ in future_params]

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
            future_days_estimate=sum(future_means) * regime_adjustment,
            regime_adjustment=regime_adjustment,
            n_simulations=n_simulations,
            simulation_time_ms=elapsed_ms,
        )

        logger.debug(
            f"{horizon}-day simulation: mean={mean:.1f}, std={std:.1f}, "
            f"regime_adj={regime_adjustment:.2f}, time={elapsed_ms:.1f}ms"
        )

        return result

    def simulate(
        self,
        events: List[TweetEvent],
        contract_date: date,
        now: datetime,
        n_simulations: Optional[int] = None,
    ) -> ForecastResult:
        """
        Run Monte Carlo simulation for 7-day sum.

        Args:
            events: Today's events so far
            contract_date: Today's contract date
            now: Current timestamp
            n_simulations: Number of simulations (default from config)

        Returns:
            ForecastResult with distribution and bin probabilities
        """
        return self.simulate_horizon(events, contract_date, now, horizon=7, n_simulations=n_simulations)

    def _compute_bin_probabilities(self, sums: np.ndarray) -> List[BinProbability]:
        """
        Compute probability for each bin.

        Args:
            sums: Array of simulated 7-day sums

        Returns:
            List of BinProbability objects
        """
        n = len(sums)
        probs = []

        for lower, upper in self.bins:
            count = np.sum((sums >= lower) & (sums <= upper))
            prob = count / n
            probs.append(BinProbability(lower=lower, upper=upper, probability=prob))

        return probs

    def simulate_at_tau(
        self,
        events: List[TweetEvent],
        contract_date: date,
        tau: int,
        n_simulations: Optional[int] = None,
    ) -> ForecastResult:
        """
        Run simulation at a specific τ value (for backtesting).

        Args:
            events: Full day's events (will be filtered to τ)
            contract_date: The contract date
            tau: Minutes since noon to simulate at
            n_simulations: Number of simulations

        Returns:
            ForecastResult
        """
        # Construct 'now' from τ
        start_dt, _ = self.contract_utils.get_contract_day_bounds(contract_date)
        now = start_dt + timedelta(minutes=tau)

        return self.simulate(events, contract_date, now, n_simulations)

    def get_bin_for_count(self, count: int) -> Optional[Tuple[int, int]]:
        """Get the bin that contains a given count."""
        for lower, upper in self.bins:
            if lower <= count <= upper:
                return (lower, upper)
        return None

    def get_bin_index(self, count: int) -> int:
        """Get the index of the bin containing a count (-1 if not found)."""
        for i, (lower, upper) in enumerate(self.bins):
            if lower <= count <= upper:
                return i
        return -1


def compute_log_score(
    forecast: ForecastResult,
    actual_count: int,
) -> float:
    """
    Compute log score for a forecast.

    Log score = log(P(actual bin))
    Higher is better, max is 0 (perfect prediction).

    Args:
        forecast: The forecast result
        actual_count: Actual 7-day sum

    Returns:
        Log score (negative, higher is better)
    """
    prob = forecast.get_bin_probability(actual_count)

    # Avoid log(0)
    prob = max(prob, 1e-10)

    return float(np.log(prob))


def compute_brier_score(
    forecast: ForecastResult,
    actual_count: int,
) -> float:
    """
    Compute Brier score for a forecast.

    Brier score = sum((p_i - y_i)^2) where y_i is 1 for actual bin, 0 otherwise.
    Lower is better, min is 0 (perfect prediction).

    Args:
        forecast: The forecast result
        actual_count: Actual 7-day sum

    Returns:
        Brier score (positive, lower is better)
    """
    score = 0.0

    for bp in forecast.bin_probabilities:
        actual = 1.0 if bp.lower <= actual_count <= bp.upper else 0.0
        score += (bp.probability - actual) ** 2

    return score


def compute_calibration_stats(
    forecasts: List[ForecastResult],
    actual_counts: List[int],
    confidence_levels: List[float] = None,
) -> Dict[str, float]:
    """
    Compute calibration statistics across multiple forecasts.

    Args:
        forecasts: List of forecasts
        actual_counts: List of actual 7-day sums
        confidence_levels: Confidence levels to check (default: [50, 80, 90, 95])

    Returns:
        Dict with calibration stats
    """
    if confidence_levels is None:
        confidence_levels = [50, 80, 90, 95]

    n = len(forecasts)
    if n == 0 or len(actual_counts) != n:
        return {}

    stats = {}

    # Check coverage at each confidence level
    for level in confidence_levels:
        lower_pct = (100 - level) / 2
        upper_pct = 100 - lower_pct

        covered = 0
        for forecast, actual in zip(forecasts, actual_counts):
            lower_bound = np.percentile([forecast.p5], lower_pct)  # Approximate
            upper_bound = np.percentile([forecast.p95], upper_pct)

            # More precise: use simulation quantiles
            # For simplicity, use p5/p95 for 90%, etc.
            if level == 90:
                if forecast.p5 <= actual <= forecast.p95:
                    covered += 1
            elif level == 50:
                if forecast.p25 <= actual <= forecast.p75:
                    covered += 1
            else:
                # Rough approximation
                margin = (100 - level) / 200  # e.g., 95% -> 0.025
                lower = forecast.mean - stats.get('avg_std', forecast.std) * 2
                upper = forecast.mean + stats.get('avg_std', forecast.std) * 2
                if lower <= actual <= upper:
                    covered += 1

        stats[f"coverage_{level}"] = covered / n

    # Mean absolute error
    errors = [abs(f.mean - a) for f, a in zip(forecasts, actual_counts)]
    stats["mae"] = float(np.mean(errors))

    # Root mean squared error
    stats["rmse"] = float(np.sqrt(np.mean([e**2 for e in errors])))

    # Average log score
    log_scores = [compute_log_score(f, a) for f, a in zip(forecasts, actual_counts)]
    stats["avg_log_score"] = float(np.mean(log_scores))

    # Average Brier score
    brier_scores = [compute_brier_score(f, a) for f, a in zip(forecasts, actual_counts)]
    stats["avg_brier_score"] = float(np.mean(brier_scores))

    return stats
