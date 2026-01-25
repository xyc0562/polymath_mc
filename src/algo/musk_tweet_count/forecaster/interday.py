"""
Interday regime model for forecasting future daily tweet counts.

Components:
- RegimeModel: EWMA-based latent intensity tracking with mean reversion
- DispersionEstimator: Rolling estimation of Negative Binomial k parameter
- WeekendEffect: Weekend adjustment factor estimation
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from .config import RegimeConfig, DispersionConfig, WeekendConfig
from .data import ContractDayUtils

logger = logging.getLogger(__name__)


@dataclass
class RegimeState:
    """Current state of the regime model."""

    log_intensity: float  # λ̂ (log scale)
    intensity: float  # exp(λ̂)
    last_update_date: Optional[date] = None


class RegimeModel:
    """
    EWMA-based regime model for latent posting intensity.

    Tracks λ̂_d = α * log(C_d + 1) + (1 - α) * λ̂_{d-1}
    with mean reversion for multi-day forecasts.
    """

    def __init__(self, config: RegimeConfig, contract_utils: ContractDayUtils):
        """
        Initialize regime model.

        Args:
            config: Regime configuration
            contract_utils: Contract-day utilities
        """
        self.config = config
        self.contract_utils = contract_utils

        # State
        self._log_intensity: float = 0.0
        self._long_term_mean: float = 0.0
        self._last_update_date: Optional[date] = None

        # Initialization flag
        self._initialized = False

    def initialize(self, historical_counts: Dict[date, int]) -> None:
        """
        Initialize regime state from historical data.

        Args:
            historical_counts: Dict mapping contract_date -> count
        """
        if not historical_counts:
            logger.warning("No historical data for regime initialization")
            self._log_intensity = np.log(50 + 1)  # Default
            self._long_term_mean = self._log_intensity
            self._initialized = True
            return

        # Sort by date
        sorted_dates = sorted(historical_counts.keys())

        # Use initialization window
        window_days = self.config.initialization_window_days
        recent_dates = sorted_dates[-window_days:] if len(sorted_dates) > window_days else sorted_dates

        # Compute log counts
        log_counts = [np.log(historical_counts[d] + 1) for d in recent_dates]

        # Initialize λ̂ as mean of recent log counts
        self._log_intensity = float(np.mean(log_counts))
        self._long_term_mean = self._log_intensity

        # Apply cap
        self._log_intensity = min(self._log_intensity, self.config.max_log_intensity)

        # Set last update date
        self._last_update_date = sorted_dates[-1]
        self._initialized = True

        logger.info(
            f"Initialized regime: λ̂={self._log_intensity:.3f}, "
            f"intensity={np.exp(self._log_intensity):.1f}, "
            f"from {len(recent_dates)} days"
        )

    def update(self, contract_date: date, count: int) -> None:
        """
        Update regime state with a new observation.

        Args:
            contract_date: The contract date
            count: Tweet count for that day
        """
        if not self._initialized:
            raise RuntimeError("Regime model not initialized")

        # Skip if already updated for this date
        if self._last_update_date and contract_date <= self._last_update_date:
            return

        # EWMA update
        log_count = np.log(count + 1)
        alpha = self.config.ewma_alpha

        self._log_intensity = alpha * log_count + (1 - alpha) * self._log_intensity

        # Apply intensity cap
        self._log_intensity = min(self._log_intensity, self.config.max_log_intensity)

        # Update long-term mean (slower adaptation)
        self._long_term_mean = 0.01 * log_count + 0.99 * self._long_term_mean

        self._last_update_date = contract_date

        logger.debug(
            f"Updated regime for {contract_date}: count={count}, "
            f"λ̂={self._log_intensity:.3f}, intensity={np.exp(self._log_intensity):.1f}"
        )

    def get_state(self) -> RegimeState:
        """Get current regime state."""
        return RegimeState(
            log_intensity=self._log_intensity,
            intensity=np.exp(self._log_intensity),
            last_update_date=self._last_update_date,
        )

    def forecast_intensity(self, horizon: int) -> float:
        """
        Forecast intensity for h days ahead.

        Applies mean reversion: λ_h = λ̂ + (1 - ρ)^h * (λ̂ - μ)
        which simplifies to exponential decay toward long-term mean.

        Args:
            horizon: Days ahead (1 = tomorrow)

        Returns:
            Forecasted log intensity
        """
        if not self._initialized:
            raise RuntimeError("Regime model not initialized")

        rho = self.config.mean_reversion_rate

        # Mean reversion: λ_h decays toward long-term mean
        decay = (1 - rho) ** horizon
        forecast_log = self._long_term_mean + decay * (self._log_intensity - self._long_term_mean)

        # Apply cap
        forecast_log = min(forecast_log, self.config.max_log_intensity)

        return forecast_log

    def forecast_intensities(self, horizons: List[int]) -> List[float]:
        """Forecast intensities for multiple horizons."""
        return [self.forecast_intensity(h) for h in horizons]

    @property
    def current_log_intensity(self) -> float:
        """Get current log intensity."""
        return self._log_intensity

    @property
    def current_intensity(self) -> float:
        """Get current intensity (exp scale)."""
        return np.exp(self._log_intensity)

    @property
    def long_term_mean(self) -> float:
        """Get long-term mean log intensity."""
        return self._long_term_mean


class DispersionEstimator:
    """
    Estimate Negative Binomial dispersion parameter k.

    k controls overdispersion: Var = μ + μ²/k
    - Large k → approaches Poisson (Var ≈ μ)
    - Small k → high overdispersion
    """

    def __init__(self, config: DispersionConfig, contract_utils: ContractDayUtils):
        """
        Initialize dispersion estimator.

        Args:
            config: Dispersion configuration
            contract_utils: Contract-day utilities
        """
        self.config = config
        self.contract_utils = contract_utils

        # Estimated k
        self._k: float = 2.0  # Default

    def estimate(self, historical_counts: Dict[date, int]) -> float:
        """
        Estimate k from historical count data.

        Uses method of moments: k = μ² / (Var - μ)

        Args:
            historical_counts: Dict mapping contract_date -> count

        Returns:
            Estimated k value
        """
        if not historical_counts:
            logger.warning("No data for dispersion estimation, using default k=2.0")
            return self._k

        # Get recent counts within window
        today = self.contract_utils.get_current_contract_date()
        window_start = today - timedelta(days=self.config.estimation_window_days)

        recent_counts = [
            count for d, count in historical_counts.items()
            if d >= window_start
        ]

        if len(recent_counts) < 10:
            logger.warning(f"Only {len(recent_counts)} days for k estimation, using default")
            return self._k

        counts = np.array(recent_counts)
        mean = np.mean(counts)
        variance = np.var(counts, ddof=1)  # Sample variance

        # Handle underdispersion or near-Poisson cases
        # Use buffer to avoid division instability
        if variance <= mean * self.config.underdispersion_buffer:
            # Underdispersed or Poisson-like: use large k
            logger.info(f"Underdispersed data (Var={variance:.1f} ≤ μ×{self.config.underdispersion_buffer}={mean * self.config.underdispersion_buffer:.1f}), using k=100")
            self._k = 100.0
            return self._k

        # Method of moments
        k = mean ** 2 / (variance - mean)

        # Apply floor
        k = max(k, self.config.min_k)

        self._k = k

        logger.info(
            f"Estimated dispersion: k={k:.2f} from {len(recent_counts)} days "
            f"(μ={mean:.1f}, Var={variance:.1f})"
        )

        return self._k

    @property
    def k(self) -> float:
        """Get current k estimate."""
        return self._k


class WeekendEffect:
    """
    Estimate weekend effect on posting intensity.

    Models multiplicative effect: E[C_weekend] = effect * E[C_weekday]
    """

    def __init__(self, config: WeekendConfig, contract_utils: ContractDayUtils):
        """
        Initialize weekend effect estimator.

        Args:
            config: Weekend configuration
            contract_utils: Contract-day utilities
        """
        self.config = config
        self.contract_utils = contract_utils

        # Estimated effect (multiplicative)
        self._effect: float = 1.0  # Default: no effect

    def estimate(self, historical_counts: Dict[date, int]) -> float:
        """
        Estimate weekend effect from historical data.

        Args:
            historical_counts: Dict mapping contract_date -> count

        Returns:
            Weekend effect (ratio of weekend to weekday mean)
        """
        if not historical_counts:
            logger.warning("No data for weekend effect estimation")
            return self._effect

        # Get recent counts within window
        today = self.contract_utils.get_current_contract_date()
        window_start = today - timedelta(days=self.config.estimation_window_days)

        weekday_counts = []
        weekend_counts = []

        for d, count in historical_counts.items():
            if d < window_start:
                continue

            if d.weekday() in self.config.weekend_days:
                weekend_counts.append(count)
            else:
                weekday_counts.append(count)

        if len(weekday_counts) < 5 or len(weekend_counts) < 2:
            logger.warning(
                f"Not enough data for weekend effect: "
                f"{len(weekday_counts)} weekdays, {len(weekend_counts)} weekends"
            )
            return self._effect

        weekday_mean = np.mean(weekday_counts)
        weekend_mean = np.mean(weekend_counts)

        if weekday_mean > 0:
            self._effect = weekend_mean / weekday_mean
        else:
            self._effect = 1.0

        logger.info(
            f"Estimated weekend effect: {self._effect:.3f} "
            f"(weekend μ={weekend_mean:.1f}, weekday μ={weekday_mean:.1f})"
        )

        return self._effect

    def get_effect(self, contract_date: date) -> float:
        """
        Get effect multiplier for a given date.

        Args:
            contract_date: The contract date

        Returns:
            Effect multiplier (1.0 for weekday, estimated for weekend)
        """
        if contract_date.weekday() in self.config.weekend_days:
            return self._effect
        return 1.0

    @property
    def effect(self) -> float:
        """Get estimated weekend effect."""
        return self._effect


class InterdayForecaster:
    """
    Combined interday forecasting model.

    Integrates regime model, dispersion estimation, and weekend effects
    to forecast future daily counts.
    """

    def __init__(
        self,
        regime_config: RegimeConfig,
        dispersion_config: DispersionConfig,
        weekend_config: WeekendConfig,
        contract_utils: ContractDayUtils,
    ):
        """
        Initialize interday forecaster.

        Args:
            regime_config: Regime model configuration
            dispersion_config: Dispersion configuration
            weekend_config: Weekend configuration
            contract_utils: Contract-day utilities
        """
        self.contract_utils = contract_utils

        # Components
        self.regime = RegimeModel(regime_config, contract_utils)
        self.dispersion = DispersionEstimator(dispersion_config, contract_utils)
        self.weekend = WeekendEffect(weekend_config, contract_utils)

        # Fitted flag
        self._fitted = False

    def fit(self, historical_counts: Dict[date, int]) -> None:
        """
        Fit all interday model components.

        Args:
            historical_counts: Dict mapping contract_date -> count
        """
        # Initialize regime
        self.regime.initialize(historical_counts)

        # Estimate dispersion
        self.dispersion.estimate(historical_counts)

        # Estimate weekend effect
        self.weekend.estimate(historical_counts)

        self._fitted = True

        logger.info("Interday forecaster fitted successfully")

    def update(self, contract_date: date, count: int) -> None:
        """
        Update model with new observation.

        Args:
            contract_date: The contract date
            count: Tweet count for that day
        """
        self.regime.update(contract_date, count)

    def forecast_day(
        self,
        horizon: int,
        base_date: Optional[date] = None,
    ) -> Tuple[float, float]:
        """
        Forecast count distribution for a single future day.

        Args:
            horizon: Days ahead (1 = tomorrow)
            base_date: Base date for weekend calculation (default: today)

        Returns:
            Tuple of (mean, k) for Negative Binomial distribution
        """
        if not self._fitted:
            raise RuntimeError("Interday forecaster not fitted")

        # Get base date
        if base_date is None:
            base_date = self.contract_utils.get_current_contract_date()

        # Forecast log intensity
        log_intensity = self.regime.forecast_intensity(horizon)

        # Convert to mean
        mean = np.exp(log_intensity)

        # Apply weekend effect
        target_date = base_date + timedelta(days=horizon)
        weekend_effect = self.weekend.get_effect(target_date)
        mean *= weekend_effect

        return mean, self.dispersion.k

    def sample_day(
        self,
        horizon: int,
        base_date: Optional[date] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """
        Sample a count for a single future day.

        Args:
            horizon: Days ahead
            base_date: Base date for calculation
            rng: Random number generator

        Returns:
            Sampled count
        """
        if rng is None:
            rng = np.random.default_rng()

        mean, k = self.forecast_day(horizon, base_date)

        # Negative Binomial parameterization
        # scipy uses (n, p) where n = k, p = k / (k + μ)
        p = k / (k + mean)

        # Sample
        count = rng.negative_binomial(k, p)

        return int(count)

    def sample_trajectory(
        self,
        horizons: List[int],
        base_date: Optional[date] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> List[int]:
        """
        Sample counts for multiple future days.

        Args:
            horizons: List of horizons (e.g., [1, 2, 3, 4, 5, 6])
            base_date: Base date for calculation
            rng: Random number generator

        Returns:
            List of sampled counts
        """
        if rng is None:
            rng = np.random.default_rng()

        return [self.sample_day(h, base_date, rng) for h in horizons]

    def get_forecast_params(
        self,
        horizons: List[int],
        base_date: Optional[date] = None,
    ) -> List[Tuple[float, float]]:
        """
        Get forecast parameters for multiple horizons.

        Args:
            horizons: List of horizons
            base_date: Base date

        Returns:
            List of (mean, k) tuples
        """
        return [self.forecast_day(h, base_date) for h in horizons]

    @property
    def current_regime_state(self) -> RegimeState:
        """Get current regime state."""
        return self.regime.get_state()
