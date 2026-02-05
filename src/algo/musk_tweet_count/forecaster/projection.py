"""
Projection models for computing bin probabilities from forecasts.

This module provides an abstraction for converting forecast distributions
into bin probabilities. Different projection models handle this differently:

- AsymptoticProjection: Uses actual Monte Carlo samples (preserves asymmetry)
- NormalProjection: Approximates with Normal distribution (symmetric)

The choice affects how probability mass is distributed across bins,
particularly in the tails.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .monte_carlo import BinProbability, ForecastResult


class ProjectionModel(ABC):
    """
    Abstract base class for projection models.

    A projection model computes bin probabilities from a forecast result,
    optionally shifted by a constant (e.g., past_count from completed days).
    """

    @abstractmethod
    def compute_bin_probabilities(
        self,
        forecast: ForecastResult,
        bins: List[Tuple[int, int]],
        shift: int = 0,
        floor: Optional[int] = None,
    ) -> List[float]:
        """
        Compute probability for each bin.

        Args:
            forecast: The forecast result containing distribution info
            bins: List of (lower, upper) bin boundaries
            shift: Amount to shift the distribution (e.g., past_count)
            floor: Optional floor value (samples are max'd with this)

        Returns:
            List of probabilities, one per bin (same order as bins)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        pass


class AsymmetricProjection(ProjectionModel):
    """
    Projection using actual Monte Carlo samples.

    This preserves the asymmetric shape of the underlying distribution
    (Log-normal + Negative Binomial sum), which is typically right-skewed.

    Requires forecast.samples to be populated.
    """

    def __init__(self, n_fallback_samples: int = 10000):
        """
        Args:
            n_fallback_samples: Number of samples for fallback (if needed)
        """
        self._n_fallback_samples = n_fallback_samples

    @property
    def name(self) -> str:
        return "asymmetric"

    def compute_bin_probabilities(
        self,
        forecast: ForecastResult,
        bins: List[Tuple[int, int]],
        shift: int = 0,
        floor: Optional[int] = None,
    ) -> List[float]:
        """
        Compute bin probabilities from actual MC samples.

        Raises:
            RuntimeError: If forecast.samples is None
        """
        if forecast.samples is None:
            raise RuntimeError(
                "AsymmetricProjection requires forecast.samples to be populated. "
                "Ensure simulate_horizon is called with return_samples=True."
            )

        # Shift samples by past_count
        samples = forecast.samples + shift

        # Apply floor if specified (samples can't be below current count)
        if floor is not None:
            samples = np.maximum(samples, floor)

        # Compute probability for each bin
        n = len(samples)
        probs = []

        for lower, upper in bins:
            count = np.sum((samples >= lower) & (samples <= upper))
            prob = count / n
            probs.append(float(prob))

        return probs


class NormalProjection(ProjectionModel):
    """
    Projection using analytic Normal CDF.

    This approximates the forecast distribution as Normal(mean, std),
    which is symmetric around the mean. Uses the exact CDF formula
    rather than sampling for precision and speed.

    Characteristics:
    - Same mean and variance as the actual distribution
    - Symmetric tails (equal probability mass above/below mean)
    - May overestimate probability in lower tail bins
    - May underestimate probability in upper tail bins (for right-skewed actual)

    Formula: P(lower <= X <= upper) = Φ((upper - μ) / σ) - Φ((lower - μ) / σ)
    """

    @property
    def name(self) -> str:
        return "normal"

    def compute_bin_probabilities(
        self,
        forecast: ForecastResult,
        bins: List[Tuple[int, int]],
        shift: int = 0,
        floor: Optional[int] = None,
    ) -> List[float]:
        """
        Compute bin probabilities using Normal CDF.
        """
        from scipy.stats import norm

        mean = forecast.mean + shift
        std = forecast.std

        if std <= 0:
            # Degenerate case: all probability at mean
            probs = []
            for lower, upper in bins:
                if lower <= mean <= upper:
                    probs.append(1.0)
                else:
                    probs.append(0.0)
            return probs

        # Handle floor (truncated normal)
        # P(X in bin | X >= floor) = P(X in bin ∩ X >= floor) / P(X >= floor)
        if floor is not None and floor > mean - 3 * std:
            p_above_floor = 1.0 - norm.cdf(floor, mean, std)
            if p_above_floor < 1e-10:
                # Everything is below floor - uniform over bins above floor
                n_live = sum(1 for l, u in bins if u >= floor)
                probs = []
                for lower, upper in bins:
                    if upper >= floor:
                        probs.append(1.0 / n_live if n_live > 0 else 0.0)
                    else:
                        probs.append(0.0)
                return probs
        else:
            p_above_floor = 1.0

        probs = []
        for lower, upper in bins:
            # Apply floor to bin boundaries
            effective_lower = lower
            if floor is not None:
                effective_lower = max(lower, floor)

            if effective_lower > upper:
                # Bin is entirely below floor
                probs.append(0.0)
            else:
                # P(effective_lower <= X <= upper)
                # +0.5/-0.5 for continuity correction (discrete bins)
                p_upper = norm.cdf(upper + 0.5, mean, std)
                p_lower = norm.cdf(effective_lower - 0.5, mean, std)
                raw_prob = p_upper - p_lower

                # Normalize by P(X >= floor) for truncated distribution
                if floor is not None:
                    raw_prob = raw_prob / p_above_floor

                probs.append(float(max(0.0, raw_prob)))

        return probs


class SkewNormalProjection(ProjectionModel):
    """
    Projection using Skew-Normal distribution.

    Extends Normal with a skewness parameter to capture the residual
    right-skew in the 7-day sum (~0.4 typically).

    The skew-normal has PDF: f(x) = 2φ(x)Φ(αx)
    where φ is Normal PDF, Φ is Normal CDF, α is skewness parameter.

    Uses scipy.stats.skewnorm for CDF calculations.
    """

    def __init__(self, skewness: float = 0.43):
        """
        Args:
            skewness: Target skewness (default 0.43 based on empirical observation)
        """
        self._target_skewness = skewness

    @property
    def name(self) -> str:
        return "skew_normal"

    def compute_bin_probabilities(
        self,
        forecast: ForecastResult,
        bins: List[Tuple[int, int]],
        shift: int = 0,
        floor: Optional[int] = None,
    ) -> List[float]:
        """
        Compute bin probabilities using Skew-Normal CDF.
        """
        from scipy.stats import skewnorm

        mean = forecast.mean + shift
        std = forecast.std

        if std <= 0:
            probs = []
            for lower, upper in bins:
                if lower <= mean <= upper:
                    probs.append(1.0)
                else:
                    probs.append(0.0)
            return probs

        # Convert target skewness to skew-normal shape parameter α
        # Skewness of skew-normal: γ = (4-π)/2 * (δ√(2/π))³ / (1 - 2δ²/π)^(3/2)
        # where δ = α / √(1 + α²)
        # For small skewness, α ≈ skewness * √(2π) / (√(4-π))
        # Approximate: α ≈ 1.5 * skewness for skewness in [0, 1]
        alpha = 1.5 * self._target_skewness

        # Skew-normal location and scale (not same as mean/std)
        # Mean = ξ + ω*δ*√(2/π), Var = ω²(1 - 2δ²/π)
        # Solve for ξ, ω given mean, std, α
        delta = alpha / np.sqrt(1 + alpha**2)
        omega = std / np.sqrt(1 - 2 * delta**2 / np.pi)
        xi = mean - omega * delta * np.sqrt(2 / np.pi)

        # Handle floor via truncation
        if floor is not None and floor > mean - 3 * std:
            p_above_floor = 1.0 - skewnorm.cdf(floor, alpha, loc=xi, scale=omega)
            if p_above_floor < 1e-10:
                n_live = sum(1 for l, u in bins if u >= floor)
                probs = []
                for lower, upper in bins:
                    if upper >= floor:
                        probs.append(1.0 / n_live if n_live > 0 else 0.0)
                    else:
                        probs.append(0.0)
                return probs
        else:
            p_above_floor = 1.0

        probs = []
        for lower, upper in bins:
            effective_lower = lower
            if floor is not None:
                effective_lower = max(lower, floor)

            if effective_lower > upper:
                probs.append(0.0)
            else:
                p_upper = skewnorm.cdf(upper + 0.5, alpha, loc=xi, scale=omega)
                p_lower = skewnorm.cdf(effective_lower - 0.5, alpha, loc=xi, scale=omega)
                raw_prob = p_upper - p_lower

                if floor is not None:
                    raw_prob = raw_prob / p_above_floor

                probs.append(float(max(0.0, raw_prob)))

        return probs


class GammaProjection(ProjectionModel):
    """
    Projection using Gamma distribution.

    Gamma is natural for sums of positive random variables and is
    always right-skewed. Parameters are derived from mean and variance:
    - shape k = (mean/std)²
    - scale θ = std²/mean

    The Gamma distribution has support [0, ∞), which matches our
    constraint that tweet counts are non-negative.
    """

    @property
    def name(self) -> str:
        return "gamma"

    def compute_bin_probabilities(
        self,
        forecast: ForecastResult,
        bins: List[Tuple[int, int]],
        shift: int = 0,
        floor: Optional[int] = None,
    ) -> List[float]:
        """
        Compute bin probabilities using Gamma CDF.
        """
        from scipy.stats import gamma

        mean = forecast.mean + shift
        std = forecast.std

        if std <= 0 or mean <= 0:
            probs = []
            for lower, upper in bins:
                if lower <= mean <= upper:
                    probs.append(1.0)
                else:
                    probs.append(0.0)
            return probs

        # Gamma parameters from mean and variance
        # Mean = k*θ, Var = k*θ²
        # => k = mean²/var, θ = var/mean
        variance = std ** 2
        k = mean ** 2 / variance  # shape
        theta = variance / mean   # scale

        # Handle floor via truncation
        if floor is not None and floor > 0:
            p_above_floor = 1.0 - gamma.cdf(floor, a=k, scale=theta)
            if p_above_floor < 1e-10:
                n_live = sum(1 for l, u in bins if u >= floor)
                probs = []
                for lower, upper in bins:
                    if upper >= floor:
                        probs.append(1.0 / n_live if n_live > 0 else 0.0)
                    else:
                        probs.append(0.0)
                return probs
        else:
            p_above_floor = 1.0

        probs = []
        for lower, upper in bins:
            effective_lower = max(lower, 0)  # Gamma is non-negative
            if floor is not None:
                effective_lower = max(effective_lower, floor)

            if effective_lower > upper:
                probs.append(0.0)
            else:
                # +0.5/-0.5 for continuity correction
                p_upper = gamma.cdf(upper + 0.5, a=k, scale=theta)
                p_lower = gamma.cdf(max(0, effective_lower - 0.5), a=k, scale=theta)
                raw_prob = p_upper - p_lower

                if floor is not None:
                    raw_prob = raw_prob / p_above_floor

                probs.append(float(max(0.0, raw_prob)))

        return probs


@dataclass
class ProjectionResult:
    """Result of a projection computation."""

    probabilities: List[float]
    bin_probabilities: List[BinProbability]
    projection_model: str
    shift: int
    floor: Optional[int]

    def get_probability(self, bin_index: int) -> float:
        """Get probability for a specific bin by index."""
        if 0 <= bin_index < len(self.probabilities):
            return self.probabilities[bin_index]
        return 0.0


def create_projection_model(
    model_type: str = "asymmetric",
    **kwargs,
) -> ProjectionModel:
    """
    Factory function to create a projection model.

    Args:
        model_type: One of "asymmetric", "normal", "skew_normal", "gamma"
        **kwargs: Additional arguments for the specific model

    Returns:
        ProjectionModel instance
    """
    if model_type == "asymmetric":
        return AsymmetricProjection(**kwargs)
    elif model_type == "normal":
        return NormalProjection()
    elif model_type == "skew_normal":
        return SkewNormalProjection(**kwargs)
    elif model_type == "gamma":
        return GammaProjection()
    else:
        raise ValueError(f"Unknown projection model type: {model_type}")


def project_forecast(
    forecast: ForecastResult,
    bins: List[Tuple[int, int]],
    model: ProjectionModel,
    shift: int = 0,
    floor: Optional[int] = None,
) -> ProjectionResult:
    """
    Convenience function to project a forecast onto bins.

    Args:
        forecast: The forecast result
        bins: List of (lower, upper) bin boundaries
        model: The projection model to use
        shift: Amount to shift distribution (e.g., past_count)
        floor: Optional floor for samples

    Returns:
        ProjectionResult with probabilities and metadata
    """
    probs = model.compute_bin_probabilities(forecast, bins, shift, floor)

    # Create BinProbability objects
    bin_probs = [
        BinProbability(lower=lower, upper=upper, probability=prob)
        for (lower, upper), prob in zip(bins, probs)
    ]

    return ProjectionResult(
        probabilities=probs,
        bin_probabilities=bin_probs,
        projection_model=model.name,
        shift=shift,
        floor=floor,
    )
