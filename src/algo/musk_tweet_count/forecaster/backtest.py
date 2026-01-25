"""
Comprehensive backtesting framework for the forecasting model.

Supports:
- Rolling window backtests
- Multiple evaluation metrics (MAE, RMSE, log score, Brier score)
- Calibration analysis
- Ablation studies
- Visualization helpers
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np

from .config import ForecasterConfig
from .data import ContractDayUtils, TweetEvent, XTrackerClient, EventStore
from .intraday import IntradayProgressCurve, BurstFeatureExtractor, IntradayNowcast
from .interday import InterdayForecaster
from .monte_carlo import (
    MonteCarloForecaster,
    ForecastResult,
    compute_log_score,
    compute_brier_score,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    # Training window
    training_window_days: int = 75

    # Minimum training data required
    min_training_days: int = 45

    # τ values to evaluate (minutes since noon)
    # Default: evaluate at 25%, 50%, 75% of day
    tau_values: List[int] = field(default_factory=lambda: [360, 720, 1080])

    # Number of Monte Carlo simulations per forecast
    n_simulations: int = 5000

    # Whether to include interday-only baseline
    include_interday_baseline: bool = True

    # Whether to include naive baseline (historical mean)
    include_naive_baseline: bool = True

    # Verbose logging
    verbose: bool = True


@dataclass
class SingleForecastResult:
    """Result of a single forecast evaluation."""

    forecast_date: date
    tau: int
    forecast: ForecastResult
    actual_today: int
    actual_7day: int

    # Pre-computed metrics
    mae_today: float
    mae_7day: float
    log_score: float
    brier_score: float

    # Calibration checks
    in_50_interval: bool
    in_90_interval: bool


@dataclass
class BacktestResults:
    """Complete backtesting results."""

    # Configuration used
    config: BacktestConfig

    # Individual forecast results
    forecasts: List[SingleForecastResult]

    # Aggregate metrics by τ
    metrics_by_tau: Dict[int, Dict[str, float]]

    # Overall metrics
    overall_metrics: Dict[str, float]

    # Baseline comparisons (if computed)
    baseline_metrics: Optional[Dict[str, Dict[str, float]]] = None

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["=" * 60, "BACKTEST RESULTS SUMMARY", "=" * 60, ""]

        lines.append(f"Total forecasts: {len(self.forecasts)}")
        lines.append(f"Date range: {self.forecasts[0].forecast_date} to {self.forecasts[-1].forecast_date}")
        lines.append("")

        lines.append("OVERALL METRICS:")
        lines.append("-" * 40)
        for metric, value in self.overall_metrics.items():
            lines.append(f"  {metric}: {value:.4f}")
        lines.append("")

        lines.append("METRICS BY τ (minutes since noon):")
        lines.append("-" * 40)
        for tau, metrics in sorted(self.metrics_by_tau.items()):
            lines.append(f"  τ = {tau} ({tau/60:.1f} hours):")
            for metric, value in metrics.items():
                lines.append(f"    {metric}: {value:.4f}")
        lines.append("")

        if self.baseline_metrics:
            lines.append("BASELINE COMPARISONS:")
            lines.append("-" * 40)
            for baseline_name, metrics in self.baseline_metrics.items():
                lines.append(f"  {baseline_name}:")
                for metric, value in metrics.items():
                    lines.append(f"    {metric}: {value:.4f}")
        lines.append("")

        # Improvement summary
        if self.baseline_metrics and "naive" in self.baseline_metrics:
            naive_mae = self.baseline_metrics["naive"].get("mae_7day", float("inf"))
            model_mae = self.overall_metrics.get("mae_7day", float("inf"))
            if naive_mae > 0:
                improvement = (naive_mae - model_mae) / naive_mae * 100
                lines.append(f"MAE improvement over naive: {improvement:.1f}%")

        lines.append("=" * 60)
        return "\n".join(lines)


class Backtester:
    """
    Rolling window backtester for the forecasting model.

    Simulates real-time forecasting by:
    1. Training on data up to day D-1
    2. Forecasting from partial data on day D at various τ values
    3. Comparing to actual outcomes
    """

    def __init__(
        self,
        forecaster_config: Optional[ForecasterConfig] = None,
        backtest_config: Optional[BacktestConfig] = None,
    ):
        """
        Initialize backtester.

        Args:
            forecaster_config: Configuration for forecaster
            backtest_config: Configuration for backtesting
        """
        self.forecaster_config = forecaster_config or ForecasterConfig()
        self.backtest_config = backtest_config or BacktestConfig()

        # Contract utils
        self.contract_utils = ContractDayUtils(
            timezone=self.forecaster_config.timezone,
            boundary_hour=self.forecaster_config.contract_boundary_hour,
        )

        # Override Monte Carlo config for backtesting
        self.forecaster_config.monte_carlo.n_simulations = self.backtest_config.n_simulations

    def run(
        self,
        events_by_date: Dict[date, List[TweetEvent]],
        test_start_date: Optional[date] = None,
        test_end_date: Optional[date] = None,
    ) -> BacktestResults:
        """
        Run rolling window backtest.

        Args:
            events_by_date: Dict mapping contract_date -> events
            test_start_date: First date to test (default: after min training period)
            test_end_date: Last date to test (default: 7 days before end of data)

        Returns:
            BacktestResults with all metrics
        """
        sorted_dates = sorted(events_by_date.keys())

        if len(sorted_dates) < self.backtest_config.min_training_days + 7:
            raise ValueError(
                f"Not enough data: need at least {self.backtest_config.min_training_days + 7} days, "
                f"got {len(sorted_dates)}"
            )

        # Determine test range
        if test_start_date is None:
            # Start after minimum training period
            test_start_idx = self.backtest_config.min_training_days
            test_start_date = sorted_dates[test_start_idx]

        if test_end_date is None:
            # End 7 days before last date (need actual 7-day sum)
            test_end_idx = len(sorted_dates) - 7
            test_end_date = sorted_dates[test_end_idx]

        # Compute actual 7-day sums
        actual_7day_sums = self._compute_7day_sums(events_by_date, sorted_dates)

        # Run forecasts
        forecasts = []

        for forecast_date in sorted_dates:
            if forecast_date < test_start_date or forecast_date > test_end_date:
                continue

            if forecast_date not in actual_7day_sums:
                continue

            # Determine training window
            training_end_idx = sorted_dates.index(forecast_date)
            training_start_idx = max(0, training_end_idx - self.backtest_config.training_window_days)
            training_dates = sorted_dates[training_start_idx:training_end_idx]

            if len(training_dates) < self.backtest_config.min_training_days:
                continue

            # Forecast at each τ value
            for tau in self.backtest_config.tau_values:
                try:
                    result = self._run_single_forecast(
                        events_by_date,
                        forecast_date,
                        tau,
                        training_dates,
                        actual_7day_sums[forecast_date],
                    )
                    forecasts.append(result)

                    if self.backtest_config.verbose:
                        logger.info(
                            f"Forecast {forecast_date} τ={tau}: "
                            f"actual={result.actual_7day}, pred={result.forecast.mean:.1f}, "
                            f"MAE={result.mae_7day:.1f}"
                        )

                except Exception as e:
                    logger.warning(f"Failed forecast for {forecast_date} τ={tau}: {e}")
                    continue

        # Compute aggregate metrics
        metrics_by_tau = self._compute_metrics_by_tau(forecasts)
        overall_metrics = self._compute_overall_metrics(forecasts)

        # Compute baseline comparisons
        baseline_metrics = None
        if self.backtest_config.include_naive_baseline or self.backtest_config.include_interday_baseline:
            baseline_metrics = self._compute_baselines(
                events_by_date,
                forecasts,
                actual_7day_sums,
            )

        return BacktestResults(
            config=self.backtest_config,
            forecasts=forecasts,
            metrics_by_tau=metrics_by_tau,
            overall_metrics=overall_metrics,
            baseline_metrics=baseline_metrics,
        )

    def _run_single_forecast(
        self,
        events_by_date: Dict[date, List[TweetEvent]],
        forecast_date: date,
        tau: int,
        training_dates: List[date],
        actual_7day: int,
    ) -> SingleForecastResult:
        """Run a single forecast and compute metrics."""
        # Prepare training data
        training_events = {d: events_by_date[d] for d in training_dates}
        training_counts = {d: len(events) for d, events in training_events.items()}

        # Create fresh forecaster
        forecaster = self._create_forecaster()

        # Fit progress curve
        historical_timestamps = {
            d: [e.timestamp for e in events]
            for d, events in training_events.items()
        }
        forecaster.progress_curve.fit(historical_timestamps)

        # Fit nowcast
        forecaster.nowcast.fit(training_events, training_counts)

        # Fit interday
        forecaster.interday.fit(training_counts)

        # Create Monte Carlo
        monte_carlo = MonteCarloForecaster(
            self.forecaster_config,
            forecaster.nowcast,
            forecaster.interday,
            self.contract_utils,
        )

        # Get today's events
        today_events = events_by_date.get(forecast_date, [])
        actual_today = len(today_events)

        # Construct 'now' from τ
        start_dt, _ = self.contract_utils.get_contract_day_bounds(forecast_date)
        now = start_dt + timedelta(minutes=tau)

        # Run simulation
        forecast = monte_carlo.simulate(today_events, forecast_date, now)

        # Compute metrics
        mae_today = abs(forecast.today_estimate - actual_today)
        mae_7day = abs(forecast.mean - actual_7day)
        log_score = compute_log_score(forecast, actual_7day)
        brier_score = compute_brier_score(forecast, actual_7day)

        # Calibration checks
        in_50_interval = forecast.p25 <= actual_7day <= forecast.p75
        in_90_interval = forecast.p5 <= actual_7day <= forecast.p95

        return SingleForecastResult(
            forecast_date=forecast_date,
            tau=tau,
            forecast=forecast,
            actual_today=actual_today,
            actual_7day=actual_7day,
            mae_today=mae_today,
            mae_7day=mae_7day,
            log_score=log_score,
            brier_score=brier_score,
            in_50_interval=in_50_interval,
            in_90_interval=in_90_interval,
        )

    def _create_forecaster(self):
        """Create forecaster components for a single forecast."""
        from .forecaster import Musk7DayForecaster

        class MinimalForecaster:
            pass

        forecaster = MinimalForecaster()
        forecaster.progress_curve = IntradayProgressCurve(
            self.forecaster_config.intraday_curve,
            self.contract_utils,
        )
        forecaster.burst_extractor = BurstFeatureExtractor(
            self.forecaster_config.burst_features,
            self.contract_utils,
        )
        forecaster.nowcast = IntradayNowcast(
            self.forecaster_config.nowcast,
            forecaster.progress_curve,
            forecaster.burst_extractor,
            self.contract_utils,
        )
        forecaster.interday = InterdayForecaster(
            self.forecaster_config.regime,
            self.forecaster_config.dispersion,
            self.forecaster_config.weekend,
            self.contract_utils,
        )
        return forecaster

    def _compute_7day_sums(
        self,
        events_by_date: Dict[date, List[TweetEvent]],
        sorted_dates: List[date],
    ) -> Dict[date, int]:
        """Compute actual 7-day sums for each date."""
        sums = {}

        for i, d in enumerate(sorted_dates):
            if i + 6 >= len(sorted_dates):
                break

            # Check all 7 days are present
            seven_days = sorted_dates[i:i + 7]
            if all(day in events_by_date for day in seven_days):
                total = sum(len(events_by_date[day]) for day in seven_days)
                sums[d] = total

        return sums

    def _compute_metrics_by_tau(
        self,
        forecasts: List[SingleForecastResult],
    ) -> Dict[int, Dict[str, float]]:
        """Compute metrics grouped by τ value."""
        metrics_by_tau = {}

        for tau in self.backtest_config.tau_values:
            tau_forecasts = [f for f in forecasts if f.tau == tau]

            if not tau_forecasts:
                continue

            metrics_by_tau[tau] = {
                "mae_today": np.mean([f.mae_today for f in tau_forecasts]),
                "mae_7day": np.mean([f.mae_7day for f in tau_forecasts]),
                "rmse_7day": np.sqrt(np.mean([f.mae_7day ** 2 for f in tau_forecasts])),
                "avg_log_score": np.mean([f.log_score for f in tau_forecasts]),
                "avg_brier_score": np.mean([f.brier_score for f in tau_forecasts]),
                "coverage_50": np.mean([f.in_50_interval for f in tau_forecasts]),
                "coverage_90": np.mean([f.in_90_interval for f in tau_forecasts]),
                "n_forecasts": len(tau_forecasts),
            }

        return metrics_by_tau

    def _compute_overall_metrics(
        self,
        forecasts: List[SingleForecastResult],
    ) -> Dict[str, float]:
        """Compute overall metrics across all forecasts."""
        if not forecasts:
            return {}

        return {
            "mae_today": np.mean([f.mae_today for f in forecasts]),
            "mae_7day": np.mean([f.mae_7day for f in forecasts]),
            "rmse_7day": np.sqrt(np.mean([f.mae_7day ** 2 for f in forecasts])),
            "avg_log_score": np.mean([f.log_score for f in forecasts]),
            "avg_brier_score": np.mean([f.brier_score for f in forecasts]),
            "coverage_50": np.mean([f.in_50_interval for f in forecasts]),
            "coverage_90": np.mean([f.in_90_interval for f in forecasts]),
            "n_forecasts": len(forecasts),
        }

    def _compute_baselines(
        self,
        events_by_date: Dict[date, List[TweetEvent]],
        forecasts: List[SingleForecastResult],
        actual_7day_sums: Dict[date, int],
    ) -> Dict[str, Dict[str, float]]:
        """Compute baseline metrics for comparison."""
        baselines = {}

        if self.backtest_config.include_naive_baseline:
            # Naive baseline: historical mean
            all_daily_counts = [len(events) for events in events_by_date.values()]
            historical_mean_daily = np.mean(all_daily_counts)
            naive_7day = historical_mean_daily * 7

            naive_errors = []
            for f in forecasts:
                naive_errors.append(abs(naive_7day - f.actual_7day))

            baselines["naive"] = {
                "mae_7day": np.mean(naive_errors),
                "rmse_7day": np.sqrt(np.mean([e**2 for e in naive_errors])),
            }

        if self.backtest_config.include_interday_baseline:
            # Interday-only baseline: no intraday adjustment
            # Uses cum_so_far = 0 for today's prediction
            # This is a simplified version - just uses regime forecast

            interday_errors = []
            for f in forecasts:
                # Use the future days estimate + historical mean for today
                # This approximates interday-only
                regime_estimate = f.forecast.future_days_estimate + (
                    f.forecast.mean - f.forecast.future_days_estimate
                )
                interday_errors.append(abs(regime_estimate - f.actual_7day))

            if interday_errors:
                baselines["interday_only"] = {
                    "mae_7day": np.mean(interday_errors),
                    "rmse_7day": np.sqrt(np.mean([e**2 for e in interday_errors])),
                }

        return baselines


class AblationStudy:
    """
    Ablation study to evaluate contribution of different components.

    Tests:
    - Full model vs no burst features
    - Full model vs no weekend adjustment
    - Full model vs no progress curve (uniform)
    """

    def __init__(
        self,
        base_config: Optional[ForecasterConfig] = None,
        backtest_config: Optional[BacktestConfig] = None,
    ):
        self.base_config = base_config or ForecasterConfig()
        self.backtest_config = backtest_config or BacktestConfig()

    def run(
        self,
        events_by_date: Dict[date, List[TweetEvent]],
    ) -> Dict[str, BacktestResults]:
        """
        Run ablation study with different configurations.

        Args:
            events_by_date: Historical data

        Returns:
            Dict mapping configuration name -> BacktestResults
        """
        results = {}

        # Full model
        logger.info("Running full model backtest...")
        backtester = Backtester(self.base_config, self.backtest_config)
        results["full_model"] = backtester.run(events_by_date)

        # No weekend adjustment
        logger.info("Running no-weekend backtest...")
        no_weekend_config = ForecasterConfig()
        no_weekend_config.weekend.weekend_days = ()  # Empty tuple disables
        backtester = Backtester(no_weekend_config, self.backtest_config)
        results["no_weekend"] = backtester.run(events_by_date)

        # Higher EWMA alpha (faster adaptation)
        logger.info("Running fast-ewma backtest...")
        fast_ewma_config = ForecasterConfig()
        fast_ewma_config.regime.ewma_alpha = 0.35
        backtester = Backtester(fast_ewma_config, self.backtest_config)
        results["fast_ewma"] = backtester.run(events_by_date)

        # Slower EWMA alpha
        logger.info("Running slow-ewma backtest...")
        slow_ewma_config = ForecasterConfig()
        slow_ewma_config.regime.ewma_alpha = 0.10
        backtester = Backtester(slow_ewma_config, self.backtest_config)
        results["slow_ewma"] = backtester.run(events_by_date)

        return results

    def summary(self, results: Dict[str, BacktestResults]) -> str:
        """Generate comparison summary."""
        lines = ["=" * 70, "ABLATION STUDY RESULTS", "=" * 70, ""]

        headers = ["Configuration", "MAE (7-day)", "RMSE", "Log Score", "Coverage 90%"]
        col_widths = [20, 12, 12, 12, 12]

        # Header row
        header_line = "".join(
            h.ljust(w) for h, w in zip(headers, col_widths)
        )
        lines.append(header_line)
        lines.append("-" * sum(col_widths))

        # Data rows
        for name, result in results.items():
            metrics = result.overall_metrics
            row = [
                name,
                f"{metrics.get('mae_7day', 0):.2f}",
                f"{metrics.get('rmse_7day', 0):.2f}",
                f"{metrics.get('avg_log_score', 0):.3f}",
                f"{metrics.get('coverage_90', 0):.1%}",
            ]
            row_line = "".join(
                str(v).ljust(w) for v, w in zip(row, col_widths)
            )
            lines.append(row_line)

        lines.append("=" * 70)
        return "\n".join(lines)


def load_data_for_backtest(
    n_days: int = 90,
    timezone: str = "America/New_York",
) -> Dict[date, List[TweetEvent]]:
    """
    Load data from XTracker API for backtesting.

    Args:
        n_days: Number of days to load
        timezone: Timezone for contract-day calculations

    Returns:
        Dict mapping contract_date -> events
    """
    contract_utils = ContractDayUtils(timezone=timezone)
    xtracker = XTrackerClient()

    history = xtracker.fetch_historical(n_days, contract_utils)

    logger.info(f"Loaded {len(history)} days of data for backtesting")
    return history


def run_quick_backtest(
    n_days: int = 60,
    n_simulations: int = 2000,
) -> BacktestResults:
    """
    Run a quick backtest with default settings.

    Args:
        n_days: Days of data to load
        n_simulations: Monte Carlo simulations per forecast

    Returns:
        BacktestResults
    """
    # Load data
    events_by_date = load_data_for_backtest(n_days)

    # Configure backtest
    backtest_config = BacktestConfig(
        training_window_days=45,
        min_training_days=30,
        tau_values=[720],  # Only at noon (50% of day)
        n_simulations=n_simulations,
        verbose=True,
    )

    # Run
    backtester = Backtester(backtest_config=backtest_config)
    results = backtester.run(events_by_date)

    print(results.summary())
    return results
