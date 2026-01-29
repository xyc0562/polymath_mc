"""
Comprehensive backtesting framework for the forecasting model.

Supports:
- Rolling window backtests
- Multiple evaluation metrics (MAE, RMSE, log score, Brier score)
- Calibration analysis
- Ablation studies
- Visualization helpers
- Ensemble forecasting (combining fast/slow EWMA models)
"""

import copy
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np

from .config import ForecasterConfig
from .data import ContractDayUtils, TweetEvent, XTrackerClient, EventStore
from .intraday import IntradayProgressCurve, BurstFeatureExtractor, IntradayNowcast
from .interday import InterdayForecaster, GASInterdayForecaster
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

    # Forecast horizons to evaluate (in days)
    # 1 = today only, 7 = full 7-day period
    # Default: test all horizons
    horizons: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])

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
    horizon: int  # Forecast horizon in days (1-7)
    forecast: ForecastResult
    actual_today: int
    actual_horizon: int  # Actual count for the horizon period

    # Pre-computed metrics
    mae_today: float
    mae_horizon: float  # MAE for the horizon forecast
    log_score: float
    brier_score: float

    # Calibration checks
    in_50_interval: bool
    in_90_interval: bool

    # Backward compatibility
    @property
    def actual_7day(self) -> int:
        """Backward compatibility: return actual_horizon."""
        return self.actual_horizon

    @property
    def mae_7day(self) -> float:
        """Backward compatibility: return mae_horizon."""
        return self.mae_horizon


@dataclass
class BacktestResults:
    """Complete backtesting results."""

    # Configuration used
    config: BacktestConfig

    # Individual forecast results
    forecasts: List[SingleForecastResult]

    # Aggregate metrics by horizon
    metrics_by_horizon: Dict[int, Dict[str, float]]

    # Aggregate metrics by τ (for backward compatibility, horizon=7 only)
    metrics_by_tau: Dict[int, Dict[str, float]]

    # Overall metrics (for backward compatibility, horizon=7 only)
    overall_metrics: Dict[str, float]

    # Baseline comparisons (if computed)
    baseline_metrics: Optional[Dict[str, Dict[str, float]]] = None

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["=" * 60, "BACKTEST RESULTS SUMMARY", "=" * 60, ""]

        lines.append(f"Total forecasts: {len(self.forecasts)}")
        lines.append(f"Date range: {self.forecasts[0].forecast_date} to {self.forecasts[-1].forecast_date}")
        lines.append("")

        lines.append("METRICS BY HORIZON:")
        lines.append("-" * 40)
        for horizon in sorted(self.metrics_by_horizon.keys()):
            metrics = self.metrics_by_horizon[horizon]
            lines.append(f"  Horizon {horizon}d:")
            lines.append(f"    mae_horizon: {metrics.get('mae_horizon', 0):.4f}")
            lines.append(f"    rmse_horizon: {metrics.get('rmse_horizon', 0):.4f}")
            lines.append(f"    coverage_50: {metrics.get('coverage_50', 0):.4f}")
            lines.append(f"    coverage_90: {metrics.get('coverage_90', 0):.4f}")
            lines.append(f"    n_forecasts: {metrics.get('n_forecasts', 0):.0f}")
        lines.append("")

        lines.append("OVERALL METRICS (7-day horizon):")
        lines.append("-" * 40)
        for metric, value in self.overall_metrics.items():
            lines.append(f"  {metric}: {value:.4f}")
        lines.append("")

        lines.append("METRICS BY τ (7-day horizon only):")
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
        use_ensemble: bool = False,
        use_gas: bool = False,
    ):
        """
        Initialize backtester.

        Args:
            forecaster_config: Configuration for forecaster
            backtest_config: Configuration for backtesting
            use_ensemble: Whether to use ensemble forecasting (fast+slow EWMA)
            use_gas: Whether to use NB-GAS regime model instead of EWMA
        """
        self.forecaster_config = forecaster_config or ForecasterConfig()
        self.backtest_config = backtest_config or BacktestConfig()
        self.use_ensemble = use_ensemble
        self.use_gas = use_gas

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
            # End max_horizon days before last date
            max_horizon = max(self.backtest_config.horizons)
            test_end_idx = len(sorted_dates) - max_horizon
            test_end_date = sorted_dates[test_end_idx]

        # Compute actual N-day sums for each horizon
        actual_sums_by_horizon = {}
        for horizon in self.backtest_config.horizons:
            actual_sums_by_horizon[horizon] = self._compute_nday_sums(
                events_by_date, sorted_dates, horizon
            )

        # Run forecasts
        forecasts = []

        for forecast_date in sorted_dates:
            if forecast_date < test_start_date or forecast_date > test_end_date:
                continue

            # Determine training window
            training_end_idx = sorted_dates.index(forecast_date)
            training_start_idx = max(0, training_end_idx - self.backtest_config.training_window_days)
            training_dates = sorted_dates[training_start_idx:training_end_idx]

            if len(training_dates) < self.backtest_config.min_training_days:
                continue

            # Forecast at each horizon and τ value
            for horizon in self.backtest_config.horizons:
                # Skip if we don't have actual data for this horizon
                if forecast_date not in actual_sums_by_horizon[horizon]:
                    continue

                for tau in self.backtest_config.tau_values:
                    try:
                        result = self._run_single_forecast(
                            events_by_date,
                            forecast_date,
                            tau,
                            horizon,
                            training_dates,
                            actual_sums_by_horizon[horizon][forecast_date],
                            use_ensemble=self.use_ensemble,
                            use_gas=self.use_gas,
                        )
                        forecasts.append(result)

                        if self.backtest_config.verbose:
                            logger.debug(
                                f"Forecast {forecast_date} h={horizon}d τ={tau}: "
                                f"actual={result.actual_horizon}, pred={result.forecast.mean:.1f}, "
                                f"MAE={result.mae_horizon:.1f}"
                            )

                    except Exception as e:
                        logger.warning(f"Failed forecast for {forecast_date} h={horizon} τ={tau}: {e}")
                        continue

        # Compute aggregate metrics by horizon
        metrics_by_horizon = self._compute_metrics_by_horizon(forecasts)

        # For backward compatibility, compute τ and overall metrics for horizon=7 only
        forecasts_7d = [f for f in forecasts if f.horizon == 7]
        metrics_by_tau = self._compute_metrics_by_tau(forecasts_7d)
        overall_metrics = self._compute_overall_metrics(forecasts_7d)

        # Compute baseline comparisons (use 7-day actuals for baselines)
        baseline_metrics = None
        if self.backtest_config.include_naive_baseline or self.backtest_config.include_interday_baseline:
            actual_7day_sums = actual_sums_by_horizon.get(7, {})
            baseline_metrics = self._compute_baselines(
                events_by_date,
                forecasts,
                actual_7day_sums,
            )

        return BacktestResults(
            config=self.backtest_config,
            forecasts=forecasts,
            metrics_by_horizon=metrics_by_horizon,
            metrics_by_tau=metrics_by_tau,
            overall_metrics=overall_metrics,
            baseline_metrics=baseline_metrics,
        )

    def _run_single_forecast(
        self,
        events_by_date: Dict[date, List[TweetEvent]],
        forecast_date: date,
        tau: int,
        horizon: int,
        training_dates: List[date],
        actual_horizon: int,
        use_ensemble: bool = False,
        use_gas: bool = False,
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

        # Get today's events
        today_events = events_by_date.get(forecast_date, [])
        actual_today = len(today_events)

        # Construct 'now' from τ
        start_dt, _ = self.contract_utils.get_contract_day_bounds(forecast_date)
        now = start_dt + timedelta(minutes=tau)

        if use_gas:
            # GAS mode: use NB-GAS regime model
            gas_interday = GASInterdayForecaster(
                self.forecaster_config.gas,
                self.forecaster_config.dispersion,
                self.forecaster_config.weekend,
                self.contract_utils,
            )
            gas_interday.fit(training_counts)

            monte_carlo = MonteCarloForecaster(
                self.forecaster_config,
                forecaster.nowcast,
                gas_interday,
                self.contract_utils,
            )

            forecast = monte_carlo.simulate_horizon(
                today_events, forecast_date, now, horizon=horizon
            )
        elif use_ensemble:
            # Ensemble mode: combine fast and slow EWMA forecasters
            from .ensemble import EnsembleMonteCarloForecaster

            ensemble_cfg = self.forecaster_config.ensemble

            # Create slow interday forecaster
            config_slow = copy.deepcopy(self.forecaster_config)
            config_slow.regime.ewma_alpha = ensemble_cfg.alpha_slow
            interday_slow = InterdayForecaster(
                config_slow.regime,
                config_slow.dispersion,
                config_slow.weekend,
                self.contract_utils,
            )
            interday_slow.fit(training_counts)

            # Create fast interday forecaster
            config_fast = copy.deepcopy(self.forecaster_config)
            config_fast.regime.ewma_alpha = ensemble_cfg.alpha_fast
            interday_fast = InterdayForecaster(
                config_fast.regime,
                config_fast.dispersion,
                config_fast.weekend,
                self.contract_utils,
            )
            interday_fast.fit(training_counts)

            # Create ensemble forecaster
            ensemble_mc = EnsembleMonteCarloForecaster(
                self.forecaster_config,
                forecaster.nowcast,
                [interday_slow, interday_fast],
                self.contract_utils,
                weights=[ensemble_cfg.weight_slow, ensemble_cfg.weight_fast],
            )

            # Run simulation
            forecast = ensemble_mc.simulate_horizon(
                today_events, forecast_date, now, horizon=horizon
            )
        else:
            # Single forecaster mode
            forecaster.interday.fit(training_counts)

            # Create Monte Carlo
            monte_carlo = MonteCarloForecaster(
                self.forecaster_config,
                forecaster.nowcast,
                forecaster.interday,
                self.contract_utils,
            )

            # Run simulation with specified horizon
            forecast = monte_carlo.simulate_horizon(
                today_events, forecast_date, now, horizon=horizon
            )

        # Compute metrics
        mae_today = abs(forecast.today_estimate - actual_today)
        mae_horizon = abs(forecast.mean - actual_horizon)
        log_score = compute_log_score(forecast, actual_horizon)
        brier_score = compute_brier_score(forecast, actual_horizon)

        # Calibration checks
        in_50_interval = forecast.p25 <= actual_horizon <= forecast.p75
        in_90_interval = forecast.p5 <= actual_horizon <= forecast.p95

        return SingleForecastResult(
            forecast_date=forecast_date,
            tau=tau,
            horizon=horizon,
            forecast=forecast,
            actual_today=actual_today,
            actual_horizon=actual_horizon,
            mae_today=mae_today,
            mae_horizon=mae_horizon,
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

    def _compute_nday_sums(
        self,
        events_by_date: Dict[date, List[TweetEvent]],
        sorted_dates: List[date],
        horizon: int,
    ) -> Dict[date, int]:
        """Compute actual N-day sums for each date."""
        sums = {}

        for i, d in enumerate(sorted_dates):
            if i + horizon - 1 >= len(sorted_dates):
                break

            # Check all N days are present
            n_days = sorted_dates[i:i + horizon]
            if all(day in events_by_date for day in n_days):
                total = sum(len(events_by_date[day]) for day in n_days)
                sums[d] = total

        return sums

    def _compute_7day_sums(
        self,
        events_by_date: Dict[date, List[TweetEvent]],
        sorted_dates: List[date],
    ) -> Dict[date, int]:
        """Compute actual 7-day sums for each date (backward compatibility)."""
        return self._compute_nday_sums(events_by_date, sorted_dates, 7)

    def _compute_metrics_by_horizon_tau(
        self,
        forecasts: List[SingleForecastResult],
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Compute metrics grouped by (horizon, τ) pairs."""
        metrics = {}

        for horizon in self.backtest_config.horizons:
            for tau in self.backtest_config.tau_values:
                key_forecasts = [f for f in forecasts if f.horizon == horizon and f.tau == tau]

                if not key_forecasts:
                    continue

                metrics[(horizon, tau)] = {
                    "mae_today": np.mean([f.mae_today for f in key_forecasts]),
                    "mae_horizon": np.mean([f.mae_horizon for f in key_forecasts]),
                    "rmse_horizon": np.sqrt(np.mean([f.mae_horizon ** 2 for f in key_forecasts])),
                    "avg_log_score": np.mean([f.log_score for f in key_forecasts]),
                    "avg_brier_score": np.mean([f.brier_score for f in key_forecasts]),
                    "coverage_50": np.mean([f.in_50_interval for f in key_forecasts]),
                    "coverage_90": np.mean([f.in_90_interval for f in key_forecasts]),
                    "n_forecasts": len(key_forecasts),
                }

        return metrics

    def _compute_metrics_by_tau(
        self,
        forecasts: List[SingleForecastResult],
    ) -> Dict[int, Dict[str, float]]:
        """Compute metrics grouped by τ value (backward compatibility)."""
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

    def _compute_metrics_by_horizon(
        self,
        forecasts: List[SingleForecastResult],
    ) -> Dict[int, Dict[str, float]]:
        """Compute metrics grouped by horizon."""
        metrics_by_horizon = {}

        for horizon in self.backtest_config.horizons:
            h_forecasts = [f for f in forecasts if f.horizon == horizon]

            if not h_forecasts:
                continue

            metrics_by_horizon[horizon] = {
                "mae_today": np.mean([f.mae_today for f in h_forecasts]),
                "mae_horizon": np.mean([f.mae_horizon for f in h_forecasts]),
                "rmse_horizon": np.sqrt(np.mean([f.mae_horizon ** 2 for f in h_forecasts])),
                "avg_log_score": np.mean([f.log_score for f in h_forecasts]),
                "avg_brier_score": np.mean([f.brier_score for f in h_forecasts]),
                "coverage_50": np.mean([f.in_50_interval for f in h_forecasts]),
                "coverage_90": np.mean([f.in_90_interval for f in h_forecasts]),
                "n_forecasts": len(h_forecasts),
            }

        return metrics_by_horizon

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

        # Filter to only 7-day horizon forecasts
        forecasts_7d = [f for f in forecasts if f.horizon == 7]
        if not forecasts_7d:
            return baselines

        if self.backtest_config.include_naive_baseline:
            # Naive baseline: historical mean
            all_daily_counts = [len(events) for events in events_by_date.values()]
            historical_mean_daily = np.mean(all_daily_counts)
            naive_7day = historical_mean_daily * 7

            naive_errors = []
            for f in forecasts_7d:
                naive_errors.append(abs(naive_7day - f.actual_7day))

            baselines["naive"] = {
                "mae_7day": np.mean(naive_errors),
                "rmse_7day": np.sqrt(np.mean([e**2 for e in naive_errors])),
            }

        if self.backtest_config.include_interday_baseline:
            # Interday-only baseline: pure regime forecast without intraday nowcast
            # Uses regime intensity for all 7 days (today from interday, not nowcast)

            interday_errors = []
            full_vs_interday_diffs = []  # Diagnostic: how much does full model differ from interday?
            for f in forecasts_7d:
                # Interday-only: regime forecast for today + pure regime forecasts for days 1-6
                today_interday = f.forecast.today_interday_estimate
                future_pure = f.forecast.future_days_pure

                interday_only_estimate = today_interday + future_pure
                interday_errors.append(abs(interday_only_estimate - f.actual_7day))
                full_vs_interday_diffs.append(f.forecast.mean - interday_only_estimate)

            if interday_errors:
                baselines["interday_only"] = {
                    "mae_7day": np.mean(interday_errors),
                    "rmse_7day": np.sqrt(np.mean([e**2 for e in interday_errors])),
                    "mean_diff_from_full": np.mean(full_vs_interday_diffs),  # +ve = full predicts higher
                    "std_diff_from_full": np.std(full_vs_interday_diffs),
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
