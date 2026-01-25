"""
Musk 7-Day Tweet Count Forecaster

A comprehensive forecasting system for Polymarket's Elon Musk tweet count markets.

Main Components:
- Musk7DayForecaster: Main forecasting interface
- ForecastResult: Result container with bin probabilities
- Backtester: Rolling window backtesting framework

Usage:
    from forecaster import Musk7DayForecaster, ForecasterConfig

    # Create and fit forecaster
    forecaster = Musk7DayForecaster()
    forecaster.fit(n_days=90)

    # Get 7-day forecast
    result = forecaster.forecast_7day_distribution()

    # Get bin probabilities for trading
    probs = forecaster.get_bin_probabilities()
"""

from .config import (
    ForecasterConfig,
    IntradayCurveConfig,
    BurstFeaturesConfig,
    NowcastConfig,
    RegimeConfig,
    DispersionConfig,
    WeekendConfig,
    MonteCarloConfig,
    UpdateConfig,
)

from .data import (
    TweetEvent,
    ContractDayUtils,
    XTrackerClient,
    EventStore,
)

from .intraday import (
    IntradayProgressCurve,
    BurstFeatures,
    BurstFeatureExtractor,
    IntradayNowcast,
)

from .interday import (
    RegimeState,
    RegimeModel,
    DispersionEstimator,
    WeekendEffect,
    InterdayForecaster,
)

from .monte_carlo import (
    BinProbability,
    ForecastResult,
    MonteCarloForecaster,
    compute_log_score,
    compute_brier_score,
    compute_calibration_stats,
)

from .forecaster import (
    Musk7DayForecaster,
    create_forecaster,
)

from .backtest import (
    BacktestConfig,
    SingleForecastResult,
    BacktestResults,
    Backtester,
    AblationStudy,
    load_data_for_backtest,
    run_quick_backtest,
)

from .run_backtest import (
    fetch_historical_data,
    run_full_backtest,
    run_quick_backtest as run_quick,
    run_ablation_study,
    XTRACKER_START_DATE,
)

__all__ = [
    # Main forecaster
    "Musk7DayForecaster",
    "create_forecaster",
    "ForecastResult",

    # Configuration
    "ForecasterConfig",
    "IntradayCurveConfig",
    "BurstFeaturesConfig",
    "NowcastConfig",
    "RegimeConfig",
    "DispersionConfig",
    "WeekendConfig",
    "MonteCarloConfig",
    "UpdateConfig",

    # Data layer
    "TweetEvent",
    "ContractDayUtils",
    "XTrackerClient",
    "EventStore",

    # Intraday components
    "IntradayProgressCurve",
    "BurstFeatures",
    "BurstFeatureExtractor",
    "IntradayNowcast",

    # Interday components
    "RegimeState",
    "RegimeModel",
    "DispersionEstimator",
    "WeekendEffect",
    "InterdayForecaster",

    # Monte Carlo
    "BinProbability",
    "MonteCarloForecaster",
    "compute_log_score",
    "compute_brier_score",
    "compute_calibration_stats",

    # Backtesting
    "BacktestConfig",
    "SingleForecastResult",
    "BacktestResults",
    "Backtester",
    "AblationStudy",
    "load_data_for_backtest",
    "run_quick_backtest",

    # Run backtest script
    "fetch_historical_data",
    "run_full_backtest",
    "run_ablation_study",
    "XTRACKER_START_DATE",
]

__version__ = "1.0.0"
