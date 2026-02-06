"""
Tweet Count Forecaster

A comprehensive forecasting system for Polymarket's Elon Musk tweet count markets.
Supports variable-length events (2-day, 7-day, monthly, etc.).

Main Components:
- TweetCountForecaster: Main forecasting interface
- ForecastResult: Result container with bin probabilities
- Backtester: Rolling window backtesting framework

Usage:
    from forecaster import TweetCountForecaster, ForecasterConfig

    # Create and fit forecaster
    forecaster = TweetCountForecaster()
    forecaster.fit(n_days=90)

    # Get forecast for event window
    result = forecaster.forecast_for_event_window(
        market_start_date=start,
        settlement_date=end,
        now=datetime.now(),
    )

    # Get bin probabilities for trading
    probs = forecaster.get_bin_probabilities()
"""

from .config import (
    ForecasterConfig,
    IntradayCurveConfig,
    BurstFeaturesConfig,
    NowcastConfig,
    BucketNowcastConfig,
    RegimeConfig,
    DispersionConfig,
    WeekendConfig,
    MonteCarloConfig,
    UpdateConfig,
    EnsembleConfig,
    GASConfig,
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
    BaseIntradayForecaster,
    IntradayNowcast,
    BucketIntradayForecaster,
    BucketDistribution,
)

from .interday import (
    RegimeState,
    RegimeModel,
    DispersionEstimator,
    WeekendEffect,
    InterdayForecaster,
    GASInterdayForecaster,
)

from .gas import (
    GASRegimeModel,
)

from .ensemble import (
    EnsembleMonteCarloForecaster,
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
    TweetCountForecaster,
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

from .trading_bot import (
    GASKellyTradingBot,
    TradingBotConfig,
)

__all__ = [
    # Main forecaster
    "TweetCountForecaster",
    "create_forecaster",
    "ForecastResult",

    # Configuration
    "ForecasterConfig",
    "IntradayCurveConfig",
    "BurstFeaturesConfig",
    "NowcastConfig",
    "BucketNowcastConfig",
    "RegimeConfig",
    "DispersionConfig",
    "WeekendConfig",
    "MonteCarloConfig",
    "UpdateConfig",
    "GASConfig",

    # Data layer
    "TweetEvent",
    "ContractDayUtils",
    "XTrackerClient",
    "EventStore",

    # Intraday components
    "IntradayProgressCurve",
    "BurstFeatures",
    "BurstFeatureExtractor",
    "BaseIntradayForecaster",
    "IntradayNowcast",
    "BucketIntradayForecaster",
    "BucketDistribution",

    # Interday components
    "RegimeState",
    "RegimeModel",
    "DispersionEstimator",
    "WeekendEffect",
    "InterdayForecaster",
    "GASInterdayForecaster",
    "GASRegimeModel",

    # Ensemble
    "EnsembleConfig",
    "EnsembleMonteCarloForecaster",

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

    # Trading bot
    "GASKellyTradingBot",
    "TradingBotConfig",
]

__version__ = "1.0.0"
