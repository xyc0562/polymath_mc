"""
Kelly criterion optimizer for Polymarket tweet-count bin trading.

This package implements a Kelly-optimal trading strategy for mutually
exclusive bin outcomes, with support for:
- Kelly reservation prices and optimal position sizing
- Edge buffers for model uncertainty protection
- WebSocket orderbook streaming
- Adaptive chunk sizing based on liquidity
- Greedy execution with utility maximization
"""

from .config import (
    KellyConfig,
    EdgeBufferConfig,
    AdaptiveDeltaConfig,
    RateLimitConfig,
    CollateralConfig,
)
from .websocket_client import WebSocketConfig
from .kelly_math import (
    compute_terminal_wealth,
    compute_normalizer_S,
    compute_reservation_price_yes,
    compute_reservation_price_no,
    compute_utility_gain,
)
from .orderbook import UnifiedOrderbook, OrderbookLevel, compute_vwap
from .portfolio import Portfolio, BinPosition
from .candidates import TradeCandidate, generate_candidates, should_trade
from .executor import KellyExecutor, OrderExecutor, ExecutionResult, TickResult
from .integration import KellyTradingBot, create_kelly_bot_from_config

__all__ = [
    # Config
    "KellyConfig",
    "EdgeBufferConfig",
    "AdaptiveDeltaConfig",
    "WebSocketConfig",
    "RateLimitConfig",
    "CollateralConfig",
    # Math
    "compute_terminal_wealth",
    "compute_normalizer_S",
    "compute_reservation_price_yes",
    "compute_reservation_price_no",
    "compute_utility_gain",
    # Orderbook
    "UnifiedOrderbook",
    "OrderbookLevel",
    "compute_vwap",
    # Portfolio
    "Portfolio",
    "BinPosition",
    # Candidates
    "TradeCandidate",
    "generate_candidates",
    "should_trade",
    # Executor
    "KellyExecutor",
    "OrderExecutor",
    "ExecutionResult",
    "TickResult",
    # Integration
    "KellyTradingBot",
    "create_kelly_bot_from_config",
]
