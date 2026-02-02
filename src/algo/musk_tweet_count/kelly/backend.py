"""
Abstract backend interfaces for Kelly trading.

These interfaces allow the same trading logic to be used for both
live trading and backtesting by swapping out the data/execution backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .orderbook import UnifiedOrderbook
from .candidates import TradeCandidate, TradeAction


@dataclass
class ExecutionResult:
    """Result of a single trade execution."""

    success: bool
    candidate: TradeCandidate
    order_id: Optional[str] = None
    filled_size: float = 0.0
    filled_price: float = 0.0
    error: Optional[str] = None


class OrderbookProvider(ABC):
    """
    Abstract interface for orderbook data.

    Implementations:
    - LiveOrderbookProvider: WebSocket streaming from Polymarket
    - BacktestOrderbookProvider: Historical prices with simulated spread
    """

    @abstractmethod
    def get_orderbook(self, bin_index: int) -> Optional[UnifiedOrderbook]:
        """
        Get current orderbook for a bin.

        Args:
            bin_index: The bin index

        Returns:
            UnifiedOrderbook or None if not available
        """
        pass

    @abstractmethod
    def get_all_orderbooks(self) -> Dict[int, UnifiedOrderbook]:
        """
        Get orderbooks for all bins.

        Returns:
            Dict mapping bin_index -> UnifiedOrderbook
        """
        pass

    @abstractmethod
    def refresh(self) -> None:
        """
        Refresh orderbook data.

        For live: fetch latest from WebSocket/REST
        For backtest: advance to next timestamp (no-op if time set externally)
        """
        pass


class TradeExecutor(ABC):
    """
    Abstract interface for trade execution.

    Implementations:
    - LiveTradeExecutor: Real order execution via Polymarket API
    - BacktestTradeExecutor: Simulated execution with slippage
    """

    @abstractmethod
    def execute(self, candidate: TradeCandidate, token_id: str) -> ExecutionResult:
        """
        Execute a trade candidate.

        Args:
            candidate: The trade to execute
            token_id: Token ID for the bin

        Returns:
            ExecutionResult with fill details
        """
        pass

    @property
    @abstractmethod
    def dry_run(self) -> bool:
        """Whether this executor is in dry-run mode."""
        pass


class PortfolioProvider(ABC):
    """
    Abstract interface for portfolio state management.

    This is typically shared between live and backtest, but the interface
    allows for different implementations if needed.
    """

    @abstractmethod
    def get_capital(self) -> float:
        """Get current available capital."""
        pass

    @abstractmethod
    def get_yes_position(self, bin_index: int) -> float:
        """Get YES shares held for a bin."""
        pass

    @abstractmethod
    def get_no_position(self, bin_index: int) -> float:
        """Get NO shares held for a bin."""
        pass

    @abstractmethod
    def update_after_trade(
        self,
        candidate: TradeCandidate,
        filled_size: float,
        filled_price: float,
        token_id: str,
    ) -> None:
        """
        Update portfolio state after a successful trade.

        Args:
            candidate: The executed trade candidate
            filled_size: Actual filled size
            filled_price: Actual fill price
            token_id: Token ID for the bin
        """
        pass


@dataclass
class TradingContext:
    """
    Context for a single trading tick.

    Contains all the information needed to make trading decisions.
    """
    current_count: int  # Current tweet count
    hours_elapsed: float  # Hours since counting started
    hours_to_settlement: float  # Hours until settlement
    timestamp: datetime  # Current timestamp
    probabilities: List[float]  # Probability for each bin
    dead_bins: List[int]  # Bins that are impossible (count exceeded upper bound)
