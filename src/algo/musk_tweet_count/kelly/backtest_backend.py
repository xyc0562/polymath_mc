"""
Backtest implementations of the abstract backend interfaces.

These implementations allow the production Kelly trading logic to run
on historical data by providing simulated orderbooks and execution.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

from .backend import (
    OrderbookProvider,
    TradeExecutor,
    PortfolioProvider,
    ExecutionResult,
    TradingContext,
)
from .orderbook import UnifiedOrderbook, OrderbookLevel
from .candidates import TradeCandidate, TradeAction
from .portfolio import Portfolio

if TYPE_CHECKING:
    from ..backtest.data_provider import SimulatedOrderbook, EventPriceData

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for market simulation in backtests."""

    # Market simulation
    spread: float = 0.02  # 2% bid-ask spread
    slippage: float = 0.005  # 0.5% slippage

    # Whether to log trade executions
    log_trades: bool = False


class BacktestOrderbookProvider(OrderbookProvider):
    """
    Orderbook provider that converts historical prices to UnifiedOrderbook.

    Takes SimulatedOrderbook from historical data and creates proper
    UnifiedOrderbook objects that the Kelly optimizer can use.
    """

    def __init__(
        self,
        config: SimulationConfig,
        token_ids: Dict[int, str],  # bin_index -> token_id
    ):
        """
        Initialize backtest orderbook provider.

        Args:
            config: Backtest configuration
            token_ids: Map of bin_index to token_id (can be placeholder strings)
        """
        self.config = config
        self.token_ids = token_ids

        # Current orderbooks from historical data
        self._orderbooks: Dict[int, UnifiedOrderbook] = {}

    def get_orderbook(self, bin_index: int) -> Optional[UnifiedOrderbook]:
        """Get orderbook for a specific bin."""
        return self._orderbooks.get(bin_index)

    def get_all_orderbooks(self) -> Dict[int, UnifiedOrderbook]:
        """Get all current orderbooks."""
        return self._orderbooks.copy()

    def refresh(self) -> None:
        """No-op for backtest - orderbooks are updated externally."""
        pass

    def update_from_simulated(
        self,
        simulated_orderbooks: Dict[int, "SimulatedOrderbook"],
        timestamp: float,
    ) -> None:
        """
        Update internal orderbooks from simulated historical data.

        Converts SimulatedOrderbook (simple mid/bid/ask) to UnifiedOrderbook
        (full orderbook with depth).

        Args:
            simulated_orderbooks: Dict of bin_index -> SimulatedOrderbook
            timestamp: Current simulation timestamp
        """
        self._orderbooks = {}

        for bin_index, sim_ob in simulated_orderbooks.items():
            token_id = self.token_ids.get(bin_index, f"token_{bin_index}")

            # Create synthetic orderbook depth from the simulated prices
            # For backtest, we create a simple orderbook with configurable depth
            unified_ob = self._create_unified_orderbook(
                bin_index=bin_index,
                token_id=token_id,
                yes_bid=sim_ob.yes_bid,
                yes_ask=sim_ob.yes_ask,
                timestamp=timestamp,
            )

            self._orderbooks[bin_index] = unified_ob

    def _create_unified_orderbook(
        self,
        bin_index: int,
        token_id: str,
        yes_bid: float,
        yes_ask: float,
        timestamp: float,
    ) -> UnifiedOrderbook:
        """
        Create a UnifiedOrderbook from simulated bid/ask prices.

        Creates synthetic depth levels around the bid/ask prices
        to simulate realistic orderbook structure.
        """
        # Create synthetic depth levels
        # In real markets, there's typically more depth at worse prices
        depth_levels = 5
        size_per_level = 100.0  # Shares per level

        # YES bids: start at best bid, decrease price for worse levels
        yes_bids = []
        for i in range(depth_levels):
            price = max(0.01, yes_bid - (i * 0.005))  # 0.5% step per level
            yes_bids.append(OrderbookLevel(price=price, size=size_per_level * (1 + i * 0.5)))

        # YES asks: start at best ask, increase price for worse levels
        yes_asks = []
        for i in range(depth_levels):
            price = min(0.99, yes_ask + (i * 0.005))  # 0.5% step per level
            yes_asks.append(OrderbookLevel(price=price, size=size_per_level * (1 + i * 0.5)))

        return UnifiedOrderbook(
            bin_index=bin_index,
            yes_token_id=token_id,
            yes_bids=yes_bids,
            yes_asks=yes_asks,
            last_updated=timestamp,
        )


class BacktestTradeExecutor(TradeExecutor):
    """
    Trade executor that simulates execution with slippage.

    Provides deterministic execution at VWAP + slippage for backtesting.
    """

    def __init__(
        self,
        config: SimulationConfig,
        portfolio: Portfolio,
    ):
        """
        Initialize backtest trade executor.

        Args:
            config: Backtest configuration
            portfolio: Portfolio to update on execution
        """
        self.config = config
        self.portfolio = portfolio
        self._dry_run = False  # Backtest always "executes" (simulates)

    @property
    def dry_run(self) -> bool:
        """Backtest executor is never dry-run (it simulates execution)."""
        return self._dry_run

    def execute(self, candidate: TradeCandidate, token_id: str) -> ExecutionResult:
        """
        Execute a trade candidate in the backtest simulation.

        Applies slippage to the candidate price and updates the portfolio.
        """
        action = candidate.action
        size = candidate.size
        base_price = candidate.price

        # Apply slippage based on action
        if action in (TradeAction.BUY_YES, TradeAction.BUY_NO):
            # Buying: we pay more (worse price)
            filled_price = base_price + self.config.slippage
        else:
            # Selling: we receive less (worse price)
            filled_price = max(0.001, base_price - self.config.slippage)

        # Clamp price to valid range
        filled_price = max(0.001, min(0.999, filled_price))

        # Check if we have enough capital for buys
        if action in (TradeAction.BUY_YES, TradeAction.BUY_NO):
            cost = size * filled_price
            if cost > self.portfolio.available_capital:
                return ExecutionResult(
                    success=False,
                    candidate=candidate,
                    error=f"Insufficient capital: need ${cost:.2f}, have ${self.portfolio.available_capital:.2f}",
                )

        # Execute trade on portfolio
        self._update_portfolio(candidate, size, filled_price, token_id)

        if self.config.log_trades:
            logger.info(
                f"[BACKTEST] Executed: {action.value} bin={candidate.bin_index} "
                f"size={size:.2f} @ {filled_price:.4f} (base={base_price:.4f})"
            )

        return ExecutionResult(
            success=True,
            candidate=candidate,
            order_id=f"backtest_{candidate.bin_index}_{int(time.time()*1000)}",
            filled_size=size,
            filled_price=filled_price,
        )

    def _update_portfolio(
        self,
        candidate: TradeCandidate,
        size: float,
        price: float,
        token_id: str,
    ) -> None:
        """Update portfolio after trade execution."""
        action = candidate.action
        bin_index = candidate.bin_index

        if action == TradeAction.BUY_YES:
            self.portfolio.execute_buy_yes(bin_index, size, price, token_id)
        elif action == TradeAction.SELL_YES:
            self.portfolio.execute_sell_yes(bin_index, size, price)
        elif action == TradeAction.BUY_NO:
            self.portfolio.execute_buy_no(bin_index, size, price, token_id)
        elif action == TradeAction.SELL_NO:
            self.portfolio.execute_sell_no(bin_index, size, price)


class BacktestPortfolioAdapter(PortfolioProvider):
    """
    Adapter that wraps Portfolio to implement PortfolioProvider interface.

    This allows the Portfolio class to be used through the abstract interface.
    """

    def __init__(self, portfolio: Portfolio):
        """
        Initialize adapter.

        Args:
            portfolio: The Portfolio instance to wrap
        """
        self.portfolio = portfolio

    def get_capital(self) -> float:
        """Get current available capital."""
        return self.portfolio.available_capital

    def get_yes_position(self, bin_index: int) -> float:
        """Get YES shares held for a bin."""
        pos = self.portfolio.get_position(bin_index)
        return pos.yes_shares if pos else 0.0

    def get_no_position(self, bin_index: int) -> float:
        """Get NO shares held for a bin."""
        pos = self.portfolio.get_position(bin_index)
        return pos.no_shares if pos else 0.0

    def update_after_trade(
        self,
        candidate: TradeCandidate,
        filled_size: float,
        filled_price: float,
        token_id: str,
    ) -> None:
        """
        Update portfolio state after a successful trade.

        Note: For backtest, this is typically handled by BacktestTradeExecutor.
        This method exists for interface completeness.
        """
        action = candidate.action
        bin_index = candidate.bin_index

        if action == TradeAction.BUY_YES:
            self.portfolio.execute_buy_yes(bin_index, filled_size, filled_price, token_id)
        elif action == TradeAction.SELL_YES:
            self.portfolio.execute_sell_yes(bin_index, filled_size, filled_price)
        elif action == TradeAction.BUY_NO:
            self.portfolio.execute_buy_no(bin_index, filled_size, filled_price, token_id)
        elif action == TradeAction.SELL_NO:
            self.portfolio.execute_sell_no(bin_index, filled_size, filled_price)


def create_backtest_portfolio(
    initial_capital: float,
    probabilities: List[float],
    bin_upper_bounds: List[int],
    dead_bins: Optional[List[int]] = None,
) -> Portfolio:
    """
    Create a Portfolio configured for backtesting.

    Args:
        initial_capital: Starting capital
        probabilities: Initial probability estimates for each bin
        bin_upper_bounds: Upper bound for each bin
        dead_bins: List of bin indices that are impossible

    Returns:
        Configured Portfolio instance
    """
    return Portfolio(
        initial_capital=initial_capital,
        capital=initial_capital,
        probabilities=probabilities,
        num_bins=len(probabilities),
        bin_upper_bounds=bin_upper_bounds,
        dead_bins=dead_bins or [],
    )


def create_backtest_context(
    current_count: int,
    hours_elapsed: float,
    hours_to_settlement: float,
    timestamp: datetime,
    probabilities: List[float],
    dead_bins: List[int],
) -> TradingContext:
    """
    Create a TradingContext for a single backtest tick.

    Args:
        current_count: Current tweet count
        hours_elapsed: Hours since counting started
        hours_to_settlement: Hours until settlement
        timestamp: Current simulation timestamp
        probabilities: Probability for each bin
        dead_bins: Bins that are impossible

    Returns:
        TradingContext instance
    """
    return TradingContext(
        current_count=current_count,
        hours_elapsed=hours_elapsed,
        hours_to_settlement=hours_to_settlement,
        timestamp=timestamp,
        probabilities=probabilities,
        dead_bins=dead_bins,
    )
