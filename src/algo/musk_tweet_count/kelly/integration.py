"""
Integration module for Kelly optimizer with the main trading bot.

Provides a KellyTradingBot that integrates the Kelly optimizer with
the existing PolymarketTradingBot infrastructure.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable

from py_clob_client.client import ClobClient

from .config import KellyConfig
from .orderbook import UnifiedOrderbook
from .portfolio import Portfolio
from .candidates import TradeCandidate
from .executor import KellyExecutor, OrderExecutor, ExecutionResult, TickResult
from .websocket_client import OrderbookManager, WebSocketConfig
from .kelly_math import identify_dead_bins, renormalize_probabilities

logger = logging.getLogger(__name__)


class KellyTradingBot:
    """
    Kelly-optimal trading bot for Polymarket tweet-count bins.

    Integrates with the existing trading infrastructure while using
    Kelly criterion optimization for position sizing.
    """

    def __init__(
        self,
        clob_client: ClobClient,
        config: KellyConfig,
        probability_model: Callable[[int, float, float], List[float]],
        dry_run: bool = False,
    ):
        """
        Initialize Kelly trading bot.

        Args:
            clob_client: Authenticated ClobClient for order execution
            config: Kelly configuration
            probability_model: Function(current_count, hours_elapsed, hours_remaining)
                              -> List[float] probabilities for each bin
            dry_run: If True, simulate trades without execution
        """
        self.clob_client = clob_client
        self.config = config
        self.probability_model = probability_model
        self.dry_run = dry_run

        # Bin metadata (set during setup)
        self.bin_upper_bounds: List[int] = []
        self.bin_token_ids: Dict[int, str] = {}  # bin_index -> YES token_id
        self.num_bins: int = 0

        # Components (initialized during setup)
        self.orderbook_manager: Optional[OrderbookManager] = None
        self.order_executor: Optional[OrderExecutor] = None
        self.portfolio: Optional[Portfolio] = None
        self.kelly_executor: Optional[KellyExecutor] = None

        # State
        self._running = False
        self._setup_complete = False

        # Lock for thread-safe tick execution
        self._tick_lock = asyncio.Lock()

    async def setup(
        self,
        initial_capital: float,
        bins: List[Dict],  # List of {"upper_bound": int, "token_id": str}
    ) -> None:
        """
        Setup the Kelly trading bot.

        Args:
            initial_capital: Initial USDC capital
            bins: List of bin definitions with upper_bound and token_id
        """
        logger.info(f"Setting up Kelly bot with {len(bins)} bins, ${initial_capital:.2f} capital")

        # Store bin metadata
        self.num_bins = len(bins)
        self.bin_upper_bounds = [b["upper_bound"] for b in bins]
        self.bin_token_ids = {i: b["token_id"] for i, b in enumerate(bins)}

        # Initialize portfolio
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            capital=initial_capital,
            num_bins=self.num_bins,
            bin_upper_bounds=self.bin_upper_bounds,
        )

        # Initialize orderbook manager
        self.orderbook_manager = OrderbookManager(
            config=self.config.websocket,
            clob_client=self.clob_client,
        )

        # Start orderbook streaming
        await self.orderbook_manager.start()

        # Subscribe to all bins
        await self.orderbook_manager.subscribe_bins([
            {"token_id": b["token_id"], "bin_index": i}
            for i, b in enumerate(bins)
        ])

        # Initialize order executor
        self.order_executor = OrderExecutor(
            clob_client=self.clob_client,
            dry_run=self.dry_run,
        )

        # Initialize Kelly executor
        self.kelly_executor = KellyExecutor(
            config=self.config,
            portfolio=self.portfolio,
            orderbook_manager=self.orderbook_manager,
            order_executor=self.order_executor,
            token_ids=self.bin_token_ids,
            on_trade=self._on_trade,
        )

        self._setup_complete = True
        logger.info("Kelly bot setup complete")

    async def shutdown(self) -> None:
        """Shutdown the Kelly trading bot."""
        logger.info("Shutting down Kelly bot")
        self._running = False

        if self.kelly_executor:
            self.kelly_executor.stop()

        if self.orderbook_manager:
            await self.orderbook_manager.stop()

        self._setup_complete = False

    def _on_trade(self, result: ExecutionResult) -> None:
        """Callback for trade execution."""
        if result.success:
            logger.info(
                f"Trade executed: {result.candidate.action.value} "
                f"bin={result.candidate.bin_index} "
                f"size={result.filled_size:.2f} @ {result.filled_price:.4f}"
            )
        else:
            logger.warning(f"Trade failed: {result.error}")

    def update_probabilities(
        self,
        current_count: int,
        hours_elapsed: float,
        hours_remaining: float,
    ) -> None:
        """
        Update probability estimates and identify dead bins.

        Args:
            current_count: Current tweet count from xtracker
            hours_elapsed: Hours since counting period started
            hours_remaining: Hours until settlement
        """
        if not self._setup_complete:
            raise RuntimeError("Bot not setup. Call setup() first.")

        # Identify dead bins (upper_bound < current_count)
        dead_bins = identify_dead_bins(self.bin_upper_bounds, current_count)
        self.portfolio.dead_bins = dead_bins

        if dead_bins:
            logger.info(f"Dead bins (count={current_count}): {dead_bins}")

        # Get raw probabilities from model
        raw_probabilities = self.probability_model(
            current_count, hours_elapsed, hours_remaining
        )

        # Renormalize if configured
        if self.config.renormalize_probabilities:
            probabilities = renormalize_probabilities(raw_probabilities, dead_bins)
        else:
            probabilities = raw_probabilities

        # Update portfolio
        self.portfolio.update_probabilities(probabilities, renormalize=False)

        logger.debug(f"Updated probabilities: {probabilities}")

    async def run_tick(
        self,
        current_count: int,
        hours_elapsed: float,
        hours_to_settlement: float,
    ) -> TickResult:
        """
        Run a single optimization tick.

        Thread-safe: uses lock to prevent concurrent tick execution.

        Args:
            current_count: Current tweet count
            hours_elapsed: Hours since counting started
            hours_to_settlement: Hours until settlement

        Returns:
            TickResult with execution summary
        """
        if not self._setup_complete:
            raise RuntimeError("Bot not setup. Call setup() first.")

        # Try to acquire lock without blocking - skip if already running
        if self._tick_lock.locked():
            logger.debug("Kelly tick already in progress, skipping")
            return TickResult(
                num_candidates=0,
                num_executed=0,
                total_utility_gain=0.0,
                executions=[],
                elapsed_seconds=0.0,
            )

        async with self._tick_lock:
            # Update probabilities (this also identifies dead bins)
            self.update_probabilities(current_count, hours_elapsed, hours_to_settlement)

            # Fetch fresh orderbooks only for LIVE bins (skip dead bins)
            dead_bins = set(self.portfolio.dead_bins)
            for bin_idx, token_id in self.bin_token_ids.items():
                if bin_idx in dead_bins:
                    continue  # Skip dead bins - no need to fetch orderbooks
                if not self.orderbook_manager.get_orderbook(token_id):
                    await self.orderbook_manager.fetch_orderbook(token_id, bin_idx)

            # Run Kelly optimization tick
            result = await self.kelly_executor.run_tick(hours_to_settlement)

            return result

    async def run_continuous(
        self,
        get_market_state: Callable[[], Dict],
        tick_interval_seconds: float = 60.0,
    ) -> None:
        """
        Run continuous Kelly optimization.

        Args:
            get_market_state: Function returning {
                "current_count": int,
                "hours_elapsed": float,
                "hours_to_settlement": float,
            }
            tick_interval_seconds: Seconds between optimization ticks
        """
        if not self._setup_complete:
            raise RuntimeError("Bot not setup. Call setup() first.")

        self._running = True
        logger.info("Starting continuous Kelly optimization")

        while self._running:
            try:
                # Get current market state
                state = get_market_state()

                current_count = state.get("current_count", 0)
                hours_elapsed = state.get("hours_elapsed", 0)
                hours_to_settlement = state.get("hours_to_settlement", 0)

                # Check T_stop
                if hours_to_settlement <= self.config.t_stop_hours:
                    logger.info("Reached T_stop. Holding positions to settlement.")
                    break

                # Run tick
                result = await self.run_tick(
                    current_count,
                    hours_elapsed,
                    hours_to_settlement,
                )

                logger.info(
                    f"Tick: candidates={result.num_candidates}, "
                    f"executed={result.num_executed}, "
                    f"utility_gain={result.total_utility_gain:.6f}"
                )

                # Wait for next tick
                await asyncio.sleep(tick_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in tick: {e}")
                await asyncio.sleep(tick_interval_seconds)

        self._running = False
        logger.info("Continuous optimization stopped")

    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary."""
        if not self.portfolio:
            return {}
        return self.portfolio.to_summary()

    def get_reservation_prices(self) -> tuple[List[float], List[float]]:
        """Get current Kelly reservation prices."""
        if not self.portfolio:
            return [], []
        return self.portfolio.get_reservation_prices(self.config.w_floor)


def create_kelly_bot_from_config(
    clob_client: ClobClient,
    yaml_config: Dict,
    probability_model: Callable[[int, float, float], List[float]],
    dry_run: bool = False,
) -> KellyTradingBot:
    """
    Create a KellyTradingBot from YAML configuration.

    Args:
        clob_client: Authenticated ClobClient
        yaml_config: Configuration dict with 'kelly' section
        probability_model: Probability model function
        dry_run: If True, simulate trades

    Returns:
        KellyTradingBot instance
    """
    kelly_config_dict = yaml_config.get("kelly", {})

    # Add collateral config if present
    if "collateral" in yaml_config:
        kelly_config_dict["collateral"] = yaml_config["collateral"]

    kelly_config = KellyConfig.from_dict(kelly_config_dict)

    return KellyTradingBot(
        clob_client=clob_client,
        config=kelly_config,
        probability_model=probability_model,
        dry_run=dry_run,
    )
