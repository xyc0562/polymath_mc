"""
Backtest runner for Musk tweet count trading strategy.

Replays historical data through the SAME trading logic used in live trading.
The trading logic is imported from kelly/ module to ensure consistency.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Callable

from .data_provider import (
    HistoricalDataProvider,
    CachedPostsProvider,
    EventPriceData,
    SimulatedOrderbook,
)

# Import shared trading logic from kelly module
from ..kelly.config import EdgeBufferConfig
from ..kelly.candidates import (
    compute_buy_yes_threshold,
    compute_buy_no_threshold,
    compute_exit_threshold,
)

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """A position in a single bin."""
    bin_index: int
    side: str  # "YES" or "NO"
    size: float  # Number of shares
    avg_price: float  # Average entry price
    collateral: float  # Collateral used

    @property
    def market_value(self) -> float:
        """Current value if we sold at avg_price (approximate)."""
        return self.size * self.avg_price


@dataclass
class Trade:
    """Record of a single trade."""
    timestamp: int
    datetime: datetime
    bin_index: int
    side: str  # "BUY_YES", "BUY_NO", "SELL_YES", "SELL_NO"
    size: float
    price: float
    collateral: float
    signal_strength: float  # How much below threshold
    realized_pnl: Optional[float] = None  # P/L for sell trades (proceeds - cost basis)


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    event_name: str
    start_date: date
    end_date: date
    winner_bin: int

    # P&L
    initial_capital: float
    final_capital: float
    total_pnl: float
    total_return: float

    # Trade stats
    num_trades: int
    num_winning_trades: int
    num_losing_trades: int
    win_rate: float

    # Position at settlement
    final_positions: Dict[int, Position]
    settlement_pnl: float

    # All trades
    trades: List[Trade] = field(default_factory=list)

    # Per-bin P&L
    pnl_by_bin: Dict[int, float] = field(default_factory=dict)

    # Bin ranges for display (bin_index -> "lower-upper")
    bin_ranges: Dict[int, str] = field(default_factory=dict)

    # Settlement details for trade history
    settlement_details: List[Dict] = field(default_factory=list)


@dataclass
class BacktestConfig:
    """Configuration for backtest."""
    # Capital
    initial_capital: float = 1000.0
    max_position_per_bin: float = 100.0

    # Market simulation
    spread: float = 0.02  # 2% bid-ask spread
    slippage: float = 0.005  # 0.5% slippage

    # Trading parameters - uses shared EdgeBufferConfig from kelly module
    # This ensures backtest uses SAME logic as live trading
    edge_buffer: EdgeBufferConfig = field(default_factory=lambda: EdgeBufferConfig(
        required_roi=0.10,  # 10% required ROI
        friction_mid=0.015,  # 1.5% friction for >= 9% probability
        friction_tail=0.03,  # 3% friction for < 9% probability
        tail_threshold=0.09,  # 9% threshold
    ))

    # Tick frequency
    tick_interval_seconds: int = 3600  # 1 hour between ticks

    # Forecaster settings
    training_days: int = 45
    tweet_data_start_date: date = field(default_factory=lambda: date(2025, 11, 1))

    # Early exit: close all positions X hours before settlement (0 = disabled)
    exit_hours_before_settlement: float = 0.0

    # Stop-loss: exit position if value drops below this fraction of entry cost
    # e.g., 0.75 means exit if value drops to 75% of entry (25% loss). 0 = disabled.
    # WARNING: Stop-loss typically hurts returns in prediction markets due to binary outcomes.
    stop_loss_pct: float = 0.0


class BacktestRunner:
    """
    Runs backtests on historical data using the same trading logic as live.

    Key design: Trading logic is imported from the live system, only data
    sources are replaced with historical/simulated data.
    """

    def __init__(
        self,
        config: BacktestConfig,
        price_data_dir: Path = Path("data/price_history"),
        cache_dir: Path = Path("data/backtest_cache"),
    ):
        """
        Initialize backtest runner.

        Args:
            config: Backtest configuration
            price_data_dir: Directory containing scraped price history
            cache_dir: Directory for caching posts data
        """
        self.config = config
        self.price_data_dir = Path(price_data_dir)
        self.cache_dir = Path(cache_dir)

        # Data providers
        self.price_provider = HistoricalDataProvider(
            price_data_dir=price_data_dir,
            spread=config.spread,
            slippage=config.slippage,
        )
        self.posts_provider = CachedPostsProvider(cache_dir=cache_dir)

        # State during backtest
        self._capital: float = 0.0
        self._positions: Dict[int, Position] = {}
        self._trades: List[Trade] = []

    def run(self, event_dir: str) -> Optional[BacktestResult]:
        """
        Run backtest on a specific event.

        Args:
            event_dir: Event directory name (e.g., "2025-11-25_Nov_18_-_Nov_25")

        Returns:
            BacktestResult or None if event not found
        """
        # Load event data
        event = self.price_provider.load_event(event_dir)
        if event is None:
            logger.error(f"Event not found: {event_dir}")
            return None

        logger.info(f"Running backtest for: {event.short_name}")
        logger.info(f"  Trading period: {event.start_date} to {event.end_date}")
        logger.info(f"  Counting period: {event.counting_start_date} to {event.counting_end_date}")
        logger.info(f"  Bins: {len(event.bins)}, Winner: bin {event.winner_bin_index}")
        logger.info(f"  Timestamps: {len(event.all_timestamps)}")

        # Validate training data availability
        training_start_needed = event.start_date - timedelta(days=self.config.training_days)
        days_available = (event.start_date - self.config.tweet_data_start_date).days

        if days_available < self.config.training_days:
            logger.error(
                f"Insufficient training data for {event.short_name}: "
                f"need {self.config.training_days} days, only have {days_available} days "
                f"(tweet data starts {self.config.tweet_data_start_date}, "
                f"training needs to start {training_start_needed})"
            )
            return None

        # Load posts data
        self.posts_provider.load_or_fetch(
            start_date=event.start_date - timedelta(days=self.config.training_days),
            end_date=event.end_date + timedelta(days=1),
        )

        # Initialize state
        self._capital = self.config.initial_capital
        self._positions = {}
        self._trades = []

        # Initialize forecaster with historical data (returns forecaster and backtest-period posts)
        forecaster, backtest_posts = self._create_forecaster(event)

        # Index for tracking which backtest posts have been added
        backtest_posts_idx = 0

        # Sample timestamps at hourly intervals instead of every tick
        # This dramatically reduces computation while still capturing market dynamics
        tick_interval = self.config.tick_interval_seconds
        sampled_timestamps = self._sample_timestamps(event.all_timestamps, tick_interval)

        logger.info(f"  Sampled {len(sampled_timestamps)} ticks (every {tick_interval}s)")

        # Calculate early exit time if configured
        exit_before_settlement = self.config.exit_hours_before_settlement
        settlement_dt = datetime.combine(
            event.end_date + timedelta(days=1),  # Settlement is at start of day after end_date
            datetime.min.time(),
            tzinfo=timezone.utc
        )
        exit_window_start = settlement_dt - timedelta(hours=exit_before_settlement)
        in_exit_mode = False

        if exit_before_settlement > 0:
            logger.info(f"  Early exit enabled: closing all positions {exit_before_settlement}h before settlement")

        # Track contract day for interday model updates
        last_contract_day = None

        # Run through sampled timestamps
        for i, ts in enumerate(sampled_timestamps):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)

            # Skip timestamps before event start
            if dt.date() < event.start_date:
                continue

            # Stop at event end
            if dt.date() > event.end_date:
                break

            # Get current contract day (using forecaster's boundary conventions)
            current_contract_day = forecaster.contract_utils.get_contract_date(dt)

            # Update interday model when a new contract day starts
            # This means the previous day is now complete
            if last_contract_day is not None and current_contract_day != last_contract_day:
                # Get the completed day's count
                completed_day_count = forecaster.event_store.get_contract_day_count(last_contract_day)

                # Update interday model with the completed day's observation
                if forecaster.interday is not None:
                    forecaster.interday.update(last_contract_day, completed_day_count)
                    logger.debug(
                        f"  Updated interday model: {last_contract_day} count={completed_day_count}"
                    )

            last_contract_day = current_contract_day

            # Add posts that have "arrived" by this tick's timestamp
            # This ensures the EventStore only contains posts that would be available at this point in time
            while backtest_posts_idx < len(backtest_posts):
                post_ts, tweet_event = backtest_posts[backtest_posts_idx]
                if post_ts <= ts:
                    forecaster.event_store.add_event(tweet_event)
                    backtest_posts_idx += 1
                else:
                    break  # Remaining posts are in the future

            # Check if we're in the exit window
            if exit_before_settlement > 0 and dt >= exit_window_start and not in_exit_mode:
                in_exit_mode = True
                logger.info(f"  Entering exit mode at {dt.strftime('%Y-%m-%d %H:%M')} - closing all positions")
                self._close_all_positions(event, ts)
                continue

            # Skip normal trading if in exit mode
            if in_exit_mode:
                continue

            # Run tick
            self._run_tick(
                event=event,
                timestamp=ts,
                forecaster=forecaster,
            )

            # Log progress periodically
            if i % 20 == 0:
                logger.info(
                    f"  Tick {i}/{len(sampled_timestamps)}: "
                    f"{dt.strftime('%Y-%m-%d %H:%M')} | "
                    f"capital=${self._capital:.2f}, "
                    f"positions={len(self._positions)}, "
                    f"trades={len(self._trades)}"
                )

        # Calculate settlement P&L and details
        settlement_pnl, settlement_details = self._calculate_settlement_pnl_with_details(event)

        # Calculate final capital
        final_capital = self._capital + settlement_pnl

        # Calculate trade stats
        pnl_by_bin = self._calculate_pnl_by_bin(event)
        winning_trades = sum(1 for pnl in pnl_by_bin.values() if pnl > 0)
        losing_trades = sum(1 for pnl in pnl_by_bin.values() if pnl < 0)

        # Build bin ranges dict for display
        bin_ranges = {
            b.bin_index: f"{b.lower_bound}-{b.upper_bound}"
            for b in event.bins
        }

        result = BacktestResult(
            event_name=event.short_name,
            start_date=event.start_date,
            end_date=event.end_date,
            winner_bin=event.winner_bin_index or -1,
            initial_capital=self.config.initial_capital,
            final_capital=final_capital,
            total_pnl=final_capital - self.config.initial_capital,
            total_return=(final_capital / self.config.initial_capital - 1) * 100,
            num_trades=len(self._trades),
            num_winning_trades=winning_trades,
            num_losing_trades=losing_trades,
            win_rate=winning_trades / max(1, winning_trades + losing_trades),
            final_positions=self._positions.copy(),
            settlement_pnl=settlement_pnl,
            trades=self._trades.copy(),
            pnl_by_bin=pnl_by_bin,
            bin_ranges=bin_ranges,
            settlement_details=settlement_details,
        )

        self._log_result(result)

        return result

    def _sample_timestamps(
        self,
        timestamps: List[int],
        interval_seconds: int,
    ) -> List[int]:
        """
        Sample timestamps at regular intervals.

        Args:
            timestamps: List of all timestamps (sorted)
            interval_seconds: Desired interval between samples

        Returns:
            Sampled list of timestamps
        """
        if not timestamps:
            return []

        sampled = [timestamps[0]]
        last_ts = timestamps[0]

        for ts in timestamps[1:]:
            if ts - last_ts >= interval_seconds:
                sampled.append(ts)
                last_ts = ts

        return sampled

    def _create_forecaster(self, event: EventPriceData):
        """
        Create and fit a forecaster for the backtest.

        Returns:
            Tuple of (forecaster, backtest_posts) where backtest_posts are posts
            from the event period that will be added incrementally during backtest.
        """
        from ..forecaster.config import ForecasterConfig, MonteCarloConfig
        from ..forecaster.forecaster import TweetCountForecaster
        from ..forecaster.data import EventStore, ContractDayUtils, XTrackerClient, TweetEvent
        from datetime import timezone as tz
        from zoneinfo import ZoneInfo

        # Use fixed seed for reproducible backtests
        mc_config = MonteCarloConfig(random_seed=42)
        config = ForecasterConfig(monte_carlo=mc_config)
        contract_utils = ContractDayUtils(
            timezone=config.timezone,
            boundary_hour=config.contract_boundary_hour,
        )

        # Create event store and populate from cached posts
        xtracker = XTrackerClient()
        event_store = EventStore(contract_utils, xtracker)

        # Calculate cutoff: start of first trading day (noon ET on counting_start_date)
        # Posts before this go into training, posts after are added incrementally
        est_tz = ZoneInfo("America/New_York")
        cutoff_dt = datetime.combine(
            event.counting_start_date,
            datetime.min.time().replace(hour=12),
            tzinfo=est_tz
        )

        # Separate posts into training period vs backtest period
        training_posts_count = 0
        backtest_posts = []  # List of (timestamp, TweetEvent)

        # Convert cached posts to TweetEvent objects and add to store
        for post in self.posts_provider._posts:
            try:
                # Parse timestamp from post - XTracker uses 'createdAt'
                ts = post.get("createdAt") or post.get("timestamp")
                if ts is None:
                    continue

                # Convert to datetime
                if isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(ts, tz=tz.utc)
                elif isinstance(ts, str):
                    # Handle ISO format like "2025-11-27T21:02:06.000Z"
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                else:
                    continue

                # Determine event type from content
                content = post.get("content", "")
                event_type = "retweet" if content.startswith("RT @") else "tweet"

                tweet_event = TweetEvent(
                    timestamp=dt,
                    event_type=event_type,
                    event_id=post.get("id"),
                )

                # Only add training period posts to EventStore initially
                # Backtest period posts will be added incrementally
                if dt < cutoff_dt:
                    event_store.add_event(tweet_event)
                    training_posts_count += 1
                else:
                    # Store for incremental addition during backtest
                    backtest_posts.append((int(dt.timestamp()), tweet_event))

            except (ValueError, KeyError) as e:
                logger.debug(f"Skipping post due to error: {e}")
                continue

        # Sort backtest posts by timestamp for efficient incremental addition
        backtest_posts.sort(key=lambda x: x[0])

        logger.info(
            f"Loaded {training_posts_count} training posts into event store, "
            f"{len(backtest_posts)} backtest posts will be added incrementally"
        )

        # Create forecaster
        forecaster = TweetCountForecaster(config, event_store=event_store)

        # Fit on historical data (skip API fetch since we have cached data)
        # Use event start_date as the reference "today" for training
        forecaster.fit(
            n_days=self.config.training_days,
            skip_fetch=True,
            as_of_date=event.start_date,
        )

        return forecaster, backtest_posts

    def _run_tick(
        self,
        event: EventPriceData,
        timestamp: int,
        forecaster,
    ) -> None:
        """Run a single trading tick."""
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)

        # Get simulated orderbooks for all bins
        orderbooks = self.price_provider.get_all_orderbooks(event, timestamp)
        if not orderbooks:
            return

        # Check stop-loss for existing positions
        if self.config.stop_loss_pct > 0:
            self._check_stop_losses(timestamp, orderbooks)

        # Get current cumulative count
        # Use posts from counting_start_date (not trading start_date)
        counting_start_ts = int(datetime.combine(
            event.counting_start_date,
            datetime.min.time(),
            tzinfo=timezone.utc
        ).timestamp())

        current_count = self.posts_provider.count_posts_in_range(
            start_ts=counting_start_ts,
            end_ts=timestamp,
        )

        # Get forecast
        # Use counting dates for the forecast window
        # settlement_date should be the day AFTER counting_end_date
        forecast = forecaster.forecast_for_event_window(
            market_start_date=event.counting_start_date,
            settlement_date=event.counting_end_date + timedelta(days=1),
            now=dt,
        )

        # Generate probabilities for each bin
        # Compute bin probabilities using the asymmetric Monte Carlo samples
        probabilities = self._compute_bin_probabilities(
            event=event,
            forecast=forecast,
            current_count=current_count,
        )

        # Check for trading signals
        for bin_index, ob in orderbooks.items():
            if bin_index >= len(probabilities):
                continue

            fair_prob = probabilities[bin_index]

            # Skip dead bins
            bin_data = event.bins[bin_index]
            if bin_data.upper_bound < current_count:
                continue

            # Check existing position
            existing_pos = self._positions.get(bin_index)

            # Compute thresholds
            yes_buy_threshold = self._compute_buy_threshold(fair_prob)
            no_fair = 1.0 - fair_prob
            no_buy_threshold = self._compute_buy_threshold(no_fair)

            # Sell thresholds: sell when price is above fair value (good exit)
            yes_sell_threshold = self._compute_sell_threshold(fair_prob)
            no_sell_threshold = self._compute_sell_threshold(no_fair)

            # --- Handle existing positions first ---
            if existing_pos:
                if existing_pos.side == "YES":
                    # Check if we should sell YES:
                    # 1. Price moved up favorably (take profit)
                    # 2. We want to buy NO (flip position)
                    should_sell_for_profit = ob.yes_bid >= yes_sell_threshold
                    want_to_buy_no = ob.no_ask <= no_buy_threshold

                    # Calculate sell price (with floor to avoid negative)
                    sell_price = max(0.001, ob.yes_bid - self.config.slippage)

                    # Only sell if we can get a reasonable price (at least 1% of entry)
                    # Don't give away positions for free
                    min_sell_price = existing_pos.avg_price * 0.10  # At least 10% of entry
                    can_sell_at_reasonable_price = sell_price >= min_sell_price

                    if should_sell_for_profit and can_sell_at_reasonable_price:
                        self._execute_sell(
                            timestamp=timestamp,
                            bin_index=bin_index,
                            price=sell_price,
                        )
                    elif want_to_buy_no and can_sell_at_reasonable_price:
                        # Flip position: sell YES, buy NO
                        self._execute_sell(
                            timestamp=timestamp,
                            bin_index=bin_index,
                            price=sell_price,
                        )
                        self._execute_buy(
                            timestamp=timestamp,
                            bin_index=bin_index,
                            side="BUY_NO",
                            price=ob.no_ask + self.config.slippage,
                            signal_strength=no_buy_threshold - ob.no_ask,
                        )

                elif existing_pos.side == "NO":
                    # Check if we should sell NO:
                    # 1. Price moved up favorably (take profit)
                    # 2. We want to buy YES (flip position)
                    should_sell_for_profit = ob.no_bid >= no_sell_threshold
                    want_to_buy_yes = ob.yes_ask <= yes_buy_threshold

                    # Calculate sell price (with floor to avoid negative)
                    sell_price = max(0.001, ob.no_bid - self.config.slippage)

                    # Only sell if we can get a reasonable price
                    min_sell_price = existing_pos.avg_price * 0.10  # At least 10% of entry
                    can_sell_at_reasonable_price = sell_price >= min_sell_price

                    if should_sell_for_profit and can_sell_at_reasonable_price:
                        self._execute_sell(
                            timestamp=timestamp,
                            bin_index=bin_index,
                            price=sell_price,
                        )
                    elif want_to_buy_yes and can_sell_at_reasonable_price:
                        # Flip position: sell NO, buy YES
                        self._execute_sell(
                            timestamp=timestamp,
                            bin_index=bin_index,
                            price=sell_price,
                        )
                        self._execute_buy(
                            timestamp=timestamp,
                            bin_index=bin_index,
                            side="BUY_YES",
                            price=ob.yes_ask + self.config.slippage,
                            signal_strength=yes_buy_threshold - ob.yes_ask,
                        )
            else:
                # No existing position - check for new buys
                # Check for BUY_YES signal
                if ob.yes_ask <= yes_buy_threshold:
                    signal_strength = yes_buy_threshold - ob.yes_ask
                    self._execute_buy(
                        timestamp=timestamp,
                        bin_index=bin_index,
                        side="BUY_YES",
                        price=ob.yes_ask + self.config.slippage,
                        signal_strength=signal_strength,
                    )
                # Check for BUY_NO signal
                elif ob.no_ask <= no_buy_threshold:
                    signal_strength = no_buy_threshold - ob.no_ask
                    self._execute_buy(
                        timestamp=timestamp,
                        bin_index=bin_index,
                        side="BUY_NO",
                        price=ob.no_ask + self.config.slippage,
                        signal_strength=signal_strength,
                    )

    def _compute_bin_probabilities(
        self,
        event: EventPriceData,
        forecast: "ForecastResult",
        current_count: int,
    ) -> List[float]:
        """
        Compute probability for each bin using forecast distribution.

        Uses the actual Monte Carlo samples from the forecaster, which preserve
        the asymmetric distribution (Log-normal + Negative Binomial). Falls back
        to Normal approximation only if samples aren't available.
        """
        import numpy as np

        # Use the actual Monte Carlo samples from the forecaster
        # These preserve the asymmetric distribution
        if forecast.samples is None:
            raise RuntimeError(
                "Forecast result has no samples - cannot compute probabilities. "
                "This indicates a bug in the forecaster."
            )

        samples = forecast.samples.copy()

        # Floor samples at current_count (can't go below current count)
        samples = np.maximum(samples, current_count)
        n_samples = len(samples)

        probabilities = []
        for bin_data in event.bins:
            if bin_data.upper_bound < current_count:
                prob = 0.0  # Dead bin
            else:
                count = np.sum(
                    (samples >= bin_data.lower_bound) &
                    (samples <= bin_data.upper_bound)
                )
                prob = count / n_samples
            probabilities.append(prob)

        # Renormalize
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]

        return probabilities

    def _compute_buy_threshold(self, fair_prob: float) -> float:
        """
        Compute the maximum price to pay for buying.

        Delegates to shared kelly module to ensure consistency with live trading.
        """
        return compute_buy_yes_threshold(fair_prob, self.config.edge_buffer)

    def _compute_sell_threshold(self, fair_prob: float) -> float:
        """
        Compute the minimum price to sell at when exiting a position.

        Delegates to shared kelly module to ensure consistency with live trading.
        Uses exit threshold (fair value) rather than sell threshold (requires edge).
        """
        return compute_exit_threshold(fair_prob, self.config.edge_buffer)

    def _execute_sell(
        self,
        timestamp: int,
        bin_index: int,
        price: float,
    ) -> None:
        """Execute a sell of existing position."""
        if bin_index not in self._positions:
            return

        pos = self._positions[bin_index]

        # Calculate proceeds from sale
        proceeds = pos.size * price

        # Calculate realized P/L (proceeds - original cost)
        realized_pnl = proceeds - pos.collateral

        # Return capital
        self._capital += proceeds

        # Record trade
        side = f"SELL_{pos.side}"
        self._trades.append(Trade(
            timestamp=timestamp,
            datetime=datetime.fromtimestamp(timestamp, tz=timezone.utc),
            bin_index=bin_index,
            side=side,
            size=pos.size,
            price=price,
            collateral=-pos.collateral,  # Negative to indicate return
            signal_strength=0.0,
            realized_pnl=realized_pnl,
        ))

        # Remove position
        del self._positions[bin_index]

    def _check_stop_losses(
        self,
        timestamp: int,
        orderbooks: Dict[int, "SimulatedOrderbook"],
    ) -> None:
        """Check all positions for stop-loss and exit if triggered."""
        if not self._positions:
            return

        stop_loss_threshold = self.config.stop_loss_pct
        positions_to_stop = []

        for bin_index, pos in self._positions.items():
            ob = orderbooks.get(bin_index)
            if ob is None:
                continue

            # Calculate current market value
            if pos.side == "YES":
                current_price = ob.yes_bid - self.config.slippage
            else:  # NO
                current_price = ob.no_bid - self.config.slippage

            current_price = max(0.001, current_price)
            current_value = pos.size * current_price
            entry_cost = pos.collateral

            # Check if we're below stop-loss threshold
            if current_value < entry_cost * stop_loss_threshold:
                positions_to_stop.append((bin_index, current_price))
                logger.debug(
                    f"Stop-loss triggered for bin {bin_index}: "
                    f"value ${current_value:.2f} < {stop_loss_threshold:.0%} of cost ${entry_cost:.2f}"
                )

        # Execute stop-loss sells
        for bin_index, sell_price in positions_to_stop:
            self._execute_sell(
                timestamp=timestamp,
                bin_index=bin_index,
                price=sell_price,
            )

        if positions_to_stop:
            logger.info(f"  Stop-loss triggered: closed {len(positions_to_stop)} positions")

    def _close_all_positions(
        self,
        event: EventPriceData,
        timestamp: int,
    ) -> None:
        """Close all open positions at current market prices (early exit)."""
        if not self._positions:
            return

        # Get current orderbooks
        orderbooks = self.price_provider.get_all_orderbooks(event, timestamp)

        # Close each position
        positions_to_close = list(self._positions.keys())
        for bin_index in positions_to_close:
            pos = self._positions[bin_index]
            ob = orderbooks.get(bin_index)

            if ob is None:
                # No orderbook, use last known price or skip
                logger.warning(f"No orderbook for bin {bin_index}, skipping early exit")
                continue

            # Calculate sell price based on position side
            if pos.side == "YES":
                sell_price = max(0.001, ob.yes_bid - self.config.slippage)
            else:  # NO
                sell_price = max(0.001, ob.no_bid - self.config.slippage)

            # Execute the sell
            self._execute_sell(
                timestamp=timestamp,
                bin_index=bin_index,
                price=sell_price,
            )

        logger.info(f"  Closed {len(positions_to_close)} positions in early exit")

    def _execute_buy(
        self,
        timestamp: int,
        bin_index: int,
        side: str,
        price: float,
        signal_strength: float,
    ) -> None:
        """Execute a simulated trade."""
        # Calculate position size (simplified Kelly)
        # Use a fraction of remaining capital, capped by max position
        available = min(
            self._capital * 0.1,  # Max 10% per trade
            self.config.max_position_per_bin,
        )

        if available < 1.0:
            return  # Not enough capital

        # Calculate shares
        size = available / price
        collateral = size * price

        if collateral > self._capital:
            return  # Not enough capital

        # Update capital
        self._capital -= collateral

        # Update or create position
        token_side = "YES" if side == "BUY_YES" else "NO"
        pos_key = (bin_index, token_side)

        if bin_index in self._positions:
            pos = self._positions[bin_index]
            if pos.side == token_side:
                # Add to existing position
                total_size = pos.size + size
                pos.avg_price = (pos.size * pos.avg_price + size * price) / total_size
                pos.size = total_size
                pos.collateral += collateral
            else:
                # Opposite side - would need to handle closing, skip for now
                self._capital += collateral  # Refund
                return
        else:
            self._positions[bin_index] = Position(
                bin_index=bin_index,
                side=token_side,
                size=size,
                avg_price=price,
                collateral=collateral,
            )

        # Record trade
        self._trades.append(Trade(
            timestamp=timestamp,
            datetime=datetime.fromtimestamp(timestamp, tz=timezone.utc),
            bin_index=bin_index,
            side=side,
            size=size,
            price=price,
            collateral=collateral,
            signal_strength=signal_strength,
        ))

    def _calculate_settlement_pnl(self, event: EventPriceData) -> float:
        """Calculate total payout at settlement."""
        total_payout, _ = self._calculate_settlement_pnl_with_details(event)
        return total_payout

    def _calculate_settlement_pnl_with_details(self, event: EventPriceData) -> tuple:
        """
        Calculate P&L from settlement with details for each position.

        At settlement, we receive payouts for winning positions.
        The collateral was already deducted when we bought, so we only
        add back what we receive (not subtract collateral again).

        Returns:
            Tuple of (total_payout, list of settlement details)
        """
        total_payout = 0.0
        details = []
        winner = event.winner_bin_index

        for bin_index, pos in self._positions.items():
            payout = 0.0
            is_win = False

            if bin_index == winner:
                # Winner bin
                if pos.side == "YES":
                    # YES wins: receive $1 per share
                    payout = pos.size * 1.0
                    is_win = True
                else:
                    # NO loses: receive $0
                    is_win = False
            else:
                # Loser bin
                if pos.side == "YES":
                    # YES loses: receive $0
                    is_win = False
                else:
                    # NO wins: receive $1 per share
                    payout = pos.size * 1.0
                    is_win = True

            total_payout += payout

            # Calculate realized P/L for this position
            realized_pnl = payout - pos.collateral

            details.append({
                'bin_index': bin_index,
                'side': pos.side,
                'size': pos.size,
                'collateral': pos.collateral,
                'payout': payout,
                'realized_pnl': realized_pnl,
                'is_win': is_win,
            })

        return total_payout, details

    def _calculate_pnl_by_bin(self, event: EventPriceData) -> Dict[int, float]:
        """Calculate P&L for each bin."""
        pnl_by_bin = {}
        winner = event.winner_bin_index

        for bin_index, pos in self._positions.items():
            if bin_index == winner:
                if pos.side == "YES":
                    pnl = pos.size * 1.0 - pos.collateral
                else:
                    pnl = -pos.collateral
            else:
                if pos.side == "YES":
                    pnl = -pos.collateral
                else:
                    pnl = pos.size * 1.0 - pos.collateral

            pnl_by_bin[bin_index] = pnl

        return pnl_by_bin

    def _log_result(self, result: BacktestResult) -> None:
        """Log backtest results."""
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Backtest Results: {result.event_name}")
        logger.info("=" * 60)
        logger.info(f"Period: {result.start_date} to {result.end_date}")
        logger.info(f"Winner bin: {result.winner_bin}")
        logger.info("-" * 60)
        logger.info(f"Initial Capital: ${result.initial_capital:.2f}")
        logger.info(f"Final Capital:   ${result.final_capital:.2f}")
        logger.info(f"Total P&L:       ${result.total_pnl:+.2f}")
        logger.info(f"Return:          {result.total_return:+.1f}%")
        logger.info("-" * 60)
        logger.info(f"Total Trades:    {result.num_trades}")
        logger.info(f"Win Rate:        {result.win_rate:.1%}")
        logger.info(f"Settlement P&L:  ${result.settlement_pnl:+.2f}")
        logger.info("=" * 60)

        if result.pnl_by_bin:
            logger.info("P&L by Bin:")
            for bin_idx, pnl in sorted(result.pnl_by_bin.items()):
                marker = " (WINNER)" if bin_idx == result.winner_bin else ""
                bin_range = result.bin_ranges.get(bin_idx, str(bin_idx))
                logger.info(f"  {bin_range}: ${pnl:+.2f}{marker}")

        # Log trade history
        if result.trades or result.settlement_details:
            logger.info("")
            logger.info("Trade History:")
            logger.info(f"  {'Time':<20} {'Action':<16} {'Range':<12} {'Size':>8} {'Price':>8} {'Cost':>10} {'P/L':>10}")
            logger.info("  " + "-" * 96)

            # Regular trades
            for t in result.trades:
                time_str = t.datetime.strftime("%Y-%m-%d %H:%M")
                bin_range = result.bin_ranges.get(t.bin_index, str(t.bin_index))
                cost = t.size * t.price if "BUY" in t.side else -t.size * t.price
                pnl_str = f"${t.realized_pnl:>+8.2f}" if t.realized_pnl is not None else "         -"
                logger.info(f"  {time_str:<20} {t.side:<16} {bin_range:<12} {t.size:>8.2f} {t.price:>8.4f} ${cost:>+9.2f} {pnl_str}")

            # Settlement entries
            if result.settlement_details:
                logger.info("  " + "-" * 96)
                logger.info("  Settlement:")
                for s in result.settlement_details:
                    bin_range = result.bin_ranges.get(s['bin_index'], str(s['bin_index']))
                    action = "SETTLEMENT_WIN" if s['is_win'] else "SETTLEMENT_LOSE"
                    payout_price = 1.0 if s['is_win'] else 0.0
                    pnl_str = f"${s['realized_pnl']:>+8.2f}"
                    logger.info(f"  {'(end)':<20} {action:<16} {bin_range:<12} {s['size']:>8.2f} {payout_price:>8.4f} ${s['collateral']:>+9.2f} {pnl_str}")

        logger.info("")


def run_backtest(
    event_dir: str,
    config: Optional[BacktestConfig] = None,
) -> Optional[BacktestResult]:
    """
    Convenience function to run a single backtest.

    Args:
        event_dir: Event directory name
        config: Optional configuration

    Returns:
        BacktestResult or None
    """
    if config is None:
        config = BacktestConfig()

    runner = BacktestRunner(config)
    return runner.run(event_dir)
