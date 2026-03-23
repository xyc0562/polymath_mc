"""
Unified backtest runner that uses the SAME trading logic as production.

This runner uses KellyExecutor with backtest backends, ensuring that
backtest and production trading logic are identical (including optimal sizing).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .data_provider import (
    HistoricalDataProvider,
    CachedPostsProvider,
    EventPriceData,
)
from .replay_seed import (
    ReplaySeedState,
    extract_seed_state_from_log,
    load_seed_state,
)

# Import Kelly trading infrastructure
from ..kelly.config import KellyConfig, EdgeBufferConfig, RateLimitConfig, EventTradingRulesConfig
from ..kelly.portfolio import Portfolio, BinPosition
from ..kelly.executor import KellyExecutor, TickResult
from ..kelly.market_signals import compute_market_consensus_blend
from ..kelly.take_profit import build_late_boundary_take_profit_context
from ..kelly.backtest_backend import (
    SimulationConfig,
    BacktestOrderbookProvider,
    BacktestTradeExecutor,
    create_backtest_portfolio,
)
from ..forecaster.projection import ProjectionModel, AsymmetricProjection, create_projection_model

logger = logging.getLogger(__name__)


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
    utility_gain: float = 0.0
    edge: float = 0.0
    realized_pnl: Optional[float] = None


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

    # Final portfolio state
    final_positions: Dict[int, BinPosition]
    settlement_pnl: float

    # All trades
    trades: List[Trade] = field(default_factory=list)

    # Per-bin P&L
    pnl_by_bin: Dict[int, float] = field(default_factory=dict)

    # Bin ranges for display
    bin_ranges: Dict[int, str] = field(default_factory=dict)

    # Settlement details
    settlement_details: List[Dict] = field(default_factory=list)


@dataclass
class UnifiedBacktestConfig:
    """Configuration for unified backtest."""

    # Capital
    initial_capital: float = 1000.0

    # Market simulation
    spread: float = 0.02  # 2% bid-ask spread
    slippage: float = 0.005  # 0.5% slippage

    # Trading configuration - uses KellyConfig directly to avoid duplication
    # Override specific fields as needed for backtest (e.g., higher rate limits)
    trading: KellyConfig = field(default_factory=lambda: KellyConfig(
        # Backtest-specific overrides
        rate_limit=RateLimitConfig(
            max_orders_per_tick=50,
            min_order_delay_seconds=0.0,
            max_orders_per_minute=1000,
        ),
        max_iters_per_tick=50,
    ))

    # Tick frequency
    tick_interval_seconds: int = 3600  # 1 hour between ticks
    late_tick_interval_seconds: Optional[int] = None
    late_tick_start_hours_before_settlement: Optional[float] = None
    resume_from_timestamp: Optional[int] = None

    # Forecaster settings
    training_days: int = 45

    # Verbose mode: print detailed trade information
    verbose: bool = False
    tweet_data_start_date: date = field(default_factory=lambda: date(2025, 11, 1))

    # Early exit: close all positions X hours before settlement (0 = disabled)
    exit_hours_before_settlement: float = 0.0

    # Projection model type: "asymmetric" (default) or "normal"
    # - asymmetric: Uses actual Monte Carlo samples (preserves right-skew)
    # - normal: Approximates with Normal distribution (symmetric)
    projection_model: str = "asymmetric"

    # Intraday forecaster mode: "ridge" (default) or "bucket"
    # - ridge: Original Ridge regression with F(τ) progress curve
    # - bucket: Bucket-based forecaster with 8 time buckets
    intraday_mode: str = "ridge"

    # Interday forecaster mode: "ewma" (default), "gas", or "pig"
    interday_model: str = "ewma"

    # Event trading rules configuration
    # Controls when trading is allowed based on event duration
    # If None, no duration-based restrictions are applied
    event_trading_rules: Optional[EventTradingRulesConfig] = None

    # CMP nu_scale for COM-Poisson distribution (higher = thinner left tail)
    cmp_nu_scale: float = 1.0

    # Optional intraday historical bootstrap overlay
    use_historical_bootstrap: bool = False
    bootstrap_start_hours: float = 6.0
    bootstrap_full_hours: float = 3.0
    bootstrap_max_blend: float = 0.35
    boundary_silence_overlay: bool = False
    boundary_silence_hours: float = 6.0
    boundary_silence_max_distance: int = 5
    boundary_silence_threshold_start: int = 180
    boundary_silence_threshold_floor: int = 90
    boundary_silence_threshold_step: float = 30.0
    boundary_silence_min_effective_n: float = 5.0

    # Optional seeded replay inputs
    seed_state_path: Optional[str] = None
    seed_log_path: Optional[str] = None
    seed_log_snapshot_timestamp: Optional[int] = None
    seed_log_event_name: Optional[str] = None


class UnifiedBacktestRunner:
    """
    Backtest runner using the SAME trading logic as production.

    Key design: This runner uses KellyExecutor with backtest backends,
    ensuring identical trading logic between backtest and production
    (including optimal sizing via _find_optimal_size_on).
    """

    def __init__(
        self,
        config: UnifiedBacktestConfig,
        price_data_dir: Path = Path("data/price_history"),
        cache_dir: Path = Path("data/backtest_cache"),
    ):
        """
        Initialize unified backtest runner.

        Args:
            config: Backtest configuration
            price_data_dir: Directory containing price history
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

        # Projection model for computing bin probabilities
        self._projection: ProjectionModel = create_projection_model(config.projection_model)

        # State during backtest
        self._trades: List[Trade] = []

    def run(self, event_dir: str) -> Optional[BacktestResult]:
        """
        Run backtest on a specific event.

        Args:
            event_dir: Event directory name

        Returns:
            BacktestResult or None if event not found
        """
        # Load event data
        event = self.price_provider.load_event(event_dir)
        if event is None:
            logger.error(f"Event not found: {event_dir}")
            return None

        logger.info(f"Running unified backtest for: {event.short_name}")
        logger.info(f"  Trading period: {event.start_date} to {event.end_date}")
        logger.info(f"  Counting period: {event.counting_start_date} to {event.counting_end_date}")
        logger.info(f"  Bins: {len(event.bins)}, Winner: bin {event.winner_bin_index}")

        seed_state = self._load_replay_seed_state(event)

        # Log event trading rules if configured
        if self.config.event_trading_rules is not None:
            event_duration_days = (event.counting_end_date - event.counting_start_date).days
            rules = self.config.event_trading_rules.get_rules_for_event(event_duration_days)
            logger.info(f"  Event duration: {event_duration_days} days → using '{rules.name}' trading rules")
            if rules.require_counting_started:
                logger.info(f"    - Require counting started: yes")
            if rules.max_hours_before_counting is not None:
                logger.info(f"    - Max hours before counting: {rules.max_hours_before_counting}")
            if rules.max_days_before_settlement is not None:
                logger.info(f"    - Max days before settlement: {rules.max_days_before_settlement}")
            logger.info(f"    - Min hours before settlement: {rules.min_hours_before_settlement}")

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

        # Initialize forecaster (returns forecaster and backtest-period posts)
        forecaster, backtest_posts = self._create_forecaster(event)

        # Index for tracking which backtest posts have been added
        backtest_posts_idx = 0

        # Initialize Kelly trading infrastructure
        kelly_config = self._create_kelly_config()
        token_ids = {b.bin_index: f"token_{b.bin_index}" for b in event.bins}

        # Create initial probability distribution (uniform before first forecast)
        num_bins = len(event.bins)
        initial_probs = [1.0 / num_bins] * num_bins
        bin_upper_bounds = [b.upper_bound for b in event.bins]

        # Create portfolio and backends
        portfolio = create_backtest_portfolio(
            initial_capital=self.config.initial_capital,
            probabilities=initial_probs,
            bin_upper_bounds=bin_upper_bounds,
        )
        if seed_state is not None:
            self._apply_replay_seed_state(
                portfolio=portfolio,
                token_ids=token_ids,
                event=event,
                seed_state=seed_state,
            )

        # Set phantom capital for Kelly utility inflation
        multiplier = kelly_config.collateral.capital_multiplier
        if multiplier > 1.0:
            portfolio.phantom_capital = self.config.initial_capital * (multiplier - 1.0)
            logger.info(f"  Phantom capital: ${portfolio.phantom_capital:.2f} (multiplier={multiplier}x)")
            # Reset collateral multiplier to 1.0 so collateral limits (c_bin_max,
            # virtual_c_event_max) stay at real capital levels. Phantom capital
            # already handles Kelly utility inflation independently. Without this,
            # collateral limits scale by multiplier, allowing oversized trades
            # that drain real capital at multiplier-x rate.
            kelly_config.collateral.capital_multiplier = 1.0

        backend_config = SimulationConfig(
            spread=self.config.spread,
            slippage=self.config.slippage,
            log_trades=False,
        )

        orderbook_provider = BacktestOrderbookProvider(
            config=backend_config,
            token_ids=token_ids,
        )

        trade_executor = BacktestTradeExecutor(
            config=backend_config,
            portfolio=portfolio,
        )

        # Create executor (same class as production, with backtest backends)
        executor = KellyExecutor(
            config=kelly_config,
            portfolio=portfolio,
            orderbook_provider=orderbook_provider,
            trade_executor=trade_executor,
            token_ids=token_ids,
            on_trade=lambda r: self._record_trade(r),
            event_name=event.short_name,
        )

        # Reset trade history and verbose flags
        self._trades = []
        self._trade_table_header_printed = False
        self._capital_exhausted_logged = False

        # Calculate settlement time for exit logic
        # Settlement is at 12pm EST (noon ET) on the end date of counting
        # Contract days use noon ET boundaries: "Dec 19 - Dec 26" means Dec 19 12:00 to Dec 26 12:00
        # So counting_end_date = Dec 26 means settlement at Dec 26 12:00 ET
        from zoneinfo import ZoneInfo
        est_tz = ZoneInfo("America/New_York")
        settlement_dt = datetime.combine(
            event.counting_end_date,
            datetime.min.time().replace(hour=12),  # 12pm noon ET = settlement
            tzinfo=est_tz
        )
        # Calculate counting start time (noon ET on counting_start_date)
        counting_start_dt = datetime.combine(
            event.counting_start_date,
            datetime.min.time().replace(hour=12),
            tzinfo=est_tz
        )
        # Also use this to stop processing ticks after settlement
        settlement_ts = int(settlement_dt.timestamp())
        exit_window_start = settlement_dt - timedelta(hours=self.config.exit_hours_before_settlement)
        in_exit_mode = False

        # Sample timestamps
        tick_interval = self.config.tick_interval_seconds
        sampled_timestamps = self._sample_timestamps(
            event.all_timestamps,
            tick_interval,
            start_at_ts=self.config.resume_from_timestamp,
            settlement_ts=settlement_ts,
        )
        if self.config.resume_from_timestamp is not None:
            resume_dt = datetime.fromtimestamp(self.config.resume_from_timestamp, tz=timezone.utc)
            logger.info(f"  Replay resume timestamp: {resume_dt.isoformat()}")
        if self.config.late_tick_interval_seconds is not None and self.config.late_tick_start_hours_before_settlement is not None:
            logger.info(
                "  Sampled %d ticks (every %ss, then every %ss inside final %.1fh)",
                len(sampled_timestamps),
                tick_interval,
                self.config.late_tick_interval_seconds,
                self.config.late_tick_start_hours_before_settlement,
            )
        else:
            logger.info(f"  Sampled {len(sampled_timestamps)} ticks (every {tick_interval}s)")
        if not sampled_timestamps:
            logger.error("No sampled ticks available after applying replay start timestamp")
            return None

        # Track contract day for interday model updates
        last_contract_day = None

        # Run through sampled timestamps
        for i, ts in enumerate(sampled_timestamps):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)

            if dt.date() < event.start_date:
                continue

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

            # Stop at settlement (noon ET on counting_end_date)
            if ts >= settlement_ts:
                break

            # Check for early exit mode
            if self.config.exit_hours_before_settlement > 0 and dt >= exit_window_start and not in_exit_mode:
                in_exit_mode = True
                logger.info(f"  Entering exit mode at {dt.strftime('%Y-%m-%d %H:%M')}")
                # Note: In unified system, we continue running but T_stop will prevent new trades
                # and the executor will hold positions to settlement
                continue

            if in_exit_mode:
                continue

            # Check event trading rules (duration-based restrictions)
            if self.config.event_trading_rules is not None:
                # Convert dt to EST for consistent comparison with counting_start_dt and settlement_dt
                dt_est = dt.astimezone(est_tz)
                is_allowed, reason = self._check_trading_rules(
                    event=event,
                    current_dt=dt_est,
                    counting_start_dt=counting_start_dt,
                    settlement_dt=settlement_dt,
                )
                if not is_allowed:
                    if i == 0 or i % 100 == 0:  # Log occasionally to avoid spam
                        logger.debug(f"  Trading blocked: {reason}")
                    continue

            # Log capital exhaustion but DON'T skip — we may still need to SELL positions
            if portfolio.available_capital < 1.0:
                if not getattr(self, '_capital_exhausted_logged', False):
                    logger.info(f"  Capital exhausted (${portfolio.available_capital:.2f} remaining) - continuing for potential sells")
                    self._capital_exhausted_logged = True

            # Update forecast and probabilities
            # The forecaster returns the TOTAL expected count (past + future).
            # It internally tracks past counts using contract day boundaries (noon ET).
            # We need to get the current count using the SAME boundaries as the forecaster
            # to properly determine dead bins.

            # counting_end_date is the settlement date (e.g., Dec 26 for "Dec 19 - Dec 26")
            # Settlement happens at noon on this date, and last counting day is the day before
            forecast = forecaster.forecast_for_event_window(
                market_start_date=event.counting_start_date,
                settlement_date=event.counting_end_date,
                now=dt,
            )

            # Get current count using the forecaster's boundary conventions
            # This ensures consistency between our dead bin detection and the forecaster's logic
            # get_contract_day_bounds returns (start_dt, end_dt) for the contract day
            counting_start_dt, _ = forecaster.contract_utils.get_contract_day_bounds(event.counting_start_date)
            counting_start_ts = int(counting_start_dt.timestamp())

            # For current count, use the current tick timestamp
            current_count = self.posts_provider.count_posts_in_range(
                start_ts=counting_start_ts,
                end_ts=ts,
            )

            # Compute bin probabilities using the configured projection model
            probabilities = self._compute_bin_probabilities(
                event=event,
                forecast=forecast,
                current_count=current_count,
            )

            # Log forecast std for debugging (projection model doesn't affect this)
            if i % 20 == 0:
                logger.debug(
                    f"  Forecast: mean={forecast.mean:.1f}, std={forecast.std:.1f}, "
                    f"projection={self._projection.name}"
                )

            # Determine dead bins
            dead_bins = [
                b.bin_index for b in event.bins
                if b.upper_bound < current_count
            ]

            # Update orderbooks from historical data
            simulated_obs = self.price_provider.get_all_orderbooks(event, ts)
            orderbook_provider.update_from_simulated(simulated_obs, ts)
            current_orderbooks = orderbook_provider.get_all_orderbooks()

            # Calculate hours to settlement before any quote-aware probability shaping.
            hours_to_settlement = (settlement_dt - dt).total_seconds() / 3600.0

            nowcast = getattr(forecaster, "nowcast", None)
            if hasattr(nowcast, "apply_late_boundary_silence_overlay"):
                contract_date = forecaster.contract_utils.get_contract_date(dt)
                current_day_events = forecaster.event_store.get_contract_day_events(contract_date)
                probabilities = nowcast.apply_late_boundary_silence_overlay(
                    probabilities=probabilities,
                    current_count=current_count,
                    events=current_day_events,
                    contract_date=contract_date,
                    now=dt,
                    hours_remaining=hours_to_settlement,
                    bin_ranges=[(b.lower_bound, b.upper_bound) for b in event.bins],
                )
                overlay_context = getattr(nowcast, "_last_boundary_overlay", None)
                if overlay_context and overlay_context.get("active"):
                    logger.debug(
                        "  Boundary silence overlay: distance=%s silence=%smin threshold=%.0f n_eff=%.2f",
                        overlay_context.get("distance_to_next_bin"),
                        overlay_context.get("silence_minutes"),
                        overlay_context.get("silence_threshold_minutes", 0.0),
                        overlay_context.get("n_eff", 0.0),
                    )

            probabilities, blend_context = compute_market_consensus_blend(
                probabilities=probabilities,
                dead_bins=dead_bins,
                orderbooks=simulated_obs,
                consensus_config=self.config.trading.market_consensus,
                hours_remaining=hours_to_settlement,
            )
            if blend_context is not None:
                if "alpha" in blend_context:
                    logger.debug(
                        "  Market consensus blend: alpha=%.3f coverage=%.2f avg_spread=%.3f gap=%.3f",
                        blend_context.get("alpha", 1.0),
                        blend_context.get("coverage_ratio", 0.0),
                        blend_context.get("avg_spread", 0.0),
                        blend_context.get("gap", 0.0),
                    )
                else:
                    logger.debug(
                        "  Market consensus blend skipped: reason=%s coverage=%.2f avg_spread=%.3f",
                        blend_context.get("skipped", "unknown"),
                        blend_context.get("coverage_ratio", 0.0),
                        blend_context.get("avg_spread", 0.0),
                    )

            # Update portfolio with new probabilities and dead bins
            portfolio.probabilities = probabilities
            portfolio.dead_bins = dead_bins
            portfolio.num_bins = num_bins

            take_profit_context = build_late_boundary_take_profit_context(
                config=self.config.trading.late_boundary_take_profit,
                current_count=current_count,
                bin_ranges=[(b.lower_bound, b.upper_bound) for b in event.bins],
                hours_remaining=hours_to_settlement,
                silence_minutes=(
                    getattr(getattr(forecaster, "nowcast", None), "_last_impulse", {}) or {}
                ).get("silence_min"),
                portfolio=portfolio,
                orderbooks=current_orderbooks,
            )
            executor.set_late_boundary_take_profit_context(take_profit_context)
            if take_profit_context.active:
                logger.debug(
                    "  Late-boundary take-profit: bin=%s silence=%smin threshold=%.0f distance=%s ref_vwap=%.3f sell=%.0f-%.0f strength=%.2f",
                    take_profit_context.current_bin_index,
                    int(take_profit_context.silence_minutes),
                    take_profit_context.silence_threshold_minutes,
                    take_profit_context.distance_to_next_bin,
                    take_profit_context.reference_vwap,
                    take_profit_context.size_floor,
                    take_profit_context.size_cap,
                    take_profit_context.strength,
                )

            # Store context for verbose logging
            self._current_tick_context = {
                'timestamp': ts,
                'datetime': dt,
                'current_count': current_count,
                'forecast_mean': forecast.mean,
                'forecast_std': forecast.std,
                'hours_to_settlement': hours_to_settlement,
                'probabilities': probabilities,
                'dead_bins': dead_bins,
                'bin_ranges': {b.bin_index: f"{b.lower_bound}-{b.upper_bound}" for b in event.bins},
                'orderbooks': simulated_obs,
                'event': event,
                'executor': executor,  # For P&L calculation on sells
            }
            self._tick_header_printed = False

            # Run Kelly executor tick
            tick_result = executor.run_tick_sync(hours_to_settlement)

            # Verbose: print tick summary if trades occurred
            if self.config.verbose and tick_result.num_executed > 0:
                logger.info(f"    → Executed {tick_result.num_executed} trades, total utility gain: {tick_result.total_utility_gain:.6f}")

            # Log progress
            if i % 20 == 0:
                logger.info(
                    f"  Tick {i}/{len(sampled_timestamps)}: "
                    f"{dt.strftime('%Y-%m-%d %H:%M')} | "
                    f"capital=${portfolio.capital:.2f}, "
                    f"trades={len(self._trades)}, "
                    f"candidates={tick_result.num_candidates}, "
                    f"executed={tick_result.num_executed}"
                )

        # Calculate settlement P&L
        settlement_pnl, settlement_details = self._calculate_settlement_pnl(
            portfolio, event.winner_bin_index
        )

        # Calculate final capital
        final_capital = portfolio.capital + settlement_pnl

        # Calculate trade stats
        pnl_by_bin = self._calculate_pnl_by_bin(portfolio, event.winner_bin_index)
        winning_trades = sum(1 for pnl in pnl_by_bin.values() if pnl > 0)
        losing_trades = sum(1 for pnl in pnl_by_bin.values() if pnl < 0)

        # Build bin ranges dict
        bin_ranges = {
            b.bin_index: f"{b.lower_bound}-{b.upper_bound}"
            for b in event.bins
        }

        # Verbose: print settlement summary table
        if self.config.verbose and settlement_details:
            logger.info(f"\n{'='*130}")
            logger.info(f"SETTLEMENT: Winner = bin {event.winner_bin_index} ({bin_ranges.get(event.winner_bin_index, '?')})")
            logger.info(f"{'='*130}")
            logger.info(f"{'Bin':>4} {'Range':<12} {'Side':<4} {'Shares':>8} {'AvgCost':>8} {'Collateral':>10} "
                  f"{'Payout':>10} {'P&L':>10} {'Result':<6}")
            logger.info(f"{'-'*130}")

            for detail in sorted(settlement_details, key=lambda x: x['bin_index']):
                bin_idx = detail['bin_index']
                rng = bin_ranges.get(bin_idx, str(bin_idx))
                side = detail['side']
                size = detail['size']
                collateral = detail['collateral']
                payout = detail['payout']
                pnl = detail['realized_pnl']
                is_win = detail['is_win']
                result_str = "WIN" if is_win else "LOSS"
                winner_mark = " <--" if bin_idx == event.winner_bin_index else ""

                # Calculate average cost
                pos = portfolio.positions.get(bin_idx)
                if pos:
                    avg_cost = pos.yes_avg_cost if side == 'YES' else pos.no_avg_cost
                else:
                    avg_cost = collateral / size if size > 0 else 0

                logger.info(f"{bin_idx:>4} {rng:<12} {side:<4} {size:>8.1f} {avg_cost:>8.3f} ${collateral:>9.2f} "
                      f"${payout:>9.2f} ${pnl:>+9.2f} {result_str:<6}{winner_mark}")

            logger.info(f"{'-'*130}")
            logger.info(f"{'TOTAL':<17} {'':<4} {'':<8} {'':<8} ${sum(d['collateral'] for d in settlement_details):>9.2f} "
                  f"${sum(d['payout'] for d in settlement_details):>9.2f} "
                  f"${sum(d['realized_pnl'] for d in settlement_details):>+9.2f}")
            logger.info(f"{'='*130}")

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
            final_positions=portfolio.positions.copy(),
            settlement_pnl=settlement_pnl,
            trades=self._trades.copy(),
            pnl_by_bin=pnl_by_bin,
            bin_ranges=bin_ranges,
            settlement_details=settlement_details,
        )

        self._log_result(result)
        return result

    def _create_kelly_config(self) -> KellyConfig:
        """Return the trading config (KellyConfig) from backtest config."""
        return self.config.trading

    def _check_trading_rules(
        self,
        event: EventPriceData,
        current_dt: datetime,
        counting_start_dt: datetime,
        settlement_dt: datetime,
    ) -> tuple:
        """
        Check if trading is allowed based on event trading rules.

        Returns:
            Tuple of (is_allowed, reason) where reason explains why trading is blocked
        """
        if self.config.event_trading_rules is None:
            return True, ""

        # Calculate event duration in days
        event_duration_days = (event.counting_end_date - event.counting_start_date).days

        # Get rules for this event duration
        rules = self.config.event_trading_rules.get_rules_for_event(event_duration_days)

        # Check min_hours_before_settlement (stop trading close to settlement)
        hours_to_settlement = (settlement_dt - current_dt).total_seconds() / 3600.0
        if hours_to_settlement < rules.min_hours_before_settlement:
            return False, f"Too close to settlement ({hours_to_settlement:.1f}h < {rules.min_hours_before_settlement}h min)"

        # Check max_days_before_settlement (don't trade too far from settlement)
        days_to_settlement = hours_to_settlement / 24.0
        if rules.max_days_before_settlement is not None:
            if days_to_settlement > rules.max_days_before_settlement:
                return False, f"Too far from settlement ({days_to_settlement:.1f}d > {rules.max_days_before_settlement}d max)"

        # Check if counting has started
        counting_started = current_dt >= counting_start_dt

        if rules.require_counting_started:
            if not counting_started:
                return False, "Counting period has not started yet"
        else:
            # Check max_hours_before_counting if set
            if rules.max_hours_before_counting is not None and not counting_started:
                hours_before_counting = (counting_start_dt - current_dt).total_seconds() / 3600.0
                if hours_before_counting > rules.max_hours_before_counting:
                    return False, f"Too far before counting ({hours_before_counting:.1f}h > {rules.max_hours_before_counting}h max)"

        return True, ""

    def _record_trade(self, result) -> None:
        """Record a trade from the executor callback."""
        from ..kelly.candidates import TradeAction

        candidate = result.candidate
        action_map = {
            TradeAction.BUY_YES: "BUY_YES",
            TradeAction.SELL_YES: "SELL_YES",
            TradeAction.BUY_NO: "BUY_NO",
            TradeAction.SELL_NO: "SELL_NO",
        }

        # Get timestamp from context if available
        ctx = getattr(self, '_current_tick_context', {})
        trade_ts = ctx.get('timestamp', int(datetime.now(timezone.utc).timestamp()))
        trade_dt = ctx.get('datetime', datetime.now(timezone.utc))

        trade = Trade(
            timestamp=trade_ts,
            datetime=trade_dt,
            bin_index=candidate.bin_index,
            side=action_map.get(candidate.action, str(candidate.action)),
            size=result.filled_size,
            price=result.filled_price,
            collateral=result.filled_size * result.filled_price,
            utility_gain=candidate.utility_gain,
            edge=candidate.edge,
        )
        self._trades.append(trade)

        # Verbose logging
        if self.config.verbose:
            self._print_trade_details(candidate, result, ctx)

    def _print_trade_details(self, candidate, result, ctx):
        """Print trade as a table row using logger for consistent output ordering."""
        # Print table header on first trade of the event
        if not getattr(self, '_trade_table_header_printed', False):
            logger.info(f"{'='*115}")
            logger.info(f"{'#':>4} {'Action':<8} {'Bin':>3} {'Range':<12} {'Size':>6} {'Price':>6} "
                       f"{'Cost':>8} {'Model%':>7} {'Mkt%':>6} {'Edge':>7} {'Odds':>5} {'P&L':>10}")
            logger.info(f"{'-'*115}")
            self._trade_table_header_printed = True

        # Print tick separator on first trade of this tick
        if not getattr(self, '_tick_header_printed', False):
            from zoneinfo import ZoneInfo
            dt = ctx.get('datetime')
            forecast_mean = ctx.get('forecast_mean', 0)
            forecast_std = ctx.get('forecast_std', 0)
            current_count = ctx.get('current_count', 0)
            hours_to_settlement = ctx.get('hours_to_settlement', 0)
            probs = ctx.get('probabilities', [])
            event = ctx.get('event')

            est_dt = dt.astimezone(ZoneInfo("America/New_York"))

            # Compact tick info
            top_bins_str = ""
            if event:
                prob_bins = [(i, p, event.bins[i]) for i, p in enumerate(probs) if p > 0.01]
                prob_bins.sort(key=lambda x: x[1], reverse=True)
                top_bins_str = " | Top: " + ", ".join(
                    f"b{idx}={prob:.0%}" for idx, prob, _ in prob_bins[:3]
                )

            logger.info(f"--- {est_dt.strftime('%m-%d %H:%M')} | T-{hours_to_settlement:.1f}h | cnt={current_count} | μ={forecast_mean:.0f}±{forecast_std:.0f}{top_bins_str} ---")
            self._tick_header_printed = True

        bin_idx = candidate.bin_index
        bin_range = ctx.get('bin_ranges', {}).get(bin_idx, str(bin_idx))
        probs = ctx.get('probabilities', [])
        prob = probs[bin_idx] if bin_idx < len(probs) else 0.0

        # Get market implied probability
        orderbooks = ctx.get('orderbooks', {})
        ob = orderbooks.get(bin_idx)
        mkt_prob = ob.yes_ask if ob else 0.0

        # Calculate odds
        price = result.filled_price
        if candidate.action.value in ['BUY_YES', 'BUY_NO']:
            odds = (1.0 - price) / price if price > 0 else 0
        else:
            odds = price / (1.0 - price) if price < 1 else 0

        # Format action (shorter)
        action_short = candidate.action.value.replace('_', ' ')

        # Calculate P&L for sells (realized P&L)
        pnl_str = "-"
        if candidate.action.value in ['SELL_YES', 'SELL_NO']:
            # Get executor to access portfolio
            executor = ctx.get('executor')
            if executor and hasattr(executor, 'portfolio'):
                pos = executor.portfolio.get_position(bin_idx)
                if pos:
                    if candidate.action.value == 'SELL_YES' and pos.yes_avg_cost > 0:
                        avg_cost = pos.yes_avg_cost
                        pnl = (price - avg_cost) * result.filled_size
                        pnl_pct = (price - avg_cost) / avg_cost * 100
                        pnl_str = f"${pnl:+.2f} ({pnl_pct:+.0f}%)"
                    elif candidate.action.value == 'SELL_NO' and pos.no_avg_cost > 0:
                        avg_cost = pos.no_avg_cost
                        pnl = (price - avg_cost) * result.filled_size
                        pnl_pct = (price - avg_cost) / avg_cost * 100
                        pnl_str = f"${pnl:+.2f} ({pnl_pct:+.0f}%)"

        logger.info(f"{len(self._trades):>4} {action_short:<8} {bin_idx:>3} {bin_range:<12} "
                   f"{result.filled_size:>6.1f} {price:>6.3f} ${result.filled_size * price:>7.2f} "
                   f"{prob:>6.1%} {mkt_prob:>5.1%} {candidate.edge:>6.1%} {odds:>5.1f}x {pnl_str:>10}")

    def _sample_timestamps(
        self,
        timestamps: List[int],
        interval_seconds: int,
        start_at_ts: Optional[int] = None,
        settlement_ts: Optional[int] = None,
    ) -> List[int]:
        """Sample timestamps at regular intervals."""
        if not timestamps:
            return []

        eligible = timestamps
        if start_at_ts is not None:
            eligible = [ts for ts in timestamps if ts >= start_at_ts]
            if not eligible:
                return []

        late_interval = self.config.late_tick_interval_seconds
        late_start_hours = self.config.late_tick_start_hours_before_settlement
        if (
            settlement_ts is None
            or late_interval is None
            or late_start_hours is None
        ):
            return self._sample_timestamps_at_interval(eligible, interval_seconds)

        switch_ts = settlement_ts - int(late_start_hours * 3600.0)
        early = [ts for ts in eligible if ts < switch_ts]
        late = [ts for ts in eligible if ts >= switch_ts]

        sampled: List[int] = []
        if early:
            sampled.extend(self._sample_timestamps_at_interval(early, interval_seconds))
        if late:
            sampled.extend(self._sample_timestamps_at_interval(late, late_interval))
        return sampled

    @staticmethod
    def _sample_timestamps_at_interval(
        timestamps: List[int],
        interval_seconds: int,
    ) -> List[int]:
        """Sample a sorted timestamp series at a fixed interval."""
        if not timestamps:
            return []

        sampled = [timestamps[0]]
        last_ts = timestamps[0]

        for ts in timestamps[1:]:
            if ts - last_ts >= interval_seconds:
                sampled.append(ts)
                last_ts = ts

        return sampled

    def _load_replay_seed_state(self, event: EventPriceData) -> Optional[ReplaySeedState]:
        """Load optional seeded replay state for this event."""
        if self.config.seed_state_path:
            seed_state = load_seed_state(Path(self.config.seed_state_path))
        elif self.config.seed_log_path:
            if self.config.seed_log_snapshot_timestamp is None:
                raise ValueError("seed_log_snapshot_timestamp is required when seed_log_path is set")
            seed_event_name = self.config.seed_log_event_name or event.short_name
            seed_state = extract_seed_state_from_log(
                log_path=Path(self.config.seed_log_path),
                event_name=seed_event_name,
                snapshot_timestamp=self.config.seed_log_snapshot_timestamp,
            )
        else:
            return None

        if seed_state.event_budget is not None and abs(seed_state.event_budget - self.config.initial_capital) > 0.05:
            logger.warning(
                "Replay seed event budget ($%.2f) differs from backtest capital ($%.2f)",
                seed_state.event_budget,
                self.config.initial_capital,
            )
        return seed_state

    def _apply_replay_seed_state(
        self,
        portfolio: Portfolio,
        token_ids: Dict[int, str],
        event: EventPriceData,
        seed_state: ReplaySeedState,
    ) -> None:
        """Apply a seeded portfolio state before replay begins."""
        for bin_index, seeded in seed_state.positions.items():
            if bin_index not in token_ids:
                raise ValueError(
                    f"Replay seed references bin {bin_index}, which is not present in event {event.short_name}"
                )

            position = portfolio.ensure_position(bin_index, token_ids[bin_index])
            position.yes_shares = seeded.yes_shares
            position.yes_avg_cost = seeded.yes_avg_cost
            position.no_shares = seeded.no_shares
            position.no_avg_cost = seeded.no_avg_cost
            position.collateral_used = seeded.collateral_used

        invested = portfolio.total_collateral_used
        if seed_state.available_capital is not None:
            portfolio.capital = max(0.0, seed_state.available_capital)
        else:
            portfolio.capital = max(0.0, self.config.initial_capital - invested)

        logger.info(
            "  Seeded replay state from %s: available=$%.2f, invested=$%.2f, positions=%d",
            seed_state.source,
            portfolio.capital,
            invested,
            len([p for p in portfolio.positions.values() if p.has_yes_position or p.has_no_position]),
        )

    def _create_forecaster(self, event: EventPriceData):
        """
        Create and fit a forecaster for the backtest.

        Returns:
            Tuple of (forecaster, backtest_posts) where backtest_posts are posts
            from the event period that will be added incrementally during backtest.
        """
        from ..forecaster.config import ForecasterConfig, MonteCarloConfig, BucketNowcastConfig
        from ..forecaster.forecaster import TweetCountForecaster
        from ..forecaster.data import EventStore, ContractDayUtils, XTrackerClient, TweetEvent
        from datetime import timezone as tz
        from zoneinfo import ZoneInfo

        # Use fixed seed for reproducible backtests
        mc_config = MonteCarloConfig(n_simulations=10000, random_seed=42,
                                     cmp_nu_scale=self.config.cmp_nu_scale)
        bucket_config = BucketNowcastConfig(
            cmp_nu_scale=self.config.cmp_nu_scale,
            use_historical_bootstrap=self.config.use_historical_bootstrap,
            bootstrap_start_hours=self.config.bootstrap_start_hours,
            bootstrap_full_hours=self.config.bootstrap_full_hours,
            bootstrap_max_blend=self.config.bootstrap_max_blend,
            late_boundary_silence=BucketNowcastConfig.LateBoundarySilenceConfig(
                enabled=self.config.boundary_silence_overlay,
                start_hours=self.config.boundary_silence_hours,
                max_distance_to_next_bin=self.config.boundary_silence_max_distance,
                silence_threshold_start_minutes=self.config.boundary_silence_threshold_start,
                silence_threshold_floor_minutes=self.config.boundary_silence_threshold_floor,
                silence_threshold_step_per_hour=self.config.boundary_silence_threshold_step,
                min_effective_n=self.config.boundary_silence_min_effective_n,
            ),
        )
        config = ForecasterConfig(
            monte_carlo=mc_config,
            bucket_nowcast=bucket_config,
            intraday_mode=self.config.intraday_mode,
            interday_model=self.config.interday_model,
        )
        contract_utils = ContractDayUtils(
            timezone=config.timezone,
            boundary_hour=config.contract_boundary_hour,
        )

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

        for post in self.posts_provider._posts:
            try:
                ts = post.get("createdAt") or post.get("timestamp")
                if ts is None:
                    continue

                if isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(ts, tz=tz.utc)
                elif isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                else:
                    continue

                content = post.get("content", "")
                event_type = "retweet" if content.startswith("RT @") else "tweet"

                tweet_event = TweetEvent(
                    timestamp=dt,
                    event_type=event_type,
                    event_id=post.get("platformId") or post.get("id"),
                    source="xtracker",
                )

                # Only add training period posts to EventStore initially
                # Backtest period posts will be added incrementally
                if dt < cutoff_dt:
                    event_store.add_event(tweet_event)
                    training_posts_count += 1
                else:
                    # Store for incremental addition during backtest
                    backtest_posts.append((int(dt.timestamp()), tweet_event))

            except (ValueError, KeyError):
                continue

        # Sort backtest posts by timestamp for efficient incremental addition
        backtest_posts.sort(key=lambda x: x[0])

        logger.info(
            f"Loaded {training_posts_count} training posts into event store, "
            f"{len(backtest_posts)} backtest posts will be added incrementally"
        )

        forecaster = TweetCountForecaster(config, event_store=event_store)
        forecaster.fit(
            n_days=self.config.training_days,
            skip_fetch=True,
            as_of_date=event.start_date,
        )

        return forecaster, backtest_posts

    def _compute_bin_probabilities(
        self,
        event: EventPriceData,
        forecast: "ForecastResult",
        current_count: int,
    ) -> List[float]:
        """
        Compute probability for each bin using the configured projection model.

        The projection model determines how bin probabilities are computed:
        - AsymmetricProjection: Uses actual MC samples (preserves right-skew)
        - NormalProjection: Approximates with Normal (symmetric)
        """
        # Build bins list from event data
        bins = [(b.lower_bound, b.upper_bound) for b in event.bins]

        # Use projection model to compute bin probabilities
        probabilities = self._projection.compute_bin_probabilities(
            forecast=forecast,
            bins=bins,
            shift=0,  # No shift - forecast already includes past counts
            floor=current_count,  # Samples can't be below current count
        )

        # Renormalize to sum to 1.0
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]

        return probabilities

    def _calculate_settlement_pnl(
        self,
        portfolio: Portfolio,
        winner_bin: Optional[int],
    ) -> tuple:
        """
        Calculate P&L from settlement.

        Returns:
            Tuple of (total_payout, list of settlement details)

        Note: YES and NO positions are reported separately since a bin can have both.
        Collateral is calculated per side as shares × avg_cost.
        """
        total_payout = 0.0
        details = []

        for bin_index, pos in portfolio.positions.items():
            # Handle YES position
            if pos.has_yes_position:
                yes_collateral = pos.yes_shares * pos.yes_avg_cost
                if bin_index == winner_bin:
                    yes_payout = pos.yes_shares * 1.0
                    yes_is_win = True
                else:
                    yes_payout = 0.0
                    yes_is_win = False

                total_payout += yes_payout
                details.append({
                    'bin_index': bin_index,
                    'side': 'YES',
                    'size': pos.yes_shares,
                    'collateral': yes_collateral,
                    'payout': yes_payout,
                    'realized_pnl': yes_payout - yes_collateral,
                    'is_win': yes_is_win,
                })

            # Handle NO position
            if pos.has_no_position:
                no_collateral = pos.no_shares * pos.no_avg_cost
                if bin_index == winner_bin:
                    # NO loses in winner bin
                    no_payout = 0.0
                    no_is_win = False
                else:
                    # NO wins in loser bin
                    no_payout = pos.no_shares * 1.0
                    no_is_win = True

                total_payout += no_payout
                details.append({
                    'bin_index': bin_index,
                    'side': 'NO',
                    'size': pos.no_shares,
                    'collateral': no_collateral,
                    'payout': no_payout,
                    'realized_pnl': no_payout - no_collateral,
                    'is_win': no_is_win,
                })

        return total_payout, details

    def _calculate_pnl_by_bin(
        self,
        portfolio: Portfolio,
        winner_bin: Optional[int],
    ) -> Dict[int, float]:
        """Calculate total P&L for each bin (realized trading P&L + settlement P&L)."""
        pnl_by_bin = {}

        for bin_index, pos in portfolio.positions.items():
            has_position = pos.has_yes_position or pos.has_no_position
            has_realized = abs(pos.realized_pnl) > 0.001
            if not (has_position or has_realized):
                continue

            # Start with realized P&L from trades already closed
            pnl = pos.realized_pnl

            # Add settlement P&L for positions still held
            is_winner = (bin_index == winner_bin)

            if pos.has_yes_position:
                yes_payout = pos.yes_shares * 1.0 if is_winner else 0.0
                yes_cost = pos.yes_shares * pos.yes_avg_cost
                pnl += yes_payout - yes_cost

            if pos.has_no_position:
                no_payout = pos.no_shares * 1.0 if not is_winner else 0.0
                no_cost = pos.no_shares * pos.no_avg_cost
                pnl += no_payout - no_cost

            pnl_by_bin[bin_index] = pnl

        return pnl_by_bin

    def _log_result(self, result: BacktestResult) -> None:
        """Log backtest results."""
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Unified Backtest Results: {result.event_name}")
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

        logger.info("")
