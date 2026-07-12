"""
Configuration dataclasses for Kelly criterion trading.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


@dataclass
class EdgeBufferConfig:
    """
    Configuration for edge buffer using stake-based ROI model.

    This model requires edge proportional to what you're RISKING, not buying:
    - Buy YES: risk p_m to win (1 - p_m)
    - Buy NO: risk (1 - p_m) to win p_m

    Formulas:
    - Buy YES if: p_f - p_m >= c + r * p_m
      → YES max price: p_m <= (p_f - c) / (1 + r)
    - Buy NO if: p_m - p_f >= c + r * (1 - p_m)
      → YES min price: p_m >= (p_f + c + r) / (1 + r)

    Where:
    - c = friction (fees + slippage + spread) in probability points
    - r = required ROI on stake for model risk

    This naturally creates asymmetry:
    - At p_f=4%: Buy YES needs ~2pp edge, Buy NO needs ~6.5pp edge
    - At p_f=96%: Buy YES needs ~6.5pp edge, Buy NO needs ~2pp edge
    """

    # Required ROI on stake (e.g., 0.05 = 5%)
    required_roi: float = 0.0

    # Friction in probability points for middle range (10% < p < 90%)
    friction_mid: float = 0.02  # 2%

    # Friction in probability points for tails (p <= 10% or p >= 90%)
    friction_tail: float = 0.04  # 4%

    # Threshold for tail zone
    tail_threshold: float = 0.10  # 10%

    # Minimum perceived probability to trade (from our model)
    # Don't trade if our model assigns probability < this threshold
    # Set to 0 to disable.
    min_perceived_prob: float = 0.03  # 3%

    # Minimum market price to trade (avoids illiquid tail bets)
    # Don't buy YES if market price < this, don't buy NO if market price < this
    # Set to 0 to disable.
    min_market_price: float = 0.01  # 1%

    # Require two-sided liquidity (both bid and ask must exist)
    # If True, skip bins where only one side has liquidity
    require_two_sided_liquidity: bool = True

    # Maximum spread as ratio of bid price
    # e.g., 2.0 means if bid=0.01, ask can be at most 0.03 (spread/bid = 0.02/0.01 = 2.0)
    # Spread ratio = (ask - bid) / bid
    # Set to 0 to disable.
    max_spread_ratio: float = 2.0

    # Sell-side friction in probability points.
    # Sells require market price >= fair_value + sell_friction.
    # Creates hysteresis: harder to exit than to enter, preventing cycling.
    sell_friction: float = 0.02  # 2pp, same as buy friction_mid


@dataclass
class MarketImpactConfig:
    """Execution-only controls for limiting market impact during fresh starts."""

    # Enable fresh-start throttling when an event first starts trading.
    fresh_start_enabled: bool = True

    # Wall-clock throttle window in minutes from the event's first trading tick.
    fresh_start_minutes: float = 20.0

    # Fraction of available edge the executor may consume in one tick.
    # 0.25 means only walk 25% of the gap from top-of-book to threshold price.
    fresh_start_edge_fraction: float = 0.25


@dataclass
class RateLimitConfig:
    """
    Configuration for order rate limiting.

    Prevents runaway execution and respects API rate limits.

    Flow per tick:
    1. Sync portfolio from API (authoritative base state)
    2. Reconcile confirmed-fill overlay against API deltas
    3. Loop: generate candidates on effective state → execute → wait → repeat
    4. Stop when capital exhausted or no more utility gains
    """

    # Maximum orders per optimization tick
    max_orders_per_tick: int = 20

    # Minimum delay between orders in seconds (used for dry-run mode)
    min_order_delay_seconds: float = 0.5

    # Maximum orders per minute (hard cap)
    max_orders_per_minute: int = 60

    # Cooldown after hitting rate limit (seconds)
    rate_limit_cooldown_seconds: float = 60.0

    # Timeout for waiting for block confirmation (seconds)
    # In live mode, we wait for MINED/CONFIRMED status before next order
    # Polygon block time is ~2 seconds, but confirmation can take 6-20 seconds
    block_confirmation_timeout_seconds: float = 15.0

    # Maximum time for a single tick (seconds)
    # If a tick exceeds this, stop placing orders to avoid stale state
    tick_timeout_seconds: float = 120.0

    # Cooldown after FAK order failure for a specific bin (seconds)
    # When a FAK order fails due to no liquidity, don't retry that bin for this duration
    # This prevents spamming failed orders when liquidity dries up
    fak_failure_cooldown_seconds: float = 60.0

    # Freeze if confirmed fills remain unreconciled for longer than this.
    # This catches stale or contradictory local overlay state while allowing
    # ordinary API propagation lag to resolve naturally.
    overlay_reconciliation_grace_seconds: float = 90.0

    # When the residual overlay for a bin/side is within this fraction of the
    # current API-reported position, trust the API as "close enough" and clear
    # the residual instead of letting tiny drifts age into an integrity freeze.
    overlay_reconciliation_api_tolerance_fraction: float = 0.05

    # Safety cap for the API-relative overlay tolerance. This keeps the
    # tolerance focused on tiny propagation/rounding drifts instead of masking
    # large stale-position mismatches on bigger inventory.
    overlay_reconciliation_api_tolerance_max_shares: float = 1.0

    # Notional (dollar) tolerance for clearing residual overlay. If the
    # residual shares * fill price is below this amount, clear the overlay
    # regardless of the share-based cap. This prevents cheap low-price
    # positions from triggering freezes over negligible dollar amounts.
    overlay_reconciliation_api_tolerance_max_notional: float = 10.0

    # Hard deadline for an event integrity freeze. Once exceeded, the executor
    # drops any residual overlay, trusts the latest API snapshot, logs a
    # critical recovery event, and resumes trading from API state.
    integrity_freeze_max_seconds: float = 180.0

    # Maximum age for unpriced shares before force-resolving with fallback
    # cost basis. Prevents indefinite buy-blocking when the positions API
    # never provides pricing for a confirmed fill.
    unpriced_max_age_seconds: float = 300.0



@dataclass
class MarketConsensusConfig:
    """
    Configuration for trusted-quote market consensus blending.

    This is an optional probability-layer guardrail that blends only across
    bins with strong two-sided quotes and can reject fresh BUY entries in bins
    whose quotes are not trusted.
    """

    enabled: bool = False

    # Time-based consensus damping.
    time_enabled: bool = False
    time_tau: float = 12.0

    # Gap-based consensus damping.
    gap_enabled: bool = False
    gap_scale: float = 0.5
    gap_gamma: float = 1.5
    gap_floor: float = 0.80

    # Global minimum model weight after combining alpha terms — a floor on the
    # model's blend weight that mainly bites near settlement (low hours
    # remaining). 0.15 (more late-stage market deference) beat 0.30 on a
    # full-history backtest: ~37% lower drawdown for ~5% less return.
    # See research/2026-07_drawdown_and_forecast_bias.md.
    min_model_weight: float = 0.15

    # Quote quality gates for the trusted-bin subset.
    min_coverage_ratio: float = 0.0
    max_avg_spread: float = 0.06
    max_bin_spread: float = 0.10

    # If enabled, new BUY entries must come from trusted bins.
    require_trusted_quote_for_buys: bool = True


@dataclass
class RobustKellyConfig:
    """
    Configuration for market-aware Kelly fraction haircuts.

    Unlike market consensus blending, this leaves the model
    probabilities unchanged and only reduces effective Kelly aggressiveness
    when the market strongly disagrees and quote quality is good enough
    to trust.
    """

    # Enable dynamic Kelly fraction haircuting.
    enabled: bool = False

    # Lowest allowed Kelly fraction as a multiplier of the configured base.
    # 0.50 means the effective Kelly fraction can shrink to at most half
    # the configured kelly_fraction for that tick.
    min_fraction_multiplier: float = 0.50

    # Minimum fraction of live bins that must have two-sided quotes.
    min_coverage_ratio: float = 0.50

    # If average mid spread exceeds this, disable the haircut.
    max_avg_spread: float = 0.06

    # Half-L1 disagreement scale at which the full haircut is reached.
    # 0.5 * sum_i |p_model_i - p_mkt_i|
    disagreement_scale: float = 0.20


@dataclass
class MarketBuyGuardConfig:
    """
    Configuration for market-aware buy threshold widening.

    This uses market disagreement only as a guardrail on new buys. It does not
    change model probabilities, sells, or Kelly fraction globally.
    """

    # Enable market-aware widening of buy thresholds.
    enabled: bool = False

    # Maximum extra threshold widening in probability points under full guard.
    max_threshold_widening: float = 0.03

    # Minimum fraction of live bins that must have two-sided quotes.
    min_coverage_ratio: float = 0.50

    # If average mid spread exceeds this, disable the guard.
    max_avg_spread: float = 0.06

    # Half-L1 disagreement scale at which the full guard is reached.
    disagreement_scale: float = 0.20


@dataclass
class LateBoundaryTakeProfitConfig:
    """
    Configuration for late-boundary majority YES take-profit behavior.

    This is a decision-layer guard. It does not change model probabilities.
    """

    enabled: bool = False
    start_hours: float = 6.0
    max_distance_to_next_bin: int = 5
    silence_threshold_start_minutes: int = 180
    silence_threshold_floor_minutes: int = 90
    silence_threshold_step_per_hour: float = 30.0
    trigger_price: float = 0.80
    min_sell_fraction: float = 0.70
    max_sell_fraction: float = 0.90


@dataclass
class MakerConfig:
    """
    Passive maker-quote configuration (resting GTD bids).

    Maker mode rests BUY orders on bins the Kelly optimizer already wants
    to accumulate, at passive prices inside the spread. Adverse selection
    is controlled by hard gates calibrated on the 2026-07 markout study:
    quiet activity state only, never in the final hours before settlement,
    minimum spread, and kill-on-signal cancellation the moment any new
    post (countable or reply activity) is detected.
    """

    # "off" = disabled, "shadow" = compute and log quotes without posting,
    # "live" = post real GTD orders.
    mode: str = "off"

    # Seconds from placement to exchange-side GTD expiration. This is the
    # crash-safety floor: no resting order outlives it without any action
    # from us. Polymarket enforces a ~60s security buffer on GTD
    # expirations, so effective resting time is roughly ttl - 60s.
    ttl_seconds: float = 360.0

    # After a kill-switch cancel (new post, gap, dispute, prob jump), do
    # not re-quote for this long even if gates pass again.
    requote_cooldown_seconds: float = 120.0

    # Fraction of the event's capital allocation that may rest in open
    # maker orders at any one time.
    budget_fraction: float = 0.15

    # Per-quote collateral cap in USD.
    max_quote_usd: float = 150.0

    # Skip quotes whose collateral would be below this (dust orders are
    # churn without meaningful capture).
    min_quote_usd: float = 10.0

    # Only quote bins whose YES spread is at least this wide (capture
    # must dominate residual toxicity; quiet drift measured ~0.1-0.4c).
    min_spread: float = 0.03

    # Activity gates: quiet = no post in quiet_window_seconds AND fewer
    # than storm_count posts in storm_window_seconds.
    quiet_window_seconds: float = 1200.0
    storm_window_seconds: float = 1800.0
    storm_count: int = 5

    # Hard no-quote zone before settlement (final-12h drift measured
    # 5-10x the earlier-week quiet baseline).
    no_quote_final_hours: float = 12.0

    # Quotable price zone; outside it books are too degenerate.
    quote_zone_min: float = 0.05
    quote_zone_max: float = 0.95

    # Reject quoting when the orderbook snapshot is older than this.
    max_book_age_seconds: float = 45.0

    # Reject quoting when the activity tracker's last successful poll is
    # older than this (blind tracker = fail closed).
    activity_staleness_seconds: float = 90.0

    # Directory for the durable, structured JSONL maker-event log (one file
    # per event). Empty disables it — the dataclass default is empty so unit
    # tests stay hermetic; the run entrypoint sets a real directory so live
    # and shadow runs are recorded by default.
    event_log_dir: str = ""

    # Forward-markout horizons in seconds: after each (would-)fill, the mid
    # of the quoted token is sampled at each of these offsets and written to
    # the event log, so realized adverse selection is measured directly.
    markout_horizons_seconds: Tuple[float, ...] = (300.0, 900.0, 1800.0)

    @property
    def enabled(self) -> bool:
        return self.mode in ("shadow", "live")

    @property
    def shadow(self) -> bool:
        return self.mode == "shadow"

    def __post_init__(self) -> None:
        if self.mode not in ("off", "shadow", "live"):
            raise ValueError(f"MakerConfig.mode must be off/shadow/live, got {self.mode!r}")


@dataclass
class CollateralConfig:
    """Configuration for collateral limits."""

    # Maximum collateral per event (USD)
    c_event_max: float = 500.0

    # Maximum collateral per bin as ratio of c_event_max
    # e.g., 0.15 means max 15% of event budget per bin
    # With $500 budget and 0.15 ratio, max per bin = $75
    c_bin_max_ratio: float = 0.15

    # Capital multiplier for phantom capital injection.
    # 1.0 = standard Kelly (no phantom capital)
    # 2.0 = Kelly sees 2x capital → ~2x bigger positions
    # Phantom capital inflates Kelly's wealth perception but real capital
    # still hard-gates execution via available_capital.
    capital_multiplier: float = 1.0

    @property
    def virtual_c_event_max(self) -> float:
        """Event-level collateral gate scaled by multiplier."""
        return self.c_event_max * self.capital_multiplier

    @property
    def c_bin_max(self) -> float:
        """Per-bin collateral limit, scaled by multiplier."""
        return self.c_event_max * self.capital_multiplier * self.c_bin_max_ratio


@dataclass
class KellyConfig:
    """
    Main configuration for Kelly criterion trading strategy.
    """

    # Enable Kelly optimizer
    enabled: bool = True

    # Fractional Kelly multiplier (1.0 = full chunk size)
    kappa: float = 1.0

    # Minimum utility gain to execute a buy trade.
    # Prevents low-utility entries that get reversed next tick.
    min_buy_utility: float = 0.003

    # Minimum utility gain to execute a sell trade.
    # Higher than buy to create hysteresis and prevent cycling.
    min_sell_utility: float = 0.005

    # Fractional Kelly parameter α ∈ (0, 1].
    # Controls risk aversion via CRRA power utility with γ = 1/α.
    # α = 1.0: full Kelly (log utility)
    # α = 0.5: half Kelly (γ = 2, more conservative)
    # α = 0.25: quarter Kelly (γ = 4, very conservative)
    kelly_fraction: float = 1.0

    # Minimum terminal wealth floor (prevents ruin)
    w_floor: float = 1.0

    # Maximum iterations per optimization tick
    max_iters_per_tick: int = 100

    # Trading cutoff hours before settlement
    t_stop_hours: float = 1.0

    # Exit mode: when True, exits use only utility check (utility_gain >= 0)
    # and skip the fair-value price threshold check.
    kelly_only_exit: bool = True

    # Renormalize probabilities after dead-bin removal
    renormalize_probabilities: bool = True

    # Edge buffer configuration
    edge_buffer: EdgeBufferConfig = field(default_factory=EdgeBufferConfig)

    # Collateral limits
    collateral: CollateralConfig = field(default_factory=CollateralConfig)

    # EMA smoothing factor for model probabilities across ticks.
    # Lower α = more smoothing. α=0.3 ≈ half-life of 2 ticks (~10 min).
    # Set to 1.0 to disable smoothing (use raw probabilities).
    prob_ema_alpha: float = 0.3

    # Bypass the EMA when any bin's probability moves by at least this
    # much between ticks: that's real information (tweet burst, boundary
    # cross), not Monte Carlo noise (~0.005 worst-case per bin at 10k
    # sims), and smoothing it lags fair value by ~1/alpha slow ticks.
    prob_ema_jump_threshold: float = 0.02

    # Execution-only market impact controls.
    market_impact: MarketImpactConfig = field(default_factory=MarketImpactConfig)

    # Rate limiting
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    # Trusted-quote market consensus blending
    market_consensus: MarketConsensusConfig = field(default_factory=MarketConsensusConfig)

    # Market-aware Kelly fraction haircuting
    robust_kelly: RobustKellyConfig = field(default_factory=RobustKellyConfig)

    # Market-aware buy guardrail
    market_buy_guard: MarketBuyGuardConfig = field(default_factory=MarketBuyGuardConfig)

    # Late-boundary majority YES take-profit guard
    late_boundary_take_profit: LateBoundaryTakeProfitConfig = field(default_factory=LateBoundaryTakeProfitConfig)

    # Passive maker quoting (resting GTD bids)
    maker: MakerConfig = field(default_factory=MakerConfig)

    # Late-stage same-bin rotation path for boxed inventory.
    use_unbox_rotations: bool = False
    unbox_start_hours_to_settlement: float = 12.0
    unbox_min_blocked_ticks: int = 3
    unbox_min_net_utility: float = 0.012
    unbox_late_relax_start_hours_to_settlement: float = 3.0
    unbox_late_net_utility_relax: float = 0.002
    unbox_repeat_net_utility_step: float = 0.003
    unbox_repeat_net_utility_cap: float = 0.006
    unbox_multi_bin_start_count: int = 2
    unbox_multi_bin_net_utility_step: float = 0.002
    unbox_multi_bin_net_utility_cap: float = 0.004
    unbox_turnover_penalty: float = 0.002
    unbox_bin_cooldown_seconds: int = 3600

    @classmethod
    def from_dict(cls, data: dict) -> "KellyConfig":
        """Create config from dictionary (e.g., from YAML)."""
        # Extract nested configs
        edge_buffer_data = data.pop("edge_buffer", {})
        market_impact_data = data.pop("market_impact", {})
        market_impact_data.pop("fresh_start_ticks", None)  # Legacy field, replaced by wall-clock window
        data.pop("adaptive_delta", None)  # Legacy field, ignored
        collateral_data = data.pop("collateral", {})
        rate_limit_data = data.pop("rate_limit", {})
        if "market_aware" in data:
            raise ValueError("market_aware config is no longer supported; use market_consensus")
        market_consensus_data = data.pop("market_consensus", {})
        robust_kelly_data = data.pop("robust_kelly", {})
        market_buy_guard_data = data.pop("market_buy_guard", {})
        late_boundary_take_profit_data = data.pop("late_boundary_take_profit", {})
        maker_data = data.pop("maker", {})

        # Backward compatibility: convert old c_bin_max (absolute) to c_bin_max_ratio
        if "c_bin_max" in collateral_data and "c_bin_max_ratio" not in collateral_data:
            old_bin_max = collateral_data.pop("c_bin_max")
            c_event_max = collateral_data.get("c_event_max", 500.0)
            if c_event_max > 0:
                collateral_data["c_bin_max_ratio"] = old_bin_max / c_event_max

        # Ignore websocket config if present (moved to websocket_client.py)
        data.pop("websocket", None)

        # Backward compatibility: map old min_utility to split buy/sell thresholds
        if "min_utility" in data and "min_buy_utility" not in data:
            old_min_utility = data.pop("min_utility")
            data["min_buy_utility"] = old_min_utility
            data["min_sell_utility"] = 2 * old_min_utility

        # Backward compatibility: the shared YAML historically used "tau"
        # for the minimum utility threshold before buy/sell were split.
        if "tau" in data:
            old_tau = data.pop("tau")
            data.setdefault("min_buy_utility", old_tau)
            data.setdefault("min_sell_utility", 2 * old_tau)

        return cls(
            edge_buffer=EdgeBufferConfig(**edge_buffer_data),
            market_impact=MarketImpactConfig(**market_impact_data),
            collateral=CollateralConfig(**collateral_data),
            rate_limit=RateLimitConfig(**rate_limit_data),
            market_consensus=MarketConsensusConfig(**market_consensus_data),
            robust_kelly=RobustKellyConfig(**robust_kelly_data),
            market_buy_guard=MarketBuyGuardConfig(**market_buy_guard_data),
            late_boundary_take_profit=LateBoundaryTakeProfitConfig(**late_boundary_take_profit_data),
            maker=MakerConfig(**maker_data),
            **data,
        )


@dataclass
class EventCategoryRules:
    """
    Trading rules for a specific event duration category.

    Controls when trading is allowed based on:
    - Whether counting period has started
    - How far before/after counting we can trade
    - Minimum time before settlement to stop trading
    """

    # Name for this category (e.g., "short", "weekly", "monthly")
    name: str

    # Duration range in days [min, max) - max is exclusive
    # e.g., [0, 3] means events lasting 0, 1, or 2 days
    duration_min_days: int
    duration_max_days: int

    # If True, only trade after counting period has started (cnt > 0 possible)
    require_counting_started: bool = True

    # Maximum hours before counting starts that we can begin trading
    # Only applies if require_counting_started=False
    # e.g., 72 means can trade up to 72h before counting starts
    max_hours_before_counting: Optional[float] = None

    # Maximum days before settlement to start trading
    # e.g., 7 means only trade when <= 7 days remain until settlement
    # This replaces the old max_event_duration_days parameter
    max_days_before_settlement: Optional[float] = None

    # Minimum hours before settlement to stop trading
    # e.g., 3 means stop trading 3h before settlement
    min_hours_before_settlement: float = 3.0

    def matches_duration(self, event_duration_days: int) -> bool:
        """Check if this category applies to an event of given duration."""
        return self.duration_min_days <= event_duration_days < self.duration_max_days


@dataclass
class EventTradingRulesConfig:
    """
    Configuration for event-specific trading rules.

    Allows different rules for different event durations (short, weekly, monthly).
    Loaded from YAML file.
    """

    # List of category rules, checked in order
    categories: List[EventCategoryRules] = field(default_factory=list)

    # Default rules if no category matches (weekly-style defaults)
    default_require_counting_started: bool = False
    default_max_hours_before_counting: Optional[float] = 96.0
    default_max_days_before_settlement: Optional[float] = None
    default_min_hours_before_settlement: float = 1.0

    def get_rules_for_event(self, event_duration_days: int) -> EventCategoryRules:
        """
        Get the trading rules for an event of given duration.

        Args:
            event_duration_days: Duration of the event in days

        Returns:
            EventCategoryRules for this event duration
        """
        for category in self.categories:
            if category.matches_duration(event_duration_days):
                return category

        # Return default rules (weekly-style)
        return EventCategoryRules(
            name="default",
            duration_min_days=0,
            duration_max_days=9999,
            require_counting_started=self.default_require_counting_started,
            max_hours_before_counting=self.default_max_hours_before_counting,
            max_days_before_settlement=self.default_max_days_before_settlement,
            min_hours_before_settlement=self.default_min_hours_before_settlement,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "EventTradingRulesConfig":
        """Load configuration from YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Event trading rules config not found: {yaml_path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "EventTradingRulesConfig":
        """Create config from dictionary."""
        default_rules = data.get("default_rules", {})
        default_require_counting_started = default_rules.get("require_counting_started", False)
        default_max_hours_before_counting = default_rules.get("max_hours_before_counting", 96.0)
        default_max_days_before_settlement = default_rules.get("max_days_before_settlement")
        default_min_hours_before_settlement = default_rules.get("min_hours_before_settlement", 1.0)

        categories = []

        for cat_data in data.get("event_categories", []):
            duration_range = cat_data.get("duration_days", [0, 9999])
            rules_data = cat_data.get("trading_rules", {})

            category = EventCategoryRules(
                name=cat_data.get("name", "unnamed"),
                duration_min_days=duration_range[0],
                duration_max_days=duration_range[1],
                require_counting_started=rules_data.get(
                    "require_counting_started",
                    default_require_counting_started,
                ),
                max_hours_before_counting=rules_data.get(
                    "max_hours_before_counting",
                    default_max_hours_before_counting,
                ),
                max_days_before_settlement=rules_data.get(
                    "max_days_before_settlement",
                    default_max_days_before_settlement,
                ),
                min_hours_before_settlement=rules_data.get(
                    "min_hours_before_settlement",
                    default_min_hours_before_settlement,
                ),
            )
            categories.append(category)

        return cls(
            categories=categories,
            default_require_counting_started=default_require_counting_started,
            default_max_hours_before_counting=default_max_hours_before_counting,
            default_max_days_before_settlement=default_max_days_before_settlement,
            default_min_hours_before_settlement=default_min_hours_before_settlement,
        )

    @classmethod
    def default(cls) -> "EventTradingRulesConfig":
        """Create default configuration."""
        return cls(
            categories=[
                # Short events (0-3 days): require counting started
                EventCategoryRules(
                    name="short",
                    duration_min_days=0,
                    duration_max_days=4,
                    require_counting_started=True,
                    min_hours_before_settlement=1.0,
                ),
                # Weekly events (4-8 days): can trade before counting, up to 72h early
                EventCategoryRules(
                    name="weekly",
                    duration_min_days=4,
                    duration_max_days=9,
                    require_counting_started=False,
                    max_hours_before_counting=96.0,
                    min_hours_before_settlement=1.0,
                ),
                # Monthly events (9+ days): wait until 7 days remain
                EventCategoryRules(
                    name="monthly",
                    duration_min_days=9,
                    duration_max_days=9999,
                    require_counting_started=False,
                    max_days_before_settlement=7.0,
                    min_hours_before_settlement=6.0,
                ),
            ],
            default_require_counting_started=False,
            default_max_hours_before_counting=96.0,
            default_min_hours_before_settlement=1.0,
        )
