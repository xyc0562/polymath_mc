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
    required_roi: float = 0.05

    # Friction in probability points for middle range (10% < p < 90%)
    friction_mid: float = 0.01  # 1%

    # Friction in probability points for tails (p <= 10% or p >= 90%)
    friction_tail: float = 0.02  # 2%

    # Threshold for tail zone
    tail_threshold: float = 0.10  # 10%

    # Minimum perceived probability to trade (from our model)
    # Don't trade if our model assigns probability < this threshold
    # Set to 0 to disable.
    min_perceived_prob: float = 0.03  # 5%

    # Minimum market price to trade (avoids illiquid tail bets)
    # Don't buy YES if market price < this, don't buy NO if market price < this
    # Set to 0 to disable.
    min_market_price: float = 0.01  # 3%

    # Require two-sided liquidity (both bid and ask must exist)
    # If True, skip bins where only one side has liquidity
    require_two_sided_liquidity: bool = True

    # Maximum spread as ratio of bid price
    # e.g., 2.0 means if bid=0.01, ask can be at most 0.03 (spread/bid = 0.02/0.01 = 2.0)
    # Spread ratio = (ask - bid) / bid
    # Set to 0 to disable.
    max_spread_ratio: float = 2.0


@dataclass
class RateLimitConfig:
    """
    Configuration for order rate limiting.

    Prevents runaway execution and respects API rate limits.

    Flow per tick:
    1. Sync portfolio from API (get true state)
    2. Loop: generate candidates → execute → optimistic update → repeat
    3. Stop when capital exhausted or no more utility gains
    """

    # Maximum orders per optimization tick
    max_orders_per_tick: int = 10

    # Minimum delay between orders in seconds (used for dry-run mode)
    min_order_delay_seconds: float = 1.0

    # Maximum orders per minute (hard cap)
    max_orders_per_minute: int = 30

    # Cooldown after hitting rate limit (seconds)
    rate_limit_cooldown_seconds: float = 60.0

    # Timeout for waiting for block confirmation (seconds)
    # In live mode, we wait for MINED/CONFIRMED status before next order
    # Polygon block time is ~2 seconds, but confirmation can take 6-20 seconds
    block_confirmation_timeout_seconds: float = 30.0

    # Cooldown after FAK order failure for a specific bin (seconds)
    # When a FAK order fails due to no liquidity, don't retry that bin for this duration
    # This prevents spamming failed orders when liquidity dries up
    fak_failure_cooldown_seconds: float = 60.0


@dataclass
class AdaptiveDeltaConfig:
    """
    Configuration for adaptive chunk sizing in USD.

    Adapts trade size based on available liquidity (never take too much of visible depth).
    Trading stops entirely once past T_stop.

    All sizes are in USD - shares are computed as: shares = usd_amount / price
    """

    # Base chunk size in USD (e.g., $50 per trade)
    base_delta_usd: float = 50.0

    # Maximum fraction of visible depth to take per trade
    # Set to 1.0 for production (no depth impact limit)
    max_depth_fraction: float = 0.3

    # Minimum chunk size in USD (must be >= $1 for Polymarket)
    min_delta_usd: float = 5.0


@dataclass
class CollateralConfig:
    """Configuration for collateral limits."""

    # Maximum collateral per event (USD)
    c_event_max: float = 500.0

    # Maximum collateral per bin as ratio of c_event_max
    # e.g., 0.15 means max 15% of event budget per bin
    # With $500 budget and 0.15 ratio, max per bin = $75
    c_bin_max_ratio: float = 0.15

    @property
    def c_bin_max(self) -> float:
        """Computed max collateral per bin."""
        return self.c_event_max * self.c_bin_max_ratio


@dataclass
class KellyConfig:
    """
    Main configuration for Kelly criterion trading strategy.
    """

    # Enable Kelly optimizer
    enabled: bool = True

    # Fractional Kelly multiplier (0.25 = quarter Kelly for safety)
    kappa: float = 0.25

    # Minimum utility gain threshold to execute a trade
    # With small position sizes and many bins, utility gains are naturally small
    # 0.0001 is appropriate for $500-1000 capital with 10-share base delta
    # min_utility: float = 0.0001
    min_utility: float = 0.00005

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

    # Adaptive delta configuration
    adaptive_delta: AdaptiveDeltaConfig = field(default_factory=AdaptiveDeltaConfig)

    # Collateral limits
    collateral: CollateralConfig = field(default_factory=CollateralConfig)

    # Rate limiting
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "KellyConfig":
        """Create config from dictionary (e.g., from YAML)."""
        # Extract nested configs
        edge_buffer_data = data.pop("edge_buffer", {})
        adaptive_delta_data = data.pop("adaptive_delta", {})
        collateral_data = data.pop("collateral", {})
        rate_limit_data = data.pop("rate_limit", {})

        # Backward compatibility: convert old c_bin_max (absolute) to c_bin_max_ratio
        if "c_bin_max" in collateral_data and "c_bin_max_ratio" not in collateral_data:
            old_bin_max = collateral_data.pop("c_bin_max")
            c_event_max = collateral_data.get("c_event_max", 500.0)
            if c_event_max > 0:
                collateral_data["c_bin_max_ratio"] = old_bin_max / c_event_max

        # Backward compatibility: convert old share-based delta to USD-based
        # Old: base_delta=60 (shares), min_delta=60 (shares)
        # New: base_delta_usd=15 (USD), min_delta_usd=5 (USD)
        if "base_delta" in adaptive_delta_data and "base_delta_usd" not in adaptive_delta_data:
            # Remove old share-based fields, use new defaults
            adaptive_delta_data.pop("base_delta", None)
            adaptive_delta_data.pop("min_delta", None)

        # Ignore websocket config if present (moved to websocket_client.py)
        data.pop("websocket", None)

        return cls(
            edge_buffer=EdgeBufferConfig(**edge_buffer_data),
            adaptive_delta=AdaptiveDeltaConfig(**adaptive_delta_data),
            collateral=CollateralConfig(**collateral_data),
            rate_limit=RateLimitConfig(**rate_limit_data),
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

    # Default rules if no category matches
    default_require_counting_started: bool = True
    default_min_hours_before_settlement: float = 3.0

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

        # Return default rules
        return EventCategoryRules(
            name="default",
            duration_min_days=0,
            duration_max_days=9999,
            require_counting_started=self.default_require_counting_started,
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
        categories = []

        for cat_data in data.get("event_categories", []):
            duration_range = cat_data.get("duration_days", [0, 9999])
            rules_data = cat_data.get("trading_rules", {})

            category = EventCategoryRules(
                name=cat_data.get("name", "unnamed"),
                duration_min_days=duration_range[0],
                duration_max_days=duration_range[1],
                require_counting_started=rules_data.get("require_counting_started", True),
                max_hours_before_counting=rules_data.get("max_hours_before_counting"),
                max_days_before_settlement=rules_data.get("max_days_before_settlement"),
                min_hours_before_settlement=rules_data.get("min_hours_before_settlement", 3.0),
            )
            categories.append(category)

        default_rules = data.get("default_rules", {})

        return cls(
            categories=categories,
            default_require_counting_started=default_rules.get("require_counting_started", True),
            default_min_hours_before_settlement=default_rules.get("min_hours_before_settlement", 3.0),
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
                    max_hours_before_counting=72.0,
                    min_hours_before_settlement=3.0,
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
            default_require_counting_started=True,
            default_min_hours_before_settlement=3.0,
        )
