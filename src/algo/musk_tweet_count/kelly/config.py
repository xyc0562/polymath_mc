"""
Configuration dataclasses for Kelly criterion trading.
"""

from dataclasses import dataclass, field
from typing import Optional


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
    min_perceived_prob: float = 0.0

    # Minimum market price to trade (avoids illiquid tail bets)
    # Don't buy YES if market price < this, don't buy NO if market price < this
    # Set to 0 to disable. Recommended: 0.15-0.20 based on backtest analysis.
    min_market_price: float = 0.0


@dataclass
class RateLimitConfig:
    """
    Configuration for order rate limiting.

    Prevents runaway execution and respects API rate limits.
    """

    # Maximum orders per optimization tick
    max_orders_per_tick: int = 10

    # Minimum delay between orders in seconds
    min_order_delay_seconds: float = 1.0

    # Maximum orders per minute (hard cap)
    max_orders_per_minute: int = 30

    # Cooldown after hitting rate limit (seconds)
    rate_limit_cooldown_seconds: float = 60.0


@dataclass
class AdaptiveDeltaConfig:
    """
    Configuration for adaptive chunk sizing.

    Adapts trade size based on:
    - Available liquidity (never take too much of visible depth)
    - Time remaining (ramp down as we approach T_stop)
    """

    # Base chunk size in shares
    base_delta: float = 10.0

    # Maximum fraction of visible depth to take per trade
    max_depth_fraction: float = 0.10

    # Hours before T_stop to start ramping down chunk size
    time_ramp_hours: float = 6.0

    # Minimum chunk size in shares
    min_delta: float = 1.0


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket orderbook streaming."""

    # Enable WebSocket streaming (vs polling)
    enabled: bool = True

    # Delay between reconnection attempts
    reconnect_delay_seconds: float = 5.0

    # Heartbeat interval to keep connection alive
    heartbeat_interval_seconds: float = 30.0

    # WebSocket endpoint
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


@dataclass
class CollateralConfig:
    """Configuration for collateral limits."""

    # Maximum collateral per event (USD)
    c_event_max: float = 500.0

    # Maximum collateral per bin (USD)
    c_bin_max: float = 100.0


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
    tau: float = 0.001

    # Minimum terminal wealth floor (prevents ruin)
    w_floor: float = 1.0

    # Maximum iterations per optimization tick
    max_iters_per_tick: int = 100

    # Trading cutoff hours before settlement
    t_stop_hours: float = 3.0

    # Renormalize probabilities after dead-bin removal
    renormalize_probabilities: bool = True

    # Edge buffer configuration
    edge_buffer: EdgeBufferConfig = field(default_factory=EdgeBufferConfig)

    # Adaptive delta configuration
    adaptive_delta: AdaptiveDeltaConfig = field(default_factory=AdaptiveDeltaConfig)

    # WebSocket configuration
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)

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
        websocket_data = data.pop("websocket", {})
        collateral_data = data.pop("collateral", {})
        rate_limit_data = data.pop("rate_limit", {})

        return cls(
            edge_buffer=EdgeBufferConfig(**edge_buffer_data),
            adaptive_delta=AdaptiveDeltaConfig(**adaptive_delta_data),
            websocket=WebSocketConfig(**websocket_data),
            collateral=CollateralConfig(**collateral_data),
            rate_limit=RateLimitConfig(**rate_limit_data),
            **data,
        )
