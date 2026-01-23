"""
Configuration dataclasses for Kelly criterion trading.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EdgeBufferConfig:
    """
    Configuration for edge buffer to protect against model uncertainty.

    The edge buffer requires a minimum edge before trading, with dynamic
    scaling at extreme prices where model errors have larger relative impact.
    """

    # Base edge percentage required (e.g., 0.05 = 5%)
    base_edge_pct: float = 0.05

    # Multiplier at extreme prices (0 or 1)
    # At p=0.5: require base_edge
    # At p=0 or p=1: require base_edge * extreme_multiplier
    extreme_multiplier: float = 2.0


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

    @classmethod
    def from_dict(cls, data: dict) -> "KellyConfig":
        """Create config from dictionary (e.g., from YAML)."""
        # Extract nested configs
        edge_buffer_data = data.pop("edge_buffer", {})
        adaptive_delta_data = data.pop("adaptive_delta", {})
        websocket_data = data.pop("websocket", {})
        collateral_data = data.pop("collateral", {})

        return cls(
            edge_buffer=EdgeBufferConfig(**edge_buffer_data),
            adaptive_delta=AdaptiveDeltaConfig(**adaptive_delta_data),
            websocket=WebSocketConfig(**websocket_data),
            collateral=CollateralConfig(**collateral_data),
            **data,
        )
