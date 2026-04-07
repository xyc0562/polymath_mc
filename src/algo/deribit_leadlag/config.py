"""Typed configuration for Deribit lead-lag trading."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ruamel.yaml import YAML


@dataclass
class DeribitConfig:
    base_url: str = "https://www.deribit.com/api/v2/public"
    currency: str = "BTC"


@dataclass
class DeribitWSConfig:
    ws_url: str = "wss://www.deribit.com/ws/api/v2"
    reconnect_delay_seconds: float = 2.0
    max_reconnect_delay_seconds: float = 60.0
    stale_threshold_seconds: float = 5.0


@dataclass
class SignalConfig:
    min_call_spread_usd: float = 15.0
    max_bounds_width_maker: float = 0.50
    max_bounds_width_taker: float = 0.30
    min_prob: float = 0.03
    max_prob: float = 0.97
    min_time_to_expiry_hours: float = 4.0
    max_time_to_expiry_days: float = 7.0
    time_adjusted_basis_haircut: float = 0.02
    no_next_day_extra_haircut: float = 0.03
    max_bracket_dk: float = 3000.0


@dataclass
class OrderConfig:
    maker_min_edge: float = 0.02
    taker_min_edge: float = 0.15
    emergency_exit_edge: float = 0.20
    reprice_edge_threshold: float = 0.02
    emergency_prob_shift: float = 0.03
    polymarket_crypto_fee_rate: float = 0.072


@dataclass
class AllocationConfig:
    per_bin_max_usd: float = 50.0
    per_date_max_usd: float = 150.0
    total_max_usd: float = 500.0
    max_order_size_usd: float = 30.0


@dataclass
class PolyWSConfig:
    orderbook_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    user_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
    heartbeat_interval_seconds: float = 10.0


@dataclass
class StrategyConfig:
    tick_interval_seconds: float = 1.5
    position_poll_interval_seconds: float = 10.0
    market_refresh_interval_seconds: float = 600.0


@dataclass
class LeadLagConfig:
    trading_enabled: bool = True
    dry_run: bool = True
    deribit: DeribitConfig = None
    deribit_ws: DeribitWSConfig = None
    signal: SignalConfig = None
    order: OrderConfig = None
    allocation: AllocationConfig = None
    poly_ws: PolyWSConfig = None
    strategy: StrategyConfig = None

    def __post_init__(self):
        if self.deribit is None:
            self.deribit = DeribitConfig()
        if self.deribit_ws is None:
            self.deribit_ws = DeribitWSConfig()
        if self.signal is None:
            self.signal = SignalConfig()
        if self.order is None:
            self.order = OrderConfig()
        if self.allocation is None:
            self.allocation = AllocationConfig()
        if self.poly_ws is None:
            self.poly_ws = PolyWSConfig()
        if self.strategy is None:
            self.strategy = StrategyConfig()

    @classmethod
    def from_yaml(cls, path: str) -> "LeadLagConfig":
        yaml = YAML()
        with open(path, "r") as f:
            raw = yaml.load(f) or {}

        trading = raw.get("trading", {})
        deribit_raw = raw.get("deribit", {})
        deribit_ws_raw = raw.get("deribit_ws", {})
        signal_raw = raw.get("signal", {})
        order_raw = raw.get("order", {})
        alloc_raw = raw.get("allocation", {})
        poly_ws_raw = raw.get("poly_ws", {})
        strategy_raw = raw.get("strategy", {})

        return cls(
            trading_enabled=trading.get("enabled", True),
            dry_run=trading.get("dry_run", True),
            deribit=DeribitConfig(
                base_url=deribit_raw.get("base_url", DeribitConfig.base_url),
                currency=deribit_raw.get("currency", DeribitConfig.currency),
            ),
            deribit_ws=DeribitWSConfig(
                ws_url=deribit_ws_raw.get("ws_url", DeribitWSConfig.ws_url),
                reconnect_delay_seconds=deribit_ws_raw.get("reconnect_delay_seconds", DeribitWSConfig.reconnect_delay_seconds),
                max_reconnect_delay_seconds=deribit_ws_raw.get("max_reconnect_delay_seconds", DeribitWSConfig.max_reconnect_delay_seconds),
                stale_threshold_seconds=deribit_ws_raw.get("stale_threshold_seconds", DeribitWSConfig.stale_threshold_seconds),
            ),
            signal=SignalConfig(
                min_call_spread_usd=signal_raw.get("min_call_spread_usd", SignalConfig.min_call_spread_usd),
                max_bounds_width_maker=signal_raw.get("max_bounds_width_maker", SignalConfig.max_bounds_width_maker),
                max_bounds_width_taker=signal_raw.get("max_bounds_width_taker", SignalConfig.max_bounds_width_taker),
                min_prob=signal_raw.get("min_prob", SignalConfig.min_prob),
                max_prob=signal_raw.get("max_prob", SignalConfig.max_prob),
                min_time_to_expiry_hours=signal_raw.get("min_time_to_expiry_hours", SignalConfig.min_time_to_expiry_hours),
                max_time_to_expiry_days=signal_raw.get("max_time_to_expiry_days", SignalConfig.max_time_to_expiry_days),
                time_adjusted_basis_haircut=signal_raw.get("time_adjusted_basis_haircut", SignalConfig.time_adjusted_basis_haircut),
                no_next_day_extra_haircut=signal_raw.get("no_next_day_extra_haircut", SignalConfig.no_next_day_extra_haircut),
                max_bracket_dk=signal_raw.get("max_bracket_dk", SignalConfig.max_bracket_dk),
            ),
            order=OrderConfig(
                maker_min_edge=order_raw.get("maker_min_edge", OrderConfig.maker_min_edge),
                taker_min_edge=order_raw.get("taker_min_edge", OrderConfig.taker_min_edge),
                emergency_exit_edge=order_raw.get("emergency_exit_edge", OrderConfig.emergency_exit_edge),
                reprice_edge_threshold=order_raw.get("reprice_edge_threshold", OrderConfig.reprice_edge_threshold),
                emergency_prob_shift=order_raw.get("emergency_prob_shift", OrderConfig.emergency_prob_shift),
                polymarket_crypto_fee_rate=order_raw.get("polymarket_crypto_fee_rate", OrderConfig.polymarket_crypto_fee_rate),
            ),
            allocation=AllocationConfig(
                per_bin_max_usd=alloc_raw.get("per_bin_max_usd", AllocationConfig.per_bin_max_usd),
                per_date_max_usd=alloc_raw.get("per_date_max_usd", AllocationConfig.per_date_max_usd),
                total_max_usd=alloc_raw.get("total_max_usd", AllocationConfig.total_max_usd),
                max_order_size_usd=alloc_raw.get("max_order_size_usd", AllocationConfig.max_order_size_usd),
            ),
            poly_ws=PolyWSConfig(
                orderbook_ws_url=poly_ws_raw.get("orderbook_ws_url", PolyWSConfig.orderbook_ws_url),
                user_ws_url=poly_ws_raw.get("user_ws_url", PolyWSConfig.user_ws_url),
                heartbeat_interval_seconds=poly_ws_raw.get("heartbeat_interval_seconds", PolyWSConfig.heartbeat_interval_seconds),
            ),
            strategy=StrategyConfig(
                tick_interval_seconds=strategy_raw.get("tick_interval_seconds", StrategyConfig.tick_interval_seconds),
                position_poll_interval_seconds=strategy_raw.get("position_poll_interval_seconds", StrategyConfig.position_poll_interval_seconds),
                market_refresh_interval_seconds=strategy_raw.get("market_refresh_interval_seconds", StrategyConfig.market_refresh_interval_seconds),
            ),
        )
