"""Typed configuration for Deribit lead-lag trading."""

from dataclasses import dataclass
from pathlib import Path

from ruamel.yaml import YAML


@dataclass
class DeribitConfig:
    base_url: str = "https://www.deribit.com/api/v2/public"
    currency: str = "BTC"


@dataclass
class SignalConfig:
    min_edge_threshold: float = 0.06
    max_edge_threshold: float = 0.40
    min_time_to_expiry_hours: float = 4.0
    max_time_to_expiry_days: float = 7.0
    min_prob: float = 0.03
    max_prob: float = 0.97
    min_deribit_volume_24h: float = 0.1
    min_deribit_open_interest: float = 1.0
    max_iv_spread: float = 30.0
    min_call_spread_usd: float = 50.0
    signal_cooldown_seconds: float = 300.0


@dataclass
class ExecutionConfig:
    max_order_size_usd: float = 20.0
    max_total_exposure_usd: float = 100.0
    polymarket_crypto_fee_rate: float = 0.072
    state_file: str = "data/deribit_leadlag_state.json"
    dry_run: bool = True


@dataclass
class LeadLagConfig:
    trading_enabled: bool = True
    dry_run: bool = True
    poll_interval_seconds: float = 30.0
    market_refresh_interval_seconds: float = 600.0
    deribit: DeribitConfig = None
    signal: SignalConfig = None
    execution: ExecutionConfig = None

    def __post_init__(self):
        if self.deribit is None:
            self.deribit = DeribitConfig()
        if self.signal is None:
            self.signal = SignalConfig()
        if self.execution is None:
            self.execution = ExecutionConfig()

    @classmethod
    def from_yaml(cls, path: str) -> "LeadLagConfig":
        yaml = YAML()
        with open(path, "r") as f:
            raw = yaml.load(f) or {}

        trading = raw.get("trading", {})
        deribit_raw = raw.get("deribit", {})
        signal_raw = raw.get("signal", {})
        exec_raw = raw.get("execution", {})

        dry_run = trading.get("dry_run", True)

        return cls(
            trading_enabled=trading.get("enabled", True),
            dry_run=dry_run,
            poll_interval_seconds=trading.get("poll_interval_seconds", 30.0),
            market_refresh_interval_seconds=trading.get(
                "market_refresh_interval_seconds", 600.0
            ),
            deribit=DeribitConfig(
                base_url=deribit_raw.get(
                    "base_url", "https://www.deribit.com/api/v2/public"
                ),
                currency=deribit_raw.get("currency", "BTC"),
            ),
            signal=SignalConfig(
                min_edge_threshold=signal_raw.get("min_edge_threshold", 0.06),
                max_edge_threshold=signal_raw.get("max_edge_threshold", 0.40),
                min_time_to_expiry_hours=signal_raw.get(
                    "min_time_to_expiry_hours", 4.0
                ),
                max_time_to_expiry_days=signal_raw.get("max_time_to_expiry_days", 7.0),
                min_prob=signal_raw.get("min_prob", 0.03),
                max_prob=signal_raw.get("max_prob", 0.97),
                min_deribit_volume_24h=signal_raw.get("min_deribit_volume_24h", 0.1),
                min_deribit_open_interest=signal_raw.get(
                    "min_deribit_open_interest", 1.0
                ),
                max_iv_spread=signal_raw.get("max_iv_spread", 30.0),
                min_call_spread_usd=signal_raw.get("min_call_spread_usd", 50.0),
                signal_cooldown_seconds=signal_raw.get("signal_cooldown_seconds", 300.0),
            ),
            execution=ExecutionConfig(
                max_order_size_usd=exec_raw.get("max_order_size_usd", 20.0),
                max_total_exposure_usd=exec_raw.get("max_total_exposure_usd", 100.0),
                polymarket_crypto_fee_rate=exec_raw.get(
                    "polymarket_crypto_fee_rate", 0.072
                ),
                state_file=exec_raw.get("state_file", "data/deribit_leadlag_state.json"),
                dry_run=dry_run,
            ),
        )
