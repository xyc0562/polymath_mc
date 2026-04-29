#!/usr/bin/env python3
"""
Run multi-event trading bot with shared capital pool.

Usage:
    python -m src.algo.musk_tweet_count.forecaster.run_multi_event \
        --capital 5000 \
        --dry-run

    # With specific events (comma-separated event IDs or "auto" for discovery)
    python -m src.algo.musk_tweet_count.forecaster.run_multi_event \
        --capital 5000 \
        --events auto
"""

import argparse
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
import logging
import os
import re
import sys
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import yaml
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):
        return False

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from py_clob_client_v2.client import ClobClient
from py_clob_client_v2.clob_types import ApiCreds, AssetType, BalanceAllowanceParams

from src.utils.crypto_utils import load_private_key
from src.algo.musk_tweet_count.forecaster.config import (
    BucketNowcastConfig,
    ForecasterConfig,
)
from src.algo.musk_tweet_count.forecaster.data import ContractDayUtils
from src.algo.musk_tweet_count.forecaster.multi_event_manager import (
    MultiEventManager,
    MultiEventConfig,
    EventInfo,
)
from src.algo.musk_tweet_count.kelly.config import (
    KellyConfig,
    EdgeBufferConfig,
    CollateralConfig,
    EventTradingRulesConfig,
    MarketImpactConfig,
    MarketConsensusConfig,
    RobustKellyConfig,
    MarketBuyGuardConfig,
    LateBoundaryTakeProfitConfig,
)
from src.algo.musk_tweet_count.kelly.capital_pool import CapitalPoolConfig
from src.algo.musk_tweet_count.notifications import SlackNotifier

logger = logging.getLogger(__name__)
CONTRACT_UTILS = ContractDayUtils()
DEFAULT_RUNNER_CONFIG_PATH = "config/musk_tweet_count.yaml"


def _resolve_project_path(path_str: str) -> Path:
    """Resolve a config path against cwd first, then project root."""
    path = Path(path_str)
    if path.exists():
        return path
    project_path = project_root / path
    if project_path.exists():
        return project_path
    return path


def _load_runner_yaml_config(config_path: str) -> dict:
    """Load the shared Musk tweet count YAML config for the live runner."""
    path = _resolve_project_path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _consensus_mode_from_config(config: MarketConsensusConfig) -> str:
    """Convert consensus booleans into the CLI preset mode string."""
    if not config.enabled or (not config.time_enabled and not config.gap_enabled):
        return "off"
    if config.time_enabled and config.gap_enabled:
        return "time_gap"
    if config.time_enabled:
        return "time_only"
    return "gap_only"


def _build_base_configs_from_yaml(
    config_path: str,
) -> tuple[dict, KellyConfig, ForecasterConfig, dict]:
    """Build base runtime configs from the shared YAML file."""
    raw_config = _load_runner_yaml_config(config_path)
    kelly_config = KellyConfig.from_dict(deepcopy(raw_config.get("kelly", {})))
    forecaster_config = ForecasterConfig.from_dict(
        deepcopy(raw_config.get("forecaster", {}))
    )

    trading_config = raw_config.get("trading", {})
    websocket_config = raw_config.get("kelly", {}).get("websocket", {})

    runtime_defaults = {
        "live": not bool(trading_config.get("dry_run", True)),
        "tick_interval": int(
            trading_config.get(
                "slow_loop_interval_seconds",
                trading_config.get("tick_interval_seconds", 3600),
            )
        ),
        "fast_tick_interval": int(
            trading_config.get("fast_loop_interval_seconds", 30)
        ),
        "forecast_cache_seconds": int(
            trading_config.get("forecast_cache_seconds", 165)
        ),
        "no_ws": not bool(websocket_config.get("enabled", True)),
    }

    cli_defaults = {
        "max_per_event": kelly_config.collateral.c_event_max,
        "tick_interval": runtime_defaults["tick_interval"],
        "fast_tick_interval": runtime_defaults["fast_tick_interval"],
        "forecast_cache_seconds": runtime_defaults["forecast_cache_seconds"],
        "live": runtime_defaults["live"],
        "no_ws": runtime_defaults["no_ws"],
        "kappa": kelly_config.kappa,
        "kelly_fraction": kelly_config.kelly_fraction,
        "required_roi": kelly_config.edge_buffer.required_roi,
        "min_prob": kelly_config.edge_buffer.min_perceived_prob,
        "min_market_price": kelly_config.edge_buffer.min_market_price,
        "max_spread_ratio": kelly_config.edge_buffer.max_spread_ratio,
        "no_require_two_sided": not kelly_config.edge_buffer.require_two_sided_liquidity,
        "capital_multiplier": kelly_config.collateral.capital_multiplier,
        "min_buy_utility": kelly_config.min_buy_utility,
        "min_sell_utility": kelly_config.min_sell_utility,
        "fresh_start_minutes": kelly_config.market_impact.fresh_start_minutes,
        "fresh_start_edge_fraction": kelly_config.market_impact.fresh_start_edge_fraction,
        "no_fresh_start_throttle": not kelly_config.market_impact.fresh_start_enabled,
        "intraday_mode": forecaster_config.intraday_mode,
        "interday_model": forecaster_config.interday_model,
        "historical_bootstrap": forecaster_config.bucket_nowcast.use_historical_bootstrap,
        "bootstrap_start_hours": forecaster_config.bucket_nowcast.bootstrap_start_hours,
        "bootstrap_full_hours": forecaster_config.bucket_nowcast.bootstrap_full_hours,
        "bootstrap_max_blend": forecaster_config.bucket_nowcast.bootstrap_max_blend,
        "boundary_silence_overlay": forecaster_config.bucket_nowcast.late_boundary_silence.enabled,
        "boundary_silence_hours": forecaster_config.bucket_nowcast.late_boundary_silence.start_hours,
        "boundary_silence_max_distance": forecaster_config.bucket_nowcast.late_boundary_silence.max_distance_to_next_bin,
        "boundary_silence_threshold_start": forecaster_config.bucket_nowcast.late_boundary_silence.silence_threshold_start_minutes,
        "boundary_silence_threshold_floor": forecaster_config.bucket_nowcast.late_boundary_silence.silence_threshold_floor_minutes,
        "boundary_silence_threshold_step": forecaster_config.bucket_nowcast.late_boundary_silence.silence_threshold_step_per_hour,
        "boundary_silence_min_effective_n": forecaster_config.bucket_nowcast.late_boundary_silence.min_effective_n,
        "late_boundary_take_profit": kelly_config.late_boundary_take_profit.enabled,
        "late_boundary_trigger_price": kelly_config.late_boundary_take_profit.trigger_price,
        "late_boundary_min_sell_fraction": kelly_config.late_boundary_take_profit.min_sell_fraction,
        "late_boundary_max_sell_fraction": kelly_config.late_boundary_take_profit.max_sell_fraction,
        "use_unbox_rotations": kelly_config.use_unbox_rotations,
        "unbox_start_hours_to_settlement": kelly_config.unbox_start_hours_to_settlement,
        "unbox_min_blocked_ticks": kelly_config.unbox_min_blocked_ticks,
        "unbox_min_net_utility": kelly_config.unbox_min_net_utility,
        "unbox_late_relax_start_hours": kelly_config.unbox_late_relax_start_hours_to_settlement,
        "unbox_late_net_utility_relax": kelly_config.unbox_late_net_utility_relax,
        "unbox_repeat_net_utility_step": kelly_config.unbox_repeat_net_utility_step,
        "unbox_repeat_net_utility_cap": kelly_config.unbox_repeat_net_utility_cap,
        "unbox_multi_bin_start_count": kelly_config.unbox_multi_bin_start_count,
        "unbox_multi_bin_net_utility_step": kelly_config.unbox_multi_bin_net_utility_step,
        "unbox_multi_bin_net_utility_cap": kelly_config.unbox_multi_bin_net_utility_cap,
        "unbox_turnover_penalty": kelly_config.unbox_turnover_penalty,
        "unbox_bin_cooldown_seconds": kelly_config.unbox_bin_cooldown_seconds,
        "consensus_mode": _consensus_mode_from_config(kelly_config.market_consensus),
        "consensus_time_tau": kelly_config.market_consensus.time_tau,
        "consensus_gap_scale": kelly_config.market_consensus.gap_scale,
        "consensus_gap_gamma": kelly_config.market_consensus.gap_gamma,
        "consensus_gap_floor": kelly_config.market_consensus.gap_floor,
        "consensus_min_model_weight": kelly_config.market_consensus.min_model_weight,
        "consensus_min_coverage": kelly_config.market_consensus.min_coverage_ratio,
        "consensus_max_avg_spread": kelly_config.market_consensus.max_avg_spread,
        "consensus_max_bin_spread": kelly_config.market_consensus.max_bin_spread,
        "consensus_allow_untrusted_buys": not kelly_config.market_consensus.require_trusted_quote_for_buys,
        "robust_kelly": kelly_config.robust_kelly.enabled,
        "robust_kelly_min_fraction_multiplier": kelly_config.robust_kelly.min_fraction_multiplier,
        "robust_kelly_min_coverage": kelly_config.robust_kelly.min_coverage_ratio,
        "robust_kelly_max_avg_spread": kelly_config.robust_kelly.max_avg_spread,
        "robust_kelly_disagreement_scale": kelly_config.robust_kelly.disagreement_scale,
        "market_buy_guard": kelly_config.market_buy_guard.enabled,
        "market_buy_guard_max_widening": kelly_config.market_buy_guard.max_threshold_widening,
        "market_buy_guard_min_coverage": kelly_config.market_buy_guard.min_coverage_ratio,
        "market_buy_guard_max_avg_spread": kelly_config.market_buy_guard.max_avg_spread,
        "market_buy_guard_disagreement_scale": kelly_config.market_buy_guard.disagreement_scale,
    }
    return raw_config, kelly_config, forecaster_config, cli_defaults


def _resolve_consensus_mode(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[bool, bool]:
    """Resolve consensus mode presets against explicit legacy flags."""
    mode_to_flags = {
        "off": (False, False),
        "time_only": (True, False),
        "gap_only": (False, True),
        "time_gap": (True, True),
    }
    mode_time, mode_gap = mode_to_flags[args.consensus_mode]
    flag_time = bool(args.consensus_time)
    flag_gap = bool(args.consensus_gap)

    if args.consensus_mode != "off":
        if flag_time != mode_time and flag_time:
            parser.error(
                f"--consensus-mode {args.consensus_mode} conflicts with --consensus-time"
            )
        if flag_gap != mode_gap and flag_gap:
            parser.error(
                f"--consensus-mode {args.consensus_mode} conflicts with --consensus-gap"
            )
        args.consensus_time = mode_time
        args.consensus_gap = mode_gap

    return bool(args.consensus_time), bool(args.consensus_gap)


def log_config_summary(
    kelly_config: KellyConfig,
    multi_event_config: "MultiEventConfig",
    forecaster_config: ForecasterConfig,
    capital_pool_config: CapitalPoolConfig,
    dry_run: bool,
) -> None:
    """Log a formatted summary of all configurable parameters at startup."""
    lines = []
    w = lines.append
    w("")
    w("=" * 72)
    w("  CONFIGURATION SUMMARY")
    w("=" * 72)

    # Mode
    mode = "DRY RUN" if dry_run else "LIVE TRADING"
    w(f"  Mode: {mode}")
    w("")

    # Capital
    w("  CAPITAL")
    w(f"    Total capital:           ${capital_pool_config.total_capital:,.2f}")
    w(f"    Min allocation:          ${capital_pool_config.min_allocation:,.2f}")
    w(f"    Max per event:           ${multi_event_config.max_per_event:,.2f}")
    cm = kelly_config.collateral.capital_multiplier
    bin_max_label = f"{kelly_config.collateral.c_bin_max_ratio:.0%} of event max" if cm == 1.0 else f"{kelly_config.collateral.c_bin_max_ratio:.0%} of virtual event max ${kelly_config.collateral.virtual_c_event_max:,.2f}"
    w(f"    Max per bin:             ${kelly_config.collateral.c_bin_max:,.2f}  ({bin_max_label})")
    w(f"    Capital multiplier:      {cm}x" + (f" (phantom capital enabled, virtual event max=${kelly_config.collateral.virtual_c_event_max:,.2f})" if cm != 1.0 else " (standard Kelly)"))
    w("")

    # Kelly
    w("  KELLY OPTIMIZER")
    w(f"    Kappa:                   {kelly_config.kappa}")
    w(f"    Kelly fraction (alpha):  {kelly_config.kelly_fraction}")
    w(f"    W floor:                 {kelly_config.w_floor}")
    w(f"    Min buy utility:         {kelly_config.min_buy_utility}")
    w(f"    Min sell utility:        {kelly_config.min_sell_utility}")
    w(f"    Kelly-only exit:         {kelly_config.kelly_only_exit}")
    w(f"    Renormalize probs:       {kelly_config.renormalize_probabilities}")
    w(f"    Prob EMA alpha:          {kelly_config.prob_ema_alpha}")
    w(f"    T_stop hours:            per-event (from trading rules, default={kelly_config.t_stop_hours})")
    w(f"    Max iters/tick:          {kelly_config.max_iters_per_tick}")
    w("")

    # Edge buffer
    eb = kelly_config.edge_buffer
    w("  EDGE BUFFER")
    w(f"    Required ROI:            {eb.required_roi:.2%}")
    w(f"    Friction mid:            {eb.friction_mid:.2%}")
    w(f"    Friction tail:           {eb.friction_tail:.2%}")
    w(f"    Tail threshold:          {eb.tail_threshold:.2%}")
    w(f"    Sell friction:           {eb.sell_friction:.2%}")
    w(f"    Min perceived prob:      {eb.min_perceived_prob:.2%}")
    w(f"    Min market price:        {eb.min_market_price:.2%}")
    w(f"    Require 2-sided liq:     {eb.require_two_sided_liquidity}")
    w(f"    Max spread ratio:        {eb.max_spread_ratio}")
    w("")

    mi = kelly_config.market_impact
    w("  MARKET IMPACT")
    w(f"    Fresh-start throttle:    {mi.fresh_start_enabled}")
    w(f"    Fresh-start window:      {mi.fresh_start_minutes:.1f}m")
    w(f"    Edge fraction cap:       {mi.fresh_start_edge_fraction:.2f}")
    w("")

    mc = kelly_config.market_consensus
    if mc.enabled:
        w("  MARKET CONSENSUS")
        w(f"    Time blend:              {mc.time_enabled} (tau={mc.time_tau:.1f}h)")
        w(f"    Gap blend:               {mc.gap_enabled} (scale={mc.gap_scale:.2f}, gamma={mc.gap_gamma:.1f}, floor={mc.gap_floor:.2f})")
        w(f"    Min model weight:        {mc.min_model_weight:.2f}")
        w(f"    Min coverage ratio:      {mc.min_coverage_ratio:.2f}")
        w(f"    Max avg spread:          {mc.max_avg_spread:.3f}")
        w(f"    Max bin spread:          {mc.max_bin_spread:.3f}")
        w(f"    Require trusted buys:    {mc.require_trusted_quote_for_buys}")
        w("")

    # Rate limit
    rl = kelly_config.rate_limit
    w("  RATE LIMITING")
    w(f"    Max orders/tick:         {rl.max_orders_per_tick}")
    w(f"    Max orders/minute:       {rl.max_orders_per_minute}")
    w(f"    Min order delay:         {rl.min_order_delay_seconds}s")
    w(f"    Block confirm timeout:   {rl.block_confirmation_timeout_seconds}s")
    w(f"    Tick timeout:            {rl.tick_timeout_seconds}s")
    w(f"    FAK failure cooldown:    {rl.fak_failure_cooldown_seconds}s")
    w("")

    # Multi-event
    w("  MULTI-EVENT")
    w(f"    Slow tick interval:      {multi_event_config.tick_interval_seconds}s")
    w(f"    Fast tick interval:      {multi_event_config.fast_tick_interval_seconds}s")
    w(f"    Forecast cache:          {multi_event_config.forecast_cache_seconds}s")
    w(f"    Orderbook websocket:     {not multi_event_config.disable_websocket}")
    w(f"    Event duration filter:   {multi_event_config.min_event_duration_days}-{multi_event_config.max_event_duration_days} days")
    w(f"    Projection model:        {multi_event_config.projection_model}")
    w(f"    Event scan interval:     {multi_event_config.event_scan_interval}s")
    w(f"    Max data age:            {multi_event_config.max_data_age_seconds}s")
    w(f"    Count validation:        every {multi_event_config.count_validation_interval}s")
    w(f"    Realtime tracker:        {multi_event_config.realtime_tracker_enabled}")
    if multi_event_config.realtime_tracker_enabled:
        w(f"    Realtime poll interval:  {multi_event_config.realtime_poll_interval_seconds}s")
        w(f"    Realtime fetch count:    {multi_event_config.realtime_fetch_count}")
        w(f"    Realtime late grace:     {multi_event_config.realtime_late_tweet_grace_seconds}s")
        w(f"    Realtime cookies path:   {multi_event_config.realtime_cookies_path or 'config/twitter_cookies.json'}")
    w("")

    # Event trading rules
    etr = multi_event_config.event_trading_rules
    if etr and etr.categories:
        w("  EVENT TRADING RULES")
        for cat in etr.categories:
            settle_str = f"stop {cat.min_hours_before_settlement}h before settlement"
            if cat.require_counting_started:
                w(f"    {cat.name:10s} ({cat.duration_min_days}-{cat.duration_max_days}d): require counting started, {settle_str}")
            else:
                before = f"trade {cat.max_hours_before_counting}h before counting" if cat.max_hours_before_counting else ""
                max_days = f"start {cat.max_days_before_settlement}d before settle" if cat.max_days_before_settlement else ""
                detail = before or max_days or "no restrictions"
                w(f"    {cat.name:10s} ({cat.duration_min_days}-{cat.duration_max_days}d): {detail}, {settle_str}")
        w("")

    # Forecaster
    w("  FORECASTER")
    w(f"    Intraday mode:           {forecaster_config.intraday_mode}")
    w(f"    Interday model:          {forecaster_config.interday_model}")
    w(f"    Projection model:        {multi_event_config.projection_model}")
    w(f"    Timezone:                {forecaster_config.timezone}")
    w(f"    Contract boundary hr:    {forecaster_config.contract_boundary_hour}")

    w("=" * 72)
    w("")

    logger.info("\n".join(lines))


# API Constants
GAMMA_API_URL = "https://gamma-api.polymarket.com"
MUSK_TWEET_TAG_ID = 972


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with compact format."""
    level = logging.DEBUG if verbose else logging.INFO

    # Custom formatter for compact output
    # Format: "26-02-03 15:20:01[I]: message"
    class CompactFormatter(logging.Formatter):
        LEVEL_MAP = {
            'DEBUG': 'D',
            'INFO': 'I',
            'WARNING': 'W',
            'ERROR': 'E',
            'CRITICAL': 'C',
        }

        def format(self, record):
            level_char = self.LEVEL_MAP.get(record.levelname, '?')
            timestamp = self.formatTime(record, "%y-%m-%d %H:%M:%S")
            message = record.getMessage()

            if record.exc_info:
                if not record.exc_text:
                    record.exc_text = self.formatException(record.exc_info)
                if record.exc_text:
                    message = f"{message}\n{record.exc_text}"

            if record.stack_info:
                message = f"{message}\n{self.formatStack(record.stack_info)}"

            return f"{timestamp}[{level_char}]: {message}"

    handler = logging.StreamHandler()
    handler.setFormatter(CompactFormatter())

    logging.root.handlers = []
    logging.root.addHandler(handler)
    logging.root.setLevel(level)

    # Reduce noise from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


def create_clob_client() -> ClobClient:
    """
    Create authenticated CLOB client from environment variables.

    Loads private key from:
    - POLYMARKET_PRIVATE_KEY (plain)
    - ENCRYPTED_POLYMARKET_PRIVATE_KEY + PK_PWD (encrypted)

    API credentials from:
    - CLOB_API_KEY, CLOB_API_SECRET, CLOB_API_PASSPHRASE

    For proxy wallets (Magic Link / browser connection):
    - POLY_FUNDER: Proxy wallet address (required for signature_type 1 or 2)
    - POLY_SIGNATURE_TYPE: 0=EOA, 1=POLY_PROXY (Magic Link), 2=GNOSIS_SAFE
    """
    host = os.getenv("CLOB_HOST", "https://clob.polymarket.com")
    key = os.getenv("CLOB_API_KEY")
    secret = os.getenv("CLOB_API_SECRET")
    passphrase = os.getenv("CLOB_API_PASSPHRASE")
    chain_id = int(os.getenv("CHAIN_ID", "137"))  # Polygon mainnet

    # Proxy wallet settings
    funder = os.getenv("POLY_FUNDER")  # Proxy wallet address
    signature_type = int(os.getenv("POLY_SIGNATURE_TYPE", "0"))  # 0=EOA, 1=POLY_PROXY, 2=GNOSIS_SAFE

    if not all([key, secret, passphrase]):
        raise ValueError(
            "Missing CLOB credentials. Set CLOB_API_KEY, CLOB_API_SECRET, CLOB_API_PASSPHRASE"
        )

    # Load private key (handles encrypted keys automatically)
    try:
        private_key = load_private_key()
        logger.info("Private key loaded successfully")
    except ValueError as e:
        raise ValueError(f"Failed to load private key: {e}")

    creds = ApiCreds(
        api_key=key,
        api_secret=secret,
        api_passphrase=passphrase,
    )

    # Create client with proxy wallet support
    if funder and signature_type > 0:
        logger.info(f"Using proxy wallet: {funder} (signature_type={signature_type})")
        return ClobClient(
            host,
            key=private_key,
            chain_id=chain_id,
            creds=creds,
            signature_type=signature_type,
            funder=funder,
        )
    else:
        logger.info("Using EOA wallet (signature_type=0)")
        return ClobClient(host, key=private_key, chain_id=chain_id, creds=creds)


PUSD_MIN_BALANCE_USD = 10.0


def assert_sufficient_pusd(clob_client: ClobClient, min_usd: float = PUSD_MIN_BALANCE_USD) -> None:
    """
    Fail fast if the wallet's pUSD collateral balance is below `min_usd`.

    Polymarket CLOB V2 settles in pUSD; USDC.e on the wallet is not
    spendable on the exchange until wrapped via the CollateralOnramp.
    Wrapping is operator-driven; if balance is insufficient, this raises
    SystemExit with a pointer at scripts/wrap_usdce_to_pusd.py.
    """
    try:
        info = clob_client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
    except Exception as e:
        logger.warning(f"Could not query pUSD balance: {e}")
        return
    raw = info.get("balance") if isinstance(info, dict) else None
    try:
        pusd = float(raw) / 1e6
    except (TypeError, ValueError):
        logger.warning(f"Unparseable balance response: {info!r}")
        return
    logger.info(f"pUSD collateral balance: ${pusd:,.2f}")
    if pusd < min_usd:
        logger.error(
            f"pUSD balance ${pusd:,.2f} < ${min_usd:,.2f}. "
            f"Wrap USDC.e first: python -m scripts.wrap_usdce_to_pusd --check"
        )
        sys.exit(1)


def get_wallet_address_for_positions() -> str:
    """
    Get wallet address for position queries.

    IMPORTANT: When using a proxy wallet (signature_type=1 or 2),
    positions are held by the PROXY wallet (POLY_FUNDER), not the main wallet.
    This is because trades are executed through the proxy.

    Returns:
        Wallet address that holds positions (proxy if configured, else main wallet)
    """
    # Check if proxy wallet is configured
    funder = os.getenv("POLY_FUNDER")
    signature_type = int(os.getenv("POLY_SIGNATURE_TYPE", "0"))

    if funder and signature_type in (1, 2):
        # Proxy wallet holds the positions
        logger.debug(f"Using proxy wallet for position queries: {funder}")
        return funder

    # No proxy - use main wallet
    wallet = os.getenv("WALLET_ADDRESS")
    if wallet:
        return wallet

    raise ValueError(
        "No wallet address configured. "
        "Set WALLET_ADDRESS or POLY_FUNDER (for proxy wallets)."
    )


def get_wallet_address(clob_client: ClobClient) -> str:
    """
    Get wallet address from CLOB client (for signing).

    NOTE: For position queries, use get_wallet_address_for_positions() instead,
    as positions may be held by a proxy wallet.
    """
    try:
        # Derive from CLOB client's signer
        address = clob_client.get_address()
        if address:
            return address
    except Exception as e:
        logger.debug(f"Could not derive address from CLOB client: {e}")

    # Fallback to env var
    wallet = os.getenv("WALLET_ADDRESS")
    if wallet:
        return wallet

    raise ValueError(
        "Could not derive wallet address from CLOB client. "
        "Ensure CLOB credentials are correct."
    )


def parse_counting_dates_from_title(title: str) -> Tuple[Optional[date], Optional[date]]:
    """
    Parse tweet counting dates from event title.

    Matches patterns like:
    - "Elon Musk # tweets January 13 - January 20, 2026?"

    Returns:
        Tuple of (start_date, end_date) as date objects, or (None, None) if not parsed
    """
    # Pattern: "Month Day - Month Day, Year"
    pattern = r"(\w+)\s+(\d{1,2})\s*[-–]\s*(\w+)\s+(\d{1,2}),?\s*(\d{4})"
    match = re.search(pattern, title)

    if not match:
        return None, None

    try:
        start_month_str = match.group(1)
        start_day = int(match.group(2))
        end_month_str = match.group(3)
        end_day = int(match.group(4))
        year = int(match.group(5))

        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }

        start_month = month_map.get(start_month_str.lower())
        end_month = month_map.get(end_month_str.lower())

        if not start_month or not end_month:
            return None, None

        start_date = date(year, start_month, start_day)
        end_date = date(year, end_month, end_day)

        return start_date, end_date

    except (ValueError, AttributeError) as e:
        logger.debug(f"Failed to parse dates from '{title}': {e}")
        return None, None


def parse_bin_bounds(outcome: str) -> Tuple[int, int]:
    """
    Parse bin bounds from outcome string.

    Examples:
        "0-74" -> (0, 74)
        "75-99" -> (75, 99)
        "500+" -> (500, inf)

    Returns:
        Tuple of (lower, upper) bounds
    """
    outcome = outcome.strip()

    # Handle "X+" format (e.g., "500+")
    if outcome.endswith("+"):
        lower = int(outcome[:-1])
        return lower, float('inf')

    # Handle "<X" format (e.g., "<40" -> (0, 39))
    if outcome.startswith("<"):
        upper = int(outcome[1:]) - 1
        return 0, upper

    # Handle "X-Y" format
    if "-" in outcome:
        parts = outcome.split("-")
        lower = int(parts[0])
        upper = int(parts[1])
        return lower, upper

    # Single number
    try:
        val = int(outcome)
        return val, val
    except ValueError:
        return 0, 0


async def discover_musk_tweet_events(clob_client: ClobClient) -> List[EventInfo]:
    """
    Discover active Musk tweet count events from Polymarket.

    Uses the Gamma API to find active events with tag_id=972 (Musk tweets).

    Returns:
        List of EventInfo for active events
    """
    logger.info("Discovering Musk tweet count events from Gamma API...")

    try:
        response = requests.get(
            f"{GAMMA_API_URL}/events",
            params={
                "tag_id": MUSK_TWEET_TAG_ID,
                "active": "true",
                "closed": "false",
                "limit": 100,
            },
            timeout=30,
        )
        response.raise_for_status()
        events_data = response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch events from Gamma API: {e}")
        return []

    events: List[EventInfo] = []

    for event_data in events_data:
        event_title = event_data.get("title", "")
        event_id = event_data.get("id", "")

        # Filter for Musk tweet events only (API may return other events under same tag)
        title_lower = event_title.lower()
        if "musk" not in title_lower and "elon" not in title_lower:
            logger.debug(f"Skipping non-Musk event: {event_title}")
            continue

        # Parse counting dates from title
        market_start, settlement = parse_counting_dates_from_title(event_title)

        if not market_start or not settlement:
            logger.debug(f"Skipping event with unparseable dates: {event_title}")
            continue

        # Check if event hasn't settled yet
        now = datetime.now(timezone.utc)
        settlement_dt, _ = CONTRACT_UTILS.get_contract_day_bounds_utc(settlement)

        if now >= settlement_dt:
            logger.debug(f"Skipping settled event: {event_title}")
            continue

        # Parse markets (bins) from event
        markets = event_data.get("markets", [])
        bins = []

        for market in markets:
            outcome = market.get("groupItemTitle", "")
            if not outcome:
                continue

            lower, upper = parse_bin_bounds(outcome)

            # Get token IDs (index 0 = YES, index 1 = NO)
            clob_token_ids = market.get("clobTokenIds", [])
            if isinstance(clob_token_ids, str):
                try:
                    clob_token_ids = json.loads(clob_token_ids)
                except json.JSONDecodeError:
                    clob_token_ids = []

            yes_token_id = clob_token_ids[0] if len(clob_token_ids) > 0 else None
            no_token_id = clob_token_ids[1] if len(clob_token_ids) > 1 else None

            if yes_token_id:
                bins.append({
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "token_id": yes_token_id,
                    "no_token_id": no_token_id,
                    "outcome": outcome,
                    "condition_id": market.get("conditionId"),
                })

        # Sort bins by lower bound
        bins.sort(key=lambda b: b.get("lower_bound", 0))

        if not bins:
            logger.debug(f"Skipping event with no valid bins: {event_title}")
            continue

        # Create short name from dates
        short_name = f"{market_start.strftime('%b %d')} - {settlement.strftime('%b %d')}"

        event_info = EventInfo(
            event_id=event_id,
            title=event_title,
            short_name=short_name,
            settlement_date=settlement,
            market_start_date=market_start,
            bins=bins,
            condition_id=event_data.get("conditionId"),
        )

        events.append(event_info)
        logger.info(f"Discovered event: {short_name} ({len(bins)} bins)")

    logger.info(f"Discovered {len(events)} active Musk tweet events")
    return events


async def fetch_event_details(
    clob_client: ClobClient,
    event_id: str,
) -> EventInfo:
    """
    Fetch event details from Polymarket by event ID.

    Args:
        clob_client: CLOB client (not used, keeping for interface compatibility)
        event_id: Event ID to fetch

    Returns:
        EventInfo with full details
    """
    logger.info(f"Fetching details for event {event_id}...")

    try:
        response = requests.get(
            f"{GAMMA_API_URL}/events/{event_id}",
            timeout=30,
        )
        response.raise_for_status()
        event_data = response.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch event {event_id}: {e}")

    event_title = event_data.get("title", "")

    # Parse counting dates from title
    market_start, settlement = parse_counting_dates_from_title(event_title)

    if not market_start or not settlement:
        raise ValueError(f"Could not parse dates from event title: {event_title}")

    # Parse markets (bins) from event
    markets = event_data.get("markets", [])
    bins = []

    for market in markets:
        outcome = market.get("groupItemTitle", "")
        if not outcome:
            continue

        lower, upper = parse_bin_bounds(outcome)

        # Get token IDs (index 0 = YES, index 1 = NO)
        clob_token_ids = market.get("clobTokenIds", [])
        if isinstance(clob_token_ids, str):
            try:
                clob_token_ids = json.loads(clob_token_ids)
            except json.JSONDecodeError:
                clob_token_ids = []

        yes_token_id = clob_token_ids[0] if len(clob_token_ids) > 0 else None
        no_token_id = clob_token_ids[1] if len(clob_token_ids) > 1 else None

        if yes_token_id:
            bins.append({
                "lower_bound": lower,
                "upper_bound": upper,
                "token_id": yes_token_id,
                "no_token_id": no_token_id,
                "outcome": outcome,
                "condition_id": market.get("conditionId"),
            })

    # Sort bins by lower bound
    bins.sort(key=lambda b: b.get("lower_bound", 0))

    if not bins:
        raise ValueError(f"No valid bins found for event: {event_title}")

    # Create short name from dates
    short_name = f"{market_start.strftime('%b %d')} - {settlement.strftime('%b %d')}"

    return EventInfo(
        event_id=event_id,
        title=event_title,
        short_name=short_name,
        settlement_date=settlement,
        market_start_date=market_start,
        bins=bins,
        condition_id=event_data.get("conditionId"),
    )


def build_collateral_config(
    *,
    max_per_event: Optional[float],
    capital_multiplier: float,
    base: Optional[CollateralConfig] = None,
) -> CollateralConfig:
    """Build Kelly collateral config from YAML-backed defaults plus CLI values."""
    base_config = base or CollateralConfig()
    return CollateralConfig(
        c_event_max=(
            max_per_event
            if max_per_event is not None else base_config.c_event_max
        ),
        c_bin_max_ratio=base_config.c_bin_max_ratio,
        capital_multiplier=capital_multiplier,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_RUNNER_CONFIG_PATH,
    )
    bootstrap_args, _ = bootstrap_parser.parse_known_args(argv)
    _, _, _, cli_defaults = _build_base_configs_from_yaml(bootstrap_args.config)

    parser = argparse.ArgumentParser(
        description="Run multi-event trading bot with shared capital pool"
    )
    parser.set_defaults(**cli_defaults)
    parser.add_argument(
        "--config",
        type=str,
        default=bootstrap_args.config,
        help=f"Path to base YAML config (default: {DEFAULT_RUNNER_CONFIG_PATH})",
    )

    # Capital configuration
    parser.add_argument(
        "--capital",
        type=float,
        default=0.0,
        help="Total capital in pool (default: 0 = auto-detect from API)",
    )
    parser.add_argument(
        "--max-per-event",
        "--c-event-max",
        dest="max_per_event",
        type=float,
        default=cli_defaults.get("max_per_event"),
        help="Maximum capital per event / Kelly c_event_max (default: from YAML config)",
    )
    parser.add_argument(
        "--min-allocation",
        type=float,
        default=100.0,
        help="Minimum capital to start an event (default: 100)",
    )

    # Trading configuration
    parser.add_argument(
        "--tick-interval",
        type=int,
        default=cli_defaults.get("tick_interval", 3600),
        help="Seconds between slow trading ticks (default: from YAML config)",
    )
    parser.add_argument(
        "--fast-tick-interval",
        type=int,
        default=cli_defaults.get("fast_tick_interval", 30),
        help="Seconds between fast orderbook checks (default: from YAML config)",
    )
    parser.add_argument(
        "--forecast-cache-seconds",
        type=int,
        default=cli_defaults.get("forecast_cache_seconds", 165),
        help="Forecast cache timeout in seconds (default: from YAML config)",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        dest="live",
        action="store_false",
        help="Run in dry-run mode (no real orders)",
    )
    mode_group.add_argument(
        "--live",
        dest="live",
        action="store_true",
        help="Run in live mode (real orders)",
    )
    ws_group = parser.add_mutually_exclusive_group()
    ws_group.add_argument(
        "--no-ws",
        dest="no_ws",
        action="store_true",
        help="Disable market-data websocket streaming and use REST orderbook fetches only.",
    )
    ws_group.add_argument(
        "--ws",
        dest="no_ws",
        action="store_false",
        help="Enable market-data websocket streaming.",
    )

    # Event selection
    parser.add_argument(
        "--events",
        type=str,
        default="auto",
        help="Event IDs to trade (comma-separated) or 'auto' for discovery",
    )
    parser.add_argument(
        "--min-event-days",
        type=int,
        default=7,
        help="Minimum event duration in days to auto-discover (inclusive, default: 7)",
    )
    parser.add_argument(
        "--max-event-days",
        type=int,
        default=7,
        help="Maximum event duration in days to auto-discover (inclusive, default: 7). "
             "Events with existing positions are always included regardless of this filter.",
    )
    parser.add_argument(
        "--event-rules",
        type=str,
        default="config/event_trading_rules.yaml",
        help="Path to event trading rules YAML config (default: config/event_trading_rules.yaml). "
             "Controls when trading is allowed based on event duration and counting status.",
    )
    realtime_group = parser.add_mutually_exclusive_group()
    realtime_group.add_argument(
        "--disable-realtime-tracker",
        dest="disable_realtime_tracker",
        action="store_true",
        help="Disable provisional twikit polling. Default: enabled.",
    )
    realtime_group.add_argument(
        "--realtime-tracker",
        dest="disable_realtime_tracker",
        action="store_false",
        help="Enable provisional twikit polling.",
    )
    parser.add_argument(
        "--realtime-poll-interval",
        type=float,
        default=None,
        help="Seconds between twikit polls for provisional tweet detection (default: from config, 10).",
    )
    parser.add_argument(
        "--realtime-fetch-count",
        type=int,
        default=None,
        help="Number of tweets to fetch per twikit poll (default: from config, 40).",
    )
    parser.add_argument(
        "--realtime-late-tweet-grace",
        type=float,
        default=None,
        help="Seconds of grace for slightly late/out-of-order tweets from twikit (default: from config, 120).",
    )
    parser.add_argument(
        "--realtime-cookies-path",
        type=str,
        default=None,
        help="Optional override for twikit cookies JSON path. Default: config/twitter_cookies.json.",
    )

    # Kelly configuration
    parser.add_argument(
        "--kappa",
        type=float,
        default=cli_defaults.get("kappa", KellyConfig.kappa),
        help="Fractional Kelly multiplier (default: from YAML config)",
    )
    parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=cli_defaults.get("kelly_fraction", KellyConfig.kelly_fraction),
        help="Fractional Kelly parameter α ∈ (0, 1]. "
             "1.0 = full Kelly, 0.5 = half Kelly, 0.25 = quarter Kelly (default: from YAML config)",
    )
    parser.add_argument(
        "--required-roi",
        type=float,
        default=cli_defaults.get("required_roi", EdgeBufferConfig.required_roi),
        help="Required ROI for edge buffer (default: from YAML config)",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=cli_defaults.get("min_prob", EdgeBufferConfig.min_perceived_prob),
        help="Minimum model probability to trade a bin (default: from YAML config)",
    )
    parser.add_argument(
        "--min-market-price",
        type=float,
        default=cli_defaults.get("min_market_price", EdgeBufferConfig.min_market_price),
        help="Minimum market price to trade (default: from YAML config)",
    )
    parser.add_argument(
        "--max-spread-ratio",
        type=float,
        default=cli_defaults.get("max_spread_ratio", EdgeBufferConfig.max_spread_ratio),
        help="Maximum spread ratio (ask-bid)/bid to trade. Default: from YAML config. Set to 0 to disable.",
    )
    two_sided_group = parser.add_mutually_exclusive_group()
    two_sided_group.add_argument(
        "--no-require-two-sided",
        dest="no_require_two_sided",
        action="store_true",
        help="Disable requirement for two-sided liquidity (both bid and ask). Default: require two-sided.",
    )
    two_sided_group.add_argument(
        "--require-two-sided",
        dest="no_require_two_sided",
        action="store_false",
        help="Require two-sided liquidity (both bid and ask).",
    )
    parser.add_argument(
        "--capital-multiplier",
        type=float,
        default=cli_defaults.get("capital_multiplier", CollateralConfig.capital_multiplier),
        help="Capital multiplier for phantom capital injection. "
             "1.0 = standard Kelly. 2.0 = Kelly sees 2x capital → bigger positions. "
             "Real capital still hard-gates execution. Default: from YAML config",
    )
    parser.add_argument(
        "--min-buy-utility",
        type=float,
        default=cli_defaults.get("min_buy_utility", KellyConfig.min_buy_utility),
        help="Minimum utility gain for buys (default: from YAML config)",
    )
    parser.add_argument(
        "--min-sell-utility",
        type=float,
        default=cli_defaults.get("min_sell_utility", KellyConfig.min_sell_utility),
        help="Minimum utility gain for sells (default: from YAML config)",
    )
    parser.add_argument(
        "--fresh-start-minutes",
        type=float,
        default=cli_defaults.get("fresh_start_minutes", MarketImpactConfig.fresh_start_minutes),
        help="Minutes to throttle after an event's first trading tick (default: from YAML config)",
    )
    parser.add_argument(
        "--fresh-start-edge-fraction",
        type=float,
        default=cli_defaults.get("fresh_start_edge_fraction", MarketImpactConfig.fresh_start_edge_fraction),
        help="Fraction of available edge the executor may consume per fresh-start tick (default: from YAML config)",
    )
    fresh_start_group = parser.add_mutually_exclusive_group()
    fresh_start_group.add_argument(
        "--no-fresh-start-throttle",
        dest="no_fresh_start_throttle",
        action="store_true",
        help="Disable fresh-start market impact throttling.",
    )
    fresh_start_group.add_argument(
        "--fresh-start-throttle",
        dest="no_fresh_start_throttle",
        action="store_false",
        help="Enable fresh-start market impact throttling.",
    )

    # Projection model
    parser.add_argument(
        "--projection",
        type=str,
        default="asymmetric",
        choices=["asymmetric", "normal", "skew_normal", "gamma"],
        help="Projection model for computing bin probabilities. "
             "'asymmetric' (default) uses actual Monte Carlo samples. "
             "'normal' uses symmetric Normal CDF. "
             "'skew_normal' uses Skew-Normal CDF (captures right-skew). "
             "'gamma' uses Gamma CDF (natural for positive sums).",
    )

    # Intraday forecaster mode
    parser.add_argument(
        "--intraday-mode",
        type=str,
        default=cli_defaults.get("intraday_mode", "ridge"),
        choices=["ridge", "bucket"],
        help="Intraday forecaster mode: 'ridge' (default) uses Ridge regression with "
             "linear F(τ) scaling, 'bucket' uses Negative Binomial per 3-hour bucket.",
    )

    parser.add_argument(
        "--interday-model",
        type=str,
        default=cli_defaults.get("interday_model", "ewma"),
        choices=["ewma", "gas", "pig"],
        help="Interday forecaster mode: "
             "'ewma' (default) uses the original EWMA regime model. "
             "'gas' uses NB-GAS regime dynamics. "
             "'pig' uses PIG-GAS with heavier-tailed future-day sampling.",
    )

    parser.add_argument(
        "--historical-bootstrap",
        action="store_true",
        default=cli_defaults.get("historical_bootstrap", False),
        help="Blend late intraday bucket forecasts with weighted historical suffix samples.",
    )

    parser.add_argument(
        "--bootstrap-start-hours",
        type=float,
        default=cli_defaults.get("bootstrap_start_hours", 6.0),
        help="Hours left threshold where historical bootstrap starts blending in. Default: 6.0.",
    )

    parser.add_argument(
        "--bootstrap-full-hours",
        type=float,
        default=cli_defaults.get("bootstrap_full_hours", 3.0),
        help="Hours left threshold where historical bootstrap reaches max blend. Default: 3.0.",
    )

    parser.add_argument(
        "--bootstrap-max-blend",
        type=float,
        default=cli_defaults.get("bootstrap_max_blend", 0.35),
        help="Maximum mixture weight for historical bootstrap samples. Default: 0.35.",
    )

    parser.add_argument(
        "--boundary-silence-overlay",
        action="store_true",
        default=cli_defaults.get("boundary_silence_overlay", False),
        help="Enable the late-boundary silence overlay in the final hours before settlement.",
    )

    parser.add_argument(
        "--boundary-silence-hours",
        type=float,
        default=cli_defaults.get(
            "boundary_silence_hours",
            BucketNowcastConfig.LateBoundarySilenceConfig.start_hours,
        ),
        help="Hours before settlement where the late-boundary silence overlay becomes eligible.",
    )

    parser.add_argument(
        "--boundary-silence-max-distance",
        type=int,
        default=cli_defaults.get(
            "boundary_silence_max_distance",
            BucketNowcastConfig.LateBoundarySilenceConfig.max_distance_to_next_bin,
        ),
        help="Maximum tweets from the next bin edge for the late-boundary silence overlay.",
    )

    parser.add_argument(
        "--boundary-silence-threshold-start",
        type=int,
        default=cli_defaults.get(
            "boundary_silence_threshold_start",
            BucketNowcastConfig.LateBoundarySilenceConfig.silence_threshold_start_minutes,
        ),
        help="Adaptive silence threshold at the overlay start window in minutes.",
    )

    parser.add_argument(
        "--boundary-silence-threshold-floor",
        type=int,
        default=cli_defaults.get(
            "boundary_silence_threshold_floor",
            BucketNowcastConfig.LateBoundarySilenceConfig.silence_threshold_floor_minutes,
        ),
        help="Minimum adaptive silence threshold in minutes once the overlay is active.",
    )

    parser.add_argument(
        "--boundary-silence-threshold-step",
        type=float,
        default=cli_defaults.get(
            "boundary_silence_threshold_step",
            BucketNowcastConfig.LateBoundarySilenceConfig.silence_threshold_step_per_hour,
        ),
        help="Minutes to reduce the silence threshold by per hour inside the overlay window.",
    )

    parser.add_argument(
        "--boundary-silence-min-effective-n",
        type=float,
        default=cli_defaults.get(
            "boundary_silence_min_effective_n",
            BucketNowcastConfig.LateBoundarySilenceConfig.min_effective_n,
        ),
        help="Minimum effective analog sample size required for the late-boundary silence overlay.",
    )

    parser.add_argument(
        "--late-boundary-take-profit",
        action="store_true",
        default=cli_defaults.get("late_boundary_take_profit", False),
        help="Enable late-boundary majority YES take-profit behavior near settlement.",
    )

    parser.add_argument(
        "--late-boundary-trigger-price",
        type=float,
        default=cli_defaults.get(
            "late_boundary_trigger_price",
            LateBoundaryTakeProfitConfig.trigger_price,
        ),
        help="Minimum majority-exit YES VWAP to trigger late-boundary take profit. Default: from YAML config.",
    )

    parser.add_argument(
        "--late-boundary-min-sell-fraction",
        type=float,
        default=cli_defaults.get(
            "late_boundary_min_sell_fraction",
            LateBoundaryTakeProfitConfig.min_sell_fraction,
        ),
        help="Minimum fraction of YES shares to sell when late-boundary take profit triggers. Default: from YAML config.",
    )

    parser.add_argument(
        "--late-boundary-max-sell-fraction",
        type=float,
        default=cli_defaults.get(
            "late_boundary_max_sell_fraction",
            LateBoundaryTakeProfitConfig.max_sell_fraction,
        ),
        help="Maximum fraction of YES shares to sell under full late-boundary take-profit strength. Default: from YAML config.",
    )

    parser.add_argument(
        "--use-unbox-rotations",
        action="store_true",
        default=cli_defaults.get("use_unbox_rotations", False),
        help="Enable same-bin unbox rotations for boxed inventory.",
    )

    parser.add_argument(
        "--unbox-start-hours-to-settlement",
        type=float,
        default=cli_defaults.get(
            "unbox_start_hours_to_settlement",
            KellyConfig.unbox_start_hours_to_settlement,
        ),
        help="Hours-to-settlement window where unbox rotations become eligible. Default: from YAML config.",
    )

    parser.add_argument(
        "--unbox-min-blocked-ticks",
        type=int,
        default=cli_defaults.get(
            "unbox_min_blocked_ticks",
            KellyConfig.unbox_min_blocked_ticks,
        ),
        help="Minimum consecutive blocked ticks before an unbox can trigger. Default: from YAML config.",
    )

    parser.add_argument(
        "--unbox-min-net-utility",
        type=float,
        default=cli_defaults.get(
            "unbox_min_net_utility",
            KellyConfig.unbox_min_net_utility,
        ),
        help="Minimum net package utility required for an unbox. Default: from YAML config.",
    )

    parser.add_argument(
        "--unbox-late-relax-start-hours",
        type=float,
        default=cli_defaults.get(
            "unbox_late_relax_start_hours",
            KellyConfig.unbox_late_relax_start_hours_to_settlement,
        ),
        help="Hours-to-settlement window where the unbox min net utility starts relaxing. Default: from YAML config.",
    )

    parser.add_argument(
        "--unbox-late-net-utility-relax",
        type=float,
        default=cli_defaults.get(
            "unbox_late_net_utility_relax",
            KellyConfig.unbox_late_net_utility_relax,
        ),
        help="Reduction applied to the unbox min net utility inside the late-relax window. Default: from YAML config.",
    )

    parser.add_argument(
        "--unbox-repeat-net-utility-step",
        type=float,
        default=cli_defaults.get(
            "unbox_repeat_net_utility_step",
            KellyConfig.unbox_repeat_net_utility_step,
        ),
        help="Additional net utility required per prior unbox on the same bin. Default: from YAML config.",
    )

    parser.add_argument(
        "--unbox-repeat-net-utility-cap",
        type=float,
        default=cli_defaults.get(
            "unbox_repeat_net_utility_cap",
            KellyConfig.unbox_repeat_net_utility_cap,
        ),
        help="Maximum repeat-unbox utility uplift. Default: from YAML config.",
    )

    parser.add_argument(
        "--unbox-multi-bin-start-count",
        type=int,
        default=cli_defaults.get(
            "unbox_multi_bin_start_count",
            KellyConfig.unbox_multi_bin_start_count,
        ),
        help="Distinct-bin count where the multi-bin unbox utility uplift starts. Default: from YAML config.",
    )

    parser.add_argument(
        "--unbox-multi-bin-net-utility-step",
        type=float,
        default=cli_defaults.get(
            "unbox_multi_bin_net_utility_step",
            KellyConfig.unbox_multi_bin_net_utility_step,
        ),
        help="Additional net utility required per prior unbox in other bins. Default: from YAML config.",
    )

    parser.add_argument(
        "--unbox-multi-bin-net-utility-cap",
        type=float,
        default=cli_defaults.get(
            "unbox_multi_bin_net_utility_cap",
            KellyConfig.unbox_multi_bin_net_utility_cap,
        ),
        help="Maximum multi-bin utility uplift. Default: from YAML config.",
    )

    parser.add_argument(
        "--unbox-turnover-penalty",
        type=float,
        default=cli_defaults.get(
            "unbox_turnover_penalty",
            KellyConfig.unbox_turnover_penalty,
        ),
        help="Turnover penalty subtracted from gross package utility for unbox rotations. Default: from YAML config.",
    )

    parser.add_argument(
        "--unbox-bin-cooldown-seconds",
        type=int,
        default=cli_defaults.get(
            "unbox_bin_cooldown_seconds",
            KellyConfig.unbox_bin_cooldown_seconds,
        ),
        help="Cooldown after executing an unbox on the same bin. Default: from YAML config.",
    )

    parser.add_argument(
        "--consensus-time",
        action="store_true",
        help="Enable time-based trusted-quote consensus blending near settlement.",
    )

    parser.add_argument(
        "--consensus-mode",
        type=str,
        choices=["off", "time_only", "gap_only", "time_gap"],
        default=cli_defaults.get("consensus_mode", "off"),
        help="Consensus preset mode. 'time_only' enables the recommended time-based blend without gap-based damping.",
    )

    parser.add_argument(
        "--consensus-gap",
        action="store_true",
        help="Enable gap-based trusted-quote consensus blending on large model-market disagreement.",
    )

    parser.add_argument(
        "--consensus-time-tau",
        type=float,
        default=cli_defaults.get("consensus_time_tau", MarketConsensusConfig.time_tau),
        help="Time constant in hours for time-based consensus alpha. Default: from YAML config.",
    )

    parser.add_argument(
        "--consensus-gap-scale",
        type=float,
        default=cli_defaults.get("consensus_gap_scale", MarketConsensusConfig.gap_scale),
        help="Half-L1 disagreement scale for gap-based consensus alpha. Default: from YAML config.",
    )

    parser.add_argument(
        "--consensus-gap-gamma",
        type=float,
        default=cli_defaults.get("consensus_gap_gamma", MarketConsensusConfig.gap_gamma),
        help="Curvature for gap-based consensus alpha. Default: from YAML config.",
    )

    parser.add_argument(
        "--consensus-gap-floor",
        type=float,
        default=cli_defaults.get("consensus_gap_floor", MarketConsensusConfig.gap_floor),
        help="Minimum gap-based model weight before the combined floor. Default: from YAML config.",
    )

    parser.add_argument(
        "--consensus-min-model-weight",
        type=float,
        default=cli_defaults.get(
            "consensus_min_model_weight",
            MarketConsensusConfig.min_model_weight,
        ),
        help="Global minimum model weight after consensus blending. Default: from YAML config.",
    )

    parser.add_argument(
        "--consensus-min-coverage",
        type=float,
        default=cli_defaults.get(
            "consensus_min_coverage",
            MarketConsensusConfig.min_coverage_ratio,
        ),
        help="Minimum live-bin trusted-quote coverage for consensus blending. Default: from YAML config.",
    )

    parser.add_argument(
        "--consensus-max-avg-spread",
        type=float,
        default=cli_defaults.get(
            "consensus_max_avg_spread",
            MarketConsensusConfig.max_avg_spread,
        ),
        help="Maximum average YES spread across trusted bins for consensus blending. Default: from YAML config.",
    )

    parser.add_argument(
        "--consensus-max-bin-spread",
        type=float,
        default=cli_defaults.get(
            "consensus_max_bin_spread",
            MarketConsensusConfig.max_bin_spread,
        ),
        help="Maximum YES spread for a bin to count as trusted by consensus. Default: from YAML config.",
    )

    parser.add_argument(
        "--consensus-allow-untrusted-buys",
        action="store_true",
        default=cli_defaults.get("consensus_allow_untrusted_buys", False),
        help="Allow fresh BUY entries in bins without trusted quotes even when consensus mode is enabled.",
    )

    parser.add_argument(
        "--robust-kelly",
        action="store_true",
        default=cli_defaults.get("robust_kelly", False),
        help="Reduce effective Kelly fraction when the market strongly disagrees and quote quality is good.",
    )

    parser.add_argument(
        "--robust-kelly-min-fraction-multiplier",
        type=float,
        default=cli_defaults.get(
            "robust_kelly_min_fraction_multiplier",
            RobustKellyConfig.min_fraction_multiplier,
        ),
        help="Minimum multiplier on Kelly fraction under full robust-Kelly haircut. Default: from YAML config.",
    )

    parser.add_argument(
        "--robust-kelly-min-coverage",
        type=float,
        default=cli_defaults.get(
            "robust_kelly_min_coverage",
            RobustKellyConfig.min_coverage_ratio,
        ),
        help="Minimum live-bin quote coverage for robust Kelly haircuting. Default: from YAML config.",
    )

    parser.add_argument(
        "--robust-kelly-max-avg-spread",
        type=float,
        default=cli_defaults.get(
            "robust_kelly_max_avg_spread",
            RobustKellyConfig.max_avg_spread,
        ),
        help="Maximum average YES mid spread to allow robust Kelly haircuting. Default: from YAML config.",
    )

    parser.add_argument(
        "--robust-kelly-disagreement-scale",
        type=float,
        default=cli_defaults.get(
            "robust_kelly_disagreement_scale",
            RobustKellyConfig.disagreement_scale,
        ),
        help="Half-L1 model-vs-market disagreement scale for full robust-Kelly haircut. Default: from YAML config.",
    )

    parser.add_argument(
        "--market-buy-guard",
        action="store_true",
        default=cli_defaults.get("market_buy_guard", False),
        help="Widen buy-entry thresholds when the market strongly disagrees and quote quality is good.",
    )

    parser.add_argument(
        "--market-buy-guard-max-widening",
        type=float,
        default=cli_defaults.get(
            "market_buy_guard_max_widening",
            MarketBuyGuardConfig.max_threshold_widening,
        ),
        help="Maximum extra buy-threshold widening in probability points. Default: from YAML config.",
    )

    parser.add_argument(
        "--market-buy-guard-min-coverage",
        type=float,
        default=cli_defaults.get(
            "market_buy_guard_min_coverage",
            MarketBuyGuardConfig.min_coverage_ratio,
        ),
        help="Minimum live-bin quote coverage for buy guard. Default: from YAML config.",
    )

    parser.add_argument(
        "--market-buy-guard-max-avg-spread",
        type=float,
        default=cli_defaults.get(
            "market_buy_guard_max_avg_spread",
            MarketBuyGuardConfig.max_avg_spread,
        ),
        help="Maximum average YES mid spread to allow buy guard. Default: from YAML config.",
    )

    parser.add_argument(
        "--market-buy-guard-disagreement-scale",
        type=float,
        default=cli_defaults.get(
            "market_buy_guard_disagreement_scale",
            MarketBuyGuardConfig.disagreement_scale,
        ),
        help="Half-L1 model-vs-market disagreement scale for full buy guard. Default: from YAML config.",
    )

    # Other
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--list-events",
        action="store_true",
        help="List all active Musk tweet events with IDs and exit",
    )

    args = parser.parse_args(argv)
    args.dry_run = not args.live
    _resolve_consensus_mode(parser, args)
    return args


async def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    load_dotenv()
    manager: Optional[MultiEventManager] = None
    slack_notifier: Optional[SlackNotifier] = None

    # Handle --list-events: list events and exit
    if args.list_events:
        # Need a minimal CLOB client just for API discovery
        try:
            clob_client = create_clob_client()
        except Exception as e:
            logger.error(f"Failed to create CLOB client: {e}")
            sys.exit(1)

        events = await discover_musk_tweet_events(clob_client)

        if not events:
            print("\nNo active Musk tweet events found.")
            sys.exit(0)

        print(f"\n{'='*80}")
        print(f"Active Musk Tweet Events ({len(events)} found)")
        print(f"{'='*80}\n")

        for event in sorted(events, key=lambda e: e.settlement_date):
            now = datetime.now(timezone.utc)
            settlement_dt, _ = CONTRACT_UTILS.get_contract_day_bounds_utc(
                event.settlement_date
            )
            hours_remaining = (settlement_dt - now).total_seconds() / 3600

            print(f"Event ID: {event.event_id}")
            print(f"  Title:      {event.title}")
            print(f"  Short Name: {event.short_name}")
            print(f"  Dates:      {event.market_start_date} to {event.settlement_date}")
            print(f"  Hours Left: {hours_remaining:.1f}h")
            print(f"  Bins:       {len(event.bins)}")
            print()

        print(f"{'='*80}")
        print("Usage: --events <id1>,<id2>,... to trade specific events")
        print(f"{'='*80}")
        sys.exit(0)

    config_path = _resolve_project_path(args.config)
    if config_path.exists():
        logger.info(f"Loaded base config from {config_path}")
    else:
        logger.warning(
            "Base config not found at %s, falling back to code defaults",
            config_path,
        )
    _, base_kelly_config, base_forecaster_config, _ = _build_base_configs_from_yaml(
        str(config_path)
    )

    dry_run = args.dry_run
    if dry_run:
        logger.info("Running in DRY-RUN mode (no real orders)")
    else:
        logger.warning("Running in LIVE mode - real orders will be placed!")

    logger.info(f"Projection model: {args.projection}")
    logger.info(f"Intraday mode: {args.intraday_mode}")

    # Create CLOB client
    try:
        clob_client = create_clob_client()
        logger.info("CLOB client created successfully")
    except Exception as e:
        logger.error(f"Failed to create CLOB client: {e}")
        sys.exit(1)

    if args.live:
        assert_sufficient_pusd(clob_client)

    # Create configurations
    # Kelly config first (source of truth for c_event_max)
    edge_buffer_config = replace(
        base_kelly_config.edge_buffer,
        required_roi=args.required_roi,
        min_perceived_prob=args.min_prob,
        min_market_price=args.min_market_price,
        max_spread_ratio=args.max_spread_ratio,
        require_two_sided_liquidity=not args.no_require_two_sided,
    )

    collateral_config = build_collateral_config(
        max_per_event=args.max_per_event,
        capital_multiplier=args.capital_multiplier,
        base=base_kelly_config.collateral,
    )

    market_impact_config = replace(
        base_kelly_config.market_impact,
        fresh_start_enabled=not args.no_fresh_start_throttle,
        fresh_start_minutes=args.fresh_start_minutes,
        fresh_start_edge_fraction=args.fresh_start_edge_fraction,
    )

    kelly_config = replace(
        base_kelly_config,
        kappa=args.kappa,
        kelly_fraction=args.kelly_fraction,
        min_buy_utility=args.min_buy_utility,
        min_sell_utility=args.min_sell_utility,
        edge_buffer=edge_buffer_config,
        market_impact=market_impact_config,
        collateral=collateral_config,
        market_consensus=replace(
            base_kelly_config.market_consensus,
            enabled=args.consensus_time or args.consensus_gap,
            time_enabled=args.consensus_time,
            time_tau=args.consensus_time_tau,
            gap_enabled=args.consensus_gap,
            gap_scale=args.consensus_gap_scale,
            gap_gamma=args.consensus_gap_gamma,
            gap_floor=args.consensus_gap_floor,
            min_model_weight=args.consensus_min_model_weight,
            min_coverage_ratio=args.consensus_min_coverage,
            max_avg_spread=args.consensus_max_avg_spread,
            max_bin_spread=args.consensus_max_bin_spread,
            require_trusted_quote_for_buys=not args.consensus_allow_untrusted_buys,
        ),
        robust_kelly=replace(
            base_kelly_config.robust_kelly,
            enabled=args.robust_kelly,
            min_fraction_multiplier=args.robust_kelly_min_fraction_multiplier,
            min_coverage_ratio=args.robust_kelly_min_coverage,
            max_avg_spread=args.robust_kelly_max_avg_spread,
            disagreement_scale=args.robust_kelly_disagreement_scale,
        ),
        market_buy_guard=replace(
            base_kelly_config.market_buy_guard,
            enabled=args.market_buy_guard,
            max_threshold_widening=args.market_buy_guard_max_widening,
            min_coverage_ratio=args.market_buy_guard_min_coverage,
            max_avg_spread=args.market_buy_guard_max_avg_spread,
            disagreement_scale=args.market_buy_guard_disagreement_scale,
        ),
        late_boundary_take_profit=replace(
            base_kelly_config.late_boundary_take_profit,
            enabled=args.late_boundary_take_profit,
            trigger_price=args.late_boundary_trigger_price,
            min_sell_fraction=args.late_boundary_min_sell_fraction,
            max_sell_fraction=args.late_boundary_max_sell_fraction,
        ),
        use_unbox_rotations=args.use_unbox_rotations,
        unbox_start_hours_to_settlement=args.unbox_start_hours_to_settlement,
        unbox_min_blocked_ticks=args.unbox_min_blocked_ticks,
        unbox_min_net_utility=args.unbox_min_net_utility,
        unbox_late_relax_start_hours_to_settlement=args.unbox_late_relax_start_hours,
        unbox_late_net_utility_relax=args.unbox_late_net_utility_relax,
        unbox_repeat_net_utility_step=args.unbox_repeat_net_utility_step,
        unbox_repeat_net_utility_cap=args.unbox_repeat_net_utility_cap,
        unbox_multi_bin_start_count=args.unbox_multi_bin_start_count,
        unbox_multi_bin_net_utility_step=args.unbox_multi_bin_net_utility_step,
        unbox_multi_bin_net_utility_cap=args.unbox_multi_bin_net_utility_cap,
        unbox_turnover_penalty=args.unbox_turnover_penalty,
        unbox_bin_cooldown_seconds=args.unbox_bin_cooldown_seconds,
    )

    # Kelly collateral cap is the source of truth for both Kelly sizing and
    # the manager's per-event allocation limit.
    max_per_event = kelly_config.collateral.c_event_max
    logger.info(f"Max capital per event: ${max_per_event:.2f}")

    capital_pool_config = CapitalPoolConfig(
        total_capital=args.capital,
        min_allocation=args.min_allocation,
    )

    bucket_nowcast_config = replace(
        base_forecaster_config.bucket_nowcast,
        use_historical_bootstrap=args.historical_bootstrap,
        bootstrap_start_hours=args.bootstrap_start_hours,
        bootstrap_full_hours=args.bootstrap_full_hours,
        bootstrap_max_blend=args.bootstrap_max_blend,
        late_boundary_silence=replace(
            base_forecaster_config.bucket_nowcast.late_boundary_silence,
            enabled=args.boundary_silence_overlay,
            start_hours=args.boundary_silence_hours,
            max_distance_to_next_bin=args.boundary_silence_max_distance,
            silence_threshold_start_minutes=args.boundary_silence_threshold_start,
            silence_threshold_floor_minutes=args.boundary_silence_threshold_floor,
            silence_threshold_step_per_hour=args.boundary_silence_threshold_step,
            min_effective_n=args.boundary_silence_min_effective_n,
        ),
    )
    forecaster_config = replace(
        base_forecaster_config,
        intraday_mode=args.intraday_mode,
        interday_model=args.interday_model,
        bucket_nowcast=bucket_nowcast_config,
    )
    if args.historical_bootstrap:
        logger.info(
            "Historical intraday bootstrap: enabled (start=%.1fh, full=%.1fh, max_blend=%.2f)",
            args.bootstrap_start_hours,
            args.bootstrap_full_hours,
            args.bootstrap_max_blend,
        )
    if args.boundary_silence_overlay:
        logger.info(
            "Late boundary silence overlay: enabled (window=%.1fh, max_distance=%d, threshold=max(%d, %d - %.1f*(%.1f-h)), min_n_eff=%.1f)",
            args.boundary_silence_hours,
            args.boundary_silence_max_distance,
            args.boundary_silence_threshold_floor,
            args.boundary_silence_threshold_start,
            args.boundary_silence_threshold_step,
            args.boundary_silence_hours,
            args.boundary_silence_min_effective_n,
        )
    if args.late_boundary_take_profit:
        logger.info(
            "Late-boundary take-profit: enabled (trigger=%.2f, sell=%.0f%%-%.0f%%)",
            args.late_boundary_trigger_price,
            args.late_boundary_min_sell_fraction * 100.0,
            args.late_boundary_max_sell_fraction * 100.0,
        )
    if args.use_unbox_rotations:
        logger.info(
            "Unbox rotations: enabled (start=%.1fh, blocked_ticks>=%d, min_net=%.3f, late_relax_start=%.1fh, late_relax=%.3f, repeat_step=%.3f cap=%.3f, multi_bin_start=%d step=%.3f cap=%.3f, turnover=%.3f, cooldown=%ds)",
            args.unbox_start_hours_to_settlement,
            args.unbox_min_blocked_ticks,
            args.unbox_min_net_utility,
            args.unbox_late_relax_start_hours,
            args.unbox_late_net_utility_relax,
            args.unbox_repeat_net_utility_step,
            args.unbox_repeat_net_utility_cap,
            args.unbox_multi_bin_start_count,
            args.unbox_multi_bin_net_utility_step,
            args.unbox_multi_bin_net_utility_cap,
            args.unbox_turnover_penalty,
            args.unbox_bin_cooldown_seconds,
        )
    if args.consensus_time or args.consensus_gap:
        logger.info(
            "Market consensus: enabled (time=%s tau=%.1fh, gap=%s scale=%.2f gamma=%.2f floor=%.2f, min_model_weight=%.2f, min_coverage=%.2f, max_avg_spread=%.3f, max_bin_spread=%.3f, require_trusted_buys=%s)",
            args.consensus_time,
            args.consensus_time_tau,
            args.consensus_gap,
            args.consensus_gap_scale,
            args.consensus_gap_gamma,
            args.consensus_gap_floor,
            args.consensus_min_model_weight,
            args.consensus_min_coverage,
            args.consensus_max_avg_spread,
            args.consensus_max_bin_spread,
            not args.consensus_allow_untrusted_buys,
        )
    if args.robust_kelly:
        logger.info(
            "Robust Kelly haircut: enabled (min_fraction_mult=%.2f, min_coverage=%.2f, max_avg_spread=%.3f, disagreement_scale=%.2f)",
            args.robust_kelly_min_fraction_multiplier,
            args.robust_kelly_min_coverage,
            args.robust_kelly_max_avg_spread,
            args.robust_kelly_disagreement_scale,
        )
    if args.market_buy_guard:
        logger.info(
            "Market buy guard: enabled (max_widening=%.3f, min_coverage=%.2f, max_avg_spread=%.3f, disagreement_scale=%.2f)",
            args.market_buy_guard_max_widening,
            args.market_buy_guard_min_coverage,
            args.market_buy_guard_max_avg_spread,
            args.market_buy_guard_disagreement_scale,
        )

    # Load event trading rules
    event_trading_rules = None
    if args.event_rules:
        try:
            from pathlib import Path
            rules_path = Path(args.event_rules)
            if rules_path.exists():
                event_trading_rules = EventTradingRulesConfig.from_yaml(str(rules_path))
                logger.info(f"Loaded event trading rules from {args.event_rules}")
                for cat in event_trading_rules.categories:
                    logger.info(
                        f"  {cat.name}: {cat.duration_min_days}-{cat.duration_max_days}d, "
                        f"require_counting={cat.require_counting_started}"
                    )
            else:
                logger.warning(f"Event rules file not found: {args.event_rules}, using defaults")
                event_trading_rules = EventTradingRulesConfig.default()
        except Exception as e:
            logger.error(f"Failed to load event rules from {args.event_rules}: {e}")
            logger.info("Using default event trading rules")
            event_trading_rules = EventTradingRulesConfig.default()

    multi_event_config = MultiEventConfig(
        capital_pool=capital_pool_config,
        max_per_event=max_per_event,
        tick_interval_seconds=args.tick_interval,
        fast_tick_interval_seconds=args.fast_tick_interval,
        forecast_cache_seconds=args.forecast_cache_seconds,
        dry_run=dry_run,
        disable_websocket=args.no_ws,
        event_trading_rules=event_trading_rules,
        projection_model=args.projection,
        min_event_duration_days=args.min_event_days,
        max_event_duration_days=args.max_event_days,
        realtime_tracker_enabled=not args.disable_realtime_tracker,
        **({"realtime_poll_interval_seconds": args.realtime_poll_interval} if args.realtime_poll_interval is not None else {}),
        **({"realtime_fetch_count": args.realtime_fetch_count} if args.realtime_fetch_count is not None else {}),
        **({"realtime_late_tweet_grace_seconds": args.realtime_late_tweet_grace} if args.realtime_late_tweet_grace is not None else {}),
        realtime_cookies_path=args.realtime_cookies_path,
    )
    logger.info(f"Event duration filter: {args.min_event_days}-{args.max_event_days} days (inclusive)")
    if multi_event_config.realtime_tracker_enabled:
        logger.info(
            "Realtime twikit tracker enabled: poll_interval=%.1fs fetch_count=%d late_grace=%.1fs cookies=%s",
            multi_event_config.realtime_poll_interval_seconds,
            multi_event_config.realtime_fetch_count,
            multi_event_config.realtime_late_tweet_grace_seconds,
            multi_event_config.realtime_cookies_path or "config/twitter_cookies.json",
        )
    else:
        logger.info("Realtime twikit tracker disabled")

    # Get wallet address for position fetching
    # IMPORTANT: Use proxy wallet if configured (positions are held there)
    try:
        wallet_address = get_wallet_address_for_positions()
        logger.info(f"Using wallet address for positions: {wallet_address}")
    except ValueError as e:
        logger.error(f"{e}")
        sys.exit(1)

    # Determine events to trade
    initial_events: List[EventInfo] = []
    discovery_callback = None

    if args.events == "auto":
        # Auto mode: use discovery callback
        async def discovery_callback() -> List[EventInfo]:
            return await discover_musk_tweet_events(clob_client)
        logger.info("Using auto event discovery mode")
    else:
        # Manual mode: fetch specific event IDs upfront
        event_ids = [e.strip() for e in args.events.split(",")]
        logger.info(f"Loading {len(event_ids)} specific events: {event_ids}")
        for event_id in event_ids:
            try:
                event_info = await fetch_event_details(clob_client, event_id)
                initial_events.append(event_info)
                logger.info(f"Loaded event: {event_info.short_name}")
            except Exception as e:
                logger.error(f"Failed to fetch event {event_id}: {e}")

        if not initial_events:
            logger.error("No events could be loaded. Exiting.")
            sys.exit(1)

    # Create manager with initial events or discovery callback
    try:
        slack_notifier = SlackNotifier.from_env(dry_run=dry_run)
        await slack_notifier.start()

        manager = MultiEventManager(
            clob_client=clob_client,
            kelly_config=kelly_config,
            forecaster_config=forecaster_config,
            config=multi_event_config,
            wallet_address=wallet_address,
            event_discovery_callback=discovery_callback,
            initial_events=initial_events if initial_events else None,
            slack_notifier=slack_notifier,
        )

        # Reconstruct state from Polymarket API (positions, balances)
        # This recovers state after a crash/restart
        logger.info("Reconstructing state from Polymarket API...")
        await manager.reconstruct_state_from_api()

        # Log full config summary
        log_config_summary(
            kelly_config=kelly_config,
            multi_event_config=multi_event_config,
            forecaster_config=forecaster_config,
            capital_pool_config=capital_pool_config,
            dry_run=dry_run,
        )

        # Run manager
        async with manager._events_lock:
            num_events = len(manager._pending_events) + len(manager._active_events)
        logger.info(f"Starting multi-event manager with {num_events} events")

        # Setup signal handlers for graceful shutdown
        import signal
        loop = asyncio.get_running_loop()
        shutdown_event = asyncio.Event()

        def signal_handler():
            logger.info("Received shutdown signal (Ctrl+C)")
            manager.stop()
            shutdown_event.set()

        # Register signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

        try:
            await manager.run()
        except asyncio.CancelledError:
            logger.info("Manager cancelled")
        except Exception as e:
            logger.error(f"Manager error: {e}", exc_info=True)
            if slack_notifier:
                slack_notifier.notify_error(
                    "Manager error",
                    [str(e)],
                    dedupe_key="manager_run_error",
                    cooldown_seconds=300.0,
                )
                await slack_notifier.flush()
        finally:
            # Remove signal handlers
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(sig)

            # Ensure manager is stopped
            manager.stop()

            # Give tasks time to clean up
            await asyncio.sleep(0.5)

            # Log final status
            try:
                status = manager.get_status()
                logger.info(f"Final status: {status}")

                performance = manager.capital_pool.get_performance_summary()
                logger.info(
                    f"Performance: {performance['num_events']} events, "
                    f"total P&L: ${performance['total_pnl']:.2f}, "
                    f"win rate: {performance['win_rate']:.1%}"
                )

                if slack_notifier:
                    health = manager.get_health()
                    slack_notifier.notify_info(
                        "Multi-event manager stopped",
                        [
                            f"status={health['status']} uptime={health['uptime_hours']:.1f}h",
                            f"events={performance['num_events']} pnl=${performance['total_pnl']:.2f} win_rate={performance['win_rate']:.1%}",
                            f"capital_available=${health['available_capital']:.2f} errors={health['errors_count']}",
                        ],
                    )
                    await slack_notifier.flush()
            except Exception as e:
                logger.debug(f"Error logging final status: {e}")

            logger.info("Shutdown complete")
    finally:
        if slack_notifier:
            await slack_notifier.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Already handled by signal handler
