import pytest
from datetime import date
from pathlib import Path

from src.algo.musk_tweet_count.forecaster.run_multi_event import (
    build_collateral_config,
    parse_args,
    parse_counting_dates_from_title,
)
from src.algo.musk_tweet_count.kelly.config import CollateralConfig


def test_build_collateral_config_uses_cli_event_cap():
    collateral = build_collateral_config(
        max_per_event=300.0,
        capital_multiplier=1.5,
    )

    assert collateral.c_event_max == 300.0
    assert collateral.capital_multiplier == 1.5
    assert collateral.c_bin_max == 300.0 * 1.5 * collateral.c_bin_max_ratio


def test_build_collateral_config_defaults_to_dataclass_cap():
    collateral = build_collateral_config(
        max_per_event=None,
        capital_multiplier=CollateralConfig.capital_multiplier,
    )

    assert collateral.c_event_max == CollateralConfig.c_event_max


def test_build_collateral_config_uses_yaml_backed_base():
    base = CollateralConfig(c_event_max=180.0, c_bin_max_ratio=0.25, capital_multiplier=1.0)

    collateral = build_collateral_config(
        max_per_event=None,
        capital_multiplier=1.4,
        base=base,
    )

    assert collateral.c_event_max == 180.0
    assert collateral.c_bin_max_ratio == 0.25
    assert collateral.capital_multiplier == 1.4


def test_parse_args_uses_yaml_defaults_for_runner(tmp_path: Path):
    config_path = tmp_path / "runner.yaml"
    config_path.write_text(
        """
trading:
  dry_run: false
  slow_loop_interval_seconds: 120
  fast_loop_interval_seconds: 12
  forecast_cache_seconds: 130
forecaster:
  intraday_mode: bucket
kelly:
  edge_buffer:
    required_roi: 0.07
  collateral:
    c_event_max: 222.0
  websocket:
    enabled: false
  use_unbox_rotations: true
  unbox_start_hours_to_settlement: 9.0
""".strip(),
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config_path)])

    assert args.live is True
    assert args.dry_run is False
    assert args.tick_interval == 120
    assert args.fast_tick_interval == 12
    assert args.forecast_cache_seconds == 130
    assert args.max_per_event == 222.0
    assert args.required_roi == 0.07
    assert args.no_ws is True
    assert args.use_unbox_rotations is True
    assert args.unbox_start_hours_to_settlement == 9.0
    assert args.intraday_mode == "bucket"


def test_parse_args_cli_overrides_yaml_defaults(tmp_path: Path):
    config_path = tmp_path / "runner.yaml"
    config_path.write_text(
        """
trading:
  dry_run: false
  slow_loop_interval_seconds: 120
kelly:
  collateral:
    c_event_max: 222.0
  websocket:
    enabled: false
""".strip(),
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--config", str(config_path),
            "--dry-run",
            "--tick-interval", "90",
            "--max-per-event", "333",
            "--ws",
        ]
    )

    assert args.live is False
    assert args.dry_run is True
    assert args.tick_interval == 90
    assert args.max_per_event == 333.0
    assert args.no_ws is False


def test_parse_args_accepts_c_event_max_alias():
    args = parse_args(["--c-event-max", "275", "--dry-run"])

    assert args.max_per_event == 275.0
    assert args.dry_run is True


def test_parse_args_accepts_max_per_event_flag():
    args = parse_args(["--max-per-event", "325"])

    assert args.max_per_event == 325.0


def test_parse_args_accepts_no_ws_flag():
    args = parse_args(["--no-ws"])

    assert args.no_ws is True


def test_parse_args_accepts_consensus_flags():
    args = parse_args(
        [
            "--consensus-time",
            "--consensus-gap",
            "--consensus-time-tau", "8",
            "--consensus-gap-scale", "0.35",
            "--consensus-gap-gamma", "2.0",
            "--consensus-gap-floor", "0.8",
            "--consensus-min-model-weight", "0.3",
            "--consensus-min-coverage", "0.9",
            "--consensus-max-avg-spread", "0.04",
            "--consensus-max-bin-spread", "0.08",
            "--consensus-allow-untrusted-buys",
        ]
    )

    assert args.consensus_time is True
    assert args.consensus_gap is True
    assert args.consensus_time_tau == 8.0
    assert args.consensus_gap_scale == 0.35
    assert args.consensus_gap_gamma == 2.0
    assert args.consensus_gap_floor == 0.8
    assert args.consensus_min_model_weight == 0.3
    assert args.consensus_min_coverage == 0.9
    assert args.consensus_max_avg_spread == 0.04
    assert args.consensus_max_bin_spread == 0.08
    assert args.consensus_allow_untrusted_buys is True


def test_parse_args_accepts_consensus_time_only_mode():
    args = parse_args(["--consensus-mode", "time_only"])

    assert args.consensus_mode == "time_only"
    assert args.consensus_time is True
    assert args.consensus_gap is False
    assert args.consensus_time_tau == 12.0
    assert args.consensus_min_model_weight == 0.3


def test_parse_args_rejects_conflicting_consensus_mode_and_flag():
    with pytest.raises(SystemExit):
        parse_args(["--consensus-mode", "time_only", "--consensus-gap"])


def test_parse_args_rejects_removed_market_aware_flag():
    with pytest.raises(SystemExit):
        parse_args(["--market-aware"])


def test_parse_args_accepts_boundary_silence_overlay_flags():
    args = parse_args(
        [
            "--boundary-silence-overlay",
            "--boundary-silence-hours", "5.5",
            "--boundary-silence-max-distance", "4",
            "--boundary-silence-threshold-start", "170",
            "--boundary-silence-threshold-floor", "80",
            "--boundary-silence-threshold-step", "25",
            "--boundary-silence-min-effective-n", "7",
        ]
    )

    assert args.boundary_silence_overlay is True
    assert args.boundary_silence_hours == 5.5
    assert args.boundary_silence_max_distance == 4
    assert args.boundary_silence_threshold_start == 170
    assert args.boundary_silence_threshold_floor == 80
    assert args.boundary_silence_threshold_step == 25.0
    assert args.boundary_silence_min_effective_n == 7.0


def test_parse_args_accepts_late_boundary_take_profit_flags():
    args = parse_args(
        [
            "--late-boundary-take-profit",
            "--late-boundary-trigger-price", "0.8",
            "--late-boundary-min-sell-fraction", "0.7",
            "--late-boundary-max-sell-fraction", "0.9",
        ]
    )

    assert args.late_boundary_take_profit is True
    assert args.late_boundary_trigger_price == 0.8
    assert args.late_boundary_min_sell_fraction == 0.7
    assert args.late_boundary_max_sell_fraction == 0.9


def test_parse_args_accepts_unbox_rotation_flags():
    args = parse_args(
        [
            "--use-unbox-rotations",
            "--unbox-start-hours-to-settlement", "18",
            "--unbox-min-blocked-ticks", "1",
            "--unbox-min-net-utility", "0.006",
            "--unbox-late-relax-start-hours", "3",
            "--unbox-late-net-utility-relax", "0.002",
            "--unbox-repeat-net-utility-step", "0.003",
            "--unbox-repeat-net-utility-cap", "0.006",
            "--unbox-multi-bin-start-count", "2",
            "--unbox-multi-bin-net-utility-step", "0.002",
            "--unbox-multi-bin-net-utility-cap", "0.004",
            "--unbox-turnover-penalty", "0.002",
            "--unbox-bin-cooldown-seconds", "1800",
        ]
    )

    assert args.use_unbox_rotations is True
    assert args.unbox_start_hours_to_settlement == 18.0
    assert args.unbox_min_blocked_ticks == 1
    assert args.unbox_min_net_utility == 0.006
    assert args.unbox_late_relax_start_hours == 3.0
    assert args.unbox_late_net_utility_relax == 0.002
    assert args.unbox_repeat_net_utility_step == 0.003
    assert args.unbox_repeat_net_utility_cap == 0.006
    assert args.unbox_multi_bin_start_count == 2
    assert args.unbox_multi_bin_net_utility_step == 0.002
    assert args.unbox_multi_bin_net_utility_cap == 0.004
    assert args.unbox_turnover_penalty == 0.002
    assert args.unbox_bin_cooldown_seconds == 1800


def test_parse_args_uses_updated_unbox_start_default():
    args = parse_args([])

    assert args.unbox_start_hours_to_settlement == 12.0


def test_parse_counting_dates_same_year():
    start, end = parse_counting_dates_from_title(
        "Elon Musk # of tweets June 2 - June 9, 2026?"
    )

    assert start == date(2026, 6, 2)
    assert end == date(2026, 6, 9)


def test_parse_counting_dates_cross_year_single_trailing_year():
    start, end = parse_counting_dates_from_title(
        "Elon Musk # of tweets December 29 - January 5, 2027?"
    )

    assert start == date(2026, 12, 29)
    assert end == date(2027, 1, 5)


def test_parse_counting_dates_cross_year_dual_years():
    start, end = parse_counting_dates_from_title(
        "Elon Musk # of tweets December 29, 2026 - January 5, 2027?"
    )

    assert start == date(2026, 12, 29)
    assert end == date(2027, 1, 5)


def test_parse_counting_dates_dual_years_same_year():
    start, end = parse_counting_dates_from_title(
        "Elon Musk # of tweets June 2, 2026 - June 9, 2026?"
    )

    assert start == date(2026, 6, 2)
    assert end == date(2026, 6, 9)


def test_parse_counting_dates_unparseable_returns_none():
    start, end = parse_counting_dates_from_title("Elon Musk # of tweets in July?")

    assert start is None
    assert end is None
