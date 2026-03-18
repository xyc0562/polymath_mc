import pytest

from src.algo.musk_tweet_count.forecaster.run_multi_event import (
    build_collateral_config,
    parse_args,
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
