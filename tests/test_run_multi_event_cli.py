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
