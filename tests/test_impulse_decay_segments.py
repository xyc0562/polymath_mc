from datetime import date, datetime, timedelta, timezone
import math

import numpy as np
import pytest

from src.algo.musk_tweet_count.forecaster.config import BucketNowcastConfig
from src.algo.musk_tweet_count.forecaster.data import ContractDayUtils, TweetEvent
from src.algo.musk_tweet_count.forecaster.intraday import (
    BucketDistribution,
    BucketIntradayForecaster,
    ImpulseOverrideWindow,
    _build_decay_schedule,
    _load_impulse_overrides,
    _piecewise_decay_factor_exact,
    _piecewise_decay_factor_tau,
    _validate_impulse_overrides,
)


def _default_overrides():
    return _validate_impulse_overrides(
        [
            ImpulseOverrideWindow(
                name="overnight",
                start=720,
                end=1080,
                impulse_decay_halflife=15,
            ),
            ImpulseOverrideWindow(
                name="near_end",
                start=1200,
                end=1440,
                floor=1.0,
            ),
        ]
    )


def _make_forecaster(config: BucketNowcastConfig | None = None):
    contract_utils = ContractDayUtils()
    config = config or BucketNowcastConfig()
    forecaster = BucketIntradayForecaster(config, contract_utils)
    forecaster._fitted = True
    forecaster._impulse_fitted = True
    forecaster._rate_curve = np.full(1440, 0.05)
    forecaster._weekday_buckets = [
        BucketDistribution(
            bucket_idx=i,
            start_tau=i * forecaster.bucket_size,
            end_tau=(i + 1) * forecaster.bucket_size,
            mean=1.0,
            std=1.0,
            dispersion_k=1.0,
        )
        for i in range(forecaster.n_buckets)
    ]
    forecaster._weekend_buckets = [
        BucketDistribution(
            bucket_idx=b.bucket_idx,
            start_tau=b.start_tau,
            end_tau=b.end_tau,
            mean=b.mean,
            std=b.std,
            dispersion_k=b.dispersion_k,
        )
        for b in forecaster._weekday_buckets
    ]
    forecaster._impulse_overrides = _default_overrides()
    return forecaster, contract_utils


def _dt_for_tau(
    contract_utils: ContractDayUtils,
    contract_date: date,
    tau: int,
    *,
    seconds: int = 0,
) -> datetime:
    start_dt, _ = contract_utils.get_contract_day_bounds(contract_date)
    return start_dt + timedelta(minutes=tau, seconds=seconds)


def test_build_decay_schedule_no_overrides_returns_single_segment():
    config = BucketNowcastConfig()
    schedule = _build_decay_schedule(
        710,
        730,
        [],
        config.impulse_decay_halflife_minutes,
        config.impulse_floor,
        config.impulse_ceiling,
    )

    assert [(segment.start_tau, segment.end_tau) for segment in schedule] == [(710, 730)]
    assert schedule[0].halflife == pytest.approx(config.impulse_decay_halflife_minutes)
    assert schedule[0].floor == pytest.approx(config.impulse_floor)
    assert schedule[0].ceiling == pytest.approx(config.impulse_ceiling)


def test_build_decay_schedule_splits_on_override_boundaries():
    config = BucketNowcastConfig()
    overrides = _default_overrides()

    overnight = _build_decay_schedule(
        710,
        730,
        overrides,
        config.impulse_decay_halflife_minutes,
        config.impulse_floor,
        config.impulse_ceiling,
    )
    assert [(segment.start_tau, segment.end_tau) for segment in overnight] == [
        (710, 720),
        (720, 730),
    ]
    assert [segment.halflife for segment in overnight] == [30.0, 15.0]

    overnight_exit = _build_decay_schedule(
        1070,
        1090,
        overrides,
        config.impulse_decay_halflife_minutes,
        config.impulse_floor,
        config.impulse_ceiling,
    )
    assert [(segment.start_tau, segment.end_tau) for segment in overnight_exit] == [
        (1070, 1080),
        (1080, 1090),
    ]
    assert [segment.halflife for segment in overnight_exit] == [15.0, 30.0]

    gap = _build_decay_schedule(
        1070,
        1210,
        overrides,
        config.impulse_decay_halflife_minutes,
        config.impulse_floor,
        config.impulse_ceiling,
    )
    assert [(segment.start_tau, segment.end_tau) for segment in gap] == [
        (1070, 1080),
        (1080, 1200),
        (1200, 1210),
    ]
    assert [segment.halflife for segment in gap] == [15.0, 30.0, 30.0]
    assert gap[-1].floor == pytest.approx(1.0)
    assert gap[-1].ceiling == pytest.approx(config.impulse_ceiling)


def test_build_decay_schedule_within_single_override_uses_override_params():
    config = BucketNowcastConfig()
    schedule = _build_decay_schedule(
        730,
        740,
        _default_overrides(),
        config.impulse_decay_halflife_minutes,
        config.impulse_floor,
        config.impulse_ceiling,
    )

    assert [(segment.start_tau, segment.end_tau) for segment in schedule] == [(730, 740)]
    assert schedule[0].halflife == pytest.approx(15.0)
    assert schedule[0].floor == pytest.approx(config.impulse_floor)
    assert schedule[0].ceiling == pytest.approx(config.impulse_ceiling)


def test_build_decay_schedule_full_day_matches_current_override_layout():
    config = BucketNowcastConfig()
    schedule = _build_decay_schedule(
        0,
        1440,
        _default_overrides(),
        config.impulse_decay_halflife_minutes,
        config.impulse_floor,
        config.impulse_ceiling,
    )

    assert [(segment.start_tau, segment.end_tau) for segment in schedule] == [
        (0, 720),
        (720, 1080),
        (1080, 1200),
        (1200, 1440),
    ]
    assert sum(segment.end_tau - segment.start_tau for segment in schedule) == 1440


def test_build_decay_schedule_reversed_range_returns_empty():
    config = BucketNowcastConfig()
    schedule = _build_decay_schedule(
        800,
        790,
        _default_overrides(),
        config.impulse_decay_halflife_minutes,
        config.impulse_floor,
        config.impulse_ceiling,
    )

    assert schedule == []


@pytest.mark.parametrize(
    "overrides, match",
    [
        (
            [
                ImpulseOverrideWindow(name="a", start=100, end=200),
                ImpulseOverrideWindow(name="b", start=150, end=250),
            ],
            "overlaps",
        ),
        (
            [ImpulseOverrideWindow(name="bad", start=200, end=200)],
            "start < end",
        ),
        (
            [ImpulseOverrideWindow(name="bad", start=-1, end=10)],
            "within \\[0, 1440\\]",
        ),
    ],
)
def test_validate_impulse_overrides_rejects_invalid_ranges(overrides, match):
    with pytest.raises(ValueError, match=match):
        _validate_impulse_overrides(overrides)


def test_piecewise_decay_factor_tau_matches_single_segment_formula():
    config = BucketNowcastConfig()
    schedule = _build_decay_schedule(
        700,
        730,
        [],
        config.impulse_decay_halflife_minutes,
        config.impulse_floor,
        config.impulse_ceiling,
    )
    decay = math.log(2) / config.impulse_decay_halflife_minutes

    assert _piecewise_decay_factor_tau(730, 730, schedule) == pytest.approx(1.0)
    assert _piecewise_decay_factor_tau(700, 730, schedule) == pytest.approx(math.exp(-decay * 30))


def test_piecewise_decay_factor_tau_matches_boundary_crossing_math():
    config = BucketNowcastConfig()
    schedule = _build_decay_schedule(
        710,
        730,
        _default_overrides(),
        config.impulse_decay_halflife_minutes,
        config.impulse_floor,
        config.impulse_ceiling,
    )
    expected = math.exp(-(math.log(2) / 30.0) * 10) * math.exp(-(math.log(2) / 15.0) * 10)

    assert _piecewise_decay_factor_tau(710, 730, schedule) == pytest.approx(expected)


def test_piecewise_decay_factor_exact_matches_single_segment_elapsed_minutes():
    config = BucketNowcastConfig()
    contract_utils = ContractDayUtils()
    contract_date = date(2026, 1, 10)
    start = _dt_for_tau(contract_utils, contract_date, 700)
    end = _dt_for_tau(contract_utils, contract_date, 720)
    schedule = _build_decay_schedule(
        700,
        720,
        [],
        config.impulse_decay_halflife_minutes,
        config.impulse_floor,
        config.impulse_ceiling,
    )
    decay = math.log(2) / config.impulse_decay_halflife_minutes

    assert _piecewise_decay_factor_exact(start, start, contract_date, schedule, contract_utils) == pytest.approx(1.0)
    assert _piecewise_decay_factor_exact(start, end, contract_date, schedule, contract_utils) == pytest.approx(
        math.exp(-decay * 20)
    )


def test_piecewise_decay_factor_exact_matches_crossing_boundary_math():
    config = BucketNowcastConfig()
    contract_utils = ContractDayUtils()
    contract_date = date(2026, 1, 10)
    start = _dt_for_tau(contract_utils, contract_date, 710)
    end = _dt_for_tau(contract_utils, contract_date, 730)
    schedule = _build_decay_schedule(
        710,
        730,
        _default_overrides(),
        config.impulse_decay_halflife_minutes,
        config.impulse_floor,
        config.impulse_ceiling,
    )
    expected = math.exp(-(math.log(2) / 30.0) * 10) * math.exp(-(math.log(2) / 15.0) * 10)

    assert _piecewise_decay_factor_exact(start, end, contract_date, schedule, contract_utils) == pytest.approx(expected)


def test_piecewise_decay_factor_exact_keeps_subminute_tweet_contribution():
    config = BucketNowcastConfig()
    contract_utils = ContractDayUtils()
    contract_date = date(2026, 1, 10)
    now = _dt_for_tau(contract_utils, contract_date, 730, seconds=30)
    start = now - timedelta(seconds=20)
    schedule = _build_decay_schedule(
        0,
        731,
        [],
        config.impulse_decay_halflife_minutes,
        config.impulse_floor,
        config.impulse_ceiling,
    )
    decay = math.log(2) / config.impulse_decay_halflife_minutes
    expected = math.exp(-decay * (20 / 60))

    factor = _piecewise_decay_factor_exact(start, now, contract_date, schedule, contract_utils)
    assert 0.0 < factor < 1.0
    assert factor == pytest.approx(expected)


def test_piecewise_decay_factor_exact_not_truncated_by_expected_lookback():
    config = BucketNowcastConfig()
    contract_utils = ContractDayUtils()
    contract_date = date(2026, 1, 10)
    start = _dt_for_tau(contract_utils, contract_date, 100)
    end = _dt_for_tau(contract_utils, contract_date, 730)
    schedule = _build_decay_schedule(
        0,
        730,
        [],
        config.impulse_decay_halflife_minutes,
        config.impulse_floor,
        config.impulse_ceiling,
    )

    assert _piecewise_decay_factor_exact(start, end, contract_date, schedule, contract_utils) > 0.0


def test_piecewise_decay_factor_exact_uses_real_elapsed_minutes_across_dst():
    config = BucketNowcastConfig()
    contract_utils = ContractDayUtils()
    contract_date = date(2026, 3, 7)
    start = datetime(2026, 3, 8, 6, 11, 5, tzinfo=timezone.utc)
    end = datetime(2026, 3, 8, 7, 1, 5, tzinfo=timezone.utc)
    schedule = _build_decay_schedule(
        0,
        1440,
        [],
        config.impulse_decay_halflife_minutes,
        config.impulse_floor,
        config.impulse_ceiling,
    )
    decay = math.log(2) / config.impulse_decay_halflife_minutes

    assert _piecewise_decay_factor_exact(start, end, contract_date, schedule, contract_utils) == pytest.approx(
        math.exp(-decay * 50.0)
    )


def test_avg_rate_mult_for_slice_piecewise_matches_single_segment_reference():
    forecaster, _ = _make_forecaster()
    config = forecaster.config

    boost_schedule = _build_decay_schedule(
        100,
        120,
        [],
        config.impulse_decay_halflife_minutes,
        config.impulse_floor,
        config.impulse_ceiling,
    )
    boost_piecewise = forecaster._avg_rate_mult_for_slice_piecewise(
        rate_mult_now=2.0,
        tau_now=100,
        abs_start=100,
        abs_end=120,
        forward_schedule=boost_schedule,
        silence_halflife=config.impulse_silence_halflife_minutes,
    )
    boost_reference = forecaster._avg_rate_mult_for_slice(
        rate_mult=2.0,
        forward_decay=math.log(2) / config.impulse_decay_halflife_minutes,
        rel_start=0.0,
        rel_end=20.0,
    )
    assert boost_piecewise == pytest.approx(boost_reference)

    silence_piecewise = forecaster._avg_rate_mult_for_slice_piecewise(
        rate_mult_now=0.5,
        tau_now=100,
        abs_start=100,
        abs_end=120,
        forward_schedule=boost_schedule,
        silence_halflife=config.impulse_silence_halflife_minutes,
    )
    silence_reference = forecaster._avg_rate_mult_for_slice(
        rate_mult=0.5,
        forward_decay=math.log(2) / config.impulse_silence_halflife_minutes,
        rel_start=0.0,
        rel_end=20.0,
    )
    assert silence_piecewise == pytest.approx(silence_reference)


def test_avg_rate_mult_for_slice_piecewise_boost_decays_faster_when_crossing_overnight():
    forecaster, _ = _make_forecaster()
    config = forecaster.config
    schedule = _build_decay_schedule(
        710,
        730,
        _default_overrides(),
        config.impulse_decay_halflife_minutes,
        config.impulse_floor,
        config.impulse_ceiling,
    )

    piecewise = forecaster._avg_rate_mult_for_slice_piecewise(
        rate_mult_now=2.0,
        tau_now=710,
        abs_start=710,
        abs_end=730,
        forward_schedule=schedule,
        silence_halflife=config.impulse_silence_halflife_minutes,
    )
    reference = forecaster._avg_rate_mult_for_slice(
        rate_mult=2.0,
        forward_decay=math.log(2) / config.impulse_decay_halflife_minutes,
        rel_start=0.0,
        rel_end=20.0,
    )

    assert piecewise < reference


def test_avg_rate_mult_for_slice_piecewise_silence_clamps_to_one_in_near_end():
    forecaster, _ = _make_forecaster()
    config = forecaster.config
    schedule = _build_decay_schedule(
        1190,
        1210,
        _default_overrides(),
        config.impulse_decay_halflife_minutes,
        config.impulse_floor,
        config.impulse_ceiling,
    )

    avg_mult = forecaster._avg_rate_mult_for_slice_piecewise(
        rate_mult_now=0.5,
        tau_now=1190,
        abs_start=1200,
        abs_end=1210,
        forward_schedule=schedule,
        silence_halflife=config.impulse_silence_halflife_minutes,
    )

    assert avg_mult == pytest.approx(1.0)


def test_sample_with_impulse_matches_old_behavior_without_overrides():
    forecaster, contract_utils = _make_forecaster()
    forecaster._impulse_overrides = []
    config = forecaster.config
    contract_date = date(2026, 1, 10)
    now = _dt_for_tau(contract_utils, contract_date, 730)
    events = [
        TweetEvent(timestamp=_dt_for_tau(contract_utils, contract_date, 700)),
        TweetEvent(timestamp=_dt_for_tau(contract_utils, contract_date, 720)),
    ]

    forecaster._sample_with_impulse(
        events=events,
        contract_date=contract_date,
        now=now,
        n_simulations=4,
        settlement_tau=790,
        rng=np.random.default_rng(1),
    )

    decay = math.log(2) / config.impulse_decay_halflife_minutes
    excitation = math.exp(-decay * 30) + math.exp(-decay * 10)
    expected_excitation = sum(0.05 * math.exp(-decay * t) for t in range(1, config.impulse_lookback_minutes + 1))
    shifted = excitation - expected_excitation * config.impulse_neutral_fraction
    rate_mult = max(
        config.impulse_floor,
        min(config.impulse_ceiling, 1.0 + config.impulse_gain * shifted),
    )

    assert forecaster._last_impulse is not None
    assert forecaster._last_impulse["override"] is None
    assert forecaster._last_impulse["excitation"] == pytest.approx(round(excitation, 2))
    assert forecaster._last_impulse["expected"] == pytest.approx(round(expected_excitation, 2))
    assert forecaster._last_impulse["shifted"] == pytest.approx(round(shifted, 2))
    assert forecaster._last_impulse["rate_mult_now"] == pytest.approx(round(rate_mult, 2))
    assert forecaster._last_impulse["n_actual_decay_segments"] == 1
    assert forecaster._last_impulse["n_expected_decay_segments"] == 1
    assert forecaster._last_impulse["n_forward_decay_segments"] == 1


def test_sample_with_impulse_reports_multi_segment_decay_counts():
    forecaster, contract_utils = _make_forecaster()
    contract_date = date(2026, 1, 10)
    now = _dt_for_tau(contract_utils, contract_date, 730)
    events = [
        TweetEvent(timestamp=_dt_for_tau(contract_utils, contract_date, 710)),
        TweetEvent(timestamp=_dt_for_tau(contract_utils, contract_date, 725)),
    ]

    forecaster._sample_with_impulse(
        events=events,
        contract_date=contract_date,
        now=now,
        n_simulations=4,
        settlement_tau=1250,
        rng=np.random.default_rng(2),
    )

    assert forecaster._last_impulse is not None
    assert forecaster._last_impulse["override"] == "overnight"
    assert forecaster._last_impulse["n_actual_decay_segments"] > 1
    assert forecaster._last_impulse["n_expected_decay_segments"] > 1
    assert forecaster._last_impulse["n_forward_decay_segments"] > 1


def test_fit_raises_immediately_on_invalid_override_config(tmp_path, monkeypatch):
    path = tmp_path / "impulse_overrides.yaml"
    path.write_text(
        "overrides:\n"
        "  first:\n"
        "    start: 100\n"
        "    end: 200\n"
        "  second:\n"
        "    start: 150\n"
        "    end: 250\n"
    )

    contract_utils = ContractDayUtils()
    config = BucketNowcastConfig(impulse_overrides_path=str(path))
    forecaster = BucketIntradayForecaster(config, contract_utils)

    dummy_buckets = [
        BucketDistribution(
            bucket_idx=i,
            start_tau=i * forecaster.bucket_size,
            end_tau=(i + 1) * forecaster.bucket_size,
            mean=1.0,
            std=1.0,
            dispersion_k=1.0,
        )
        for i in range(forecaster.n_buckets)
    ]

    monkeypatch.setattr(forecaster, "_fit_bucket_distributions", lambda bucket_counts: dummy_buckets)
    monkeypatch.setattr(
        forecaster,
        "_build_historical_suffix_profiles",
        lambda historical_events, historical_counts, as_of_date: [],
    )
    monkeypatch.setattr(forecaster, "_fit_impulse", lambda historical_events, as_of_date: None)

    historical_events = {date(2026, 1, 5): []}
    historical_counts = {date(2026, 1, 5): 0}

    with pytest.raises(ValueError, match="overlaps"):
        forecaster.fit(historical_events, historical_counts, as_of_date=date(2026, 1, 6))


def test_load_impulse_overrides_rejects_bad_yaml_shape(tmp_path):
    path = tmp_path / "impulse_overrides.yaml"
    path.write_text("overrides:\n  - not-a-mapping\n")

    with pytest.raises(ValueError, match="must be a mapping"):
        _load_impulse_overrides(str(path))
