from datetime import date, timedelta

import numpy as np
import pytest

from src.algo.musk_tweet_count.forecaster.config import BucketNowcastConfig
from src.algo.musk_tweet_count.forecaster.data import ContractDayUtils, TweetEvent
from src.algo.musk_tweet_count.forecaster.intraday import (
    BucketDistribution,
    BucketIntradayForecaster,
    HistoricalSuffixProfile,
)


def _make_forecaster() -> tuple[BucketIntradayForecaster, ContractDayUtils]:
    contract_utils = ContractDayUtils()
    config = BucketNowcastConfig(
        late_boundary_silence=BucketNowcastConfig.LateBoundarySilenceConfig(
            enabled=True,
            min_effective_n=1.0,
        )
    )
    forecaster = BucketIntradayForecaster(config, contract_utils)
    forecaster._fitted = True
    forecaster._weekday_buckets = [
        BucketDistribution(
            bucket_idx=i,
            start_tau=i * forecaster.bucket_size,
            end_tau=(i + 1) * forecaster.bucket_size,
            mean=10.0,
            std=2.0,
            dispersion_k=1.0,
        )
        for i in range(forecaster.n_buckets)
    ]
    forecaster._weekend_buckets = list(forecaster._weekday_buckets)
    return forecaster, contract_utils


def _dt_for_tau(contract_utils: ContractDayUtils, contract_date: date, tau: int):
    start_dt, _ = contract_utils.get_contract_day_bounds(contract_date)
    return start_dt + timedelta(minutes=tau)


def _events_for_taus(
    contract_utils: ContractDayUtils,
    contract_date: date,
    taus: list[int],
) -> list[TweetEvent]:
    return [
        TweetEvent(timestamp=_dt_for_tau(contract_utils, contract_date, tau))
        for tau in taus
    ]


def _set_profiles(
    forecaster: BucketIntradayForecaster,
    contract_date: date,
    remaining_values: list[int],
    prefix_taus: list[int],
) -> None:
    profiles = []
    for idx, remaining in enumerate(remaining_values):
        future_taus = [1210 + i for i in range(remaining)]
        profiles.append(
            HistoricalSuffixProfile(
                contract_date=contract_date - timedelta(days=idx + 1),
                is_weekend=False,
                taus=np.asarray(prefix_taus + future_taus, dtype=np.int16),
                final_count=len(prefix_taus) + remaining,
            )
        )
    forecaster._historical_suffix_profiles = profiles


def test_adaptive_threshold_schedule_matches_defaults():
    forecaster, _ = _make_forecaster()

    assert forecaster._compute_late_boundary_silence_threshold(6.0) == pytest.approx(180.0)
    assert forecaster._compute_late_boundary_silence_threshold(5.0) == pytest.approx(150.0)
    assert forecaster._compute_late_boundary_silence_threshold(4.0) == pytest.approx(120.0)
    assert forecaster._compute_late_boundary_silence_threshold(3.0) == pytest.approx(90.0)
    assert forecaster._compute_late_boundary_silence_threshold(2.0) == pytest.approx(90.0)


def test_overlay_requires_window_distance_and_adaptive_silence():
    forecaster, contract_utils = _make_forecaster()
    contract_date = date(2026, 3, 16)
    now = _dt_for_tau(contract_utils, contract_date, 1200)
    prefix = [100, 200, 400, 900, 1050]
    _set_profiles(forecaster, contract_date, [0, 5, 25], prefix)
    events = _events_for_taus(contract_utils, contract_date, prefix)
    probs = [0.2, 0.6, 0.2]
    bins = [(240, 259), (260, 279), (280, 299)]

    updated = forecaster.apply_late_boundary_silence_overlay(
        probabilities=probs,
        current_count=258,
        events=events,
        contract_date=contract_date,
        now=now,
        hours_remaining=4.0,
        bin_ranges=bins,
    )
    assert forecaster._last_boundary_overlay["active"] is True
    assert sum(updated) == pytest.approx(1.0)

    recent_events = _events_for_taus(contract_utils, contract_date, prefix + [1180])
    updated_recent = forecaster.apply_late_boundary_silence_overlay(
        probabilities=probs,
        current_count=258,
        events=recent_events,
        contract_date=contract_date,
        now=now,
        hours_remaining=4.0,
        bin_ranges=bins,
    )
    assert updated_recent == pytest.approx(probs)
    assert forecaster._last_boundary_overlay["active"] is False
    assert forecaster._last_boundary_overlay["skipped_reason"] == "insufficient_silence"


def test_overlay_preserves_probability_mass_and_non_local_bins():
    forecaster, contract_utils = _make_forecaster()
    contract_date = date(2026, 3, 16)
    now = _dt_for_tau(contract_utils, contract_date, 1200)
    prefix = [100, 200, 400, 900, 1050]
    _set_profiles(forecaster, contract_date, [0, 5, 25], prefix)
    events = _events_for_taus(contract_utils, contract_date, prefix)
    probs = [0.1, 0.05, 0.7, 0.1, 0.05]
    bins = [(200, 219), (220, 239), (240, 259), (260, 279), (280, 299)]

    updated = forecaster.apply_late_boundary_silence_overlay(
        probabilities=probs,
        current_count=259,
        events=events,
        contract_date=contract_date,
        now=now,
        hours_remaining=4.0,
        bin_ranges=bins,
    )

    assert sum(updated) == pytest.approx(1.0)
    assert updated[0] == pytest.approx(probs[0])
    assert updated[1] == pytest.approx(probs[1])
    assert updated[2] + updated[3] + updated[4] == pytest.approx(
        probs[2] + probs[3] + probs[4]
    )


def test_overlay_reduced_top_window_still_sums_to_one():
    forecaster, contract_utils = _make_forecaster()
    contract_date = date(2026, 3, 16)
    now = _dt_for_tau(contract_utils, contract_date, 1200)
    prefix = [100, 200, 400, 900, 1050]
    _set_profiles(forecaster, contract_date, [0, 25], prefix)
    events = _events_for_taus(contract_utils, contract_date, prefix)
    probs = [0.25, 0.75]
    bins = [(240, 259), (260, 10000)]

    updated = forecaster.apply_late_boundary_silence_overlay(
        probabilities=probs,
        current_count=259,
        events=events,
        contract_date=contract_date,
        now=now,
        hours_remaining=4.0,
        bin_ranges=bins,
    )

    assert sum(updated) == pytest.approx(1.0)
    assert updated[0] == pytest.approx(0.5)
    assert updated[1] == pytest.approx(0.5)


def test_overlay_skips_when_effective_sample_size_is_too_small():
    forecaster, contract_utils = _make_forecaster()
    forecaster.config.late_boundary_silence.min_effective_n = 4.0
    contract_date = date(2026, 3, 16)
    now = _dt_for_tau(contract_utils, contract_date, 1200)
    prefix = [100, 200, 400, 900, 1050]
    _set_profiles(forecaster, contract_date, [0, 5, 25], prefix)
    events = _events_for_taus(contract_utils, contract_date, prefix)
    probs = [0.2, 0.6, 0.2]
    bins = [(240, 259), (260, 279), (280, 299)]

    updated = forecaster.apply_late_boundary_silence_overlay(
        probabilities=probs,
        current_count=259,
        events=events,
        contract_date=contract_date,
        now=now,
        hours_remaining=4.0,
        bin_ranges=bins,
    )

    assert updated == pytest.approx(probs)
    assert forecaster._last_boundary_overlay["skipped_reason"] == "insufficient_effective_n"


@pytest.mark.parametrize(
    ("current_count", "remaining_values", "expected"),
    [
        (259, [0, 5, 25], [1 / 3, 1 / 3, 1 / 3]),
        (258, [0, 1, 25], [2 / 3, 0.0, 1 / 3]),
        (257, [0, 1, 2], [1.0, 0.0, 0.0]),
        (256, [0, 1, 2, 4], [3 / 4, 1 / 4, 0.0]),
        (255, [0, 1, 2, 4, 5], [4 / 5, 1 / 5, 0.0]),
    ],
)
def test_overlay_handles_one_to_five_tweets_from_boundary(
    current_count: int,
    remaining_values: list[int],
    expected: list[float],
):
    forecaster, contract_utils = _make_forecaster()
    contract_date = date(2026, 3, 16)
    now = _dt_for_tau(contract_utils, contract_date, 1200)
    prefix = [100, 200, 400, 900, 1050]
    _set_profiles(forecaster, contract_date, remaining_values, prefix)
    events = _events_for_taus(contract_utils, contract_date, prefix)
    probs = [0.2, 0.6, 0.2]
    bins = [(240, 259), (260, 279), (280, 299)]

    updated = forecaster.apply_late_boundary_silence_overlay(
        probabilities=probs,
        current_count=current_count,
        events=events,
        contract_date=contract_date,
        now=now,
        hours_remaining=4.0,
        bin_ranges=bins,
    )

    assert updated[:3] == pytest.approx(expected)


def test_overlay_state_is_restart_stable_and_turns_off_when_silence_breaks():
    contract_date = date(2026, 3, 16)
    prefix = [100, 200, 400, 900, 1050]
    probs = [0.1, 0.2, 0.7]
    bins = [(240, 259), (260, 279), (280, 299)]

    forecaster_a, contract_utils = _make_forecaster()
    _set_profiles(forecaster_a, contract_date, [0, 5, 25], prefix)
    now = _dt_for_tau(contract_utils, contract_date, 1200)
    silent_events = _events_for_taus(contract_utils, contract_date, prefix)
    updated_a = forecaster_a.apply_late_boundary_silence_overlay(
        probabilities=probs,
        current_count=259,
        events=silent_events,
        contract_date=contract_date,
        now=now,
        hours_remaining=4.0,
        bin_ranges=bins,
    )

    forecaster_b, _ = _make_forecaster()
    _set_profiles(forecaster_b, contract_date, [0, 5, 25], prefix)
    updated_b = forecaster_b.apply_late_boundary_silence_overlay(
        probabilities=probs,
        current_count=259,
        events=silent_events,
        contract_date=contract_date,
        now=now,
        hours_remaining=4.0,
        bin_ranges=bins,
    )

    assert updated_a == pytest.approx(updated_b)
    assert forecaster_a._last_boundary_overlay["active"] is True
    assert forecaster_b._last_boundary_overlay["active"] is True

    broken_silence_events = _events_for_taus(contract_utils, contract_date, prefix + [1170])
    updated_c = forecaster_b.apply_late_boundary_silence_overlay(
        probabilities=probs,
        current_count=259,
        events=broken_silence_events,
        contract_date=contract_date,
        now=now,
        hours_remaining=4.0,
        bin_ranges=bins,
    )
    assert updated_c == pytest.approx(probs)
    assert forecaster_b._last_boundary_overlay["active"] is False
