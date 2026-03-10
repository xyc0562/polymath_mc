from datetime import date, timedelta

import numpy as np
import pytest

from src.algo.musk_tweet_count.forecaster.config import BucketNowcastConfig
from src.algo.musk_tweet_count.forecaster.data import ContractDayUtils, TweetEvent
from src.algo.musk_tweet_count.forecaster.intraday import BucketDistribution, BucketIntradayForecaster


def _dt_for_tau(contract_utils: ContractDayUtils, contract_date: date, tau: int):
    start_dt, _ = contract_utils.get_contract_day_bounds(contract_date)
    return start_dt + timedelta(minutes=tau)


def _make_events(contract_utils: ContractDayUtils, contract_date: date, taus: list[int]) -> list[TweetEvent]:
    return [TweetEvent(timestamp=_dt_for_tau(contract_utils, contract_date, tau)) for tau in sorted(taus)]


def _historical_events_by_pattern(contract_utils: ContractDayUtils, as_of_date: date):
    events_by_date: dict[date, list[TweetEvent]] = {}
    counts_by_date: dict[date, int] = {}

    for idx in range(18):
        contract_date = as_of_date - timedelta(days=idx + 1)
        if idx % 4 == 0:
            taus = [300, 360, 420, 510, 730, 740, 755, 770]
        elif idx % 4 == 1:
            taus = [300, 360, 420, 510]
        elif idx % 4 == 2:
            taus = [300, 360, 940, 945, 952, 958, 967, 975, 1005, 1030]
        else:
            taus = [300, 360, 910, 918]

        events = _make_events(contract_utils, contract_date, taus)
        events_by_date[contract_date] = events
        counts_by_date[contract_date] = len(events)

    return events_by_date, counts_by_date


def _constant_bucket_forecaster(
    use_overnight_quiet: bool = True,
    runtime_mask: bool = True,
    regime_damping: bool = True,
    bootstrap_matching: bool = True,
) -> tuple[BucketIntradayForecaster, ContractDayUtils]:
    contract_utils = ContractDayUtils()
    config = BucketNowcastConfig(
        use_overnight_quiet=use_overnight_quiet,
        use_overnight_quiet_runtime_mask=runtime_mask,
        use_overnight_quiet_regime_damping=regime_damping,
        use_overnight_quiet_bootstrap_matching=bootstrap_matching,
        impulse_min_tweets_for_fit=10,
    )
    forecaster = BucketIntradayForecaster(config, contract_utils)
    as_of_date = date(2026, 3, 10)
    historical_events, historical_counts = _historical_events_by_pattern(contract_utils, as_of_date)
    forecaster.fit(historical_events, historical_counts, as_of_date=as_of_date)

    buckets = [
        BucketDistribution(
            bucket_idx=i,
            start_tau=i * forecaster.bucket_size,
            end_tau=(i + 1) * forecaster.bucket_size,
            mean=2.0,
            std=1.0,
            dispersion_k=4.0,
        )
        for i in range(forecaster.n_buckets)
    ]
    forecaster._weekday_buckets = list(buckets)
    forecaster._weekend_buckets = list(buckets)
    forecaster._rate_curve = np.full(1440, 0.05)
    forecaster._impulse_fitted = False
    forecaster._fitted = True
    return forecaster, contract_utils


def _baseline_daytime_taus() -> list[int]:
    return [
        30, 60, 90, 120, 150, 180, 210, 240, 270, 300,
        330, 360, 390, 420, 450, 480, 510, 540, 570, 600,
    ]


def test_overnight_profile_learns_idle_and_wake_relief() -> None:
    forecaster, _ = _constant_bucket_forecaster()
    profile = forecaster._weekday_overnight_profile

    assert profile is not None
    assert profile.night_idle_prior[forecaster._tau_to_overnight_bin(840)] > 0.2
    assert profile.wake_relief_curve[1030] > profile.wake_relief_curve[840]


def test_active_then_silent_has_stronger_sleep_onset_than_quiet_all_night() -> None:
    forecaster, contract_utils = _constant_bucket_forecaster()
    contract_date = date(2026, 3, 9)
    regime = 1.15

    quiet_events = _make_events(contract_utils, contract_date, _baseline_daytime_taus())
    active_then_silent = _make_events(
        contract_utils,
        contract_date,
        _baseline_daytime_taus() + [730, 740, 755, 770],
    )

    quiet_state = forecaster._compute_overnight_quiet_state(
        quiet_events,
        contract_date,
        _dt_for_tau(contract_utils, contract_date, 840),
        regime=regime,
    )
    active_state = forecaster._compute_overnight_quiet_state(
        active_then_silent,
        contract_date,
        _dt_for_tau(contract_utils, contract_date, 840),
        regime=regime,
    )

    assert quiet_state.night_idle_prior > 0.2
    assert active_state.sleep_onset_confidence > quiet_state.sleep_onset_confidence
    assert active_state.regime_eff < quiet_state.regime_eff
    assert active_state.regime_eff < regime


def test_real_wake_burst_releases_overnight_suppression() -> None:
    forecaster, contract_utils = _constant_bucket_forecaster()
    contract_date = date(2026, 3, 9)
    pre_sleep_events = _make_events(
        contract_utils,
        contract_date,
        _baseline_daytime_taus() + [730, 740, 755, 770],
    )
    wake_events = _make_events(
        contract_utils,
        contract_date,
        _baseline_daytime_taus() + [730, 740, 755, 770, 940, 945, 952, 958, 967],
    )

    quiet_state = forecaster._compute_overnight_quiet_state(
        pre_sleep_events,
        contract_date,
        _dt_for_tau(contract_utils, contract_date, 935),
        regime=1.15,
    )
    wake_state = forecaster._compute_overnight_quiet_state(
        wake_events,
        contract_date,
        _dt_for_tau(contract_utils, contract_date, 970),
        regime=1.15,
    )

    assert wake_state.wake_continuation_confidence > 0.2
    assert wake_state.quiet_strength < quiet_state.quiet_strength
    assert wake_state.state in {"night_wake", "awake_active"}


def test_brief_wake_then_silence_resuppresses_quickly() -> None:
    forecaster, contract_utils = _constant_bucket_forecaster()
    contract_date = date(2026, 3, 9)
    brief_wake_events = _make_events(
        contract_utils,
        contract_date,
        _baseline_daytime_taus() + [730, 740, 755, 770, 910, 918],
    )

    wake_state = forecaster._compute_overnight_quiet_state(
        brief_wake_events,
        contract_date,
        _dt_for_tau(contract_utils, contract_date, 919),
        regime=1.15,
    )
    resleep_state = forecaster._compute_overnight_quiet_state(
        brief_wake_events,
        contract_date,
        _dt_for_tau(contract_utils, contract_date, 940),
        regime=1.15,
    )

    assert wake_state.wake_continuation_confidence >= resleep_state.wake_continuation_confidence
    assert resleep_state.quiet_strength > wake_state.quiet_strength
    assert resleep_state.state == "overnight_quiet"


def test_daytime_quiet_state_is_inactive() -> None:
    forecaster, contract_utils = _constant_bucket_forecaster()
    contract_date = date(2026, 3, 9)
    quiet_events = _make_events(contract_utils, contract_date, _baseline_daytime_taus())

    state = forecaster._compute_overnight_quiet_state(
        quiet_events,
        contract_date,
        _dt_for_tau(contract_utils, contract_date, 60),
        regime=1.15,
    )

    assert state.state == "inactive"
    assert state.quiet_strength == 0.0


def test_runtime_quiet_is_disabled_by_default_but_state_is_still_inferred() -> None:
    contract_utils = ContractDayUtils()
    config = BucketNowcastConfig(
        use_overnight_quiet=True,
        impulse_min_tweets_for_fit=10,
    )
    forecaster = BucketIntradayForecaster(config, contract_utils)
    as_of_date = date(2026, 3, 10)
    historical_events, historical_counts = _historical_events_by_pattern(contract_utils, as_of_date)
    forecaster.fit(historical_events, historical_counts, as_of_date=as_of_date)

    contract_date = date(2026, 3, 9)
    events = _make_events(
        contract_utils,
        contract_date,
        _baseline_daytime_taus() + [730, 740, 755, 770],
    )
    regime = 1.15
    state = forecaster._compute_overnight_quiet_state(
        events,
        contract_date,
        _dt_for_tau(contract_utils, contract_date, 840),
        regime=regime,
    )

    assert state.state == "overnight_quiet"
    assert state.sleep_onset_confidence > 0.2
    assert state.quiet_strength == 0.0
    assert state.regime_eff == pytest.approx(regime)


def test_overnight_quiet_lowers_remaining_forecast_vs_baseline() -> None:
    quiet_forecaster, contract_utils = _constant_bucket_forecaster(
        use_overnight_quiet=True,
        runtime_mask=True,
        regime_damping=True,
    )
    baseline_forecaster, _ = _constant_bucket_forecaster(use_overnight_quiet=False)
    contract_date = date(2026, 3, 9)
    events = _make_events(
        contract_utils,
        contract_date,
        _baseline_daytime_taus() + [730, 740, 755, 770],
    )
    now = _dt_for_tau(contract_utils, contract_date, 840)

    quiet_samples = quiet_forecaster._sample_with_impulse(
        events=events,
        contract_date=contract_date,
        now=now,
        n_simulations=4000,
        rng=np.random.default_rng(1),
    )
    baseline_samples = baseline_forecaster._sample_with_impulse(
        events=events,
        contract_date=contract_date,
        now=now,
        n_simulations=4000,
        rng=np.random.default_rng(1),
    )

    assert float(np.mean(quiet_samples)) < float(np.mean(baseline_samples)) - 1.0
