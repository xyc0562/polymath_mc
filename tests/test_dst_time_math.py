from datetime import date, datetime, timezone

import numpy as np
import pytest

from src.algo.musk_tweet_count.forecaster.config import BucketNowcastConfig, ForecasterConfig
from src.algo.musk_tweet_count.forecaster.data import ContractDayUtils, TweetEvent
from src.algo.musk_tweet_count.forecaster.forecaster import TweetCountForecaster
from src.algo.musk_tweet_count.forecaster.intraday import BucketDistribution, BucketIntradayForecaster


def _make_bucket_forecaster() -> tuple[BucketIntradayForecaster, ContractDayUtils]:
    contract_utils = ContractDayUtils()
    forecaster = BucketIntradayForecaster(BucketNowcastConfig(), contract_utils)
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
    forecaster._weekend_buckets = list(forecaster._weekday_buckets)
    return forecaster, contract_utils


def _capture_silence_minutes(
    contract_date: date,
    last_tweet_at: datetime,
    now: datetime,
) -> float:
    forecaster, contract_utils = _make_bucket_forecaster()
    events = [TweetEvent(timestamp=last_tweet_at)]
    settlement_tau = contract_utils.get_tau(now, contract_date)
    forecaster._sample_with_impulse(
        events=events,
        contract_date=contract_date,
        now=now,
        n_simulations=1,
        settlement_tau=settlement_tau,
    )
    return float(forecaster._last_impulse["silence_min"])


def test_contract_day_bounds_utc_follow_dst() -> None:
    contract_utils = ContractDayUtils()

    spring_start, _ = contract_utils.get_contract_day_bounds_utc(date(2026, 3, 8))
    fallback_start, _ = contract_utils.get_contract_day_bounds_utc(date(2026, 11, 1))

    assert spring_start == datetime(2026, 3, 8, 16, 0, tzinfo=timezone.utc)
    assert fallback_start == datetime(2026, 11, 1, 17, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("contract_date", "last_tweet_at", "now", "expected_minutes"),
    [
        (
            date(2026, 3, 7),
            datetime(2026, 3, 8, 6, 11, 5, tzinfo=timezone.utc),
            datetime(2026, 3, 8, 6, 56, 5, tzinfo=timezone.utc),
            45.0,
        ),
        (
            date(2026, 3, 7),
            datetime(2026, 3, 8, 6, 11, 5, tzinfo=timezone.utc),
            datetime(2026, 3, 8, 7, 1, 5, tzinfo=timezone.utc),
            50.0,
        ),
        (
            date(2026, 10, 31),
            datetime(2026, 11, 1, 5, 11, 5, tzinfo=timezone.utc),
            datetime(2026, 11, 1, 5, 56, 5, tzinfo=timezone.utc),
            45.0,
        ),
        (
            date(2026, 10, 31),
            datetime(2026, 11, 1, 5, 11, 5, tzinfo=timezone.utc),
            datetime(2026, 11, 1, 6, 1, 5, tzinfo=timezone.utc),
            50.0,
        ),
    ],
)
def test_bucket_impulse_silence_uses_real_elapsed_minutes(
    contract_date: date,
    last_tweet_at: datetime,
    now: datetime,
    expected_minutes: float,
) -> None:
    silence_min = _capture_silence_minutes(contract_date, last_tweet_at, now)

    assert silence_min == expected_minutes


def test_get_settlement_timing_uses_real_elapsed_hours_across_dst() -> None:
    forecaster = TweetCountForecaster(ForecasterConfig())

    spring_before = datetime(2026, 3, 8, 6, 56, 5, tzinfo=timezone.utc)
    spring_after = datetime(2026, 3, 8, 7, 1, 5, tzinfo=timezone.utc)
    _, spring_remaining_before = forecaster.get_settlement_timing(date(2026, 3, 8), spring_before)
    _, spring_remaining_after = forecaster.get_settlement_timing(date(2026, 3, 8), spring_after)

    fallback_before = datetime(2026, 11, 1, 5, 56, 5, tzinfo=timezone.utc)
    fallback_after = datetime(2026, 11, 1, 6, 1, 5, tzinfo=timezone.utc)
    _, fallback_remaining_before = forecaster.get_settlement_timing(date(2026, 11, 1), fallback_before)
    _, fallback_remaining_after = forecaster.get_settlement_timing(date(2026, 11, 1), fallback_after)

    assert spring_remaining_before - spring_remaining_after == pytest.approx(5 / 60, abs=1e-9)
    assert fallback_remaining_before - fallback_remaining_after == pytest.approx(5 / 60, abs=1e-9)
