from src.algo.musk_tweet_count.backtest.unified_runner import (
    UnifiedBacktestConfig,
    UnifiedBacktestRunner,
)


def _make_runner(**config_overrides) -> UnifiedBacktestRunner:
    runner = UnifiedBacktestRunner.__new__(UnifiedBacktestRunner)
    runner.config = UnifiedBacktestConfig(**config_overrides)
    return runner


def test_sample_timestamps_uses_fixed_interval_by_default():
    runner = _make_runner(tick_interval_seconds=3600)
    timestamps = [0, 600, 1800, 3600, 4200, 7200]

    sampled = runner._sample_timestamps(timestamps, 3600)

    assert sampled == [0, 3600, 7200]


def test_sample_timestamps_switches_to_late_interval_near_settlement():
    runner = _make_runner(
        tick_interval_seconds=3600,
        late_tick_interval_seconds=300,
        late_tick_start_hours_before_settlement=1.0,
    )
    settlement_ts = 7200
    timestamps = [
        0,
        1800,
        3000,
        3600,
        3900,
        4200,
        4500,
        4800,
        5100,
        5400,
        5700,
        6000,
        6300,
        6600,
        6900,
    ]

    sampled = runner._sample_timestamps(
        timestamps,
        3600,
        settlement_ts=settlement_ts,
    )

    assert sampled == [0, 3600, 3900, 4200, 4500, 4800, 5100, 5400, 5700, 6000, 6300, 6600, 6900]


def test_sample_timestamps_respects_resume_inside_late_window():
    runner = _make_runner(
        tick_interval_seconds=3600,
        late_tick_interval_seconds=300,
        late_tick_start_hours_before_settlement=1.0,
    )
    settlement_ts = 7200
    timestamps = [0, 1800, 3600, 3900, 4200, 4500, 4800, 5100, 5400, 5700, 6000]

    sampled = runner._sample_timestamps(
        timestamps,
        3600,
        start_at_ts=4200,
        settlement_ts=settlement_ts,
    )

    assert sampled == [4200, 4500, 4800, 5100, 5400, 5700, 6000]
