import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.algo.musk_tweet_count.forecaster.distributions import (
    CMP_NU_MIN,
    _cmp_nu_from_k,
    sample_com_poisson,
)
from src.algo.musk_tweet_count.kelly.config import KellyConfig
from src.algo.musk_tweet_count.kelly.integration import KellyTradingBot


def _delivered(mean, k, n=200_000, seed=7):
    rng = np.random.default_rng(seed)
    samples = sample_com_poisson(mean, k, n, rng)
    return samples.mean(), samples.std()


def _nb_std(mean, k):
    return math.sqrt(mean * (1 + mean / k))


# ---------- item 3: CMP variance calibration ----------


def test_cmp_delivers_exact_negbin_std_when_attainable():
    # Targets with k >= 1 sit within the CMP family's variance ceiling
    for mean, k in [(50.0, 1.0), (60.0, 2.0), (35.0, 1.5), (20.0, 1.0)]:
        mean_dlv, std_dlv = _delivered(mean, k)
        assert std_dlv == pytest.approx(_nb_std(mean, k), rel=0.06), (
            f"(mean={mean}, k={k}): delivered {std_dlv:.2f} "
            f"vs intended {_nb_std(mean, k):.2f}"
        )
        assert mean_dlv == pytest.approx(mean, rel=0.02)


def test_cmp_saturates_at_geometric_ceiling_for_small_k():
    # k < 1 targets exceed the mean-matched CMP variance ceiling
    # var <= mean*(1+mean); the sampler must deliver that best-attainable
    # width (the old 0.05 nu floor delivered substantially less).
    for mean, k in [(20.0, 0.4), (8.0, 0.2), (100.0, 0.4)]:
        _, std_dlv = _delivered(mean, k)
        ceiling_std = math.sqrt(mean * (1 + mean))
        assert std_dlv == pytest.approx(ceiling_std, rel=0.06), (
            f"(mean={mean}, k={k}): delivered {std_dlv:.2f} "
            f"vs ceiling {ceiling_std:.2f}"
        )


def test_cmp_wider_than_old_floored_behavior():
    # Regression vs the 0.05-floored mapping: evening-bucket params must
    # now deliver more spread (old delivered ~17.4 at mean=20, k=0.4)
    _, std_dlv = _delivered(20.0, 0.4)
    assert std_dlv > 19.0


def test_dispersion_knob_effective_within_attainable_range():
    _, std_wide = _delivered(60.0, 2.0)   # intended 43.1
    _, std_thin = _delivered(60.0, 5.0)   # intended 27.9
    assert std_wide > std_thin * 1.3


def test_solved_nu_within_bounds_and_cached():
    nu = _cmp_nu_from_k(20.0, 0.4, 1.0)
    assert CMP_NU_MIN <= nu <= 5.0
    # identical rounded args must hit the lru cache (same object semantics
    # aren't observable; at least verify determinism)
    assert _cmp_nu_from_k(20.0, 0.4, 1.0) == nu


def test_pathological_k_does_not_crash():
    rng = np.random.default_rng(11)
    samples = sample_com_poisson(20.0, 1e-9, 1000, rng)
    assert np.isfinite(samples).all()


# ---------- item 4: prob EMA persistence-gated jump snap ----------


def _ema_bot(alpha=0.3, threshold=0.02, confirm_ticks=3, confirm_seconds=120.0) -> KellyTradingBot:
    bot = object.__new__(KellyTradingBot)
    bot.config = KellyConfig(
        prob_ema_alpha=alpha,
        prob_ema_jump_threshold=threshold,
        prob_ema_jump_confirm_ticks=confirm_ticks,
        prob_ema_jump_confirm_seconds=confirm_seconds,
    )
    bot.event_name = "test-event"
    bot._ema_probabilities = None
    bot._prob_jump_baseline = None
    bot._prob_jump_ticks = 0
    bot._prob_jump_first_ts = 0.0
    return bot


def _blend(new, old, alpha=0.3):
    mixed = [alpha * n + (1 - alpha) * o for n, o in zip(new, old)]
    total = sum(mixed)
    return [m / total for m in mixed]


def test_first_vector_passes_through():
    bot = _ema_bot()
    probs = [0.2, 0.3, 0.5]

    result = bot._apply_prob_ema(probs, now=0.0)

    assert result == probs
    assert bot._ema_probabilities == probs


def test_small_jitter_is_smoothed():
    bot = _ema_bot()
    bot._apply_prob_ema([0.2, 0.3, 0.5], now=0.0)

    result = bot._apply_prob_ema([0.21, 0.29, 0.5], now=30.0)  # max jump 0.01 < 0.02

    # Blended: 0.3*new + 0.7*old (then renormalized; sums already 1)
    assert result[0] == pytest.approx(0.3 * 0.21 + 0.7 * 0.2, abs=1e-9)
    assert result[1] == pytest.approx(0.3 * 0.29 + 0.7 * 0.3, abs=1e-9)
    assert bot._prob_jump_baseline is None


def test_jump_is_smoothed_while_pending_and_kills_quotes():
    bot = _ema_bot()
    kills = []
    bot.quote_manager = SimpleNamespace(kill=lambda reason: kills.append(reason))
    bot._apply_prob_ema([0.2, 0.3, 0.5], now=0.0)

    jumped = [0.05, 0.15, 0.8]  # max jump 0.3 >= threshold
    result = bot._apply_prob_ema(jumped, now=30.0)

    # NOT passed through: still the damped path
    assert result == pytest.approx(_blend(jumped, [0.2, 0.3, 0.5]))
    # Jump tracked, quotes killed exactly once at detection
    assert bot._prob_jump_baseline == [0.2, 0.3, 0.5]
    assert kills == ["prob_jump"]

    bot._apply_prob_ema(jumped, now=60.0)
    assert kills == ["prob_jump"]  # no re-kill while pending


def test_flicker_reverts_without_ever_passing_raw():
    bot = _ema_bot()
    base = [0.2, 0.3, 0.5]
    bot._apply_prob_ema(base, now=0.0)

    r1 = bot._apply_prob_ema([0.05, 0.15, 0.8], now=30.0)   # jump detected
    r2 = bot._apply_prob_ema([0.21, 0.29, 0.5], now=60.0)   # nowcast flickers back

    assert r1 == pytest.approx(_blend([0.05, 0.15, 0.8], base))
    # Revert clears the pending jump and keeps smoothing off the drifted EMA
    assert bot._prob_jump_baseline is None
    assert r2 == pytest.approx(_blend([0.21, 0.29, 0.5], r1))


def test_persistent_jump_snaps_after_ticks_and_seconds():
    bot = _ema_bot(confirm_ticks=3, confirm_seconds=120.0)
    bot._apply_prob_ema([0.2, 0.3, 0.5], now=0.0)

    jumped = [0.05, 0.15, 0.8]
    r1 = bot._apply_prob_ema(jumped, now=10.0)    # tick 1: detect, smoothed
    r2 = bot._apply_prob_ema(jumped, now=70.0)    # tick 2: smoothed
    r3 = bot._apply_prob_ema(jumped, now=129.0)   # tick 3 but only 119s elapsed
    r4 = bot._apply_prob_ema(jumped, now=135.0)   # tick 4, 125s elapsed: SNAP

    for r in (r1, r2, r3):
        assert r != jumped
    assert r4 == jumped
    assert bot._ema_probabilities == jumped
    assert bot._prob_jump_baseline is None

    # EMA restarts FROM the snapped vector
    followup = bot._apply_prob_ema([0.06, 0.14, 0.8], now=165.0)
    assert followup == pytest.approx(_blend([0.06, 0.14, 0.8], jumped))


def test_confirm_requires_elapsed_seconds_not_just_ticks():
    bot = _ema_bot(confirm_ticks=3, confirm_seconds=120.0)
    bot._apply_prob_ema([0.2, 0.3, 0.5], now=0.0)

    jumped = [0.05, 0.15, 0.8]
    # Rapid ticks: 5 confirmations inside 20 seconds must NOT snap
    for i in range(5):
        result = bot._apply_prob_ema(jumped, now=1.0 + i * 4.0)
        assert result != jumped
    assert bot._prob_jump_baseline is not None


def test_alpha_one_disables_smoothing():
    bot = _ema_bot(alpha=1.0)
    bot._apply_prob_ema([0.2, 0.3, 0.5], now=0.0)

    result = bot._apply_prob_ema([0.21, 0.29, 0.5], now=30.0)

    assert result == [0.21, 0.29, 0.5]
