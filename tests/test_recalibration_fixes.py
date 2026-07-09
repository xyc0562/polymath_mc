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


# ---------- item 4: prob EMA jump gate ----------


def _ema_bot(alpha=0.3, threshold=0.02) -> KellyTradingBot:
    bot = object.__new__(KellyTradingBot)
    bot.config = KellyConfig(
        prob_ema_alpha=alpha,
        prob_ema_jump_threshold=threshold,
    )
    bot.event_name = "test-event"
    bot._ema_probabilities = None
    return bot


def test_first_vector_passes_through():
    bot = _ema_bot()
    probs = [0.2, 0.3, 0.5]

    result = bot._apply_prob_ema(probs)

    assert result == probs
    assert bot._ema_probabilities == probs


def test_small_jitter_is_smoothed():
    bot = _ema_bot()
    bot._apply_prob_ema([0.2, 0.3, 0.5])

    result = bot._apply_prob_ema([0.21, 0.29, 0.5])  # max jump 0.01 < 0.02

    # Blended: 0.3*new + 0.7*old (then renormalized; sums already 1)
    assert result[0] == pytest.approx(0.3 * 0.21 + 0.7 * 0.2, abs=1e-9)
    assert result[1] == pytest.approx(0.3 * 0.29 + 0.7 * 0.3, abs=1e-9)


def test_real_jump_bypasses_smoothing_and_resets_state():
    bot = _ema_bot()
    bot._apply_prob_ema([0.2, 0.3, 0.5])

    jumped = [0.05, 0.15, 0.8]  # max jump 0.3 >= threshold
    result = bot._apply_prob_ema(jumped)

    assert result == jumped  # passed through unsmoothed

    # EMA restarts FROM the jumped vector
    followup = bot._apply_prob_ema([0.06, 0.14, 0.8])
    assert followup[0] == pytest.approx(0.3 * 0.06 + 0.7 * 0.05, abs=1e-9)


def test_jump_just_above_threshold_bypasses():
    # (exact-threshold equality is not testable with binary floats:
    # 0.22 - 0.2 == 0.01999...; use a clearly-above jump)
    bot = _ema_bot(threshold=0.02)
    bot._apply_prob_ema([0.2, 0.3, 0.5])

    result = bot._apply_prob_ema([0.221, 0.279, 0.5])  # max jump 0.021

    assert result == [0.221, 0.279, 0.5]


def test_alpha_one_disables_smoothing():
    bot = _ema_bot(alpha=1.0)
    bot._apply_prob_ema([0.2, 0.3, 0.5])

    result = bot._apply_prob_ema([0.21, 0.29, 0.5])

    assert result == [0.21, 0.29, 0.5]
