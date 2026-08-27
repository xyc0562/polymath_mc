from types import SimpleNamespace

import numpy as np
import pytest

from src.algo.musk_tweet_count.forecaster.config import GASConfig
from src.algo.musk_tweet_count.forecaster import gas as gas_module
from src.algo.musk_tweet_count.forecaster.gas import GASRegimeModel


def _nb_series(mean: float, k: float, n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.negative_binomial(k, k / (k + mean), size=n)]


def test_fit_high_mean_series_avoids_cap_plateau():
    """Single-start L-BFGS-B lands on the exp(5.70)=299/day plateau for
    high-activity series; multi-start with degenerate-fit rejection must not."""
    config = GASConfig()
    cap_intensity = np.exp(config.max_log_intensity)

    for seed in (1, 2, 3, 4, 5):
        model = GASRegimeModel(config)
        series = _nb_series(mean=150.0, k=2.0, n=40, seed=seed)
        model.fit(series, k=2.0)

        assert model.intensity < cap_intensity * 0.97, (
            f"seed={seed}: fit pegged at intensity cap "
            f"(mu={model.intensity:.1f})"
        )
        for horizon in range(1, 7):
            mu_h = np.exp(model.forecast_intensity(horizon))
            assert mu_h < cap_intensity * 0.97, (
                f"seed={seed} h={horizon}: forecast pegged at cap (mu={mu_h:.1f})"
            )


def test_fit_moderate_series_stays_near_sample_mean():
    config = GASConfig()
    model = GASRegimeModel(config)
    series = _nb_series(mean=50.0, k=2.0, n=40, seed=7)
    model.fit(series, k=2.0)

    sample_mean = np.mean(series)
    assert 0.4 * sample_mean < model.intensity < 2.5 * sample_mean


def test_fit_absorbs_final_observation_into_state():
    """self.f must be one filter step AHEAD of _run_filter's last state,
    i.e. it has absorbed counts[-1] (previously only in the likelihood)."""
    config = GASConfig()
    model = GASRegimeModel(config)
    series = _nb_series(mean=50.0, k=2.0, n=40, seed=11)
    model.fit(series, k=2.0)

    counts = np.array(series, dtype=np.float64)
    f_values = model._run_filter(model.omega, model.alpha, model.beta, counts, 2.0)
    mu_last = max(np.exp(f_values[-1]), 1e-6)
    score = (counts[-1] - mu_last) / mu_last
    expected = min(
        model.omega + model.beta * f_values[-1] + model.alpha * score,
        config.max_log_intensity,
    )

    assert model.f == pytest.approx(expected, abs=1e-12)


def test_final_observation_moves_state_with_fixed_params(monkeypatch):
    """With identical fitted params, a different last-day count must move
    the exported state (the pre-fix code returned bit-identical states)."""
    fixed = np.array([0.4, 0.3, 0.9])

    def fake_minimize(fun, x0, args=(), **kwargs):
        return SimpleNamespace(fun=float(fun(fixed, *args)), x=fixed, success=True)

    monkeypatch.setattr(gas_module.optimize, "minimize", fake_minimize)

    base = _nb_series(mean=50.0, k=2.0, n=40, seed=13)
    quiet, surge = list(base), list(base)
    quiet[-1] = 30
    surge[-1] = 150

    config = GASConfig()
    model_quiet = GASRegimeModel(config)
    model_quiet.fit(quiet, k=2.0)
    model_surge = GASRegimeModel(config)
    model_surge.fit(surge, k=2.0)

    assert model_surge.f > model_quiet.f
    # Difference equals alpha * (score_surge - score_quiet)
    counts = np.array(base, dtype=np.float64)
    f_values = model_quiet._run_filter(0.4, 0.3, 0.9, counts[:], 2.0)
    mu_last = np.exp(f_values[-1])
    expected_delta = 0.3 * ((150 - mu_last) / mu_last - (30 - mu_last) / mu_last)
    assert model_surge.f - model_quiet.f == pytest.approx(expected_delta, rel=1e-9)


def test_fit_falls_back_when_all_starts_degenerate(monkeypatch):
    config = GASConfig()
    model = GASRegimeModel(config)
    monkeypatch.setattr(
        GASRegimeModel, "_is_degenerate_fit", lambda self, params, counts, k: True
    )
    series = _nb_series(mean=50.0, k=2.0, n=40, seed=17)

    model.fit(series, k=2.0)

    # Moment-anchored fallback: unconditional mean equals the sample log-mean
    sample_mean = max(np.mean(series), 1.0)
    assert model.beta == config.beta_init
    assert model.omega == pytest.approx((1 - config.beta_init) * np.log(sample_mean))
    assert 0.3 * sample_mean < model.intensity < 3.0 * sample_mean


def test_short_series_uses_mean_fallback():
    config = GASConfig()
    model = GASRegimeModel(config)

    model.fit([40, 50, 60], k=2.0)

    assert model.intensity == pytest.approx(50.0)
