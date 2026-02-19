"""
Compute Bayesian impulse predictions across various scenarios.
Shows predicted tweet counts for different times-of-day, elapsed silence,
and recent tweet patterns.
"""
import sys
sys.path.insert(0, "src")

import numpy as np
from datetime import date
from scipy.ndimage import gaussian_filter1d

from algo.musk_tweet_count.forecaster.data import load_events_from_csv, ContractDayUtils
from algo.musk_tweet_count.forecaster.config import BucketNowcastConfig

# --- Load data and build rate curve ---
contract_utils = ContractDayUtils()
events_by_date = load_events_from_csv(
    "data/musk_posts_2024-01-01_to_2026-01-21_ET_merged.csv",
    contract_utils=contract_utils,
)

config = BucketNowcastConfig()
as_of_date = date(2026, 1, 21)  # latest date in CSV

# Build rate curve (same logic as _fit_impulse)
day_counts = []
day_weights = []
total_tweets = 0

for contract_date, events in events_by_date.items():
    days_ago = (as_of_date - contract_date).days
    if days_ago > config.training_window_days:
        continue
    counts = np.zeros(1440)
    for e in events:
        tau = contract_utils.get_tau(e.timestamp, contract_date)
        tau = max(0, min(tau, 1439))
        counts[tau] += 1
        total_tweets += 1
    weight = np.exp(-days_ago / config.weight_half_life_days * np.log(2))
    day_counts.append(counts)
    day_weights.append(weight)

weights_arr = np.array(day_weights)
weights_arr /= weights_arr.sum()
raw_rate = np.zeros(1440)
for counts, w in zip(day_counts, weights_arr):
    raw_rate += counts * w

rate_curve = gaussian_filter1d(raw_rate, sigma=config.impulse_rate_curve_sigma, mode="wrap")
rate_curve = np.maximum(rate_curve, 1e-6)

print(f"Data: {len(day_counts)} training days, {total_tweets} tweets")
print(f"Rate curve total: {rate_curve.sum():.1f} tweets/day")
print()

# --- Print rate curve at key hours ---
print("=" * 70)
print("RATE CURVE λ(τ) — Average tweets/minute at each hour")
print("=" * 70)
print(f"{'Hour (ET)':>12} {'τ':>6} {'λ (tweets/min)':>16} {'tweets/hr':>12}")
print("-" * 50)
for tau in range(0, 1440, 60):
    hour_et = (12 + tau // 60) % 24
    rate_at_tau = rate_curve[tau]
    tweets_per_hour = rate_curve[tau:tau+60].sum()
    print(f"  {hour_et:02d}:00 ET  {tau:>6}  {rate_at_tau:>14.4f}  {tweets_per_hour:>10.1f}")

print()

# --- Bayesian prediction function ---
def predict_impulse(tau_now, tau_last, k_obs, rate_curve, config):
    """Compute Bayesian impulse prediction stats."""
    alpha0 = config.impulse_prior_concentration
    beta0 = alpha0
    cutoff = config.impulse_cutoff_minutes

    elapsed = tau_now - tau_last

    # Observation window
    tau_last_c = max(0, min(int(tau_last), 1439))
    tau_now_c = max(0, min(int(tau_now), 1440))
    E_obs = float(rate_curve[tau_last_c:tau_now_c].sum())

    # Posterior
    alpha_post = alpha0 + k_obs
    beta_post = beta0 + E_obs
    posterior_mean = alpha_post / beta_post

    # Remaining impulse window
    impulse_end = int(min(tau_last + cutoff, 1440))
    impulse_end = max(impulse_end, tau_now_c)
    tau_impulse_end_c = max(0, min(impulse_end, 1440))
    E_remaining = float(rate_curve[tau_now_c:tau_impulse_end_c].sum())

    # Predictive mean (NB mean = alpha_post * E_remaining / beta_post)
    pred_mean = alpha_post * E_remaining / beta_post if beta_post > 0 else E_remaining

    # Predictive std (NB variance = alpha_post * E_remaining * (beta_post + E_remaining) / beta_post^2)
    if beta_post > 0:
        pred_var = alpha_post * E_remaining * (beta_post + E_remaining) / (beta_post ** 2)
    else:
        pred_var = E_remaining
    pred_std = np.sqrt(pred_var)

    return {
        "elapsed": elapsed,
        "E_obs": E_obs,
        "k_obs": k_obs,
        "alpha_post": alpha_post,
        "beta_post": beta_post,
        "posterior_mean": posterior_mean,
        "impulse_end": impulse_end,
        "E_remaining": E_remaining,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "base_rate_remaining": E_remaining,  # what you'd predict at r=1
    }

# --- Scenario table ---
print("=" * 70)
print("BAYESIAN IMPULSE PREDICTIONS")
print(f"  Prior: α₀=β₀={config.impulse_prior_concentration}, cutoff={config.impulse_cutoff_minutes}min")
print(f"  Observation window = [τ_last, τ_now] (backward from now to last tweet)")
print("=" * 70)

# Time-of-day points
tau_points = [
    (0,   "12:00 PM (noon)"),
    (180, " 3:00 PM"),
    (360, " 6:00 PM"),
    (540, " 9:00 PM"),
    (720, "12:00 AM (midnight)"),
    (900, " 3:00 AM"),
    (1080, " 6:00 AM"),
    (1260, " 9:00 AM"),
]

# Scenarios: (elapsed_min, k_obs, label)
scenarios = [
    (5,   0,  "Silent 5min"),
    (5,   3,  "Burst: 3 tweets in 5min"),
    (5,   8,  "Heavy burst: 8 in 5min"),
    (15,  0,  "Silent 15min"),
    (15,  2,  "Normal: 2 in 15min"),
    (15,  6,  "Burst: 6 in 15min"),
    (30,  0,  "Silent 30min"),
    (30,  3,  "Normal: 3 in 30min"),
    (30,  10, "Burst: 10 in 30min"),
    (60,  0,  "Silent 1hr"),
    (60,  5,  "Normal: 5 in 1hr"),
    (60,  15, "Burst: 15 in 1hr"),
    (120, 0,  "Silent 2hr"),
    (120, 5,  "Sparse: 5 in 2hr"),
    (120, 20, "Burst: 20 in 2hr"),
    (180, 0,  "Silent 3hr (at cutoff)"),
]

for tau_now, time_label in tau_points:
    print(f"\n--- τ_now={tau_now} ({time_label}) ---")
    print(f"{'Scenario':<30} {'Elapsed':>7} {'E_obs':>7} {'k_obs':>5} "
          f"{'post_mean':>10} {'E_rem':>7} {'Pred':>7} {'±Std':>7} "
          f"{'Base':>7} {'Ratio':>7}")
    print("-" * 110)

    for elapsed, k_obs, label in scenarios:
        tau_last = tau_now - elapsed
        if tau_last < 0:
            continue  # Can't have last tweet before day start
        if elapsed >= config.impulse_cutoff_minutes:
            # At cutoff, impulse is disabled
            print(f"{label:<30} {elapsed:>6}m  {'—':>7} {'—':>5} "
                  f"{'DISABLED':>10} {'—':>7} {'—':>7} {'—':>7} "
                  f"{'—':>7} {'—':>7}")
            continue

        r = predict_impulse(tau_now, tau_last, k_obs, rate_curve, config)
        ratio = r["pred_mean"] / r["base_rate_remaining"] if r["base_rate_remaining"] > 0.01 else float('inf')
        print(f"{label:<30} {elapsed:>6}m  {r['E_obs']:>7.2f} {k_obs:>5} "
              f"{r['posterior_mean']:>10.3f} {r['E_remaining']:>7.1f} "
              f"{r['pred_mean']:>7.1f} {r['pred_std']:>7.1f} "
              f"{r['base_rate_remaining']:>7.1f} {ratio:>7.2f}")


# --- Summary: how the observation window works ---
print()
print("=" * 70)
print("HOW THE BACKWARD OBSERVATION WINDOW WORKS")
print("=" * 70)
print("""
The model looks backward from τ_now to τ_last (the most recent tweet).
This is NOT a fixed lookback — it's the gap since the last tweet.

  Observation window = τ_now - τ_last  (0 to 180 min)

If elapsed >= 180 min (cutoff): impulse disabled, pure bucket sampling.
If no tweets today: impulse disabled.

Within the window:
  E_obs = ∫ rate_curve(τ) dτ from τ_last to τ_now  (expected tweets at base rate)
  k_obs = actual tweet count in (τ_last, τ_now)     (what really happened)

Posterior rate multiplier:
  r_posterior = (α₀ + k_obs) / (β₀ + E_obs)  where α₀=β₀=1.0

  - If k_obs >> E_obs: r >> 1 (burst detected, predict more)
  - If k_obs << E_obs: r << 1 (silence detected, predict fewer)
  - If k_obs ≈ E_obs: r ≈ 1 (normal, predict base rate)

Predicted tweets in remaining impulse window [τ_now, τ_last+180]:
  E[remaining] = r_posterior × ∫ rate_curve(τ) dτ from τ_now to impulse_end
""")
