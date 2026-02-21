"""
Compute Hawkes excitation impulse predictions across various scenarios.
Shows predicted rate multipliers for different times-of-day and recent tweet patterns.
"""
import sys
sys.path.insert(0, "src")

import math
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

# --- Hawkes excitation prediction function ---
def predict_hawkes(tau_now, tweet_taus, rate_curve, config):
    """Compute Hawkes excitation impulse prediction stats."""
    halflife = config.impulse_decay_halflife_minutes
    decay = math.log(2) / halflife
    cutoff = config.impulse_cutoff_minutes

    # Hawkes excitation: each tweet adds a decaying boost
    excitation = 0.0
    for t in tweet_taus:
        if t < tau_now:
            excitation += math.exp(-decay * (tau_now - t))

    # Saturating rate multiplier
    floor = config.impulse_floor
    max_boost = config.impulse_max_boost
    scale = config.impulse_scale
    rate_mult = floor + max_boost * math.tanh(excitation / scale)

    # Prediction window
    impulse_end = int(min(tau_now + cutoff, 1440))
    impulse_end = max(impulse_end, tau_now)

    tau_now_c = max(0, min(tau_now, 1440))
    tau_impulse_end_c = max(0, min(impulse_end, 1440))
    window_len = tau_impulse_end_c - tau_now_c

    E_impulse = 0.0
    if window_len > 0:
        if rate_mult >= 1.0:
            forward_decay = decay
        else:
            forward_decay = math.log(2) / config.impulse_silence_halflife_minutes

        minutes_ahead = np.arange(window_len)
        decay_factors = np.exp(-forward_decay * minutes_ahead)
        future_rates = 1.0 + (rate_mult - 1.0) * decay_factors

        rate_curve_slice = rate_curve[tau_now_c:tau_impulse_end_c]
        E_impulse = float((future_rates * rate_curve_slice).sum())

    base_rate = float(rate_curve[tau_now_c:tau_impulse_end_c].sum())

    return {
        "excitation": excitation,
        "rate_mult": rate_mult,
        "impulse_end": impulse_end,
        "E_impulse": E_impulse,
        "base_rate": base_rate,
        "ratio": E_impulse / base_rate if base_rate > 0.01 else float('inf'),
    }

# --- Scenario table ---
print("=" * 70)
print("HAWKES EXCITATION IMPULSE PREDICTIONS")
print(f"  Halflife={config.impulse_decay_halflife_minutes}min, "
      f"floor={config.impulse_floor}, max_boost={config.impulse_max_boost}, "
      f"scale={config.impulse_scale}")
print(f"  Cutoff={config.impulse_cutoff_minutes}min, "
      f"silence_halflife={config.impulse_silence_halflife_minutes}min")
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

# Scenarios: (description, list of tweet offsets before tau_now)
scenarios = [
    ("No tweets 2h",               []),
    ("1 tweet 30min ago",           [30]),
    ("1 tweet 5min ago",            [5]),
    ("1 tweet 3min ago",            [3]),
    ("2 tweets (5+10min ago)",      [5, 10]),
    ("3 tweets in 10min",           [3, 6, 10]),
    ("5 tweets in 15min",           [2, 5, 8, 11, 15]),
    ("10 tweets in 30min",          [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]),
]

for tau_now, time_label in tau_points:
    print(f"\n--- τ_now={tau_now} ({time_label}) ---")
    print(f"{'Scenario':<30} {'Excit':>7} {'rate_mult':>10} {'E_impulse':>10} "
          f"{'Base':>7} {'Ratio':>7}")
    print("-" * 80)

    for label, offsets in scenarios:
        tweet_taus = [tau_now - offset for offset in offsets if tau_now - offset >= 0]

        r = predict_hawkes(tau_now, tweet_taus, rate_curve, config)
        print(f"{label:<30} {r['excitation']:>7.2f} {r['rate_mult']:>10.3f} "
              f"{r['E_impulse']:>10.1f} {r['base_rate']:>7.1f} {r['ratio']:>7.2f}")


# --- Summary ---
print()
print("=" * 70)
print("HOW THE HAWKES EXCITATION MODEL WORKS")
print("=" * 70)
print("""
Each recent tweet contributes a decaying excitation:
  excitation = Σ exp(-ln2 * Δ_i / halflife)   for each recent tweet

The rate multiplier uses tanh saturation:
  rate_mult = floor + max_boost * tanh(excitation / scale)

  - No recent tweets: rate_mult → floor (0.4, silence penalty)
  - 1 tweet 30min ago: rate_mult ≈ 1.0 (baseline)
  - Recent burst: rate_mult → floor + max_boost = 3.5 (saturated)

Forward prediction decays toward baseline (1.0):
  future_rate_mult(t) = 1.0 + (rate_mult_now - 1.0) × exp(-decay × t)

  - Boost (rate_mult > 1): decays down to 1.0 (halflife = 30 min)
  - Silence (rate_mult < 1): decays up to 1.0 (halflife = 90 min)
""")
