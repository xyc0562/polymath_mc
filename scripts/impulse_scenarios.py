"""
Compute shifted-linear Hawkes excitation predictions across various scenarios.
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

# --- Shifted-linear Hawkes excitation prediction function ---
def predict_hawkes(tau_now, tweet_taus, rate_curve, config):
    """Compute shifted-linear Hawkes excitation prediction stats."""
    halflife = config.impulse_decay_halflife_minutes
    decay = math.log(2) / halflife
    lookback = config.impulse_lookback_minutes

    # Actual excitation: each tweet adds a decaying boost
    excitation = 0.0
    for t in tweet_taus:
        if t < tau_now:
            excitation += math.exp(-decay * (tau_now - t))

    # Expected excitation from rate curve lookback
    expected_excitation = 0.0
    for t in range(1, lookback + 1):
        past_tau = tau_now - t
        if 0 <= past_tau < 1440:
            expected_excitation += rate_curve[past_tau] * math.exp(-decay * t)

    # Shifted-linear mapping
    min_expected = config.impulse_min_expected_excitation
    if expected_excitation < min_expected:
        if excitation > min_expected:
            neutral = min_expected * config.impulse_neutral_fraction
        else:
            neutral = excitation  # forces shifted=0, rate_mult=1.0
    else:
        neutral = expected_excitation * config.impulse_neutral_fraction

    shifted = excitation - neutral
    rate_mult = 1.0 + config.impulse_gain * shifted
    rate_mult = max(config.impulse_floor, min(config.impulse_ceiling, rate_mult))

    # Forward decay (asymmetric)
    if rate_mult >= 1.0:
        forward_decay = decay
    else:
        forward_decay = math.log(2) / config.impulse_silence_halflife_minutes

    # Compute avg_rate_mult for a 180-min bucket from now
    cutoff = config.impulse_cutoff_minutes
    delta = rate_mult - 1.0
    if abs(delta) > 1e-6 and forward_decay > 0:
        integral = (delta / forward_decay) * (1.0 - math.exp(-forward_decay * cutoff))
        avg_mult_current = 1.0 + integral / cutoff
    else:
        avg_mult_current = 1.0

    # Also compute for "next" 180-min bucket (180-360 min from now)
    if abs(delta) > 1e-6 and forward_decay > 0:
        integral_next = (delta / forward_decay) * (
            math.exp(-forward_decay * cutoff) - math.exp(-forward_decay * 2 * cutoff)
        )
        avg_mult_next = 1.0 + integral_next / cutoff
    else:
        avg_mult_next = 1.0

    return {
        "excitation": excitation,
        "expected": expected_excitation,
        "neutral": expected_excitation * config.impulse_neutral_fraction if expected_excitation >= min_expected else (min_expected * config.impulse_neutral_fraction if excitation > min_expected else excitation),
        "shifted": shifted,
        "rate_mult": rate_mult,
        "avg_mult_current": avg_mult_current,
        "avg_mult_next": avg_mult_next,
    }

# --- Scenario table ---
print("=" * 70)
print("SHIFTED-LINEAR HAWKES EXCITATION MODEL")
print(f"  Halflife={config.impulse_decay_halflife_minutes}min, "
      f"neutral_frac={config.impulse_neutral_fraction}, gain={config.impulse_gain}")
print(f"  Floor={config.impulse_floor}, ceiling={config.impulse_ceiling}, "
      f"lookback={config.impulse_lookback_minutes}min")
print(f"  Silence halflife={config.impulse_silence_halflife_minutes}min, "
      f"cutoff={config.impulse_cutoff_minutes}min")
print(f"  Formula: rate_mult = clamp(1.0 + {config.impulse_gain} * (exc - exp*{config.impulse_neutral_fraction}), "
      f"{config.impulse_floor}, {config.impulse_ceiling})")
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
    ("No tweets 6h (overnight)",    []),   # same as no tweets but at overnight taus
    ("1 tweet 30min ago",           [30]),
    ("1 tweet 5min ago",            [5]),
    ("1 tweet 3min ago",            [3]),
    ("2 tweets (5+10min ago)",      [5, 10]),
    ("3 tweets in 10min",           [3, 6, 10]),
    ("5 tweets in 15min",           [2, 5, 8, 11, 15]),
    ("10 tweets in 30min",          [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]),
    ("Steady 1/15min 2h",           list(range(15, 121, 15))),
]

for tau_now, time_label in tau_points:
    print(f"\n--- τ_now={tau_now} ({time_label}) ---")
    print(f"{'Scenario':<30} {'exc':>7} {'exp':>7} {'neutral':>8} {'shifted':>8} "
          f"{'rate_m':>7} {'avg_cur':>8} {'avg_nxt':>8}")
    print("-" * 95)

    for label, offsets in scenarios:
        tweet_taus = [tau_now - offset for offset in offsets if tau_now - offset >= 0]

        r = predict_hawkes(tau_now, tweet_taus, rate_curve, config)
        shifted_sign = "+" if r['shifted'] >= 0 else ""
        print(f"{label:<30} {r['excitation']:>7.2f} {r['expected']:>7.2f} {r['neutral']:>8.2f} "
              f"{shifted_sign}{r['shifted']:>7.2f} {r['rate_mult']:>7.2f} "
              f"{r['avg_mult_current']:>8.2f} {r['avg_mult_next']:>8.2f}")


# --- Summary ---
print()
print("=" * 70)
print("HOW THE SHIFTED-LINEAR EXCITATION MODEL WORKS")
print("=" * 70)
print(f"""
Each recent tweet contributes a decaying excitation:
  excitation = Σ exp(-ln2 * Δ_i / halflife)   for each recent tweet

Expected excitation from rate curve:
  expected = Σ rate_curve[τ-t] * exp(-ln2 * t / halflife)   for t in [1, lookback]

Neutral baseline (30% of expected):
  neutral = expected * {config.impulse_neutral_fraction}

Shifted excitation:
  shifted = excitation - neutral

Rate multiplier (linear with clamp):
  rate_mult = clamp(1.0 + {config.impulse_gain} * shifted, {config.impulse_floor}, {config.impulse_ceiling})

Key properties:
  - Any excitation above neutral → rate_mult > 1.0 (boost)
  - "1 tweet 3min ago" always boosts, even at busy times
  - Overnight silence gets mild suppression (neutral bar is low)
  - Bursts scale naturally up to ceiling

Bucket scaling (replaces Poisson impulse window):
  - Each bucket mean *= avg_rate_mult for that time slice
  - avg_rate_mult decays toward 1.0 over the impulse cutoff window
  - Boost halflife: {config.impulse_decay_halflife_minutes}min, Silence halflife: {config.impulse_silence_halflife_minutes}min
""")
