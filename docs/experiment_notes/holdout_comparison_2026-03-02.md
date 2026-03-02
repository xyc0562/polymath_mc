# Holdout Comparison Plan

## Goal

Evaluate intraday historical-bootstrap variants on data that was not used while tuning:

1. baseline
2. `historical-bootstrap`, `6h -> 3h`, `max_blend=0.18`
3. `historical-bootstrap`, `12h -> 3h`, `max_blend=0.20`

Do not tune more variants on the holdout.

## Clean Holdout Rule

Use:

- `counting_start > 2026-02-27`

This is the clean rule because all tuning used events whose counting windows were already in-sample by then.

If there are not enough such events yet, a provisional diagnostic can use:

- `settlement > 2026-02-27`

but that is weaker and should not be treated as final validation.

## Fresh Runs

Write fresh dumps after the historical-data update.

Baseline:

```bash
python3 -m src.algo.musk_tweet_count.backtest.run_backtest \
  --start-date 2025-12-15 \
  --projection asymmetric \
  --intraday-mode bucket \
  --trade-verbose \
  --capital-multiplier 1.5 \
  --duration 7 \
  > data/dumps/holdout_baseline_current.txt
```

Late-window bootstrap candidate:

```bash
python3 -m src.algo.musk_tweet_count.backtest.run_backtest \
  --start-date 2025-12-15 \
  --projection asymmetric \
  --intraday-mode bucket \
  --trade-verbose \
  --capital-multiplier 1.5 \
  --duration 7 \
  --historical-bootstrap \
  --bootstrap-start-hours 6 \
  --bootstrap-full-hours 3 \
  --bootstrap-max-blend 0.18 \
  > data/dumps/holdout_hist_bs_6h_3h_blend018.txt
```

Earlier-start candidate:

```bash
python3 -m src.algo.musk_tweet_count.backtest.run_backtest \
  --start-date 2025-12-15 \
  --projection asymmetric \
  --intraday-mode bucket \
  --trade-verbose \
  --capital-multiplier 1.5 \
  --duration 7 \
  --historical-bootstrap \
  --bootstrap-start-hours 12 \
  --bootstrap-full-hours 3 \
  --bootstrap-max-blend 0.20 \
  > data/dumps/holdout_hist_bs_12h_3h_blend020.txt
```

## Compare On The Holdout

Clean holdout:

```bash
python3 -m scripts.compare_holdout \
  --baseline data/dumps/holdout_baseline_current.txt \
  --candidate data/dumps/holdout_hist_bs_6h_3h_blend018.txt \
  --candidate data/dumps/holdout_hist_bs_12h_3h_blend020.txt \
  --cutoff-date 2026-02-27 \
  --cutoff-field counting_start
```

Provisional fallback:

```bash
python3 -m scripts.compare_holdout \
  --baseline data/dumps/holdout_baseline_current.txt \
  --candidate data/dumps/holdout_hist_bs_6h_3h_blend018.txt \
  --candidate data/dumps/holdout_hist_bs_12h_3h_blend020.txt \
  --cutoff-date 2026-02-27 \
  --cutoff-field settlement
```

## Promotion Rule

Promote a candidate only if the clean holdout shows:

1. total holdout P&L above baseline
2. median event P&L delta above zero
3. improved events >= regressed events
4. no obvious collapse in short-event subgroup results

If those do not hold, keep baseline.
