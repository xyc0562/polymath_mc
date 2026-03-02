# Intraday Forecasting And Trading Experiments

Date: 2026-03-02

This note summarizes the recent experiments on Musk tweet-count intraday forecasting and Kelly trading behavior. It focuses on what materially helped, what did not, and what is currently worth keeping.

## Baseline

Important: the event trading rules YAML changed during this work. Older backtest totals are not comparable to the current baseline.

Current aligned baseline command:

```bash
python3 -m src.algo.musk_tweet_count.backtest.run_backtest \
  --start-date 2025-12-15 \
  --projection asymmetric \
  --intraday-mode bucket \
  --trade-verbose \
  --capital-multiplier 1.5 \
  --duration 7
```

Current aligned baseline result:

- Total P&L: `+$94,906.81`
- Trades: `1514`
- Win rate: `48.6%`
- Avg return: `+22.1%`

Dump:

- `data/dumps/backtest_bucket_interday_ewma_duration7_currentrules.txt`

## Things That Did Not Work

### 1. Global recency-weighting of bucket distributions

Idea:

- Weight recent training days more heavily when fitting bucket means and dispersion.

Result:

- Helped some late-Feb events.
- Hurt aggregate backtest badly.

Why it likely failed:

- It changed the entire intraday template, not just the problematic late-window behavior.

Status:

- Reverted.

### 2. Finer late buckets

Idea:

- Keep most of the day coarse and split the final 3 hours into smaller buckets.

Variants tested:

- `7 x 3h + 3 x 1h`
- pooled late-bucket dispersion

Result:

- Better than global recency-weighting.
- Still worse than the original baseline.
- Did not solve the target failure mode cleanly.

Why it likely failed:

- Added noise.
- Did not fix the core issue in how late-day state is inferred.

Status:

- Reverted.

### 3. Overlap/local-regime overlays

Idea:

- Use overlapping windows or local regime overlays on top of the existing bucket + impulse model.

Result:

- Tended to double-count with the existing impulse and regime machinery.
- Worse on the target event in smoke testing.

Status:

- Reverted.

### 4. Sampled latent day-factor

Idea:

- Replace the raw `observed / expected_so_far` runtime scale factor with a sampled shrunk day-level factor.

Result:

- Helped some target late-Feb events.
- Hurt full-sweep backtest materially.

Why it likely failed:

- Reduced conviction too much.
- Flattened probabilities by widening the Monte Carlo distribution.

Status:

- Reverted.

### 5. Deterministic shrunk day-factor

Idea:

- Keep a deterministic same-day factor, but shrink it toward `1.0` instead of using the raw `observed / expected` ratio.

Best variant before the YAML alignment:

- `prior=10`
- `day_factor_max=1.25`

Result after re-aligning to current YAML:

- No-gate version improved the target late-Feb cases.
- But it underperformed the current baseline overall by about `-$10.7k`.

Status:

- Reverted.

### 6. Hard gate / ramped day-factor

Idea:

- Only turn the day-factor on late, or ramp it in.

Best aggregate variant under current YAML:

- ramp from `tau=720` to `tau=1080`

Result:

- Slightly beat the current aligned baseline overall.
- Did not preserve the target late-Feb improvements.
- Gain size was small enough that it may be noise.

Status:

- Reverted.

### 7. Late-window buy-utility ramp

Idea:

- Leave the forecast alone.
- Increase `min_buy_utility` in the last 6 to 3 hours before settlement.

Tested:

- `2x` ramp
- `3x` ramp

Results:

- `2x`: `+$95,614.40`
- `3x`: `+$95,680.96`

Compared with baseline:

- Small aggregate improvement only.
- No change at all on the target events:
  - `Feb 20 - Feb 27`
  - `Feb 23 - Feb 25`

Interpretation:

- This is probably noise or a very weak effect.
- It does not hit the late-forecast failure mode we care about.

Status:

- Code currently exists as an opt-in path, but it is not recommended based on current evidence.

## Things That Did Help, But Not Enough

### 1. Explicit interday model selection

Added explicit support for:

- `ewma`
- `gas`
- `pig`

Result on unified trading backtest:

- `ewma` remained best overall.
- `gas` and `pig` were worse.

Takeaway:

- Interday model choice is not the main bottleneck for the current strategy.

### 2. Deterministic shrinkage of the regime scalar

The deterministic shrunk day-factor was the most interesting intraday idea tested.

What it showed:

- The raw runtime `observed / expected_so_far` scalar is probably too unstable.
- But smoothing or shrinking it inside the current architecture tends to trade off:
  - better behavior on the specific late-Feb cases
  - weaker aggregate backtest performance

Takeaway:

- The instability is real.
- Fixing it cleanly without losing edge is still unsolved.

## Things Worth Keeping

### 1. Regime diagnostics in logs

Added no-behavior-change diagnostics so the live Monte Carlo log can show:

- observed count so far
- expected count so far
- raw regime scalar
- clipped regime scalar
- existing impulse diagnostics

Why this is worth keeping:

- It lets future production logs tell us whether a bad late forecast came from:
  - the regime scalar
  - the impulse multiplier
  - or both

This is the highest-confidence keep from the recent work.

### 2. Keep the current YAML-aligned baseline as the benchmark

Do not compare against older `+$81k`-style baselines anymore.

Use:

- `data/dumps/backtest_bucket_interday_ewma_duration7_currentrules.txt`

as the current reference benchmark unless the event rules or trade rules change again.

## Current Interpretation

The strongest evidence so far is:

- The current model's edge appears to come partly from a blunt but high-conviction same-day signal.
- Making that signal more statistically elegant often reduces edge.
- Simpler trade-side throttles do not seem to address the actual failure mode.
- The most useful next step is better diagnosis from live logs, not more complexity in the forecast model.

## Current Recommendation

1. Keep the current baseline behavior for production.
2. Keep the regime diagnostics in logging.
3. Do not ship the day-factor variants based on current evidence.
4. Do not spend more time on bucket reshaping, overlap windows, or late buy-utility ramps without new evidence from production logs.
5. Collect a fresh production log with the new diagnostics and re-evaluate the late-window behavior using those logs.

## Relevant Dumps

- `data/dumps/backtest_bucket_interday_ewma_duration7_currentrules.txt`
- `data/dumps/backtest_bucket_interday_ewma_dayfactor_prior10_max1p25_currentrules.txt`
- `data/dumps/backtest_bucket_interday_ewma_dayfactor_prior10_max1p25_start900_currentrules.txt`
- `data/dumps/backtest_bucket_interday_ewma_dayfactor_prior10_max1p25_ramp720_1080_currentrules.txt`
- `data/dumps/backtest_bucket_latebuyramp_6h_3h_2x_currentrules.txt`
- `data/dumps/backtest_bucket_latebuyramp_6h_3h_3x_currentrules.txt`
