# Long-drawdown & forecast-bias investigation (Jul 2026)

Research log. Question that started it: *the account value sometimes drops for
multiple days in a row — can we exit preemptively, and does it hint at a
systematic forecast bias?*

## Data & method

- Logs: `new_idea_{4..9}.log` — six contiguous live runs, **Apr 5 → Jul 10 2026**.
- Realized outcomes: `data/price_history/2026-*/event_info.json` carry the
  resolved winning bin (`is_winner` / `final_price=1.0`). This is ground truth.
- NAV: `[CAPITAL][SYNC_API] baseline_total` (hourly). Noisy marks — required
  Hampel de-glitching (e.g. a fake 32k→14k→32k round-trip from a book outage).
- Forecast: `Forecast: … = <mean> (std: N)` lines, paired with the preceding
  `Event: <name> | Count | Time Left` header.
- Fills: `[FILL CONFIRMED]` with side/bin/price/size/$.

Prereq that unblocked everything: settlement P&L was only recorded for ~27% of
events ("authoritative count unavailable"). Fixed by adding market resolution
as the settlement source (see Infra fix below); realized outcomes then
recovered from the historical records for the rest.

## Findings (what's true)

1. **Not a constant over/under bias — it's regime lag.** Aggregate forecast
   error ≈ 0 (mean +1 to +8 tweets). But the error is strongly *time-clustered*:
   the forecaster lags Musk's posting-rate regime shifts, so it under-forecasts
   during ramp-ups (bets low, count comes high) and over-forecasts during
   slow-downs. Errors are large: stdev ≈ 55, peaking at **+152** (May 15: actual
   290, forecast 138). Musk's rate swung ~30–44/day → ~14–24/day repeatedly.

2. **The long drawdowns are TEMPORAL, not cross-sectional.**
   - Only **~2 weekly events open at once** (median 2, max 3).
   - Co-open events' errors are only **ρ ≈ 0.16** correlated → Kelly treating
     them independent over-sizes by ~1.08× (negligible).
   - But consecutive events (by settle date) form **runs of the same-sign error
     up to 5 long (~18 days)**. The drawdown is a *sequence* of events each
     inheriting the same stale regime bias and losing in turn — not a
     simultaneous pile-up.

3. **The only real-time signal that works is within-event.** By mid-event
   (2–4d lead) the model has revised its forecast and **67% of the early-error
   variance is already resolved**; the revision correlates **+0.85** with the
   eventual early error (sign matches 21/26). Daily counts *are* the surprise
   signal, surfacing as forecast revisions.

4. **Entry edge decays with time-to-settlement.** Per-$ ROI by entry lead:
   ≥5d +18%, 3–5d +31%, 1–3d +3%, <1d +8%. Early entries are the *most*
   profitable (market hasn't repriced yet); most capital ($195k/$239k) goes in
   late at low edge.

5. **Strategy is profitable.** +6.8% ROI on deployed buy volume (hold-to-
   settlement, 5.4k fills), consistent with the +8.5–11.5% on the clean
   settled-event set. The −766 "net loss" seen at first was an artifact of the
   loser-biased 9-event sample that survived the broken settlement logging.

## Levers tested and REJECTED (don't re-run)

| Lever | Result |
|---|---|
| Sharp-drop NAV exit (flat 24h after −X%/12h) | Costs ~10% of return; drops rebound (+1.9% avg fwd-24h after a −4% drop). |
| MA-overlay grind detector (flat while NAV<48h SMA) | Halves DD (−17%→−9%) *frictionless*, but ~12–18 round-trips × 3–5% Musk-book spreads (drag 11–30%) erase it. |
| Grind-brake (pause new buys in downtrend) | Premise real (bot deploys 44% of buys in downtrends) but ~zero correlation of downtrend-deploy with realized loss. |
| Trade the cross-event autocorrelation | +0.46 is a **window-overlap artifact** (events overlap ~50%); the *tradeable* (prior-settled) lag is **−0.16**. |
| Cross-event correlation cap | ~2 co-open, ρ=0.16 → ~1.08× over-leverage. Negligible. |
| Early-σ inflation / entry-maturity scaling | **Backwards**: early entries are the high-ROI ones. Downsizing them cost $3.7k profit to save $0.5k DD. |
| Faster model adaptation (shorter half-life) | Not re-tested here — **backfired in prior backtests** (per YX). Leave alone. |

## Meta-conclusion

The long drawdowns appear **structural**: a temporal regime-lag realized as
streaks of successive events positioned before the model catches up. Every
risk overlay tried either costs more return than the drawdown it removes, or
targets the wrong thing. Current best read: **the drawdowns are the price of
the edge**, and effort is better spent on execution/capital than on an
equity-curve or forecast-bias overlay (consistent with the standing
"target risk/execution, not the forecaster" verdict).

## Model LEADS the market — validated (the edge is real)

Reconstructed model-implied count (from `Forecast:` lines) vs market-implied
count (Σ price×bin-center from `price_history/*/prices.csv`), Apr–Jul:
- `corr(model−market gap, next-12h market move) = +0.59`, and **+0.58 on
  de-confounded windows** (bot idle on the event → exogenous move only). So the
  lead is NOT the bot's own footprint.
- Model revision → next market move: +0.49 (slope 0.35). Gap→move monotonic.
- **Model is closer to the realized count than the market at every lead**
  (44.6 vs 47.9 @5d+, 28.9 vs 31.8 @3–5d, 17.1 vs 19.6 @1–3d, 9.6 vs 11.0 @<1d).
  Edge is real but modest (2–3 counts vs 17–45 error) and shrinks near settle.

Corrected an earlier wrong claim: acting on the revision is NOT "selling into an
already-repriced book" — the market lags the model 79/21, so you move ahead of it.

## Market-deference guard is a DE-LEVER knob, not free edge (backtested)

Engine: `src/algo/musk_tweet_count/backtest/` (`run_backtest.py` → `unified_runner.py`,
same Kelly + `compute_market_consensus_blend` as prod; reads `price_history`).
A/B over Apr 1–Jul 8, weekly events, prod flags (`--projection asymmetric --kappa 1
--intraday-mode bucket --capital 4000 --quick`):

| consensus | final P&L | max DD | return/DD |
|---|---|---|---|
| `time_only` (prod) | +$35,405 | −$7,275 | 4.87 |
| `off`             | +$41,687 | −$8,509 | 4.90 |

`off` makes +18% more return but +17% more drawdown — **return/DD flat ≈4.9**.
The consensus blend just sizes the (real) model edge down; it is a risk-appetite
dial, equivalent to Kelly fraction. No setting of it beats the frontier. The
backtest also reproduces the ~−17–20% drawdown, confirming it's structural.
NOTE: single `--quick` run; 4.87 vs 4.90 is within backtest noise — confirm with
a full run + `min_model_weight` sweep before acting.

## min_model_weight frontier (backtested, deterministic)

Swept `--consensus-min-model-weight` 0→1 (consensus time_only; 1.0 = off),
weekly events Apr–Jul, non-quick. Backtest is deterministic (3 replicates at
0.30 gave identical P&L), so points are exact but P&L is *jagged* in mmw
(discrete event flips).

| mmw | final P&L | max DD | ret/DD |
|---|---|---|---|
| 0.0  | +37,134 | −6,845 | 5.43 |
| 0.15 | +38,204 | −6,845 | **5.58** |
| 0.30 (prod) | +35,405 | −7,275 | 4.87 |
| 0.50 | +39,009 | −7,489 | 5.21 |
| 0.70 | +42,792 | −7,744 | 5.53 |
| 0.85 | +41,182 | −7,946 | 5.18 |
| 1.0 (off) | +41,687 | −8,509 | 4.90 |

- **Max DD is monotonic in mmw** — deference reliably cuts drawdown (−6.8k heavy
  → −8.5k off).
- ret/DD broadly flat (~4.9–5.6) but **mildly favors MORE deference**: off→0.15
  drops P&L 8% while cutting DD 20% (risk falls ~2.5× faster than return).
- **Turning the guard OFF is the WORST for drawdown.** For drawdown-aversion the
  one small backtested win is nudging `--consensus-min-model-weight` 0.30 → ~0.15
  (roughly dominates prod on both axes). NOT turning it off.

### Confirmed on FULL history (138 weekly events, Nov 2025 – Jul 2026)

Re-ran the sweep over the whole available `price_history` (the earlier Apr–Jul
window was an unjustified restriction — the backtest never used the logs). Both
trends are now CLEAN and monotonic (the Apr–Jul-only frontier looked flat only
because of the small sample):

| mmw | final P&L | max DD | ret/DD |
|---|---|---|---|
| 0.0  | +73,388 | −6,845  | 10.72 |
| 0.15 | +74,420 | −6,845  | **10.87** |
| 0.30 (prod) | +72,437 | −7,275 | 9.96 |
| 0.70 | +77,627 | −9,184  | 8.45 |
| 1.0 (off) | +78,719 | −10,872 | 7.24 |

- Max DD monotonic ↑ with model weight; ret/DD monotonic ↓. Off is WORST on both.
- off → 0.15: −5.5% return, −37% drawdown (deference buys DD reduction ~7× cheap).
- Holds out-of-sample: **2026-Q1** (65 ev) off nearly DOUBLES DD (−9,223 vs −4,905)
  for the SAME P&L (+35.2k vs +35.5k). 2026-Q2 same direction, milder. late-2025
  is only 3 events → ignore.
- Verdict: market-deference is a genuine risk-adjusted lever, not a wash.
  `--consensus-min-model-weight 0.30 → 0.15` is the one validated drawdown win.
  Still: one projection (asymmetric), weekly-only, jagged P&L (soft adjacent ranks).

## Still open / untested

- **State-dependent deference**: defer to market more during regime transitions
  (model confidently wrong) and less when calibrated — could beat the flat
  frontier, but needs a transition signal; only the within-event revision
  (+0.85) qualifies, and cross-event transition prediction failed (−0.16).
- Why late entries ($195k at 1–3d/<1d) earn so little edge.

## Backtest engine — usable for all of the above

`python -m src.algo.musk_tweet_count.backtest.run_backtest --start-date … --end-date …
--duration 7 --consensus-mode {off,time_only} --projection asymmetric --kappa 1
--intraday-mode bucket --capital 4000 --quick --parallel 8` reads `data/price_history`,
runs prod Kelly logic, prints per-event + total P&L. ~27s/event (`--quick`).

## Infra fix (landed on `fable-fixes`)

Settlement P&L now falls back to (and prefers) the market's own resolution
(Gamma `outcomePrices`) over the XTracker count, which disappears at
settlement. Closes the ~73% "count unavailable" gap so realized forecast error
is measurable going forward. Commits: `5cff048`, `b709dca`.
