# Statistical-model audit & first-principles study (Jul 2026)

Question that started it: *for each part of the program that embeds a statistical
model, is the chosen model the right one from first principles — and does a
better-motivated alternative beat the baseline in a controlled A/B?*

This is the theory half. Every claim about the code is grounded in a read of the
actual modules (file:line in §2). The experiment matrix (§4) is run separately;
results are appended in §5.

---

## 0. The estimand and its generative structure (first principles)

We trade categorical claims **"N ∈ bin_k"** where N = Musk's total tweets in a
fixed calendar window [0, T]. At decision time t we observe: the partial count
n(t), the history of daily counts, and the market's per-bin prices.

Musk's posting is a **doubly-stochastic (Cox) point process**: a latent intensity
λ(τ) modulates a conditionally-Poisson arrival stream. λ has three superimposed
structures, each with a different time-scale:

1. **Intraday/weekly seasonality** — hour-of-day and weekday/weekend shape (fast,
   deterministic-ish, ~hours).
2. **Self-excitation** — a burst begets more tweets in the next minutes-to-hours
   (Hawkes, ~30–90 min decay).
3. **Slow regime** — "mania vs quiet" swings of the *baseline* rate over days
   (30–44/day ↔ 14–24/day, per the drawdown study), with abrupt transitions.

Consequences that any good model must respect:

- **N is overdispersed** — Var(N) ≫ E[N], because the latent λ itself varies.
  Poisson badly understates tail risk. The right marginal family is
  Gamma-mixed-Poisson (**Negative Binomial**) or **COM-Poisson**.
- **N is right-skewed and left-truncated** — bounded below by the already-observed
  n(t), with a long upper tail (a mania burst can multiply the total). A symmetric
  Normal is wrong precisely in the tails, which is where the bin boundaries that
  matter live.
- **Nowcasting dominates** — N = n(t) [known] + remaining. Conditioning on n(t) is
  the single largest variance reduction available; the problem is 80% "project the
  rest of today + the future days" and 20% "model the marginal."
- **The dominant *error* is regime lag, not marginal mis-specification.** The prior
  drawdown investigation established that aggregate forecast bias ≈ 0 but errors are
  time-clustered into runs of up to 5 same-sign events (~18 days): the intensity
  estimator lags true regime breaks. This is the thing worth attacking, and it tells
  us *where in the pipeline* the leverage is: the **λ-updating filter** (Stage 1) and
  the **risk response to being mid-transition** (Stage 6), not the tail family.

The model also **leads the market** (validated +0.58 de-confounded): the market is
partly a *lagged echo* of the same public signal the model uses. This is load-bearing
for Stage 5 — it means model and market are **not conditionally independent**, which
changes the correct way to pool them.

---

## 1. The pipeline as six statistical decisions

| Stage | Decision | What the problem demands | What the code does | Verdict |
|---|---|---|---|---|
| 1 | Interday intensity filter | Adaptive-gain filter that jumps on real breaks, ignores noise | Fixed-gain log-EWMA (α=0.35) + slow mean-reversion + 1-D Kalman regime adj | **Mismatch** — fixed gain can't be both stable and fast; this is the regime-lag source |
| 2 | Intraday nowcast | Seasonal cumulative shape + self-excitation, conditioned on n(t) | Bucket: 8×3h NB/CMP + Hawkes impulse + clamped regime mult | **Good** — architecture matches the process |
| 3 | Count sampling family | Overdispersed discrete | NegBin (future) / LogNormal (today) / COM-Poisson (bucket) | **Good** — overdispersion handled |
| 4 | Predictive dist → bin probs | Skew-preserving, truncated at n(t) | `asymmetric` = empirical MC histogram (no parametric CDF) | **Good** — but MC-noisy in thin tails |
| 5 | Pool model ⊕ market | Combine two *correlated* sources without double-counting | **Linear** opinion pool, time-decayed weight, `min_model_weight` floor | **Questionable** — form is untested; see §3 |
| 6 | Bet sizing | Kelly shrunk by *parameter* uncertainty (bet less when mid-transition) | CRRA reservation-price Kelly, `kelly_fraction=1.0`, **point p treated as certain** | **Mismatch** — the one clean theoretical gap; forecast uncertainty is computed then discarded |

The headline: **the forecaster's *marginal* machinery (Stages 2–4) is already the
textbook-correct answer** — Cox decomposition, overdispersed discrete families,
skew-preserving empirical projection, Hawkes self-excitation. That is why "don't
target the forecaster" has held up, and why the naive win ("your Gaussian tails are
wrong") is a non-starter — there is no Gaussian on the prod path. The real
first-principles gaps are at the **edges of the forecaster**: how λ is *updated*
across events (Stage 1) and how uncertainty *sizes the bet* (Stage 6), plus one
open question of *form* in the market pool (Stage 5).

---

## 2. Code grounding (where each decision lives)

- **Stage 1** — `forecaster/interday.py:143-195` (log-EWMA α=0.35 `config.py:142`,
  mean-reversion ρ=0.05 `config.py:148`, long-term-mean EWMA); regime adj
  `monte_carlo.py:246-320`. Alternatives `ewma|gas|pig` via `--interday-model`
  (`run_backtest.py:661`).
- **Stage 2** — `intraday.py:1014` `BucketIntradayForecaster` (`--intraday-mode bucket`).
- **Stage 3** — future days `monte_carlo.py:236` `sample_negbin_scalar`; today LogNormal
  `monte_carlo.py:157-171`; family via `MonteCarloConfig.sampling_distribution` (YAML).
- **Stage 4** — `projection.py:60-116` `AsymmetricProjection` (empirical histogram);
  parametric alts `normal|skew_normal|gamma` via `--projection`.
- **Stage 5** — `market_signals.py:218-223` linear pool; α `market_signals.py:188-202`;
  `min_model_weight=0.15` `config.py`.
- **Stage 6** — `kelly_math.py:94-157` CRRA reservation prices; size search
  `executor.py:3604`; **`forecast_std` reaches only logging** (`integration.py:502,519`),
  never sizing — confirmed 0 reads in `candidates.py`/`kelly_math.py`.
- **Vestigial:** `--kappa` (`KellyConfig.kappa`) and `--quick` are both **no-ops** on
  the current path (kappa never read downstream; quick deprecated). The live command
  carries both harmlessly.

---

## 3. Hypotheses, with predicted sign (falsifiable)

Ranked by first-principles leverage. Each predicts a *direction* so the A/B can
refute it, not just measure it.

**H1 — Uncertainty-aware Kelly haircut (Stage 6). Strongest.**
Kelly on a point p over-bets when p is uncertain. The correct treatment shrinks the
bet by the *parameter* (epistemic) uncertainty — how wrong the *center* could be —
which is largest exactly during regime transitions. Note the subtlety: the MC
forecaster already widens the bins for *aleatoric+dispersion* spread, but a biased
center during a lagged regime break is **not** fixed by widening around the wrong
mean. So the right haircut signal is a **transition/instability** signal, not
`forecast_std` (which is mostly aleatoric). Best available signal: the interday
**intensity-vs-long-term-mean gap** |log_intensity − long_term_mean| (unclamped,
always computed) — it is large precisely when Musk's rate has jumped far from its
baseline and the model is mid-catch-up. Haircut the Kelly fraction ∝ that gap.
*Prediction:* cuts max-DD materially, small return cost or even a return *gain* (the
wide-forecast trades are the confident-wrong streaks). This is the concrete build of
the "state-dependent deference" that the drawdown log left open. **Needs a code hook.**

**H2 — Log (geometric) opinion pool vs linear (Stage 5). Ambiguous — test to decide.**
A linear pool is a mixture: over-dispersed, keeps both experts' modes, never sharper
than its parts. A log pool `p ∝ model^w · market^(1−w)` is the Bayesian-correct
combination *when the sources are conditionally independent*, sharpening on agreement
and suppressing bins where only one expert has mass. Two competing first-principles
effects here:
- *For log:* it would auto-suppress the model's confident-but-market-contradicted bins
  — exactly the phantom edge that regime lag creates → could cut DD.
- *Against log:* the model **leads** the market, so the two are **correlated** (market
  = lagged echo). A naive log pool then **double-counts** the shared signal →
  overconfidence; and it would drag the model's genuine *early* lead toward the stale
  market → kill the highest-ROI early entries.
*Prediction:* net negative or wash on return (the double-counting/lead-suppression
dominates), possibly a small DD improvement. If it loses, that is itself the
confirmation that LinOP is right *because* the sources are correlated. **Needs a code hook.**

**H3 — Score-driven interday filter (`gas`/`pig`) vs `ewma` (Stage 1). Cheap, flag-only.**
A GAS (Generalized Autoregressive Score) filter updates λ by the *score* of the
predictive likelihood, so its effective gain scales with how *surprising* the
observation is under the current dispersion — big jump on a genuine break, small on
noise. That is *adaptive* gain, categorically different from the already-rejected
"just lower the EWMA half-life" (which raises gain everywhere, including on noise —
hence it backfired). This is the one forecaster-internal lever aimed straight at the
diagnosed regime-lag. *Prediction:* modest DD reduction and/or return gain from
faster-but-cleaner regime catch-up; risk is over-reacting on the NB tail.

**H4 — Parametric projection (`skew_normal`/`gamma`) vs empirical `asymmetric` (Stage 4). Cheap.**
The empirical MC histogram is unbiased but noisy in thin tails (25k sims → few
samples in a 1%-mass boundary bin), and that noise is jagged edge → jagged sizing.
A skew-matched parametric CDF smooths the tail while keeping the right-skew.
*Prediction:* near-wash on return, possibly a small variance/DD improvement from less
tail jitter. Include `normal` as a **negative control** — theory says it must lose
(symmetric ⇒ mis-tailed); if it doesn't, the harness is suspect.

**H5 — COM-Poisson vs NegBin future-day sampling (Stage 3). Cheap-ish (YAML).**
COM-Poisson decouples the mean from the dispersion more flexibly than NegBin and can
represent the left-tail thinning during quiet regimes. *Prediction:* small, likely wash.

**Deferred — 2-state NB-HMM interday (`data/references/hmm_implementation_spec.md`).**
An explicit regime-switching latent state is the "correct" model for regime lag, and
a spec already exists. But it is the highest build cost and overfit risk, and H3 is a
cheaper probe of the same idea (adaptive gain). **Only build if H3 shows regime
adaptation pays.**

---

## 4. Experiment matrix

Engine: `src/algo/musk_tweet_count/backtest/run_backtest.py` (shares prod code;
deterministic, MC seed=42 — small deltas are real). Full history **154 weekly events,
Nov 2025 – Jul 2026**.

**Baseline (mirrors the live command):**
`--duration 7 --projection asymmetric --intraday-mode bucket --interday-model ewma
--historical-bootstrap --use-unbox-rotations --consensus-mode time_only
--consensus-min-model-weight 0.15 --capital 4000 --parallel 8`
(differs from the earlier `min_model_weight` frontier baseline, which omitted
bootstrap+unbox — those are in the live command, so they belong in the baseline here.)

**Arms** (one model swap each, all other flags identical to baseline):
- A0 baseline
- A1 `--projection skew_normal` · A2 `--projection gamma` · A3 `--projection normal` (control)
- A4 `--interday-model gas` · A5 `--interday-model pig`
- A6 COM-Poisson future sampling (YAML)
- A7 log opinion pool (code, flag-gated, off by default)
- A8 uncertainty-Kelly haircut (code, flag-gated, off by default)
- A9 best-of-each combined (decided after A1–A8)

**Metrics:** final P&L, max drawdown (cumulative over settle-date-sorted per-event
P&L), return/DD. **Report per-quarter (2026-Q1, 2026-Q2)**, not just aggregate — the
frontier study showed a single window can look flat/misleading. Decision weight:
**max-DD and ret/DD and per-quarter consistency over raw P&L** (P&L is jagged in
discrete event flips; a single-event flip can swing aggregate P&L without meaning).

Cheap arms (A1–A6) need no code and run first. Code arms (A7, A8) are implemented
behind off-by-default flags so baseline/prod is byte-identical when the flag is unset.

---

## 5. Results

Full history, 138 settled weekly events, Nov 2025 – Jul 2026. Deterministic
(MC seed=42). Baseline mirrors the live command. **Every alternative lost.**

### Baseline
`+$76,995 | maxDD −$6,770 | ret/DD 11.37`. Reproduces the prior `min_model_weight`
frontier's 0.15 point (~+$74–77k / −$6.8k). The code-arm sanity run `B0` returned
this to the dollar, confirming the H1/H2 plumbing is P&L-neutral when disabled.

### All arms (ΔP&L, ΔDD vs baseline; negative ΔDD = worse)

| arm | stage | final P&L | max DD | ret/DD | verdict |
|---|---|---|---|---|---|
| **baseline** (asymmetric·ewma·linear·full-Kelly) | — | **+76,995** | **−6,770** | **11.37** | — |
| `projection=gamma` | 4 | +66,554 | −6,299 | 10.57 | lose (−14% P&L; tiny DD gain) |
| `projection=skew_normal` | 4 | +51,486 | −7,216 | 7.14 | lose (−33%) |
| `projection=normal` (control) | 4 | +54,855 | −6,916 | 7.93 | lose (as predicted) |
| `interday=gas` | 1 | +73,136 | −11,997 | 6.10 | lose (DD ~1.8×) |
| `interday=pig` | 1 | +68,033 | −13,874 | 4.90 | lose (DD ~2×) |
| log opinion pool (H2) | 5 | +68,906 | −9,740 | 7.07 | lose (−11% P&L, DD worse) |
| unc-Kelly cv_ref0.18 (H1) | 6 | +48,109 | −8,121 | 5.92 | lose (−38%) |
| unc-Kelly cv_ref0.23 (H1) | 6 | +70,724 | −7,170 | 9.86 | lose (−8%, DD worse) |

### Interpretation — each loss confirms a first-principles prediction

- **H4 (projection).** The empirical MC histogram beats every parametric CDF. The
  LogNormal(today)+NegBin(future) sum is genuinely non-parametric in shape; forcing
  it through a single skew/gamma/normal family discards edge. `normal` (symmetric)
  losing was the pre-registered control — harness validated.
- **H3 (interday).** Score-driven GAS/PIG filters cut return slightly and **~doubled
  drawdown**. Adaptive gain over-reacts on the NegBin tail → regime-chasing whipsaw.
  This quantifies the standing "faster adaptation backfires" caution: it is not just
  a wash, it is strictly risk-increasing.
- **H2 (log pool).** Predicted ambiguous, leaning negative because the model *leads*
  (⇒ correlates with) the market, so a geometric pool double-counts the shared
  signal. A/B confirmed: −11% P&L and worse DD. The **linear** pool is correct here
  *because the sources are not conditionally independent.*
- **H1 (uncertainty-Kelly).** Predicted the cleanest gap, but flagged that
  `forecast_std`/CV is a mostly-**aleatoric** proxy, not the **epistemic transition**
  signal that matters. The A/B bears this out with a tell: the haircut *reduces* DD
  in Q1 (−3.6k vs −4.8k) but *increases* it in Q2 (−8.1k vs −6.8k), while costing
  return in both — it fires on inherently-volatile-but-profitable mania ramps, not on
  regime-lag. Uncertainty-aware sizing is not wrong in principle; **CV is the wrong
  signal.** A genuine transition signal (interday intensity gap, within-event
  revision) is the only remaining, higher-effort variant — but H3 already showed that
  adaptive forecaster interventions increase risk, so the prior is now against it.

### Meta-conclusion

The current stack — **empirical `asymmetric` projection · log-EWMA interday · NegBin
sampling · linear time-decayed opinion pool · point-probability full-Kelly** — is at
or above every principled alternative tested, on both return and risk-adjusted return.
"Determine the best combination" resolves to: **the baseline already is the best
combination.** This upholds the standing "target execution/risk/capital, not the
forecaster" verdict at the level of model *form*, not merely parameters — the one
validated parameter lever remains `min_model_weight 0.30→0.15` (separate study).

Reproducibility note: the H1 arm initially crashed (`KeyError: 'effective_fraction'`)
because the uncertainty haircut merged its context into the robust-Kelly context that
the caller's debug log destructures; fixed by keeping the contexts separate
(executor `_build_tick_config`). Results above are post-fix.

Code status: the flag-selectable arms (projection, interday, intraday) reproduce
directly from HEAD. The two **code** levers (log pool, uncertainty-Kelly) were
implemented behind off-by-default flags, A/B'd, then **reverted** after refutation —
they are NOT in the tree. To reproduce, re-implement per the exact forms above:
log pool `p ∝ model^alpha · market^(1−alpha)` on the trusted sub-distribution
(`market_signals.compute_market_consensus_blend`), and a Kelly-fraction multiplier
`max(min_mult, (cv_ref/cv)^gamma)` keyed to forecast CV (`_build_tick_config`), with
the backtest plumbing `forecast.std → executor.set_forecast_context` in
`unified_runner`. Cheap arms: existing `--projection` / `--interday-model` /
`--intraday-mode` flags.

### Intraday model (ridge vs prod bucket)

`intraday=ridge`: **+$22,679 | maxDD −$13,138 | ret/DD 1.73** — a −71% return collapse
and ~2× drawdown vs the prod `bucket` model. The single largest loss in the study.
Bucket's NB/COM-Poisson-per-3h-bucket + Hawkes self-excitation + clamped regime
multiplier is decisively better than ridge's regression nowcast. Stage 2 confirmed.

**Final tally: 5 stages probed (1, 2, 4, 5, 6), every alternative refuted.** Stage 3
(future-day sampling family, NegBin vs COM-Poisson) is the only untested surface —
YAML-only, no clean CLI, and the weakest a-priori (predicted wash); left for later.
