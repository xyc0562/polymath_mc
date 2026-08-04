# Execution-gap forensics (Aug 2026)

Question: live lost −$10,261 realized (−62% NAV peak-to-trough) over Jul 21–Aug 1
on events where the backtest — same flags, same price history — makes +$37.6k
(+$14.9k on the settled-9 alone). The known "2–4× optimistic" bias became a sign
flip. Where does the ~$25k gap on the settled events come from?

Data: 1,298 live fills parsed from both instance logs (Jul 10–Aug 2); 1-minute
price bars rescraped for all 17 Jul–Aug events; deterministic backtest replays.

## Loss anatomy (live, 9 settled events, −$10,261)

FIFO reconstruction over all fills:

| component | $ | evidence |
|---|---|---|
| Dead-bin ladders held to expiry | ≈ −8–9k | per-bin tables: Jul 21–28 (count 280) bought YES on EVERY rung 80→339 as the count blew through; sells ≈ $0 in dead bins; `bins_held=8` at settle |
| Flicker churn (round-trips ≤6h) | −2.1k | FIFO realized: <10min −$855, 1–6h −$1,244 |
| Long-hold sells | +5.2k | >6h holds realized POSITIVE (+$4.9k >24h) — selling per se is fine |

The regime was violent: weekly counts 158→200→280→260 (≈40/day sustained = top
of historical range), with 2-day events whipsawing low (Jul 25–27 count 40).
Regime-lag remains the primary *direction* loss — but the backtest faced the
same regime and still profited. The gap is execution:

## Gap leg 1 — friction reality (measured, not assumed)

Every fill priced vs the contemporaneous minute bar (no interpolation):

| zone | live cost (% notional) | backtest charges |
|---|---|---|
| mid prices | −3.51% | 1.5% |
| **tails (<$0.10)** | **−13.22%** | 1.5% |
| high (>$0.90) | −0.29% | 1.5% |
| **volume-weighted** | **−4.20%** | 1.5% |

Dollars: **−$5,547 actual on $132k gross volume vs ~$1,982 modeled → ≈$3.5k
unmodeled drag** in 3 weeks. Markout −3.7% @ +15min (fills adversely timed).
The tail number matters most: the backtest's wins are made by loading eventual
winners at 5–9¢ — at 1.5% assumed friction and infinite depth. Real cost there
is ~13% + book impact. Jul 20–22 alone: −$1,407 execution cost vs −$123 total
event P&L (gross trading was profitable; execution ate it).

## Gap leg 2 — flicker trading (mechanism caught in the act)

Jul 21 16:37:52, event Jul 20–22: count ticks 18→20 (TWO tweets) →
`Prob EMA bypassed: max bin jump 0.3313` → within seconds, full-Kelly
reposition: dump 1,633 sh bin 0, buy 3,739 sh bin 2 @ 0.098. Three minutes
later the nowcast flickers back → sell all bin 2 @ 0.05 (−50% realized in
3 min), re-buy @ 0.11 one minute later. Live eventually abandoned its last
winner-bin shares at 0.53. The bin settled at $1.00; the backtest (hourly
ticks, no intra-hour flicker visible) held ~4,900 sh to 0.957 → +$13.5k.

- EMA jump-bypass (ec67953, deployed Jul 12) fired **1,396×** in 3 weeks
  (p50 jump 0.034, p90 0.103, max 0.36). Near-boundary 2-day events: a single
  tweet arrival moves bins >30pp, and every such move now repositions at full
  size with zero damping. 367 FIFO lot-matches were sold ≤5min after a bypass
  ($9.4k gross).
- Honest quantification: net realized flicker cost is the −$2.1k above — real
  but secondary as a *direct* cost. The larger effect is **winner abandonment**
  (opportunity cost, Jul 20–22 case) and gross-volume inflation: settled-9
  traded **$81,680 gross to hold $11,177 net (7.3× churn, 93 same-bin
  re-buys)**, and every gross dollar pays leg-1 friction.
- Revision of the earlier audit: the bypass was cleared on theory
  ("lag-reducing"). At live cadence the EMA was cheap flicker insurance;
  removing it is a plausibly material *amplifier*. It did not cause the regime
  misses.

## Gap leg 3 — THE BIG ONE: price-data granularity (backtest was lying)

Re-ran the identical config on 1-minute bars instead of hourly (same 3600s
tick interval, ticks are sampled from the data's own timestamps):

| data | settled-9 P&L | all-events P&L |
|---|---|---|
| hourly bars (all prior backtests) | **+$14,884** | +$37.6k |
| minute bars, same ticks | **−$5,496** | +$5.4k (18 ev) |
| LIVE | **−$10,261** | — |

(Audit-trail note: the settled-9 −$5,496 survived two false alarms — a
dataset race that only added the out-of-scope monthly event, and a parser
settle-date collision (monthly "tweets in July" settles the same date as
"Jul 30 - Aug 1"; date-keyed parsing misattributed +$6,126). Final numbers
are event-NAME-matched; engine determinism verified by exact reproduction,
and pre/post-change code verified byte-identical via git-stash A/B.)

Bar *values* agree exactly at shared timestamps — the artifact is density: the
market whipsawed violently between hourly bars (bin 2 of Jul 20–22: 0.043 →
0.0605 → 0.1655 → 0.118 within 90 min; hourly grid never shows 0.1655). The
hourly-grid sim glides between coherent snapshots of a chaotic path, never
trading through the adverse intermediates, and systematically rides winners it
would have been shaken out of. On minute bars the sim's per-event results
finally CORRELATE with live (Jul 17–24: sim −1,756 vs live −1,774; Jul 25–27:
−732 vs −670; Jul 24–31: −1,644 vs −2,217).

**Consequence: every historical hourly-data backtest number (the +$77k full
history, the model-audit and mmw-frontier magnitudes) is inflated to an unknown
degree. Minute-fidelity data must be the standard from now on.**

### Cadence itself is NOT the poison (surprising)

On minute data: tick300 +$10.8k vs tick900 +$5.2k ≈ tick3600 +$5.4k (final
data; the 3600 arm's first −$0.75k read was on partial data) — faster
re-decision does NOT hurt and at 300s helps, at the sim's assumed 1.5%
friction/infinite depth. The model's short-horizon reactions carry real edge
(consistent with the model-leads-market study). The toxin is **cadence × real
friction**: live pays 4.2% (13% tails) per crossing and moves the book, so the
same reactions bleed. Fix the cost of acting, not (only) the acting.

### mmw 0.15 vs 0.30 re-verified on honest data

Matched 17 events, minute bars: 0.15 → −$752 vs 0.30 → −$1,611. The deployed
0.15 remains equal-or-better. Exonerated on both hourly and minute data.

## Waterfall (settled-9, Jul 21–Aug 1)

```
 +$14,884   hourly-data backtest        ← the mirage
 −$20,380   granularity honesty         → −$5,496 minute-data sim
  −$4,765   real execution vs sim       → −$10,261 live
            (closed by calibration: depth cap $150/side + 3¢ slippage
             reproduces live to −$4/event bias — see below)
```

≈80% of the perceived live/backtest "gap" was the simulator flattering itself;
≈20% is real execution drag beyond the sim's assumptions — and that remainder
is fully absorbed by two physical knobs (top-of-book depth + per-fill
slippage). The live loss itself decomposes as regime-lag dead-bin ladders
(≈−$8–9k) + flicker churn (−$2.1k) + friction above model (−$3.7k),
overlapping categories summing over the −$10.3k with winners partially
offsetting.

### Friction calibration status (first pass)

New flag-gated sim knobs (legacy-default, stash-verified neutral):
`--tail-spread/--tail-zone` (zone-dependent spread), `--max-fill-usd`
(top-of-book depth cap). First grid vs live settled-9 ground truth: depth cap
$250 + slippage help most (settled-9 total −$413 vs live −$10,261 — total
close, but per-event MAE still ~$1.3k and bias ~+$1.1k/event optimistic);
tail-spread interacted non-monotonically (wider tail quotes also *block* the
sim's worst tail entries via edge thresholds — not a pure cost knob).
Round 2 (depth × slippage grid), event-name-matched vs live settled-9:

| arm | settled-9 | MAE/event | bias/event |
|---|---|---|---|
| legacy sim | −$5,496 | $825 | +$529 |
| max_fill $250, slip 2¢ | −$8,704 | $642 | +$173 |
| max_fill $150, slip 2¢ | −$8,752 | $546 | +$168 |
| max_fill $250, slip 3¢ | −$10,132 | $606 | +$14 |
| **max_fill $150, slip 3¢ (D4 = adopted)** | **−$10,293** | **$590** | **−$4** |
| LIVE (target) | **−$10,261** | — | — |

**D4 (`--max-fill-usd 150 --slippage 0.03`) reproduces live almost exactly**:
aggregate within $32 of the real loss, per-event MAE $590, bias −$4/event.
Two physical knobs close the entire execution gap. Caveats: (a) calibrated on
the same 9 events it is scored on — must be validated out-of-sample on full
history (minute rescrape in progress); (b) the 3¢ slippage overshoots the
measured per-fill cost to absorb unmodeled impact/adverse timing — right for
RANKING variants, unproven for absolute P&L in other regimes.

### First strategy question under the calibrated sim: required_roi

(all arms = D4 friction; live ran required_roi=0)

| required_roi | settled-9 |
|---|---|
| 0.00 (live) | −$10,293 |
| 0.03 | −$9,819 |
| 0.05 | −$9,905 |
| 0.08 | −$8,026 |

**Modest, not a rescue**: 0.08 saves ~$2.3k (~22% of the loss), 0.03–0.05
~5%. The bleeding trades were NOT marginal-edge trades — the regime-lag
ladders carried large *perceived* edge (the model was confidently wrong), and
an ROI floor cannot block those. Edge-threshold levers treat friction bleed,
not regime misses. Full-history validation needed before recommending 0.08
(it may cost real profit in normal regimes).

## Full-history validation (155 events, Nov 2025 – Aug 2026, minute bars)

| arm | total | maxDD | Q1 | Q2 | Jul+ |
|---|---|---|---|---|---|
| hourly-data claim (old) | ≈ +$77k | −$6.8k | +$35k | +$38k | — |
| FH_legacy (minute, 1.5% friction) | **+$53,786** | −$7.9k | +$30.5k | +$15.7k | +$8.2k |
| FH_D4 (July-calibrated friction) | **−$44,153** | monotone | −$9.0k | −$25.0k | −$6.4k |

- Honest prices alone cut the historical edge ~30% (77k → 54k).
- **D4 FAILS out-of-sample**: it reproduces July but destroys the calm months
  where live was genuinely profitable (June +32%/mo real). July-fitted
  friction is a STRESS calibration, not a universal one.

### Why: friction is regime-dependent (measured, 9,057 fills Apr–Aug)

| month | overall cost | tail (<$0.10) cost |
|---|---|---|
| 2026-04 | 1.80% | 8.7% |
| 2026-05 | 2.56% | 7.5% |
| 2026-06 | 3.36% | 11.5% |
| 2026-07 | 3.85–4.2% | 11.6% |

Execution cost DOUBLED Apr→Jul (rising churn/aggressiveness + chaotic regime
forcing crossings at bad moments). April ≈ the sim's 1.5% assumption; July
2.5× it. Tails were never cheap in ANY month — the binding tail-realism knob
is depth (max_fill_usd), not spread. Reconciles: spring live profits (edge >
cost), July bleed (cost spike × regime miss), D4's out-of-sample failure.

**Sim policy going forward: bracket, don't point-estimate.**

| world | total (9 mo) | maxDD | Q1 | Q2 | Jul+ |
|---|---|---|---|---|---|
| FH_legacy (calm bound) | +$53,786 | −$7.9k | +$30.5k | +$15.7k | +$8.2k |
| FH_MID (avg friction) | **+$4,667** | −$11.7k | +$9.8k | −$6.6k | +$2.3k |
| FH_D4 (chaos bound) | −$44,153 | −$44.2k | −$9.0k | −$25.0k | −$6.4k |

The single-world average (FH_MID ≈ breakeven) is itself misleading: live
ground truth SWITCHES worlds. Live Apr–Jul10 realized ≈ +$16k — close to
FH_legacy's Q2, not MID's; live Jul 21–Aug 1 = −$10.3k — close to FH_D4's
Jul+. In calm regimes real execution ≈ the optimistic sim; in chaos it ≈ the
stress sim. Forward expectation = a regime-frequency-weighted mix of the
bounds, and both the friction trend (1.8%→4.2% Apr→Jul) and Musk's rate
volatility have been pushing the mix toward the chaos world.

Bottom line for the strategy: the edge is real but thin and regime-gated —
roughly "earn like FH_legacy in quiet months, pay like FH_D4 in chaotic ones,
at current sizing". The levers that move the whole bracket up are execution-
cost levers (passive entries, churn limits, tail-depth respect); the lever
that bounds the chaos losses is sizing.

## Implications / actions

1. **Sim honesty first**: minute-fidelity price data + friction calibrated to
   measured values (mid ≈3.5%, tails ≈13%, zone-dependent) before ANY further
   parameter decision. Re-validate standing conclusions that matter.
2. **required_roi = 0 today.** With measured 4.2% real friction, a 4–5% ROI
   floor would have blocked most bleeding trades. Cheapest candidate lever —
   validate under the calibrated sim.
3. **Rate-limit full-size repositioning in tail books** (<$0.10): measured 13%
   crossing cost exceeds any plausible per-flip edge; EMA jump-bypass should be
   zone-aware or persistence-gated there (bypass fired 1,396× in 3 weeks).
4. **Passive entries (maker) target exactly the 4.2–13% crossing cost** — the
   maker study graduates from experiment to strategic fix.
5. Sizing: fixed $4k/event on a shrinking NAV = 100%+ deployment at the trough;
   pool top-up on restore (ec67953) re-levers on restart. Scale to NAV.

## Verdict on "did our post-May-22 changes cause the loss?"

No change caused the regime misses (the direction loss). mmw 0.15: exonerated
twice. Maker: shadow, 0 real orders. Accounting: reporting only. Two changes
made the loss somewhat WORSE than it had to be: the EMA jump-bypass (ec67953)
removed flicker damping and amplified churn/winner-abandonment at live cadence
(−$2.1k realized churn + volume inflation; revised from the earlier theory-only
clearing), and the pool top-up restored full $4k allocations into the drawdown
at the Jul 21 restart. The dominant causes — a near-record posting-rate ramp
(280/wk) the forecaster lagged, real execution friction 3–9× the modeled
value, and over-deployment relative to NAV — predate May 22.

