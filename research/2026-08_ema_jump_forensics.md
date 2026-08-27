# EMA jump forensics: what damping wins, what the bypass wins (Aug 2026)

Transaction-level autopsy behind the dual-path (market-corroboration OR
persistence) jump confirmation in `_apply_prob_ema`. Compares every sold
lot of the two live eras on minute-fidelity price data:

- **MOD era** (EMA jump-bypass live): Jul 21 – Aug 7, `maker_shadow_2/3.log`,
  561 FIFO-matched sold lots.
- **REV era** (47b122d, always-EMA): Aug 7 – Aug 27, `revert.log`,
  756 sold lots (679 on settled events).

Each lot carries: FIFO cost basis and realized P&L, the bin's market mid at
±5/15/30/60/240 min around the sell, the bin's settlement value, and
(MOD) minutes since the nearest `Prob EMA bypassed` firing.

## Headline sell-quality gap (settled events)

| | MOD (bypass) | REV (damped) |
|---|---|---|
| Realized P&L of sells vs cost | +$419 | +$5,309 |
| Sell price vs settlement value | −$4,813 | −$554 |
| Loss-sold shares market-confirmed* | 46% | 57% |
| Winner-bin YES shares sold → foregone | 18,220 sh / $14,272 | 17,070 sh / $9,745 |

*market-confirmed = the bin's mid had itself fallen ≥2¢ in the 30 min
before the sell.

## The four cells (sell price vs settlement)

MOD era, sells within 10 min of a bypass firing:

| Cell | Shares | vs settlement | Verdict |
|---|---|---|---|
| LOSS-sell, bin eventually DEAD | 33,655 | **+$2,057** | fast exits were right |
| LOSS-sell, bin eventually WON | 12,857 | **−$7,012** | the disaster cell |
| GAIN-sell, bin eventually WON | 6,245 | −$2,116 | premature profit-taking |
| GAIN-sell, bin eventually DEAD | 14,499 | −$260 | fine |

REV era (all sells): the disaster cell shrinks ~5× (LOSS+WIN −$1,366 on
10,575 sh) — damping holds through the dips that the bypass sold into.

## The decision-time discriminator: market corroboration

Within MOD-era near-bypass LOSS-sells, the market's own move in the 30 min
before the sell separates the good exits from the disasters:

| | dead bins (good exits) | WINNER bins (disasters) |
|---|---|---|
| avg mid move into the sell (30m) | −8.9¢ | −1.8¢ |
| share falling ≥2¢ | 67% | 24% |
| share flat (±2¢) | 23% | 58% |

Rule replay — "block a jump-triggered loss-sell unless the bin's mid also
fell ≥2¢ in the prior 30m":

- BLOCKED sells: would have avoided **−$8,682** of settlement-value
  destruction (these were the model dumping winners while the book sat
  still).
- ALLOWED sells: kept **+$3,726** of correct fast exits.

## What the bypass genuinely wins (the flip side)

- Dead-bin loss exits: bypass exits captured **+3.4¢/sh** vs settlement;
  the damped version's late exits captured **+0.04¢/sh** (price had
  already collapsed by the time the EMA converged).
- REV-era loss-sells with a detectable prior jump (≥3¢/5m within 90m):
  selling at the jump moment instead of the damped later sell would have
  been **+2.0¢/sh** better (−$355 total). On gain-sells the lag *earned*
  +2.4¢/sh (the V-recovery premium), net damp advantage +$320.

So the bypass's edge is real but small (~$1.5–2k/era) and lives entirely
in market-corroborated moves; its damage (~$9k/era on winner bins) lives
entirely in uncorroborated, model-only moves.

## Design consequence (implemented)

`_apply_prob_ema` dual-path confirmation:

1. **Market corroboration (fast path)**: max-displacement bin's mid moved
   the model's direction by ≥ `prob_ema_jump_market_confirm_move` (2¢)
   within `prob_ema_jump_market_confirm_window_seconds` (30 min) → snap
   immediately. Captures the dead-bin fast-exit edge.
2. **Persistence (slow path)**: displacement from the frozen pre-jump EMA
   stays ≥ threshold for ≥3 ticks spanning ≥120 s → snap. Uncorroborated
   real moves still get priced, ~2 min late.

Everything else (uncorroborated, non-persistent flicker) only ever sees
the damped path. Maker quotes are killed at first detection either way.

## Caveats

- Regime is not fully controlled; however the REV era contained a count
  ramp (~320/wk, Aug 18–25) *worse* than the MOD era's peak (280/wk) and
  still showed the better sell profile.
- "vs settlement" benchmarks assume holding to settlement as the
  counterfactual; both eras also de-risk winners deliberately (GAIN+WIN
  is negative in both). The cross-era *comparison* of the same cell is
  the meaningful signal, not the absolute sign.
- The backtest does not model the EMA path at all, so this logic is
  validated by unit tests + the live pilot, not by backtest.

Analysis artifacts: scratchpad `revert_cmp/` (`sell_forensics.py`,
`{mod,rev}_sell_lots.tsv`, fill/bypass extracts); minute price data in
`data/price_history_minute/` through Aug 27.
