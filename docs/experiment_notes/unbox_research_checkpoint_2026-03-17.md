## Unbox Research Checkpoint (2026-03-17)

Branch: `bucket-modification`

### Current code state
- Base feature: same-bin unbox rotations for boxed inventory.
- Current leading research variant:
  - `use_unbox_rotations = True`
  - `unbox_min_blocked_ticks = 1`
  - `unbox_min_net_utility = 0.006`
  - multi-bin uplift enabled:
    - `unbox_multi_bin_start_count = 2`
    - `unbox_multi_bin_net_utility_step = 0.002`
    - `unbox_multi_bin_net_utility_cap = 0.004`
  - late relaxation enabled:
    - `unbox_late_relax_start_hours_to_settlement = 3.0`
    - `unbox_late_net_utility_relax = 0.002`

### Important implementation notes
- Checked base commit `5102a2a` (`Reprice Kelly sizing from live orderbooks`) before layering more unbox logic.
- Conclusion:
  - repriced ordinary sells do not conflict with unbox thresholds directly
  - unbox sells were intentionally left as full exits, not peak-sized partial exits, because unboxing must fully unblock the opposite-side buy
- Fixed unrelated backtest runner crash in `src/algo/musk_tweet_count/backtest/unified_runner.py`:
  - `blend_context["alpha"]` was assumed to exist even when consensus blend returned a structured skipped context

### Relevant validation
- `python3 -m pytest tests/test_market_consensus_blend.py tests/test_kelly_executor_integrity.py -q`
- Result at last run: `66 passed`

### Focused 10-event comparison
Reference logs:
- Baseline:
  - `data/dumps/backtest_5m_baseline_time_only.log`
- Multi-bin focus:
  - `data/dumps/unbox_multibin_focus/`
- Multi-bin + late relaxation focus:
  - `data/dumps/unbox_time_focus/`

Focused 10-event totals:
- baseline: `+$71,361.31`, `583` trades
- multi-bin: `+$83,561.31`, `613` trades
- multi-bin + late relaxation: `+$86,035.36`, `628` trades

Delta:
- late relaxation vs baseline: `+$14,674.05`
- late relaxation vs multi-bin: `+$2,474.05`

### Important caveat
- Two late-relax rows were inferred unchanged from the prior multi-bin run rather than fully re-finished in the quiet reruns:
  - `Dec 26 - Jan 2`
  - `Mar 3 - Mar 10`
- Reason for inference:
  - accepted unboxes in the prior multi-bin logs occur well before the new `3h` late-relax window
  - no evidence yet that late relaxation changes those events

### Event-level takeaways
- Strong improvements from late relaxation vs prior multi-bin:
  - `Mar 10 - Mar 11`: `+576.60`
  - `Jan 12 - Jan 14`: `+581.86`
  - `Feb 2 - Feb 4`: `+1,890.76`
  - `Mar 9 - Mar 11`: `+637.92`
- Regressions vs prior multi-bin:
  - `Mar 5 - Mar 7`: `-828.85`
  - `Feb 23 - Feb 25`: `-194.28`
  - `Jan 26 - Jan 28`: `-113.49`
  - `Jan 8 - Jan 10`: `-76.47`
- `Dec 26 - Jan 2` still not fixed:
  - this remains an early bad unbox, not a late boxed-state issue

### Next investigation target
- Inspect `Mar 5 - Mar 7` in detail.
- This is the clearest new regression introduced by the late-relax schedule.
- Compare:
  - baseline
  - multi-bin
  - multi-bin + late relaxation
- Goal:
  - identify whether late relaxation adds a new low-quality near-settlement unbox
  - or changes the path indirectly by enabling later same-bin flips that alter subsequent inventory

### Ongoing follow-up: `Mar 5 - Mar 7`
- Current finding after direct log comparison:
  - the previous `multi-bin` focus log had one accepted unbox on bin `4` at about `T-8.6h`
  - the current rerun with `late relaxation disabled` still finishes at `+$15,376.08`, `64` trades, matching the current late-relax result exactly
  - therefore the regression is **not** caused by the late-relax window itself
  - the current late-relax log and the current no-late rerun have identical trade lines for this event
- Practical implication:
  - the earlier attribution to late relaxation was too strong
  - the current code path has diverged from the older focused `multi-bin` run for some other reason
- Evidence from logs:
  - old focused multi-bin:
    - `data/dumps/unbox_multibin_focus/2026-03-07_Mar_5_-_Mar_7.multibin.log`
    - `+$16,204.93`, `57` trades
    - one accepted unbox on bin `4`
  - current code, no late relaxation:
    - `data/dumps/unbox_time_focus/2026-03-07_Mar_5_-_Mar_7.multibin_recheck.log`
    - `+$15,376.08`, `64` trades
    - no accepted unboxes
- Working hypothesis:
  - the event path is diverging much earlier than the late-relax window, likely through ordinary trade sizing/selection sensitivity rather than the late unbox floor itself

## Follow-up Update (2026-03-18)

### Operational change
- Added temporary CLI exposure for unbox controls in:
  - `src/algo/musk_tweet_count/backtest/run_backtest.py`
  - `src/algo/musk_tweet_count/forecaster/run_multi_event.py`
- Goal:
  - exercise the existing unbox path directly in normal replay commands
  - remove ambiguity about whether late boxed events were failing because the path was disabled vs. because thresholds were too strict

### Seeded replay: `Mar 10 - Mar 17`
- Used the live boxed snapshot from `2026-03-17 12:06:09 UTC` in `data/dumps/new_idea.log`.
- Seeded replay log:
  - `data/dumps/backtest_mar10_mar17_seeded_unbox.log`
- Main finding:
  - the event was immediately recognized as boxed on bin `13`
  - earliest rejection reason was `no_opposite_buy_candidate`
  - once the opposite buy candidate existed, the next blocker was `net_package_utility_below_min`
  - the current research-leading variant eventually accepted an unbox at about `T-3.41h`
- This answered the earlier uncertainty:
  - the event was not failing because it was never eligible
  - it was eligible, but the package economics were too weak until later

### Seeded calibration result
- Tested a small seeded grid around the live boxed snapshot.
- Key negative finding:
  - changing only the late-relax schedule while keeping `unbox_min_net_utility = 0.006` did **not** move the first accepted unbox
  - the effective floor at `T-3.57h` stayed well above the first plausible earlier package net (`0.002630`)
- First variant that materially changed the event:
  - `use_unbox_rotations = True`
  - `unbox_min_blocked_ticks = 1`
  - `unbox_min_net_utility = 0.003`
  - `unbox_late_relax_start_hours_to_settlement = 5.0`
  - `unbox_late_net_utility_relax = 0.002`
  - multi-bin uplift unchanged
  - repeat uplift unchanged
  - `unbox_turnover_penalty = 0.002`
- Effect on the seeded event:
  - no unbox: `-$8,873.55`
  - current research-leading unbox variant: `-$8,405.44`
  - tuned seeded variant above: `-$8,320.43`
- Interpretation:
  - unboxing helps this live-style trapped state
  - lowering the base unbox floor to `0.003` and opening the relax window to `5h` helps a bit more
  - but the useful boundary is narrow: the first meaningfully earlier accepted package was at about `T-3.57h`, not much earlier

### Churn / broader sanity check
- Ran a small hourly matrix with `--consensus-mode time_only` on:
  - `Feb 2 - Feb 4`
  - `Mar 5 - Mar 7`
  - `Mar 10 - Mar 17`
- Summary file:
  - `data/dumps/unbox_matrix_hourly_summary.txt`
- Results:
  - `Feb 2 - Feb 4`: baseline, current unbox, and tuned unbox were identical; `0` accepted unboxes
  - `Mar 5 - Mar 7`: baseline, current unbox, and tuned unbox were identical; `0` accepted unboxes
  - `Mar 10 - Mar 17`:
    - baseline: `+$11,518.93`, `37` trades
    - current research-leading unbox variant: `+$9,673.01`, `40` trades, `1` accepted unbox
    - tuned seeded variant: `+$9,673.01`, `40` trades, `1` accepted unbox
- Takeaway:
  - the tuned variant did **not** increase churn relative to the current research-leading unbox settings on this small hourly matrix
  - but it also did **not** improve over the current research-leading settings there
  - both unbox variants underperformed baseline on the ordinary `Mar 10 - Mar 17` replay

### Current conclusion
- The strongest pre-2026-03-18 basket result is still the focused 10-event `multi-bin + late relaxation` variant above.
- The new 2026-03-18 seeded calibration result suggests:
  - unbox remains promising as a rescue path for specific live-style trapped boxed states
  - but broadening thresholds further is not ready yet
  - the next likely improvement path is narrower triggering or better opposite-side buy construction, not a blanket lowering of the unbox floor

## Follow-up Update (2026-03-19)

### Combined-unbox start-window sweep
- Ran a focused 7-event hourly sweep keeping the combined package logic intact and changing only:
  - `unbox_start_hours_to_settlement` in `{24h, 12h, 8h, 6h}`
- All runs used:
  - `--consensus-mode time_only`
  - `use_unbox_rotations = True`
  - `unbox_min_blocked_ticks = 1`
  - `unbox_min_net_utility = 0.006`
  - `unbox_late_relax_start_hours_to_settlement = 3.0`
  - `unbox_late_net_utility_relax = 0.002`
- Summary file:
  - `data/dumps/unbox_start_window_20260318_summary.txt`

### Main finding
- `12h` is the best current start-window setting for the combined approach on this basket.
- Totals across the 7-event basket:
  - baseline: `+$42,235.73`, `230` trades, `0` accepted unboxes
  - `24h`: `+$41,232.42`, `243` trades, `4` accepted unboxes
  - `12h`: `+$43,356.78`, `237` trades, `3` accepted unboxes
  - `8h`: `+$42,235.73`, `230` trades, `0` accepted unboxes
  - `6h`: `+$42,235.73`, `230` trades, `0` accepted unboxes

### Interpretation
- The harmful early `Mar 10 - Mar 17` unbox happened at about `T-23.66h` under the `24h` window and disappears completely at `12h`.
- The `12h` window still preserves the more useful mid-late accepted unboxes around `10-12h`, notably on:
  - `Feb 23 - Feb 25`
  - `Mar 10 - Mar 11`
- `8h` and `6h` are too restrictive and suppress all combined unboxes on this basket.
- Practical conclusion:
  - narrowing the eligibility window is a better lever than lowering the combined utility floor further
  - `12h` is a reasonable stopping point and does not look like a knife-edge overfit

### Default-setting decision
- Updated the code/config default for `unbox_start_hours_to_settlement` from `24.0` to `12.0`.
- `use_unbox_rotations` remains off by default.
- So baseline behavior is still unchanged unless unbox is explicitly enabled.

### Broad sanity check with the `12h` window
- Ran the standard multi-event backtest command family with:
  - `--start-date 2025-12-15`
  - `--projection asymmetric`
  - `--intraday-mode bucket`
  - `--capital-multiplier 1.5`
  - `--duration 7`
  - `--historical-bootstrap`
  - `--consensus-mode time_only`
- Comparison file:
  - `data/dumps/unbox_start12_sanity_compare.txt`
- Result:
  - baseline: `+$137,630.21`, `2,059` trades
  - combined unbox with `12h` start window:
    - `+$140,102.58`, `2,084` trades, `12` accepted unboxes
  - delta:
    - `+$2,472.37`
    - `+25` trades
- Interpretation:
  - the `12h` window did not just fix the focused basket; it also improved the broader replay set
  - the changed-event count stayed small (`11/57` events)
  - the most important earlier concern, `Mar 10 - Mar 17`, no longer regressed under the `12h` window
  - there are still some negative changed events (`Jan 19 - Jan 21`, `Jan 9 - Jan 16`, `Jan 12 - Jan 14`), so this is a better default window, not proof that unbox should be on by default
