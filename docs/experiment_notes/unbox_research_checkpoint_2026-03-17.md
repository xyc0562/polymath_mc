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
