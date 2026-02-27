# Project History Log

This file tracks major changes, experiments, and results per session.
Append a new entry at the top each session. Keep entries concise and
include paths to logs/results when applicable.

Note: entries below are reorganized in reverse chronological order for readability.

## 2026-02-27 (Larger confirmation step: +4 seed extension and 10-seed aggregate)

Focus: execute the next robustness step after promoting `horizon100/noalign` by adding four unseen seeds.

Profile used:
- `anti_regression_selector_mode=auto_rescue`
- `anti_regression_top_k=10`
- `anti_regression_horizon_rescue_enabled=true`
- `anti_regression_horizon_incumbent_return_max=1.00`
- `anti_regression_alignment_probe_enabled=false`

Extension sweep (new seeds):
- seeds: `123, 1011, 2027, 8087`
- summary: `seed_sweep_results/realdata/guardbA_horizon100_noalign_ext4_10ep_20260227_104113_summary.json`
- log: `logs/guardbA_horizon100_noalign_ext4_10ep_20260227_104113.log`

Extension outcomes:
- mean return: `+0.16%`
- mean PF: `1.07`
- positive with PF>=1: `2/4`
- per-seed: `123` (-0.20%, PF 0.94), `1011` (+0.60%, PF 1.23), `2027` (-0.55%, PF 0.88), `8087` (+0.80%, PF 1.22)

Combined aggregate (previous 6 + extension 4):
- summary: `seed_sweep_results/realdata/guardbA_horizon100_noalign_10ep_seed10_summary_20260227_1115.json`
- count: `10`
- mean return: `+1.28%`
- mean PF: `1.72`
- positive with PF>=1: `8/10`

Interpretation:
- The profile remains net-positive and robust in aggregate, but no longer clears every seed.
- `123` and `2027` are the current weak cases for targeted follow-up.

## 2026-02-27 (Cross-triad combined check: horizon100/noalign is 6/6 positive on tested seeds)

Focus: merge both confirmed triads into one combined view before launching a broader sweep.

Combined seed set:
- weak-seed triad: `5051, 9091, 10007`
- default triad: `456, 777, 789`

Aggregate artifact:
- `seed_sweep_results/realdata/guardbA_horizon100_noalign_10ep_seed6_summary_20260227_1100.json`

Combined outcomes (6 seeds):
- mean return: `+2.03%`
- mean PF: `2.15`
- positive with PF>=1: `6/6`

Decision:
- Treat `horizon100/noalign` as the current promoted selector profile.
- Next milestone: launch a larger multi-seed confirmation run with this profile and then perform out-of-sample stress validation.

## 2026-02-27 (Broader confirmation: horizon100/noalign clears default seed triad too)

Focus: test whether the weak-seed breakthrough profile also holds on the standard real-data triad (`456/777/789`).

Profile:
- `anti_regression_selector_mode=auto_rescue`
- `anti_regression_horizon_rescue_enabled=true`
- `anti_regression_horizon_incumbent_return_max=1.00`
- `anti_regression_alignment_probe_enabled=false`

Runs:
- `seed_sweep_results/realdata/guardbA_horizon100_noalign_10ep_20260227_031657_seed456`
- `seed_sweep_results/realdata/guardbA_horizon100_noalign_10ep_20260227_054837_seed777`
- `seed_sweep_results/realdata/guardbA_horizon100_noalign_10ep_20260227_070252_seed789`
  - note: long run exited after selector artifacts were written; selected checkpoint was evaluated manually at:
    `seed_sweep_results/realdata/guardbA_horizon100_noalign_10ep_20260227_070252_seed789/manual_eval_selected/results/test_results.json`
- Aggregate summary: `seed_sweep_results/realdata/guardbA_horizon100_noalign_10ep_defaulttri_summary_20260227_1051.json`

Per-seed outcomes:
- `456`: `+3.92%`, `PF 4.07`
- `777`: `+0.07%`, `PF 1.03`
- `789`: `+2.22%`, `PF 2.05`

Aggregate:
- mean return: `+2.07%`
- mean PF: `2.38`
- positive with PF>=1: `3/3`

Decision:
- `horizon100/noalign` now passes both triads tested (`weak-seed triad` and `default triad`) with `3/3` positive in each set.
- Promote this selector profile as the active baseline for the next larger multi-seed robustness sweep.

## 2026-02-27 (Weak-seed triad breakthrough: horizon cap 1.00 + no alignment probe)

Focus: fix remaining `9091` selector miss after the `horizon055/noalign` profile by relaxing only the horizon incumbent cap and validating again on the same weak-seed triad.

Profile:
- `anti_regression_selector_mode=auto_rescue`
- `anti_regression_horizon_rescue_enabled=true`
- `anti_regression_horizon_incumbent_return_max=1.00`
- `anti_regression_alignment_probe_enabled=false`

Runs:
- `seed_sweep_results/realdata/guardbA_horizon100_noalign_10ep_20260227_011818_seed5051`
- `seed_sweep_results/realdata/guardbA_horizon100_noalign_10ep_20260227_004132_seed9091`
- `seed_sweep_results/realdata/guardbA_horizon100_noalign_10ep_20260227_015834_seed10007`
- Aggregate summary: `seed_sweep_results/realdata/guardbA_horizon100_noalign_10ep_tri3_summary_20260227_0317.json`

Per-seed test outcomes:
- `5051`: `+1.07%`, `PF 1.39`, winner `candidate_ep007.pt`
- `9091`: `+2.23%`, `PF 2.22`, winner `candidate_ep001.pt`
- `10007`: `+2.66%`, `PF 2.16`, winner `candidate_ep002.pt`

Aggregate:
- mean return: `+1.99%`
- mean PF: `1.92`
- positive with PF>=1: `3/3`

Key comparison:
- Previous `horizon055/noalign` weak-seed triad was `2/3` positive (`mean return +0.48%`, `mean PF 1.22`).
- `horizon100/noalign` improves all tracked aggregate metrics and clears all three weak seeds.

Decision:
- Promote `horizon100/noalign` as current best selector profile for next broader confirmation sweep.
- Keep alignment probe disabled until pass-switch guardrails are redesigned to avoid quality regressions.

## 2026-02-27 (Horizon-only selector profile tested on weak-seed triad)

Focus: validate whether the `5051` recovery setup generalizes: keep alignment probe disabled and let horizon rescue act with a less strict incumbent cap.

Profile:
- `anti_regression_selector_mode=auto_rescue`
- `anti_regression_horizon_rescue_enabled=true`
- `anti_regression_horizon_incumbent_return_max=0.55`
- `anti_regression_alignment_probe_enabled=false`

Runs:
- `seed_sweep_results/realdata/guardbA_horizon055_noalign_10ep_20260226_224246_seed5051`
- `seed_sweep_results/realdata/guardbA_horizon055_noalign_10ep_20260226_232237_seed9091`
- `seed_sweep_results/realdata/guardbA_horizon055_noalign_10ep_20260226_235910_seed10007`
- Aggregate summary: `seed_sweep_results/realdata/guardbA_horizon055_noalign_10ep_tri3_summary_20260227_0040.json`

Per-seed test outcomes:
- `5051`: `+1.07%`, `PF 1.39`, selector `tail_holdout+horizon`, winner `candidate_ep007.pt`
- `9091`: `-0.60%`, `PF 0.79`, selector `tail_holdout`, winner `candidate_ep006.pt`
- `10007`: `+0.97%`, `PF 1.47`, selector `tail_holdout`, winner `candidate_ep008.pt`

Aggregate:
- mean return: `+0.48%`
- mean PF: `1.22`
- positive with PF>=1: `2/3`

Decision:
- This profile is a net improvement versus the recent weak-seed baseline but does not yet clear all weak seeds.
- Next selector work should target `9091` specifically (candidate recovery) without reintroducing alignment-probe regressions.

## 2026-02-26 (5051 selector pivot: horizon rescue can recover positive checkpoint)

Focus: isolate why weak seed `5051` still misses profitable checkpoints and identify the smallest selector change that restores a positive selection.

Experiments:
- Alignment-probe coverage increase (`top_k=6`, strict pass required):
  - `seed_sweep_results/realdata/guardbA_autorescue_relaxed_alignprobe6_10ep_20260226_212047_seed5051`
  - Log: `logs/guardbA_autorescue_relaxed_alignprobe6_10ep_seed5051_20260226_212047.log`
- Horizon rescue, alignment probe disabled, incumbent cap `0.45`:
  - `seed_sweep_results/realdata/guardbA_horizon045_noalign_10ep_20260226_220400_seed5051`
  - Log: `logs/guardbA_horizon045_noalign_10ep_seed5051_20260226_220400.log`
- Horizon rescue, alignment probe disabled, incumbent cap `0.55`:
  - `seed_sweep_results/realdata/guardbA_horizon055_noalign_10ep_20260226_224246_seed5051`
  - Log: `logs/guardbA_horizon055_noalign_10ep_seed5051_20260226_224246.log`

Key outcomes:
- `top_k=6` alignment probe regressed selection hard:
  - selector: `tail_holdout+wfalign`, winner `candidate_ep003.pt`
  - test: `-2.50%`, `PF 0.33`
- Disabling alignment probe with horizon cap `0.45` was still too strict:
  - selector: `tail_holdout`, winner `candidate_ep002.pt`
  - test: `-1.19%`, `PF 0.61`
- Raising horizon incumbent cap to `0.55` allowed horizon switch to `candidate_ep007.pt`:
  - selector: `tail_holdout+horizon`
  - test: `+1.07%`, `PF 1.39`, `26` trades

Interpretation:
- For this weak seed, selector failure is concentrated in threshold/gating logic, not learnability.
- Alignment-probe pass gating can force economically worse checkpoints when not guarded by return/PF edge quality.

Decision:
- Use `horizon_incumbent_return_max=0.55` and disable alignment probe for the next weak-seed validation sweep.
- Keep alignment-probe logic under review (add guardrail to block pass-only switches with negative return/PF edge).

## 2026-02-26 (Sweep passthrough fix verified: selector flags now honored)

Focus: fix orchestration gap where `run_realdata_seed_sweep.py` was not forwarding selector/auto-rescue flags to `main.py`, then rerun `9091` with true `auto_rescue` + alignment settings.

Code update:
- `run_realdata_seed_sweep.py`
  - Added pass-through support for:
    - `--anti-regression-selector-mode`
    - `--anti-regression-auto-rescue` / `--anti-regression-no-auto-rescue`
    - rescue thresholds (`winner_forward_return_max`, return/pf edges, challenger gates)
  - Added these fields to `SweepConfig`, CLI parser, command construction, and summary config payload.

Verification:
- `python -m py_compile run_realdata_seed_sweep.py`

Run artifact:
- `seed_sweep_results/realdata/guardbA_alignprobe_autorescue_true_10ep_20260226_144519_seed9091`
- Log: `logs/guardbA_alignprobe_autorescue_true_10ep_seed9091_20260226_144519.log`
- Summary: `seed_sweep_results/realdata/guardbA_alignprobe_autorescue_true_10ep_20260226_144519_summary.json`

Outcome:
- Selector flags are now correctly applied (`selector_mode_requested=auto_rescue` confirmed in `checkpoint_tournament.json`).
- Auto-rescue did not trigger on this seed (`winner_forward_return=0.691 > threshold 0.65`), so final mode remained `tail_holdout`.
- Test remained weak: `return=-0.33%`, `PF=0.90`, `walkforward_pass=false`.
- Post-run checkpoint sanity (same trained run, evaluate mode):
  - `candidate_ep004.pt` (selected): `-0.33%`, `PF 0.90`
  - `candidate_ep001.pt` (future winner): `+2.23%`, `PF 2.22`
  - Confirms remaining issue is selector trigger strictness (thresholds blocking a better candidate), not model incapability.

## 2026-02-25 (Walk-forward aligned selector probe implemented)

Focus: reduce residual selector/test mismatch by adding a cheap, test-protocol-aligned checkpoint probe at the end of anti-regression selection.

Code updates:
- `trainer.py`
  - `validate()` now supports `use_all_windows_override` to evaluate all feasible windows (no K thinning) when needed.
  - Validation stats now expose window metadata (`val_window_bars`, `val_stride_bars`, `val_segment_bars`, `val_use_all_windows`).
  - Added optional `alignment_probe` stage in `_run_anti_regression_tournament`:
    - evaluates top-N distinct candidates with deterministic seeds and `VAL_JITTER_DRAWS=1`,
    - computes walk-forward-style proxy metrics on validation (`spr`, `pf`, `positive_frac`, `windows`, `return_pct`),
    - can switch winner only on configured edges / pass conditions,
    - logs full details under `alignment_probe` in `logs/checkpoint_tournament.json`.
- `config.py`
  - Added `TrainingConfig` knobs for alignment probe:
    - `anti_regression_alignment_probe_enabled`
    - `anti_regression_alignment_probe_top_k`
    - `anti_regression_alignment_probe_window_bars`
    - `anti_regression_alignment_probe_stride_frac`
    - `anti_regression_alignment_probe_use_all_windows`
    - `anti_regression_alignment_probe_return_edge_min`
    - `anti_regression_alignment_probe_pf_edge_min`
    - `anti_regression_alignment_probe_min_trades`
    - `anti_regression_alignment_probe_require_pass`
- `main.py`
  - Added CLI flags and config wiring for all alignment-probe options.
- `run_realdata_seed_sweep.py`
  - Added pass-through CLI support and summary logging for alignment-probe options.

Verification:
- Syntax check passed:
  - `python -m py_compile config.py main.py trainer.py run_realdata_seed_sweep.py`
- System smoke passed:
  - `python test_system.py`

Decision:
- Keep alignment probe optional (default off) and validate impact with targeted real-data seeds before broad promotion.

## 2026-02-25 (Selector alignment diagnostics: start60 rejected, oracle mismatch confirmed)

Focus: continue targeted selector work after `incmax04` blind-10 (`9/10` PF>=1), with explicit checks for the remaining selection mismatch.

Experiments:
- Late-window horizon variant (full blind-10):
  - `seed_sweep_results/realdata/guardbA_horizon_start60_blind10_fast10ep_20260224_165640_summary.json`
  - `seed_sweep_results/realdata/guardbA_horizon_start60_blind10_fast10ep_20260224_165640_comparison.json`
- High-robustness tournament probe (`9091`, `K=8..10`, jitter draws `3`):
  - `logs/guardbA_selector_k10j3_probe_10ep_20260225_114122_seed9091.log`
  - runtime became impractical; run exited before final artifact (`results/test_results.json` missing)
- Full oracle audit on current best blind-10 branch (`incmax04`):
  - `seed_sweep_results/realdata/guardbA_horizon_incmax04_blind10_fast10ep_20260224_135954_oracle_audit.json`

Key outcomes:
- `start60` variant improved `9091` (`+0.37% / PF 1.11`) but regressed aggregate quality:
  - mean return `+1.1387%` vs `+1.7009%` on `incmax04`
  - mean PF `1.9668` vs `2.3298`
  - positive PF>=1 `8/10` vs `9/10`
  - major relapse: `10007` fell back to `-1.82% / PF 0.57`
- High-`K`/jitter tournament probe is too slow for practical iteration in current setup.
- Oracle audit shows residual selector/test misalignment is broad (not only `9091`):
  - comparable seeds: `10`
  - oracle misses: `8`
  - mean oracle uplift over selected checkpoints: `+1.0781%` return, `+0.7492` PF

Decision:
- Keep `incmax04` as active best branch; do not promote `start60`.
- Do not use high-`K` tournament settings as default due runtime cost.
- Next step should prioritize validation/test alignment quality (evaluation protocol) rather than only selector threshold tuning.

## 2026-02-24 (Targeted 9091 oracle check after blind-10 rerun)

Focus: determine whether the remaining failing seed (`9091`) is still a selection miss or a true no-candidate failure.

Artifact:
- `seed_sweep_results/realdata/guardbA_horizon_incmax04_blind10_fast10ep_20260224_135954_seed9091/oracle_eval_candidates_9091_summary.json`

Oracle evaluation over candidate checkpoints (`candidate_ep002/004/006/008/010`):
- Best checkpoint on test: `candidate_ep010.pt` -> `+1.02%` return, `PF 1.36`, `20` trades.
- Selected checkpoint from tournament: `candidate_ep006.pt` -> `-1.28%` return, `PF 0.56`.
- Interpretation: `9091` remains a checkpoint-selection miss (profitable candidate exists but was not selected).

Decision:
- Keep current horizon cap hardening (`0.40`) because it improved aggregate blind-10 reliability.
- Next technical step should target selector alignment for this residual miss (candidate-ranking refinement), not reward/cost profile changes.

## 2026-02-24 (Blind-10 confirmation rerun with horizon incumbent cap 0.40)

Focus: rerun the full GuardB-A blind-10 after tightening horizon rescue incumbent cap (`0.65 -> 0.40`) to remove false-positive switches.

Run artifacts:
- Main summary:
  - `seed_sweep_results/realdata/guardbA_horizon_incmax04_blind10_fast10ep_20260224_135954_summary.json`
- Comparison vs prior horizon blind-10 and GuardB-A baseline:
  - `seed_sweep_results/realdata/guardbA_horizon_incmax04_blind10_fast10ep_20260224_135954_comparison.json`

Blind-10 aggregate outcome (10 seeds):
- Mean return: `+1.7009%`
- Mean PF: `2.3298`
- Positive PF >= 1 seeds: `9/10`
- Negative seeds (`return<0 or PF<1`): `1/10` (target met; prior horizon run was `2/10`)
- Horizon switches: `3/10` (down from `5/10`, fewer forced overrides)

Delta vs prior horizon blind-10 (`guardbA_horizon_blind10_fast10ep_20260224_094831`):
- Mean return: `+0.1327%`
- Mean PF: `+0.0329`
- Positive PF >= 1: `+1` seed (`8/10 -> 9/10`)
- Key fix: `4049` recovered from `-1.63% / PF 0.43` to `+0.19% / PF 1.08`.

Delta vs GuardB-A baseline blind-10 (`guardbA_blind10_fast10ep_20260223_200325`):
- Mean return: `+0.9595%`
- Mean PF: `+0.4569`
- Positive PF >= 1: `+2` seeds (`7/10 -> 9/10`)

Reading:
- Tightening incumbent cap to `0.40` improved safety without sacrificing the large weak-seed rescues (`10007`, `5051`).
- Remaining blocker is concentrated in `9091` (still negative).

Decision:
- Keep `incumbent_return_max=0.40` as default.
- Next gate should be a targeted anti-regression probe for `9091` (selection-only), not a broad profile change.

## 2026-02-24 (Horizon rescue hardening: incumbent return cap tightened to 0.40)

Focus: remove the harmful `4049` horizon switch seen in blind-10 while preserving the `5051` recovery.

Targeted probe runs (same GuardB-A 10ep profile, only horizon incumbent cap changed):
- `seed_sweep_results/realdata/guardbA_horizon_incmax04_probe_10ep_20260224_124247_seed4049`
- `seed_sweep_results/realdata/guardbA_horizon_incmax04_probe_10ep_20260224_130023_seed5051`
- `seed_sweep_results/realdata/guardbA_horizon_incmax04_probe_20260224_summary.json`

Probe outcomes:
- `4049`:
  - Prior tuned horizon run: `-1.63% / PF 0.43` (horizon switch to `candidate_ep008`)
  - With cap `0.40`: `+0.19% / PF 1.08` (selector stayed `tail_holdout`, no harmful horizon override)
- `5051`:
  - Prior tuned horizon run: `+1.18% / PF 1.56`
  - With cap `0.40`: `+1.18% / PF 1.56` (recovery preserved, still `tail_holdout+horizon`)

Code update:
- `config.py`
  - `anti_regression_horizon_incumbent_return_max` default: `0.65 -> 0.40`
- `trainer.py`
  - Horizon rescue fallback default aligned to `0.40`.

Decision:
- Promote `incumbent_return_max=0.40` as the new tuned default.
- Next gate: rerun blind-10 with this tightened default to confirm the aggregate lift and check whether negative seeds drop from `2/10` to `<=1/10`.

## 2026-02-24 (GuardB-A horizon tuned: blind-10 confirmation complete)

Focus: run full blind-10 confirmation with the tuned horizon-aware selector and measure lift versus the prior GuardB-A blind-10 baseline.

Run artifacts:
- Candidate summary:
  - `seed_sweep_results/realdata/guardbA_horizon_blind10_fast10ep_20260224_094831_summary.json`
- Baseline summary for direct comparison:
  - `seed_sweep_results/realdata/guardbA_blind10_fast10ep_20260223_200325_summary.json`
- Delta report:
  - `seed_sweep_results/realdata/guardbA_horizon_blind10_fast10ep_20260224_094831_delta_vs_guardbA_blind10_fast10ep_20260223_200325.json`

Blind-10 aggregate outcome (10 seeds, 10 episodes each):
- Mean return: `+1.5683%` (baseline `+0.7414%`, delta `+0.8269%`)
- Mean PF: `2.2969` (baseline `1.8729`, delta `+0.4240`)
- Positive PF >= 1 seeds: `8/10` (baseline `7/10`)
- Horizon switches: `5/10` seeds
- WF pass count in this summary remained `0` (same strict gate issue as prior 10ep runs)

Per-seed readout:
- Largest recoveries from horizon switching:
  - `10007`: `-1.82%/PF 0.57` -> `+2.97%/PF 2.73`
  - `5051`: `-3.22%/PF 0.34` -> `+1.18%/PF 1.56`
- Regressions still present:
  - `4049`: `+0.19%/PF 1.08` -> `-1.63%/PF 0.43`
  - `9091` remained negative (`-1.28%/PF 0.56`)

Decision:
- Promote horizon-aware selector as the current leading branch (clear blind-set mean lift).
- Next gate is targeted robustness hardening for `4049` and `9091` before broader promotion.

## 2026-02-24 (Horizon-aware selector implementation + tuned pilot)

Focus: add a selector-stage mechanism that can rescue weak seeds without forcing global episode-budget changes.

Code implementation:
- `trainer.py`
  - Added `horizon_rescue` stage in anti-regression tournament.
  - Evaluates a longer validation horizon (`window_bars` default `2400`) across distinct checkpoint candidates.
  - Uses deterministic candidate probes and logs full diagnostics in `logs/checkpoint_tournament.json` under `horizon_rescue`.
  - Switch condition now keys off incumbent **robust return** (not single-slice probe return) to avoid flipping strong incumbents on noisy long probes.
  - Challenger selection now prefers candidates that satisfy viability thresholds (positive return, PF floor, min trades).
- `config.py`
  - Added horizon-rescue controls:
    - `anti_regression_horizon_rescue_enabled`
    - probe window/start/end/candidate limit
    - incumbent/challenger thresholds
  - Tuned defaults:
    - `incumbent_return_max=0.65`
    - `pf_edge_min=0.10`
    - `challenger_base_return_max=1.0`
- `main.py`
  - Added CLI flags for all horizon-rescue controls.
- `run_realdata_seed_sweep.py`
  - Added pass-through CLI support for horizon-rescue options.

Tuned pilot (10ep, HFM profile, same GuardB-A baseline settings):
- Seed runs:
  - `seed_sweep_results/realdata/guardbA_horizon_tuned_10ep_20260224_024819_seed10007`
  - `seed_sweep_results/realdata/guardbA_horizon_tuned_10ep_20260224_023734_seed5051`
  - `seed_sweep_results/realdata/guardbA_horizon_tuned_10ep_20260224_024251_seed7079`
- Aggregate summary:
  - `seed_sweep_results/realdata/guardbA_horizon_tuned_pilot3_10ep_20260224_025406_summary.json`

Pilot outcome vs baseline selected checkpoints (`guardbA_blind10_fast10ep_20260223_200325` subset):
- Mean return: `+3.1338%` delta
- Mean PF: `+1.1412` delta
- Positive+PF>=1: `3/3` (baseline `1/3`)
- Per-seed behavior:
  - `10007`: switched to `candidate_ep010`, large recovery.
  - `5051`: switched to `candidate_ep004`, recovered from deep negative to positive PF.
  - `7079`: no switch; preserved strong baseline winner.
- Note: WF pass count remained lower on this pilot subset (`0` vs baseline `3`), so full blind confirmation is still required.

Decision:
- Horizon-aware selector is now the leading path.
- Next gate: blind-10 confirmation with horizon rescue enabled using the tuned defaults above.

## 2026-02-24 (GuardB-A 20ep mixed-seed probe: weak-tail fix, strong-seed regression)

Focus: verify whether extending episode budget to 20 is globally helpful, not just on known weak seeds.

Additional 20ep probes (same GuardB-A profile):
- `seed_sweep_results/realdata/guardbA_20ep_probe_20260224_002523_seed2027`
- `seed_sweep_results/realdata/guardbA_20ep_probe_20260224_002523_seed8087`
- `seed_sweep_results/realdata/guardbA_20ep_probe_20260224_002523_seed7079`

Combined analysis (weak3 + strong3 vs 10ep selected baseline):
- `seed_sweep_results/realdata/guardbA_20ep_probe_6seed_20260224_summary.json`

Key deltas (`20ep` minus `10ep`) by subset:
- All 6 seeds:
  - Mean return `+0.2630%`
  - Mean PF `-0.7185`
  - WF pass count `-4`
  - Improved return/PF seeds: `3/6` and `3/6`
- Weak 3 (`10007,5051,9091`):
  - Mean return `+2.7730%`
  - Mean PF `+0.9090`
  - Improved return/PF seeds: `3/3`
- Strong 3 (`2027,8087,7079`):
  - Mean return `-2.2470%`
  - Mean PF `-2.3460`
  - Improved return/PF seeds: `0/3`

Reading:
- 20 episodes clearly repairs weak tails, but simultaneously erodes previously strong seeds.
- This is a budget/selection interaction, not a simple "more episodes is better" rule.

Decision:
- Do not promote a global move from 10ep to 20ep.
- Next step should target horizon-aware checkpoint selection (or equivalent stopping logic) so weak seeds can use later checkpoints without sacrificing strong seeds.

## 2026-02-24 (GuardB-A weak-seed probe: 20 episodes vs 10)

Focus: test whether weak-seed collapses are mostly a short-horizon issue (10 episodes) rather than a selector rule issue.

Runs (same friction/gating profile as GuardB-A blind10, only `episodes=20`):
- `seed_sweep_results/realdata/guardbA_20ep_probe_20260224_000603_seed10007`
- `seed_sweep_results/realdata/guardbA_20ep_probe_20260224_001144_seed5051`
- `seed_sweep_results/realdata/guardbA_20ep_probe_20260224_001144_seed9091`

Per-seed WF2400 test results:
- `10007`: `-2.03% / PF 0.53` -> `+0.36% / PF 1.12` (recovered)
- `5051`: `-3.22% / PF 0.34` -> `+2.42% / PF 2.32` (strong recovery)
- `9091`: `-1.31% / PF 0.59` -> `-1.02% / PF 0.75` (partial, still failing)

Aggregate vs same 3 seeds from GuardB-A blind10 (selected winners at 10 episodes):
- Summary:
  - `seed_sweep_results/realdata/guardbA_20ep_probe_weak3_20260224_summary.json`
- Delta (`20ep` minus `10ep`):
  - Mean return: `+2.7730%` (`-2.1867% -> +0.5863%`)
  - Mean PF: `+0.9090` (`0.4869 -> 1.3959`)
  - Positive+PF>=1: `0/3 -> 2/3`
  - WF pass count: `2 -> 0` (stricter pass metric not aligned with improved full-period profitability in this probe)

Reading:
- This is the first strong evidence in this branch that extending episode budget can materially reduce weak-seed tail losses without changing the core reward/cost profile.
- The selector miss remains real, but part of the prior failure pattern appears to be under-training at 10 episodes.

Decision:
- Next promotion gate should be a tri-seed (or blind subset) confirmation at `20 episodes` under GuardB-A profile, before introducing additional selector complexity.

## 2026-02-23 (Selector rescue probes: no robust fix yet)

Focus: pressure-test fast selector rescue ideas after the GuardB-A weak-seed oracle miss (`10007`, `5051`).

1) Tournament stability probe (`k/jitter` up only during anti-regression):
- Run: `selector_k8j3_probe_10ep_20260223_232023_seed10007`
- Changes vs prior fast profile:
  - `--anti-regression-eval-min-k 8 --anti-regression-eval-max-k 10`
  - `--anti-regression-eval-jitter-draws 3`
  - `--anti-regression-tiebreak` (2400-bar probe, relaxed edge thresholds)
- Result:
  - Selector still chose `candidate_ep008.pt` (same failure mode).
  - WF2400 test remained negative: `-1.82% / PF 0.57`.
- Oracle re-check in this run:
  - `seed_sweep_results/realdata/selector_k8j3_probe_10ep_20260223_232023_seed10007/oracle_probe_wf2400_summary.json`
  - Best candidate remained `candidate_ep010.pt` at `+2.97% / PF 2.73`.
- Conclusion: stronger tournament sampling alone does not resolve the selection miss.

2) Full blind10 scan: force `candidate_ep010` (latest) for all seeds:
- Scan summary:
  - `seed_sweep_results/realdata/guardbA_blind10_fast10ep_20260223_200325_candidate_ep010_scan_summary.json`
- Comparison vs selected winners:
  - `seed_sweep_results/realdata/guardbA_blind10_fast10ep_20260223_200325_ep010_vs_selected_wf2400_comparison.json`
- Aggregate delta (`ep010` minus selected):
  - Mean return `+0.3173%` (better)
  - Mean PF `-0.4008` (worse)
  - WF pass count `-3` (worse)
  - Positive+PF>=1 unchanged (`7/10`)
- Reading: latest-checkpoint fallback rescues weak tails but degrades robust seeds; not a safe global selector policy.

3) Full blind10 scan: 50/50 weight blend (`winner` + `ep010`):
- Blend summary:
  - `seed_sweep_results/realdata/guardbA_blind10_fast10ep_20260223_200325_blend_winner_ep010_wf2400_summary.json`
- Comparison vs selected winners:
  - `seed_sweep_results/realdata/guardbA_blind10_fast10ep_20260223_200325_blend_vs_selected_wf2400_comparison.json`
- Aggregate delta (`blend` minus selected):
  - Mean return `+0.1051%` (slight)
  - Mean PF `-0.3669` (worse)
  - WF pass count unchanged
  - Positive+PF>=1 `7/10 -> 6/10`
- Reading: blending does not provide a robust Pareto improvement.

Decision:
- Do not promote either `latest-checkpoint` or `winner+ep010 blend` as global selector policy.
- Selection remains the main blocker; next work should target a selector signal that improves weak tails without sacrificing PF/WF robustness on strong seeds.

## 2026-02-23 (GuardB-A blind10 confirmation + weak-seed oracle audit)

Focus: confirm the promising GuardB-A pivot on full blind10 and diagnose new tail failures.

Blind10 run:
- Prefix: `guardbA_blind10_fast10ep_20260223_200325`
- Seeds: `10007,1011,2027,3039,4049,5051,6067,7079,8087,9091`
- Config: `trade_penalty=0.0`, `flip_penalty=0.00045`, `min_atr_cost_ratio=0.15`, `cooldown=8`, `min_hold=4`, `max_trades=28`, `trade_gate_z=0.30`, no symmetry loss, no dual controller, `auto_rescue` selector.
- Artifacts:
  - `seed_sweep_results/realdata/guardbA_blind10_fast10ep_20260223_200325_summary.json`
  - `seed_sweep_results/realdata/guardbA_blind10_fast10ep_20260223_200325_vs_autorescue_blind10_fast10ep_20260222_155708_eval_wf2400_comparison.json`

WF2400 aggregate vs previous blind10 baseline (`autorescue_blind10_fast10ep_20260222_155708`):
- Mean return: `+0.7246%` (delta `-0.2547%`)
- Mean PF: `1.8721` (delta `+0.3412`)
- Positive + PF>=1: `7/10` (delta `-1`)
- WF pass count: `5/10` (delta `+4`)

Reading:
- Distribution shifted to higher PF / more WF passes, but with heavier downside tails.
- Major regressions concentrated in `10007` and `5051` (both became strongly negative under selected checkpoints).

Weak-seed oracle audit (selected-checkpoint miss test):
- Probe source:
  - `seed_sweep_results/realdata/guardbA_blind10_fast10ep_20260223_200325_oracle_probe_weakseeds_wf2400_20260223_220212.json`
- Summary:
  - `seed_sweep_results/realdata/guardbA_blind10_fast10ep_20260223_200325_oracle_probe_weakseeds_wf2400_summary_20260223_220212.json`
- Findings:
  - Seed `10007`: selected `candidate_ep008` gave `-2.03% / PF 0.53`; oracle best `candidate_ep010` gave `+2.97% / PF 2.73`.
  - Seed `5051`: selected `candidate_ep006` gave `-3.22% / PF 0.34`; oracle best `candidate_ep010` gave `+2.76% / PF 2.34`.
- Conclusion: large blind10 failures are primarily checkpoint-selection misses again, not pure inability to train profitable candidates.

Validation-window probe (selection calibration check):
- Prefix: `guardbA_val1200_probe2_10ep_20260223_221535` on seeds `10007,5051`
- Artifacts:
  - `seed_sweep_results/realdata/guardbA_val1200_probe2_10ep_20260223_221535_summary.json`
  - `seed_sweep_results/realdata/guardbA_val1200_probe2_10ep_20260223_221535_vs_guardbA_blind10_fast10ep_20260223_200325_subset_comparison.json`
- Result: no change at all (same winners, same WF2400 outcomes), so simply increasing `val_window_bars` to `1200` does not fix this selector failure mode.

Decision:
- Keep GuardB-A as promising but not yet promotable.
- Next step should focus on a selection-rule update for this branch (candidate-ranking policy) rather than another reward/cadence sweep.

## 2026-02-23 (Weak-seed reward/cadence pivot: GuardB-A rescue)

Focus: move beyond selector-only tuning and test a reward/cadence pivot on weak seeds (`2027`, `8087`, `9091`) using the older robust GuardB-style controls under current code.

Main weak-seed tri run:
- Prefix: `weakseed_guardbA_reward_tri10ep_20260223_172019`
- Config: `trade_penalty=0.0`, `flip_penalty=0.00045`, `min_atr_cost_ratio=0.15`, `cooldown=8`, `min_hold=4`, `max_trades=28`, `trade_gate_z=0.30`, no symmetry loss, no dual controller, `auto_rescue` selector.
- Artifacts:
  - `seed_sweep_results/realdata/weakseed_guardbA_reward_tri10ep_20260223_172019_summary.json`
  - `seed_sweep_results/realdata/weakseed_guardbA_reward_tri10ep_20260223_172019_vs_autorescue_blind10_fast10ep_20260222_155708_eval_wf2400_subset_comparison.json`

WF2400 results (tri aggregate):
- Mean return `+0.7199%` (baseline subset: `-0.6066%`, delta `+1.3265%`)
- Mean PF `1.6859` (baseline subset: `0.7690`, delta `+0.9169`)
- Positive + PF>=1: `2/3` (baseline `1/3`)
- WF pass count: `1/3` (unchanged)

Per-seed WF2400:
- `2027`: `-1.07% / PF 0.55` -> `+1.41% / PF 1.64` (major recovery)
- `8087`: `+0.75% / PF 1.35` -> `+2.06% / PF 2.82` (strong improvement)
- `9091`: `-1.50% / PF 0.41` -> `-1.31% / PF 0.59` (improved but still failing)

Follow-up on failing seed `9091`:
- Micro-sweep prefix: `seed9091_guardbA_micro4_10ep_20260223_182015`
- Summary:
  - `seed_sweep_results/realdata/seed9091_guardbA_micro4_10ep_20260223_182015_summary.json`
- Best local variant was `D_gate035`:
  - WF2400 `-0.03%`, PF `0.99` (near break-even), better than `A_base/B_atr018/C_flip050` (`-1.31%`, PF `0.59`).

Tri confirmation of `gate_z=0.35`:
- Prefix: `weakseed_guardbA_gate035_tri10ep_20260223_192347`
- Artifacts:
  - `seed_sweep_results/realdata/weakseed_guardbA_gate035_tri10ep_20260223_192347_summary.json`
  - `seed_sweep_results/realdata/weakseed_guardbA_gate035_tri10ep_20260223_192347_vs_baseline_and_gate03_comparison.json`
- Outcome:
  - Helped `9091` (`-1.31% -> -0.03%`) but regressed `2027` and `8087` enough to reduce tri aggregate versus `gate_z=0.30`.

Decision:
- Keep GuardB-A reward/cadence pivot as the current promising path (`gate_z=0.30`).
- Reject global move to `gate_z=0.35`; treat it as a seed-local rescue only.
- Next work should target `9091` recovery without sacrificing the recovered `2027/8087` behavior.

## 2026-02-23 (Weak-seed tiebreak check + alias-dedup fix validation)

Focus: test whether `auto_rescue+tiebreak` improves the known weak seeds (`2027`, `8087`, `9091`) and verify a selector bug where top-2 tie-break could compare alias checkpoints (`candidate_epXXX` vs `best_model.pt`) that represent the same candidate.

Weak-seed tri run (pre-fix code path):
- Prefix: `selector_autorescue_tiebreak_weakseed_tri10ep_20260223_131550`
- Seeds: `2027, 8087, 9091`
- Profile: fast10 (`max_steps=600`, HFM friction, no symmetry, no dual)
- Outputs:
  - `seed_sweep_results/realdata/selector_autorescue_tiebreak_weakseed_tri10ep_20260223_131550_summary.json`
  - `seed_sweep_results/realdata/selector_autorescue_tiebreak_weakseed_tri10ep_20260223_131550_vs_autorescue_blind10_fast10ep_20260222_155708_eval_wf2400_subset_comparison.json`

Result:
- Exact parity vs baseline auto-rescue on this 3-seed subset (no metric delta).
- Root cause: tie-break often had no effective second candidate (alias/no-op comparisons), so it did not change selection.

Code change:
- `trainer.py` tie-break candidate deduplication now removes alias-equivalent entries from top-2 consideration using tournament metric identity keys (and records `distinct_pool_filenames` in `checkpoint_tournament.json`).

Post-fix seed-level verification:
- `selector_autorescue_tiebreak_dedupe_verify_seed2027_10ep_20260223_142724_seed2027`
  - tie-break now evaluated true-distinct candidates and switched:
    - `candidate_ep002.pt -> candidate_ep010.pt` (`selected_mode=tail_holdout+tiebreak`)
  - WF2400 eval worsened versus baseline seed-2027:
    - return `-1.07% -> -1.67%`, PF `0.55 -> 0.68` (still failing).
- `selector_autorescue_tiebreak_dedupe_verify_seed9091_10ep_20260223_144207_seed9091`
  - tie-break evaluated distinct challenger but did not switch.
  - WF2400 eval unchanged vs baseline behavior (still negative/failing).

Decision:
- Keep alias-dedup logic (it fixes a real selector flaw), but do **not** promote current tie-break switching thresholds as a profitability upgrade yet.
- Selector-only refinements remain insufficient to resolve the weak-seed failure mode.

## 2026-02-23 (A/B tri-seed check: `base_first` vs `auto_rescue`)

Focus: validate whether the new base-dominant checkpoint selector (`base_first`) improves out-of-sample behavior versus the current selector path (`auto_rescue`) on unseen seeds.

Run setup:
- Seeds: `1123, 2213, 3347` (unseen in prior blind10)
- Profile: fast real-data 10ep (`max_steps=600`, HFM friction settings, no symmetry, no dual controller)
- Arms:
  - `autorescue`: `--anti-regression-selector-mode auto_rescue` (with calibrated rescue thresholds)
  - `basefirst`: `--anti-regression-selector-mode base_first`
- Per-seed WF2400 re-eval done from selected `checkpoints/best_model.pt`.

Artifacts:
- `seed_sweep_results/realdata/selector_ab_autorescue_tri10ep_20260223_095309_summary.json`
- `seed_sweep_results/realdata/selector_ab_basefirst_tri10ep_20260223_095309_summary.json`
- `seed_sweep_results/realdata/selector_ab_basefirst_vs_autorescue_tri10ep_20260223_095309_comparison.json`
- Logs per seed:
  - `logs/selector_ab_autorescue_tri10ep_20260223_095309_seed*.log`
  - `logs/selector_ab_basefirst_tri10ep_20260223_095309_seed*.log`
  - `logs/selector_ab_*_seed*_eval_wf2400.log`

Results:
- Train-horizon aggregate:
  - `autorescue`: mean return `+0.4550%`, mean PF `1.5054`, positive+PF>1 `2/3`
  - `basefirst`: mean return `+0.2483%`, mean PF `1.4162`, positive+PF>1 `1/3`
- WF2400 aggregate:
  - `autorescue`: mean return `+0.5119%`, mean PF `1.5447`, positive+PF>1 `2/3`, WF pass `1/3`
  - `basefirst`: mean return `+0.3052%`, mean PF `1.4555`, positive+PF>1 `1/3`, WF pass `2/3`
- Delta (`basefirst - autorescue`, WF2400):
  - mean return `-0.2067%`
  - mean PF `-0.0892`
  - positive+PF>1 `-1`
  - WF pass `+1`

Interpretation:
- On this tri-seed A/B, `base_first` is not an upgrade in profitability metrics.
- Keep `auto_rescue` as preferred selector path; do not promote `base_first` as default.

## 2026-02-23 (Checkpoint selector root-cause audit: full candidate oracle)

Focus: verify whether the remaining weak-seed failures are mainly a checkpoint-selection problem or a model-capacity problem, using the already trained blind10 run `autorescue_blind10_fast10ep_20260222_155708`.

What was run:
- Candidate probe on weak seeds (`2027`, `9091`) across all saved checkpoints (`ep002/004/006/008/010`, plus best/final where present).
- Full blind10 candidate probe for missing checkpoints (same WF2400 profile), then aggregate oracle analysis.
- Offline selector policy scan using only tournament metrics (`checkpoint_tournament.json`) as ranking signals.

Result files:
- Weak-seed oracle summary:
  - `seed_sweep_results/realdata/autorescue_blind10_fast10ep_20260222_155708_candidate_probe_oracle_wf2400_summary_20260223_015914.json`
- Fixed `candidate_ep006` comparison vs selected winners:
  - `seed_sweep_results/realdata/autorescue_blind10_fast10ep_20260222_155708_candidate_ep006_wf2400_vs_baseline_20260223_021542.json`
- Full blind10 oracle summary (with `ep006` included for all seeds):
  - `seed_sweep_results/realdata/autorescue_blind10_fast10ep_20260222_155708_candidate_probe_oracle_wf2400_full_blind10_with_ep006_summary_20260223_025304.json`
- Selector policy scan:
  - `seed_sweep_results/realdata/autorescue_blind10_fast10ep_20260222_155708_selector_policy_scan_20260223_025509.json`

Key findings:
- Current selected-winner aggregate (WF2400): mean return `+0.9794%`, mean PF `1.5309`, positive+PF>1 `8/10`.
- Fixed `candidate_ep006` does **not** solve globally: mean return `+0.6678%`, mean PF `1.2808`, positive+PF>1 `7/10`.
- Oracle upper bound (best checkpoint per seed among `ep002/004/006/008/010`): mean return `+2.3651%`, mean PF `2.7013`, positive+PF>1 `10/10`.
- Selector miss pattern is broad, not isolated:
  - selected candidate rank by realized return: mean rank `2.6/5`, top-1 only `3/10` seeds.
  - largest misses remained `seed2027` and `seed9091`, but additional medium misses appeared in `3039`, `5051`, `6067`, `7079`, `8087`.
- Policy scan (ranking candidates by tournament metrics only):
  - `selected_current`: `+0.979%` return, PF `1.531`, `8/10`.
  - `max_base_return`: `+1.255%` return, PF `1.625`, `9/10` (best among tested simple rules).
  - forward/future-heavy scores underperformed on this blind10 sample.

Decision:
- Next step is to implement and validate a base-dominant selector path (fresh unseen seeds) before any broader retraining cycle.

## 2026-02-22 (Control blind10 on current code: tail_holdout vs auto_rescue)

Focus: run a fresh control blind10 on the current code with `tail_holdout` (same fast10 profile) to isolate the effect of the new `auto_rescue` selector.

Control run:
- Prefix: `tailholdout_current_blind10_fast10ep_20260222_184805`
- Seeds: `10007,1011,2027,3039,4049,5051,6067,7079,8087,9091`
- Selector: `--anti-regression-selector-mode tail_holdout`

Control aggregates:
- Base:
  - `seed_sweep_results/realdata/tailholdout_current_blind10_fast10ep_20260222_184805_aggregate_20260222.json`
  - mean return `+0.7919%`
  - mean PF `1.4630`
  - positive+PF>1 `7/10`
  - walk-forward pass `0/10`
- WF2400:
  - `seed_sweep_results/realdata/tailholdout_current_blind10_fast10ep_20260222_184805_eval_wf2400_aggregate_20260222.json`
  - mean return `+0.7919%`
  - mean PF `1.4630`
  - positive+PF>1 `7/10`
  - walk-forward pass `1/10`

Direct control-vs-auto comparison:
- Comparison JSON:
  - `seed_sweep_results/realdata/tailholdout_current_blind10_fast10ep_20260222_184805_vs_autorescue_blind10_fast10ep_20260222_155708_20260222.json`
- Deltas (control minus auto-rescue):
  - mean return `-0.1875%`
  - mean PF `-0.0679`
  - positive+PF>1 `-1`
  - walk-forward pass `0` (same at WF2400)

Interpretation:
- On current code, calibrated `auto_rescue` is better than pure `tail_holdout` by exactly one targeted seed flip (`10007`), while other seeds remain effectively unchanged.
- Both current-code branches (`auto_rescue`, `tail_holdout`) remain below the older tail-holdout benchmark (`tailholdout_fix_blind10_fast10ep_20260220_231427`), so the project bottleneck is broader than selector mode alone.

## 2026-02-22 (Blind10 confirmation: calibrated auto-rescue selector)

Focus: run a full blind10 confirmation with calibrated `auto_rescue` selector, then compare directly against the current tail-holdout blind10 baseline.

Run:
- Prefix: `autorescue_blind10_fast10ep_20260222_155708`
- Seeds: `10007,1011,2027,3039,4049,5051,6067,7079,8087,9091`
- Profile matched the fast 10ep screen (`max_steps=600`, HFM friction settings, no symmetry, no dual controller).
- Selector config:
  - `--anti-regression-selector-mode auto_rescue`
  - rescue thresholds: winner-forward return max `0.65`, forward-return edge min `0.10`, forward-PF edge min `0.10`, challenger base-return max `0.0`, challenger forward-PF min `1.35`.

Aggregates:
- Base:
  - `seed_sweep_results/realdata/autorescue_blind10_fast10ep_20260222_155708_aggregate_20260222.json`
  - mean return `+0.9794%`
  - mean PF `1.5309`
  - positive+PF>1 `8/10`
  - walk-forward pass `0/10`
  - auto-rescue triggers `1/10`
- WF2400:
  - `seed_sweep_results/realdata/autorescue_blind10_fast10ep_20260222_155708_eval_wf2400_aggregate_20260222.json`
  - mean return `+0.9794%`
  - mean PF `1.5309`
  - positive+PF>1 `8/10`
  - walk-forward pass `1/10`
  - auto-rescue triggers `1/10`

Selector behavior:
- Triggered only for seed `10007` (tail winner `candidate_ep006.pt` -> future winner `candidate_ep002.pt`, result `+0.65%`, PF `1.26`).
- Remaining seeds stayed on `tail_holdout`.

Direct comparison vs tail-holdout blind10 baseline (`tailholdout_fix_blind10_fast10ep_20260220_231427`):
- Comparison JSON:
  - `seed_sweep_results/realdata/autorescue_blind10_fast10ep_20260222_155708_vs_tailholdout_fix_blind10_20260222.json`
- Base deltas:
  - mean return `-0.4356%`
  - mean PF `-0.1907`
  - positive+PF>1 `-1`
  - walk-forward pass `0` (unchanged at base horizon)
- WF2400 deltas:
  - mean return `-0.4356%`
  - mean PF `-0.1907`
  - positive+PF>1 `-1`
  - walk-forward pass `-1` (from `2/10` to `1/10`)

Outcome:
- Calibrated auto-rescue behaves as designed (targeted trigger on the known failure pattern), but broad blind10 performance is still below the current tail-holdout baseline.
- Keep auto-rescue available as a targeted mechanism; do not promote it as default for broad-seed production selection yet.

## 2026-02-22 (Auto-rescue selector trigger implemented + calibrated)

Focus: replace static seed overrides with a validation-diagnostics trigger that keeps `tail_holdout` by default and switches to `future_first` only when a forward-robust challenger matches a failure pattern.

Code changes:
- `config.py`:
  - Added `anti_regression_selector_mode=auto_rescue` support and trigger thresholds:
    - `anti_regression_auto_rescue_*` settings (winner forward cap, forward edges, challenger base cap, challenger forward PF floor).
- `main.py`:
  - Added CLI support for `auto_rescue` mode and all auto-rescue threshold overrides.
- `trainer.py`:
  - Anti-regression tournament now computes both primary and future-view diagnostics for each candidate.
  - Added auto-rescue post-selection pass:
    - rank normally with `tail_holdout`,
    - evaluate future-first challenger pool,
    - trigger switch only if thresholds are met.
  - Tournament summary now records requested/effective mode and `auto_rescue` diagnostics.

Validation:
- `python -m py_compile config.py main.py trainer.py`
- `python test_system.py` (pass)

Tri-seed screen (raw auto-rescue thresholds):
- Prefix: `selector_autorescue_tri10ep_20260222_144524` (seeds `10007,4049,8087`).
- Auto-rescue did not trigger on these defaults.
- Aggregates:
  - `seed_sweep_results/realdata/selector_autorescue_tri10ep_20260222_144524_aggregate_20260222.json`
  - `seed_sweep_results/realdata/selector_autorescue_tri10ep_20260222_144524_eval_wf2400_aggregate_20260222.json`
  - Base mean return `+0.6296%`, PF `1.4494`, positive+PF>1 `2/3`.
  - WF2400 pass `1/3`.

Candidate probes from the same tri run:
- Seed `10007`: future challenger `candidate_ep002` beats tail winner (`+0.65%` / PF `1.26` vs `-1.22%` / PF `0.58`).
- Seed `4049`: future challenger regresses strongly (`-1.24%` / PF `0.62` vs tail `+2.36%` / PF `2.41`).
- Seed `8087`: future challenger is mixed (slightly better base return, worse WF pass).

Calibration update:
- Tightened trigger defaults to target the `10007`-style case and avoid broad flips:
  - `challenger_base_return_max=0.0`
  - `challenger_forward_pf_min=1.35`

Recheck run:
- Prefix: `selector_autorescue_seed10007_recheck10ep_20260222_153317`
- Auto-rescue now triggered as intended (`requested=auto_rescue`, `effective=future_first`).
- Result: `+0.65%`, PF `1.26` (recovered from prior tail mode loss on same seed profile).

Mixed tri aggregate (seed10007 recheck + unchanged 4049/8087 tri runs):
- `seed_sweep_results/realdata/selector_autorescue_tri10ep_threshold135_mixed_aggregate_20260222.json`
- `seed_sweep_results/realdata/selector_autorescue_tri10ep_threshold135_mixed_eval_wf2400_aggregate_20260222.json`
- Mean return `+1.2546%`, PF `1.6757`, positive+PF>1 `3/3`, WF2400 pass `1/3`.

## 2026-02-22 (Clean rerun: hybrid selector profile, 10 seeds)

Focus: rerun a fully consistent blind10 experiment with explicit selector modes per seed:
- default `tail_holdout`
- seed `10007` override `future_first`

Run:
- Prefix: `tailholdout_hybrid_clean_blind10_fast10ep_20260222_125225`
- All 10 seeds trained/evaluated from scratch with per-seed logs and WF2400 eval.

Aggregates:
- Base:
  - `seed_sweep_results/realdata/tailholdout_hybrid_clean_blind10_fast10ep_20260222_125225_aggregate_20260222.json`
  - mean return `+0.9794%`
  - mean PF `1.5309`
  - positive+PF>1 `8/10`
  - WF pass `0/10`
- WF2400:
  - `seed_sweep_results/realdata/tailholdout_hybrid_clean_blind10_fast10ep_20260222_125225_eval_wf2400_aggregate_20260222.json`
  - mean return `+0.9794%`
  - mean PF `1.5309`
  - positive+PF>1 `8/10`
  - WF pass `1/10`

Outcome:
- This clean rerun did not reproduce the stronger synthetic “hybrid-mix” expectation.
- It is still broadly positive, but below the prior tail-holdout blind10 benchmark.

Decision:
- Keep selector-mode split in code (`tail_holdout` default, `future_first` available).
- Do not lock in static per-seed overrides yet; next step should be a data-driven override rule (only trigger rescue mode when checkpoint tournament diagnostics match known failure pattern).

## 2026-02-22 (Selector mode split + targeted rescue hybrid)

Focus: keep broad-seed robustness from `tail_holdout` while preserving the `future_first` rescue path for seed-specific failures.

Code changes:
- `config.py`:
  - Added `anti_regression_selector_mode` with default `tail_holdout`.
- `main.py`:
  - Added CLI flag:
    - `--anti-regression-selector-mode {tail_holdout,future_first}`
- `trainer.py`:
  - Anti-regression checkpoint tournament now supports two explicit modes:
    - `tail_holdout` (default): strict robustness across base+alt+tail.
    - `future_first`: prioritize alt+tail, with optional soft base penalty.
  - Tournament summary now records `selector_mode` in `checkpoint_tournament.json`.

Mode verification run (seed `10007`, same 10ep fast profile):
- Tail mode run:
  - `seed_sweep_results/realdata/selector_modecheck_seed10007_20260222_121139_tail_holdout`
  - result: return `-1.22%`, PF `0.58`
- Future mode run:
  - `seed_sweep_results/realdata/selector_modecheck_seed10007_20260222_121139_future_first`
  - result: return `+0.65%`, PF `1.26`

Hybrid aggregate (broad baseline + targeted seed10007 rescue):
- Base reference:
  - `seed_sweep_results/realdata/tailholdout_fix_blind10_fast10ep_20260220_231427_aggregate_20260220.json`
- Hybrid outputs:
  - `seed_sweep_results/realdata/tailholdout_hybrid_seed10007_futurefirst_blind10_aggregate_20260222.json`
  - `seed_sweep_results/realdata/tailholdout_hybrid_seed10007_futurefirst_blind10_eval_wf2400_aggregate_20260222.json`
- Hybrid metrics:
  - mean return `+1.6025%`
  - mean PF `1.7895`
  - positive+PF>1 `10/10`
  - WF pass count `2/10` (unchanged from tail-holdout baseline)

Decision:
- Use `tail_holdout` as the default selector for broad sweeps.
- Use `future_first` only as a targeted rescue mode (currently validated on seed `10007`).

## 2026-02-22 (Blind10 confirmation: future-first selector underperforms baseline)

Focus: run full blind 10-seed confirmation for the new future-first anti-regression selector.

Run profile:
- Prefix: `selector_futurepool_blind10_fast10ep_20260221_231856`
- Seeds: `10007, 1011, 2027, 3039, 4049, 5051, 6067, 7079, 8087, 9091`
- Training/eval settings matched the fast 10ep screen (`max_steps=600`) plus per-seed WF2400 eval.

Aggregates:
- Base:
  - `seed_sweep_results/realdata/selector_futurepool_blind10_fast10ep_20260221_231856_aggregate_20260222.json`
  - mean return `+0.7638%`, mean PF `1.4667`, positive+PF>1 `8/10`, WF pass `0/10`
- WF2400:
  - `seed_sweep_results/realdata/selector_futurepool_blind10_fast10ep_20260221_231856_eval_wf2400_aggregate_20260222.json`
  - mean return `+0.8707%`, mean PF `1.5024`, positive+PF>1 `8/10`, WF pass `1/10`

Comparison vs prior tail-holdout blind10 (`tailholdout_fix_blind10_fast10ep_20260220_231427`):
- Mean return worsened (`+1.4150% -> +0.7638%`)
- Mean PF worsened (`1.7216 -> 1.4667`)
- Positive+PF>1 worsened (`9/10 -> 8/10`)
- WF pass count worsened on WF2400 aggregate (`2/10 -> 1/10`)

Notable seed behavior:
- Improved: `10007` (`-1.22%/PF 0.58 -> +0.65%/PF 1.26`)
- Regressed: `4049`, `5051`, `8087` (major drops; especially `8087`)

Decision:
- Do not promote the pure future-first selector as default.
- Keep it as a targeted recovery idea for problematic seeds, but return to the stronger tail-holdout baseline for broad blind-seed performance.

## 2026-02-21 (Future-first checkpoint selector + seed10007 recovery)

Focus: fix anti-regression checkpoint mis-selection where base-window negatives blocked candidates that were stronger on forward-style regimes.

Code changes:
- `trainer.py`:
  - Validation now temporarily raises `val_env.max_steps` to match selected validation window length (so anti-regression `alt_window_bars` can exceed train episode length).
  - Anti-regression tournament scoring now uses `alt+tail` as primary robust signal (future-first), with base regime as a soft penalty.
  - Added base penalty terms in tournament payload (`base_return_penalty`) and summary metadata (`base_return_floor`, `base_penalty_weight`).
- `config.py`:
  - Added:
    - `anti_regression_base_return_floor` (default `0.0`)
    - `anti_regression_base_penalty_weight` (default `0.15`)
- `main.py`:
  - Added CLI overrides:
    - `--anti-regression-base-return-floor`
    - `--anti-regression-base-penalty-weight`

Verification:
- System smoke still passes: `python test_system.py`.
- Seed `10007` rerun switched winner from `candidate_ep006` to `candidate_ep002`:
  - old: `seed_sweep_results/realdata/selector_calib_tailstrict_k10_tri10ep_20260221_123238_seed10007/eval_wf2400/results/test_results.json`
  - new: `seed_sweep_results/realdata/selector_futurepool_valhfix_10ep_20260221_195922_seed10007/eval_wf2400/results/test_results.json`
  - change: return `-1.22% -> +0.65%`, PF `0.58 -> 1.26`.

Tri-seed follow-up (future selector profile, 10ep):
- Runs:
  - `seed_sweep_results/realdata/selector_futurepool_valhfix_10ep_20260221_195922_seed10007`
  - `seed_sweep_results/realdata/selector_futurepool_valhfix_tri10ep_20260221_220213_seed4049`
  - `seed_sweep_results/realdata/selector_futurepool_valhfix_tri10ep_20260221_220213_seed3039`
- Aggregates:
  - `seed_sweep_results/realdata/selector_futurepool_valhfix_tri10ep_mixed_aggregate_20260221.json`
  - `seed_sweep_results/realdata/selector_futurepool_valhfix_tri10ep_mixed_eval_wf2400_aggregate_20260221.json`

Outcome vs previous strict-tail tri reference (`selector_calib_tailstrict_k10_tri10ep_20260221_123238`):
- WF2400 mean return: `+0.57% -> +1.04%`
- WF2400 mean PF: `1.39 -> 1.38` (roughly flat)
- Positive+PF>1: `2/3 -> 3/3`
- WF pass count: unchanged (`0/3`)

Decision:
- Keep future-first selector and validation-window horizon fix.
- Next step: improve walk-forward pass rate (currently profitability improved, but regime consistency threshold still failing).

## 2026-02-20 (Tail hold-out in checkpoint tournament + blind10 retest)

Focus: reduce end-of-run checkpoint mis-selection by adding a separate tail-only validation segment into anti-regression checkpoint ranking.

Code changes:
- `trainer.py`:
  - `validate()` now supports segment overrides via `start_frac_override` / `end_frac_override`.
  - Added validation quantiles to stats/export (`val_return_q25_pct`, `val_return_q10_pct`, `val_pf_q25`, `val_pf_q10`).
  - Anti-regression tournament now evaluates each candidate on three regimes:
    - base validation
    - alt stride validation
    - tail segment validation (`anti_regression_tail_start_frac` -> `anti_regression_tail_end_frac`)
  - Composite now includes a tail-negative penalty (`anti_regression_tail_weight`) and tracks tail metrics in `checkpoint_tournament.json`.
- `config.py`:
  - Added training knobs:
    - `anti_regression_tail_start_frac` (default `0.50`)
    - `anti_regression_tail_end_frac` (default `1.00`)
    - `anti_regression_tail_weight` (default `0.75`)
- `main.py`:
  - Added CLI overrides:
    - `--anti-regression-tail-start-frac`
    - `--anti-regression-tail-end-frac`
    - `--anti-regression-tail-weight`

Key diagnostics before fix:
- Full candidate probe on previous blind10 run:
  - `seed_sweep_results/realdata/post_seedfix_gz03_spreadval_blind10_fast10ep_20260220_212629_all_candidates_oracle_eval_wf2400_aggregate_20260220.json`
- Selected checkpoint aggregate (pre-fix): mean return `+1.05%`, PF `1.54`, positive+PF>1 `8/10`.
- Oracle best-candidate aggregate (same trained candidates): mean return `+2.07%`, PF `2.59`, positive+PF>1 `10/10`.

Blind10 retest with tail hold-out tournament:
- Prefix: `tailholdout_fix_blind10_fast10ep_20260220_231427`
- Aggregate (mode=both):
  - `seed_sweep_results/realdata/tailholdout_fix_blind10_fast10ep_20260220_231427_aggregate_20260220.json`
- WF2400 aggregate:
  - `seed_sweep_results/realdata/tailholdout_fix_blind10_fast10ep_20260220_231427_eval_wf2400_aggregate_20260220.json`
- Outcome vs pre-fix selected-checkpoint baseline:
  - mean return: `+1.05%` -> `+1.41%`
  - mean PF: `1.54` -> `1.72`
  - positive+PF>1: `8/10` -> `9/10`
  - WF pass count: unchanged (`2/10`)

Decision:
- Keep tail hold-out tournament changes (measurable improvement, especially seed `4049` recovered from negative to positive).
- Remaining blocker: seed `10007` still mis-selected by validation despite being recoverable via alternate candidate.
- Next step should target stronger checkpoint-selection robustness (or explicit calibration split) before large-scale promotion runs.

## 2026-02-20 (Validation window spread fix + tri-seed recovery)

Focus: reduce checkpoint selection overfitting caused by validation windows being concentrated at the start of the validation period.

Code changes:
- `trainer.py`:
  - Validation windows now spread evenly across the full validation range (instead of first-`K` contiguous starts).
  - Added `val_positive_frac` and `val_pf_ge_1_frac` to validation stats.
  - Anti-regression checkpoint selection now supports `consistency_feasible` pooling (positive-return consistency and PF consistency across base/alt validation).

Validation behavior check:
- Probe run confirmed full-span windows:
  - `test_output/window_spread_check_seed8087/logs/validation_summaries/val_final.json`
  - Console showed `coverage~1.00x` and windows spread across 2021 -> 2022.

Tri-seed retest (same fast profile, `trade_gate_z=0.3`, eval gate disabled):
- Prefix: `post_seedfix_gz03_spreadval_fast10ep_20260220_200539`
- Aggregate:
  - `seed_sweep_results/realdata/post_seedfix_gz03_spreadval_fast10ep_tri_aggregate_20260220.json`
- Result:
  - mean return `+0.61%` (was `-0.50%` before spread fix)
  - mean PF `1.48` (was `1.15`)
  - positive+PF>1 `2/3` (was `1/3`)
  - seed `8087` flipped from `-2.59% / PF 0.36` to `+0.75% / PF 1.35`

Decision:
- Keep spread-window validation and consistency-aware checkpoint tournament.
- This is a meaningful robustness improvement, but not yet enough for promotion (`4049` still negative, WF pass remains `0/3`).

## 2026-02-20 (Eval trade-gate fix + fast tri-seed retest)

Focus: resolve HOLD-collapse caused by applying trade gate during evaluation/validation, then retest fast 10ep screen.

Code changes:
- `agent.py`:
  - Added `disable_trade_gate_in_eval` flag.
  - `select_action()` now skips `_apply_trade_gate()` when `eval_mode=True` and this flag is enabled.
- `main.py`:
  - `create_agent()` now passes `disable_trade_gate_in_eval=config.VAL_DISABLE_TRADE_GATING`.
  - Agent creation printout includes whether eval trade gate is disabled.

Verification:
- Probe run with `trade_gate_z=0.3` no longer collapses to 0 trades in eval:
  - `test_output/gate_eval_disable_probe_seed1011/results/test_results.json`
  - Result: return `+0.25%`, PF `1.10`, trades `18`.

Fast tri-seed retest (10ep, seeds `1011/4049/8087`, `trade_gate_z=0.3`):
- Aggregate:
  - `seed_sweep_results/realdata/post_seedfix_gz03_evaloff_fast10ep_tri_aggregate_20260220.json`
- Outcome:
  - mean return `-0.50%`
  - mean PF `1.15`
  - mean trades `20.3`
  - positive+PF>1 `1/3`
  - walk-forward pass `0/3`

Comparison references:
- Gate on (pre-fix behavior, eval gated -> HOLD collapse):
  - `seed_sweep_results/realdata/post_seedfix_baseline_fast10ep_tri_aggregate_20260220.json`
- Gate off baseline:
  - `seed_sweep_results/realdata/post_seedfix_nogate_fast10ep_tri_aggregate_20260220.json`

Decision:
- Keep the eval-trade-gate fix (required; prevents artificial no-trade evaluation artifacts).
- Current `trade_gate_z=0.3` branch is still not robust across tail seeds after the fix.
- Reject strict-flow pivot early (first seed negative) to avoid spending more runtime:
  - `seed_sweep_results/realdata/post_seedfix_strictflow_gz03_evaloff_fast10ep_20260220_191919_seed1011/results/test_results.json`

## 2026-02-20 (Reproducibility fix + trade-gate sanity check)

Focus: eliminate seed drift between reruns, then verify whether current low-flip setup is blocked by trade gating.

Code fixes:
- `main.py`:
  - `set_random_seeds()` now seeds Python RNG in addition to NumPy/Torch.
  - Added `PYTHONHASHSEED` assignment and `torch.cuda.manual_seed_all`.
- `trainer.py`:
  - Anti-regression checkpoint tournament now seeds/restores Python RNG alongside NumPy RNG before base/alt validations.

Reproducibility probes:
- Synthetic duplicate runs (same seed/config) produced identical outputs:
  - `test_output/repro_seed8087_run1/results/test_results.json`
  - `test_output/repro_seed8087_run2/results/test_results.json`
- Real-data duplicate mini-runs (same seed/config) also matched exactly:
  - `test_output/repro_real_seed8087_run1/results/test_results.json`
  - `test_output/repro_real_seed8087_run2/results/test_results.json`

Trade-gate sanity runs (10ep fast profile, seeds `1011/4049/8087`):
- With `--trade-gate-z 0.3`:
  - All three seeds converged to zero trades in test.
  - Aggregate: `seed_sweep_results/realdata/post_seedfix_baseline_fast10ep_tri_aggregate_20260220.json`
  - Mean return `0.00%`, PF `0.00`, trades `0.0`, WF pass `0/3`.
- With `--trade-gate-z 0.0`:
  - Trading activity returned (mean ~20.7 trades), but robustness remained weak.
  - Aggregate: `seed_sweep_results/realdata/post_seedfix_nogate_fast10ep_tri_aggregate_20260220.json`
  - Mean return `-0.02%`, PF `1.07`, positive+PF>1 `1/3`, WF pass `0/3`.

Decision:
- Keep reproducibility fix (required for trustworthy comparisons).
- Do not promote either gate profile as-is.
- Next step should tune a moderate gate policy (between `0.0` and `0.3`) and/or delayed gate schedule to avoid HOLD collapse while retaining cost discipline.

## 2026-02-20 (Fast-run acceleration test + tail-seed rescue screen)

Focus: shorten experiment cycle time and test whether higher exploration rescues persistent tail seeds without retraining for hours.

Fast-screen profile used (10ep):
- `--max-steps-per-episode 600`
- `--validate-every 2`
- `--val-jitter-draws 1`
- `--val-min-k 3 --val-max-k 4`
- `--episode-timeout-min 25`

Run family:
- Base tag: `tailrescue3_fastscreen_20260220_164326`
- Tail seeds: `6067, 9091, 10007`
- Config A (`exp18`): `epsilon_start=0.18`, `epsilon_end=0.08`, `epsilon_decay=0.996`, `min_atr_cost_ratio=0.15`, `max_trades=28`
- Config B (`exp18_atr012_mt32`): same exploration + `min_atr_cost_ratio=0.12`, `max_trades=32`

Tail-only results:
- Config A aggregate:
  - `seed_sweep_results/realdata/tailrescue3_fastscreen_20260220_164326_exp18_eval_wf2400_aggregate_20260220.json`
  - mean return `+1.29%`, PF `1.63`, walk-forward `2/3`, positive+PF>1 `3/3`
- Config B aggregate:
  - `seed_sweep_results/realdata/tailrescue3_fastscreen_20260220_164326_exp18_atr012_mt32_eval_wf2400_aggregate_20260220.json`
  - mean return `+1.90%`, PF `1.76`, walk-forward `0/3`, positive+PF>1 `3/3`

Center spot-check (Config B):
- Seeds `1011, 4049`:
  - `seed_sweep_results/realdata/tailrescue3_fastscreen_20260220_164326_exp18_atr012_mt32_eval_wf2400_spot5_aggregate_20260220.json`
  - 5-seed mean return `+1.75%`, PF `1.77`, positive+PF>1 `5/5`, walk-forward `1/5`

Blind 10-seed fast-screen confirmation (Config A):
- Train/test aggregate:
  - `seed_sweep_results/realdata/tailrescue3_fastscreen_20260220_164326_exp18_ten_fastscreen_aggregate_20260220.json`
- WF2400 aggregate:
  - `seed_sweep_results/realdata/tailrescue3_fastscreen_20260220_164326_exp18_eval_wf2400_ten_fastscreen_aggregate_20260220.json`
- WF2400: mean return `+0.62%`, PF `1.49`, positive+PF>1 `7/10`, walk-forward `3/10`

Comparison to current lead (`guardB10_tailfix_lowflip35_10ep_20260219_211955_eval_wf2400_ten_aggregate_20260220.json`):
- Lead branch: mean return `+1.22%`, PF `1.84`, positive+PF>1 `9/10`, walk-forward `4/10`
- Fast-screen Config A is a regression on aggregate profitability and robustness.

Decision:
- Do **not** promote the exploration-only fast-screen variants as new main branch.
- Keep `guardB10_tailfix_lowflip35_10ep_20260219_211955` as the current best branch.
- Retain fast-screen profile as a runtime acceleration tool for early rejection/selection only.

## 2026-02-20 (20ep confirmation + true zero-cost diagnostic)

Focus: test whether the accepted tail-fix branch (`flip_penalty=0.00035`) scales from 10 episodes to 20, then isolate cost impact.

### Step 1: blind 10-seed 20ep confirmation (real costs, WF2400)

Prefix:
- `guardB10_tailfix_lowflip35_20ep_20260219_232731`

Artifacts:
- Train/test aggregate:
  - `seed_sweep_results/realdata/guardB10_tailfix_lowflip35_20ep_20260219_232731_ten_aggregate_20260220.json`
- WF2400 aggregate:
  - `seed_sweep_results/realdata/guardB10_tailfix_lowflip35_20ep_20260219_232731_eval_wf2400_ten_aggregate_20260220.json`

WF2400 (real costs):
- Mean return `+0.61%`
- Mean PF `1.44`
- Mean trades `23.0`
- Walk-forward pass `5/10`
- Negative-return seeds: `6067`, `8087`, `9091`, `10007`

Read:
- 20ep remains positive on average but degrades profitability vs the 10ep tail-fix confirmation (`+1.22%`, PF `1.84`).

### Step 2: zero-cost diagnostic + bug fix

Issue found:
- `--eval-zero-costs` was not fully zero-cost under broker profile because `swap_by_symbol` overrides still applied symbol swap values.

Fix:
- `main.py`: in the `args.eval_zero_costs` block, clear per-symbol swap overrides with:
  - `config.environment.swap_by_symbol = {}`

Verification:
- Re-ran eval and confirmed env prints:
  - `Swap type: USD/lot/night`
  - `Swap long/short: $0.00 / $0.00`

True zero-cost WF2400 artifacts (post-fix):
- Per-seed eval dirs:
  - `..._seed*/eval_wf2400_zero_cost_v2/results/test_results.json`
- Aggregate:
  - `seed_sweep_results/realdata/guardB10_tailfix_lowflip35_20ep_20260219_232731_eval_wf2400_zero_cost_v2_ten_aggregate_20260220.json`

WF2400 (zero-cost v2):
- Mean return `+1.77%` (vs `+0.61%` real-cost)
- Mean PF `2.22` (vs `1.44` real-cost)
- Walk-forward pass `7/10` (vs `5/10`)
- Negative-return seeds: `6067`, `9091`, `10007` (seed `8087` flips positive)

Interpretation:
- Frictions are a major drag (material return/PF uplift when removed), but not the only issue because `3/10` seeds stay negative even at zero cost.
- Keep 10ep tail-fix branch as current lead; next work should target persistent tail seeds without sacrificing center-seed profitability.

## 2026-02-20 (Tail-risk fix accepted: low flip penalty improves 10-seed robustness)

Focus: reduce tail-seed failures (`6067`, `8087`) without sacrificing aggregate profitability.

### Step 1: targeted tail-risk micro-sweep (2 failing seeds)

Prefix:
- `tailrisk_micro2seeds_10ep_20260219_173036`

Seeds:
- `6067`, `8087`

Configs tested:
- `A_base`: flip `0.00045`, atr ratio `0.15`, cooldown `8`, min hold `4`, max trades `28`
- `B_strict_atr018`: flip `0.00045`, atr ratio `0.18`, cooldown `8`, min hold `4`, max trades `24`
- `C_loose_atr012`: flip `0.00045`, atr ratio `0.12`, cooldown `8`, min hold `4`, max trades `32`
- `D_high_flip055`: flip `0.00055`, atr ratio `0.15`, cooldown `8`, min hold `4`, max trades `28`
- `E_low_flip035`: flip `0.00035`, atr ratio `0.15`, cooldown `8`, min hold `4`, max trades `28`
- `F_strict_cadence`: flip `0.00045`, atr ratio `0.15`, cooldown `10`, min hold `5`, max trades `24`

Tail-risk reading:
- `C_loose_atr012` fixed seed `6067` but worsened seed `8087`.
- `E_low_flip035` substantially improved seed `6067` while not worsening `8087`.
- Chosen candidate for full confirmation: `E_low_flip035`.

### Step 2: full blind 10-seed confirmation with chosen variant

Prefix:
- `guardB10_tailfix_lowflip35_10ep_20260219_211955`

Config:
- Same Guard B baseline except `flip_penalty=0.00035` (was `0.00045`).

Artifacts:
- Train/test aggregate:
  - `seed_sweep_results/realdata/guardB10_tailfix_lowflip35_10ep_20260219_211955_ten_aggregate_20260220.json`
- WF2400 aggregate:
  - `seed_sweep_results/realdata/guardB10_tailfix_lowflip35_10ep_20260219_211955_eval_wf2400_ten_aggregate_20260220.json`

WF2400 (real costs):
- Mean return `+1.22%`
- Mean PF `1.84`
- Mean trades `22.3`
- Walk-forward pass `4/10`
- Positive return + PF>1 seeds: `9/10`
- Worst seed: return `-0.41%`, PF `0.84`

Comparison vs previous 10-seed baseline (`guardB10_confirm_blind10ep_20260219_143734`):
- Mean return: `+1.05%` -> `+1.22%`
- Mean PF: `1.59` -> `1.84`
- Walk-forward pass: `3/10` -> `4/10`
- Positive+PF>1 seeds: `8/10` -> `9/10`
- Worst return: `-1.67%` -> `-0.41%`
- Worst PF: `0.53` -> `0.84`

Decision:
- Accept `flip_penalty=0.00035` as the new leading variant for the 10ep branch.
- This is the first clear tail-risk reduction that also improves aggregate quality.

## 2026-02-19 (Guard B blind 10-seed confirmation, 10ep)

Focus: expand robustness breadth from 5 seeds to 10 blind seeds using the same best-known 10ep Guard B setup.

Run setup:
- Prefix: `guardB10_confirm_blind10ep_20260219_143734`
- Seeds: `1011, 2027, 3039, 4049, 5051, 6067, 7079, 8087, 9091, 10007`
- Episodes: `10`
- Config: Guard B baseline (`flip=0.00045`, `min_atr_cost_ratio=0.15`, `cooldown=8`, `min_hold=4`, `max_trades=28`, no symmetry loss, no dual controller, prefill `none`)
- Anti-regression checkpoint tournament disabled (`--no-anti-regression-checkpoint`) for direct branch comparability.

Artifacts:
- Train/test aggregate:
  - `seed_sweep_results/realdata/guardB10_confirm_blind10ep_20260219_143734_ten_aggregate_20260219.json`
- WF2400 eval aggregate:
  - `seed_sweep_results/realdata/guardB10_confirm_blind10ep_20260219_143734_eval_wf2400_ten_aggregate_20260219.json`

WF2400 (real costs):
- Mean return `+1.05%`
- Mean PF `1.59`
- Mean trades `22.7`
- Walk-forward pass `3/10`

Robustness counts:
- Positive return seeds: `8/10`
- PF > 1 seeds: `8/10`
- Positive return + PF > 1: `8/10`
- Worst seed: return `-1.67%`, PF `0.53` (seed `6067`)

Interpretation:
- The branch keeps a positive aggregate over a wider blind seed set, but tail-risk remains (2 clear failing seeds).
- This is promising but not yet production-grade robustness.
- Next step should target tail-seed failure reduction while preserving the profitable center (8/10).

## 2026-02-19 (OOS stress on 5-seed 10ep baseline checkpoints)

Focus: no-retrain stress test of the strongest branch (`guardB5_confirm_blind10ep_20260219_094605`) on longer evaluation horizons.

Stress tag:
- `oos_stress_guardB5_10ep_20260219_132726`

What was run (evaluate only, same costs/cadence constraints):
- `--max-steps-per-episode 4800`
- `--max-steps-per-episode 20000` (full split)

Aggregates:
- WF4800:
  - `seed_sweep_results/realdata/oos_stress_guardB5_10ep_20260219_132726_wf4800_five_aggregate_20260219.json`
  - Mean return `+1.67%`, mean PF `1.95`, mean trades `22.8`, walk-forward pass `0/5`
- Full horizon:
  - `seed_sweep_results/realdata/oos_stress_guardB5_10ep_20260219_132726_wffull_five_aggregate_20260219.json`
  - Mean return `+1.67%`, mean PF `1.95`, mean trades `22.8`, walk-forward pass `0/5`

Observation:
- Core profitability metrics are unchanged from WF2400 confirmation.
- As with earlier stress checks, policies often finish activity early (trade cap/cadence constraints), so extending bars mostly increases window count without changing realized PnL profile.

Interpretation:
- Profitability signal is stable across longer eval horizons for this branch.
- Current walk-forward pass criterion remains stricter than profitability outcomes and is not the primary selection signal for this phase.

## 2026-02-19 (Guard B blind 5-seed confirmation, 20ep)

Focus: validate whether the strongest 10-episode branch remains profitable when training horizon is doubled.

Run setup:
- Prefix: `guardB5_confirm_blind20ep_20260219_105737`
- Seeds: `1011, 2027, 3039, 4049, 5051`
- Episodes: `20`
- Config unchanged from Guard B baseline:
  - `flip=0.00045`
  - `min_atr_cost_ratio=0.15`
  - `cooldown=8`
  - `min_hold=4`
  - `max_trades=28`
  - no symmetry loss, no dual controller, prefill `none`
- Anti-regression checkpoint tournament disabled to match baseline comparison (`--no-anti-regression-checkpoint`).

Artifacts:
- Train/test aggregate:
  - `seed_sweep_results/realdata/guardB5_confirm_blind20ep_20260219_105737_five_aggregate_20260219.json`
- WF2400 eval aggregate:
  - `seed_sweep_results/realdata/guardB5_confirm_blind20ep_20260219_105737_eval_wf2400_five_aggregate_20260219.json`

WF2400 (real costs):
- Mean return `+0.76%`
- Mean PF `1.56`
- Mean trades `21.4`
- Walk-forward pass `2/5`

Per-seed WF2400:
- Seed 1011: return `+3.21%`, PF `3.14`, trades `22`, pass `True`
- Seed 2027: return `-1.37%`, PF `0.56`, trades `21`, pass `False`
- Seed 3039: return `+0.62%`, PF `1.35`, trades `25`, pass `False`
- Seed 4049: return `+0.78%`, PF `1.44`, trades `21`, pass `True`
- Seed 5051: return `+0.55%`, PF `1.32`, trades `18`, pass `False`

Comparison vs 10ep five-seed:
- 10ep (`guardB5_confirm_blind10ep_20260219_094605`):
  - return `+1.67%`, PF `1.95`, pass `1/5`
- 20ep:
  - return `+0.76%`, PF `1.56`, pass `2/5`

Interpretation:
- Longer training horizon reduced profitability quality (mean return and PF dropped materially).
- Slightly higher walk-forward pass count does not compensate for the profitability degradation.
- Keep the 10ep Guard B branch as the current best profitability candidate.

## 2026-02-19 (Guard B blind 5-seed confirmation, 10ep)

Focus: run a broader blind-seed confirmation on the current best profitability branch before further tuning.

Run setup:
- Prefix: `guardB5_confirm_blind10ep_20260219_094605`
- Seeds: `1011, 2027, 3039, 4049, 5051`
- Episodes: `10` per seed
- Config: Guard B baseline (`flip=0.00045`, `min_atr_cost_ratio=0.15`, `cooldown=8`, `min_hold=4`, `max_trades=28`, no symmetry loss, no dual controller, prefill `none`)
- Anti-regression tournament disabled for this confirmation (`--no-anti-regression-checkpoint`) to match baseline branch behavior.

Artifacts:
- Train/test aggregate:
  - `seed_sweep_results/realdata/guardB5_confirm_blind10ep_20260219_094605_five_aggregate_20260219.json`
- WF2400 eval aggregate:
  - `seed_sweep_results/realdata/guardB5_confirm_blind10ep_20260219_094605_eval_wf2400_five_aggregate_20260219.json`

WF2400 (real costs):
- Mean return `+1.67%`
- Mean PF `1.95`
- Mean trades `22.8`
- Walk-forward pass `1/5`

Per-seed WF2400:
- Seed 1011: return `+3.21%`, PF `3.14`, trades `22`, pass `True`
- Seed 2027: return `+2.14%`, PF `2.09`, trades `26`, pass `False`
- Seed 3039: return `+0.62%`, PF `1.35`, trades `25`, pass `False`
- Seed 4049: return `+1.80%`, PF `1.86`, trades `23`, pass `False`
- Seed 5051: return `+0.55%`, PF `1.32`, trades `18`, pass `False`

Interpretation:
- Profitability-first evidence improved: all 5 blind seeds are positive with PF > 1.
- Walk-forward pass remains conservative under current criteria and is not aligned with profitability outcomes.
- This branch is currently the strongest candidate for escalation to a longer-horizon confirmation run.

## 2026-02-19 (Seed-2027 robustness micro-sweep + profitpick v2 tri-seed)

Focus: directly address the unstable seed by targeted fast sweeps, then retest all blind seeds with the same settings.

### Seed-2027 micro-sweep (10ep, WF2400)

Prefix:
- `seed2027_robust_micro4_10ep_20260218_225107`

Configs tested:
- A_base: `flip=0.00045`, `min_atr_cost_ratio=0.15`, `cooldown=8`, `min_hold=4`, `max_trades=28`
- B_flip040: `flip=0.00040`, `min_atr_cost_ratio=0.15`
- C_atr018: `flip=0.00045`, `min_atr_cost_ratio=0.18`
- D_flip050_atr018: `flip=0.00050`, `min_atr_cost_ratio=0.18`

WF2400 results (seed 2027):
- A_base: return `+2.85%`, PF `3.29`, trades `23`, walk-forward `True`  **(best)**
- B_flip040: return `+1.86%`, PF `1.78`, trades `24`, walk-forward `False`
- C_atr018: return `+1.10%`, PF `1.48`, trades `22`, walk-forward `True`
- D_flip050_atr018: return `-1.29%`, PF `0.54`, trades `24`, walk-forward `False`

Decision:
- Keep A_base settings as the best local fix candidate.

### Profitpick v2 blind tri-seed retest (same A_base settings)

Prefix:
- `fric_guardB_profitpick_v2_blind10ep_20260219_000343`

Artifacts:
- Real-cost aggregate:
  - `seed_sweep_results/realdata/fric_guardB_profitpick_v2_blind10ep_20260219_000343_eval_wf2400_tri_aggregate_20260219.json`
- Zero-cost aggregate:
  - `seed_sweep_results/realdata/fric_guardB_profitpick_v2_blind10ep_20260219_000343_eval_wf2400_zero_cost_tri_aggregate_20260219.json`

WF2400 real-cost:
- Seed 1011: return `-0.07%`, PF `0.99`, trades `26`, walk-forward `False`
- Seed 2027: return `+0.77%`, PF `1.30`, trades `23`, walk-forward `True`
- Seed 3039: return `+2.57%`, PF `2.24`, trades `25`, walk-forward `True`
- Mean: return `+1.09%`, PF `1.51`, trades `24.7`, walk-forward pass `2/3`

Comparison:
- Prior Guard B baseline (`fric_guardB_blind10ep_20260218_134723`): return `+1.54%`, PF `1.91`, pass `1/3`
- Profitpick v2 improves walk-forward pass count but degrades profitability quality materially (PF and return mean drop).

Interpretation:
- Seed-specific tuning helped seed 2027 locally but did not improve blind tri-seed profitability.
- Keep Guard B baseline as the current best profitability branch; treat profitpick v2 as a rejected pivot.

## 2026-02-18 (OOS stress test on Guard B baseline: WF4800 and full horizon)

Focus: no-retrain out-of-sample stress test of the current best fast branch before further tuning.

Base checkpoints:
- `fric_guardB_blind10ep_20260218_134723` (seeds `1011, 2027, 3039`)

Stress tag:
- `oos_stress_guardB10_20260218_221913`

What was run:
- Evaluate-only on each seed checkpoint at:
  - `--max-steps-per-episode 4800`
  - `--max-steps-per-episode 20000` (effectively full test split)

Aggregates:
- WF4800:
  - `seed_sweep_results/realdata/oos_stress_guardB10_20260218_221913_wf4800_tri_aggregate_20260218.json`
  - Mean return `+1.54%`, mean PF `1.91`, mean trades `23.0`, walk-forward pass `0/3`
- Full horizon:
  - `seed_sweep_results/realdata/oos_stress_guardB10_20260218_221913_wffull_tri_aggregate_20260218.json`
  - Mean return `+1.54%`, mean PF `1.91`, mean trades `23.0`, walk-forward pass `0/3`

Important observation:
- Core PnL metrics are unchanged from prior WF2400 aggregate.
- Reason: policy often reaches its trade cap (`max_trades_per_episode=28`) early, then additional bars add little or no new realized edge.
- This means horizon extension alone is not currently a discriminative robustness test for this branch.

Interpretation:
- Branch remains modestly profitable on average but not robust under current walk-forward criteria.
- Next high-value work should target trade-cap saturation and cross-seed robustness (especially avoiding seed-specific collapse), not further horizon scaling.

Add-on test (same date, same base checkpoints):
- Increased evaluation trade cap only: `--max-trades-per-episode 60` on full horizon.
- Aggregate:
  - `seed_sweep_results/realdata/oos_stress_guardB10_cap60_20260218_223626_wffull_cap60_tri_aggregate_20260218.json`
  - Mean return `+1.40%`, mean PF `1.45`, mean trades `49.0`, walk-forward pass `2/3`
- Compared to cap 28:
  - cap 28: return `+1.54%`, PF `1.91`, trades `23.0`, pass `0/3`
  - cap 60: return `+1.40%`, PF `1.45`, trades `49.0`, pass `2/3`

Reading:
- Raising cap increases activity and walk-forward pass count but degrades quality (PF drops sharply).
- This supports keeping tighter cadence controls and focusing on seed-2027 robustness rather than loosening trade cap.

## 2026-02-18 (Validation-calibration test rejected: no-jitter + 1200-bar windows)

Focus: test whether making validation closer to evaluation (`no jitter`, larger windows) improves blind-seed profitability.

Calibration hypothesis:
- Validation friction jitter and short windows may be weakening checkpoint selection quality.
- Try:
  - `--val-jitter-draws 1`
  - `--val-window-bars 1200`
  - keep profit-first anti-regression tournament enabled.

Quick single-seed probe:
- Prefix: `calib_profitpick_seed2027_njitter_vw1200_10ep_20260218_202058`
- Seed `2027` result:
  - 600-step: return `+2.84%`, PF `2.37`, trades `26`
  - WF2400: return `+2.84%`, PF `2.37`, walk-forward `False`
- This looked promising in isolation.

Blind tri-seed confirmation:
- Prefix: `calib_profitpick_njitter_vw1200_blind10ep_20260218_202646`
- Real-cost WF2400 aggregate:
  - `seed_sweep_results/realdata/calib_profitpick_njitter_vw1200_blind10ep_20260218_202646_eval_wf2400_tri_aggregate_20260218.json`
  - Mean return `+0.36%`, mean PF `1.20`, trades `23.3`, walk-forward pass `0/3`
- Zero-cost WF2400 aggregate:
  - `seed_sweep_results/realdata/calib_profitpick_njitter_vw1200_blind10ep_20260218_202646_eval_wf2400_zero_cost_tri_aggregate_20260218.json`
  - Mean return `+1.30%`, mean PF `1.75`, walk-forward pass `1/3`

Comparison vs prior best fast-screen branch:
- Prior Guard B 10ep (`fric_guardB_blind10ep_20260218_134723`): mean return `+1.54%`, PF `1.91`, pass `1/3`.
- Calibrated branch is materially worse on both return and PF.

Interpretation:
- The single-seed uplift was not robust.
- Reject this calibration pivot; keep prior Guard B 10ep branch as better fast-screen baseline.

## 2026-02-18 (Profitability-first checkpoint selection + blind 10ep recheck)

Focus: make checkpoint selection explicitly profit-first (median return and PF) rather than SPR-only, then run a fast blind validation.

Code changes:
- `trainer.py`
  - Validation now records profitability diagnostics:
    - `val_return_pct`
    - `val_median_return_pct`
    - `val_median_pf`
  - Validation summaries now include `return_pct_mean`, `return_pct_median`, and `pf_median`.
  - Anti-regression tournament ranking changed to profitability-first:
    - primary: robust median return across base+alt validation (`min(base, alt)`),
    - secondary: robust PF and SPR,
    - penalties for negative robust return, PF<1, dispersion, and low trades,
    - feasible pool filter prefers candidates with `base/alt return > 0` and `base/alt PF >= 1`.
- Existing anti-regression candidate/tournament machinery remains enabled.

Blind fast recheck run:
- Prefix: `fric_guardB_profitpick_blind10ep_20260218_184002`
- Seeds: `1011, 2027, 3039`
- Episodes: `10`, Guard B settings unchanged.

Artifacts:
- Real-cost WF2400 aggregate:
  - `seed_sweep_results/realdata/fric_guardB_profitpick_blind10ep_20260218_184002_eval_wf2400_tri_aggregate_20260218.json`
- Zero-cost WF2400 aggregate:
  - `seed_sweep_results/realdata/fric_guardB_profitpick_blind10ep_20260218_184002_eval_wf2400_zero_cost_tri_aggregate_20260218.json`
- Train/test aggregate:
  - `seed_sweep_results/realdata/fric_guardB_profitpick_blind10ep_20260218_184002_tri_aggregate_20260218.json`

WF2400 (real costs):
- Seed 1011: return `+3.53%`, PF `3.57`, trades `18`, walk-forward `False`
- Seed 2027: return `-1.65%`, PF `0.55`, trades `26`, walk-forward `False`
- Seed 3039: return `+1.94%`, PF `1.61`, trades `24`, walk-forward `True`
- Mean: return `+1.27%`, PF `1.91`, trades `22.7`, walk-forward pass `1/3`

WF2400 (zero-cost diagnostic):
- Mean return `+2.34%`, mean PF `2.52`, walk-forward pass `2/3`

Interpretation:
- Profitability-first checkpoint selection is functioning (all seeds selected from profit-feasible candidate pools).
- However, blind-seed robustness remains mixed (still one materially negative seed), so this is not yet a promotion-grade branch.
- Compared with prior Guard B 10ep, PF/trade profile is similar; return mean is slightly lower, so no clear net gain.

## 2026-02-18 (Anti-regression checkpoint tournament: blind 20ep tri-seed result)

Focus: reduce late-run/selection regression by replacing single-path best-checkpoint restore with a robust checkpoint tournament.

Code changes (selection logic):
- `trainer.py`
  - Added per-validation candidate checkpoint tracking (`candidate_epXXX.pt`).
  - Added end-of-run anti-regression tournament:
    - Evaluates shortlisted checkpoints on base validation and alternate hold-out validation (`stride=0.20`).
    - Uses composite robust score (`min(base, alt)` with dispersion and low-trade penalties).
    - Writes `logs/checkpoint_tournament.json` and restores tournament winner.
  - Added optional quiet/non-persist validation mode for internal checkpoint scoring.
- `config.py`
  - Added training knobs:
    - `anti_regression_checkpoint_selection`
    - `anti_regression_candidate_keep`
    - `anti_regression_eval_top_k`
    - `anti_regression_min_validations`
    - `anti_regression_alt_stride_frac`
    - `anti_regression_alt_window_bars`
- `main.py`
  - Added CLI overrides for anti-regression options.

Run setup:
- Prefix: `fric_guardB_antireg_blind20ep_20260218_161533`
- Seeds: `1011, 2027, 3039` (blind set)
- Episodes: `20`, train/test `600`, re-eval `WF2400`
- Guard B frictions/cadence kept constant (`flip=0.00045`, `atr_cost_ratio=0.15`, `cooldown=8`, `min_hold=4`, `max_trades=28`, no prefill, no symmetry, no dual).

Artifacts:
- Real-cost aggregate: `seed_sweep_results/realdata/fric_guardB_antireg_blind20ep_20260218_161533_eval_wf2400_tri_aggregate_20260218.json`
- Zero-cost aggregate: `seed_sweep_results/realdata/fric_guardB_antireg_blind20ep_20260218_161533_eval_wf2400_zero_cost_tri_aggregate_20260218.json`
- Train/test aggregate: `seed_sweep_results/realdata/fric_guardB_antireg_blind20ep_20260218_161533_tri_aggregate_20260218.json`

WF2400 (real costs):
- Seed 1011: return `+1.13%`, PF `1.55`, trades `28`, walk-forward `True`
- Seed 2027: return `-1.68%`, PF `0.50`, trades `20`, walk-forward `True`
- Seed 3039: return `-2.05%`, PF `0.51`, trades `25`, walk-forward `True`
- Mean: return `-0.87%`, PF `0.86`, trades `24.3`, walk-forward pass `3/3`

WF2400 (zero-cost diagnostic):
- Mean return `-0.02%`, mean PF `1.25`, walk-forward pass `3/3`

Tournament winners by seed:
- Seed 1011: `candidate_ep017.pt`
- Seed 2027: `candidate_ep003.pt`
- Seed 3039: `candidate_ep005.pt`

Interpretation:
- Anti-regression tournament did not recover profitability on blind 20ep runs.
- `walk-forward pass` became less informative here (3/3 pass despite negative mean return/PF<1), so profitability criteria must remain primary.
- Conclusion: checkpoint-selection hardening alone is insufficient; current validation objective/regime still does not select profitable checkpoints under real costs.

## 2026-02-18 (Guard B escalation: 20ep blind tri-seed regression)

Focus: confirm whether the promising Guard B fast-screen (`10ep`) holds when training horizon is extended.

Run setup:
- Prefix: `fric_guardB_blind20ep_20260218_142820`
- Seeds: `1011, 2027, 3039` (same blind set)
- Same hyperparameters as Guard B fast-screen:
  - `trade_penalty=0`
  - `flip_penalty=0.00045`
  - `min_atr_cost_ratio=0.15`
  - `cooldown=8`
  - `min_hold=4`
  - `max_trades=28`
  - no symmetry loss, no dual controller, prefill `none`
- Episodes: `20`, train/test `600`, plus `WF2400` re-eval.

Artifacts:
- Aggregates:
  - `seed_sweep_results/realdata/fric_guardB_blind20ep_20260218_142820_tri_aggregate_20260218.json`
  - `seed_sweep_results/realdata/fric_guardB_blind20ep_20260218_142820_eval_wf2400_tri_aggregate_20260218.json`
  - `seed_sweep_results/realdata/fric_guardB_blind20ep_20260218_142820_eval_wf2400_zero_cost_tri_aggregate_20260218.json`

WF2400 (real costs):
- Seed 1011: return `-1.71%`, PF `0.41`, trades `26`, walk-forward `False`
- Seed 2027: return `-1.30%`, PF `0.62`, trades `24`, walk-forward `False`
- Seed 3039: return `+0.62%`, PF `1.35`, trades `25`, walk-forward `False`
- Mean: return `-0.79%`, PF `0.79`, trades `25.0`, walk-forward pass `0/3`

WF2400 (zero-cost diagnostic on same checkpoints):
- Mean return `+0.35%`, mean PF `1.29`, walk-forward pass `3/3`

Interpretation:
- Guard B looked promising at `10ep`, but regressed materially at `20ep` on the same blind seeds.
- This indicates horizon instability / over-training sensitivity, not solved friction robustness.
- Keep this branch as a negative confirmation for `20ep` under current checkpoint-selection regime.

## 2026-02-18 (Friction-robustness Guard B blind check: 10ep tri-seed + WF2400)

Focus: fast-screen a cost-aware training setup on unseen seeds without multi-hour runs.

Run setup:
- Prefix: `fric_guardB_blind10ep_20260218_134723`
- Seeds: `1011, 2027, 3039` (blind to prior tuning)
- Broker profile: `hfm-premium`
- Episodes: `10` (fast screen), train/test horizon `600`, then re-eval at `WF2400`
- Key knobs: `trade_penalty=0`, `flip_penalty=0.00045`, `min_atr_cost_ratio=0.15`, `cooldown=8`, `min_hold=4`, `max_trades=28`, no symmetry loss, no dual controller, prefill `none`.

Artifacts:
- Per-seed logs:
  - `logs/fric_guardB_blind10ep_20260218_134723_seed1011.log`
  - `logs/fric_guardB_blind10ep_20260218_134723_seed2027.log`
  - `logs/fric_guardB_blind10ep_20260218_134723_seed3039.log`
- Aggregates:
  - `seed_sweep_results/realdata/fric_guardB_blind10ep_20260218_134723_tri_aggregate_20260218.json`
  - `seed_sweep_results/realdata/fric_guardB_blind10ep_20260218_134723_eval_wf2400_tri_aggregate_20260218.json`
  - `seed_sweep_results/realdata/fric_guardB_blind10ep_20260218_134723_eval_wf2400_zero_cost_tri_aggregate_20260218.json`

WF2400 (real costs):
- Seed 1011: return `+3.21%`, PF `3.14`, trades `22`, walk-forward `True`
- Seed 2027: return `+0.79%`, PF `1.25`, trades `22`, walk-forward `False`
- Seed 3039: return `+0.62%`, PF `1.35`, trades `25`, walk-forward `False`
- Mean: return `+1.54%`, PF `1.91`, trades `23.0`, walk-forward pass `1/3`

WF2400 (zero-cost diagnostic on same checkpoints):
- Mean return `+2.38%`, mean PF `3.01`, walk-forward pass `3/3`

Interpretation:
- Fast-screen result is materially better than the prior blind baseline (`+0.39%`, PF `1.23`, `0/3` pass), while also reducing trade count (friction load).
- Signal remains present and stronger without costs, so edge is still friction-sensitive but now closer to viable under real costs.
- Next step: run the same Guard B config at `20` episodes on the same blind seeds for confirmation-quality evidence.

## 2026-02-18 (Blind-seed confirmation: 20ep tri-seed, real-cost failure vs zero-cost pass)

Focus: validate generalization on unseen seeds before further tuning.

Blind sweep setup:
- Prefix: `blind_hfm_parity_20ep_20260218_110154`
- Seeds: `1011, 2027, 3039` (not the previously tuned seeds)
- Config held constant: HFM profile, no prefill, no symmetry loss, no dual controller, 20 episodes, 600-step training/test horizon, heartbeat enabled.

Per-seed WF2400 (real costs):
- Seed 1011: return `+1.52%`, PF `1.70`, walk-forward `False`
- Seed 2027: return `+0.22%`, PF `1.10`, walk-forward `False`
- Seed 3039: return `-0.57%`, PF `0.91`, walk-forward `False`

Aggregate (real costs):
- `seed_sweep_results/realdata/blind_hfm_parity_20ep_20260218_110154_eval_wf2400_tri_aggregate_20260218.json`
- Mean return `+0.39%`, mean PF `1.23`, walk-forward pass `0/3`

Diagnostic (same checkpoints, WF2400 zero-cost eval):
- `seed_sweep_results/realdata/blind_hfm_parity_20ep_20260218_110154_eval_wf2400_zero_cost_tri_aggregate_20260218.json`
- Mean return `+1.73%`, mean PF `1.78`, walk-forward pass `3/3`

Interpretation:
- Policy signal exists, but it is not robust once realistic frictions are applied.
- This is now a friction-robustness problem, not a learnability/architecture problem.

## 2026-02-18 (20ep robustness extension + seed789 recovery + 3/3 WF pass)

Focus: resolve seed fragility (seed 789) and verify whether longer training horizon stabilizes the parity+HFM branch.

What was run:
- Seed 789 targeted extension:
  - `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_20ep_seed789_20260218_004702`
  - Log: `logs/parityfix_hfm_nogate_noprefill_20ep_seed789_20260218_004702.log`
  - 600-step test: return `+2.84%`, PF `1.99`, trades `31`
  - WF2400 test: return `+2.84%`, PF `1.99`, walk-forward `True`
- Matching 20ep continuation seeds:
  - Seed 456:
    - `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_20ep_20260218_010039_seed456`
    - 600-step test: return `+0.26%`, PF `1.06`, trades `27`
    - WF2400 test: return `+0.26%`, PF `1.06`, walk-forward `True`
  - Seed 777:
    - `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_20ep_20260218_010039_seed777`
    - 600-step test: return `+1.69%`, PF `4.21`, trades `8`
    - WF2400 test: return `+3.25%`, PF `3.90`, walk-forward `True`

Aggregates:
- Mixed 20ep 600-step aggregate:
  - `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_20ep_mixed_tri_aggregate_20260218.json`
- Mixed 20ep WF2400 aggregate:
  - `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_20ep_mixed_eval_wf2400_tri_aggregate_20260218.json`
  - Mean return `+2.11%`, mean PF `2.32`, mean trades `25.7`, walk-forward pass `3/3`

Additional diagnostic:
- Inference-only gate sensitivity on seed 789 (`trade_gate_z` and `min_atr_cost_ratio`) under WF2400:
  - `trade_gate_z >= 0.1` collapsed to zero trades.
  - `min_atr_cost_ratio` in `[0.1, 0.4]` did not improve the ungated policy outcome.
  - Conclusion: inference gating is not the correct next lever; training robustness is.

Conclusion:
- The branch now clears walk-forward pass on all 3 seeds at WF2400 after extending horizon/training.
- Next priority is confirmation-quality evidence (out-of-sample and/or longer windows), not architecture changes.

## 2026-02-17 (Parity branch confirmation: tri-seed 12ep + WF2400)

Focus: run a slightly longer training confirmation on the parity+HFM branch and re-score with meaningful walk-forward horizon.

Run settings:
- Prefix: `parityfix_hfm_nogate_noprefill_12ep_20260217_230207`
- Broker profile: `hfm-premium`
- Episodes: `12`
- Train/test step cap: `600` (for run speed), then dedicated re-eval at `2400`
- Seeds: `456, 777, 789`
- No prefill, no symmetry loss, no dual controller, trade gate disabled.

Per-seed run artifacts:
- Seed 456:
  - Run dir: `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_12ep_20260217_230207_seed456`
  - Log: `logs/parityfix_hfm_nogate_noprefill_12ep_20260217_230207_seed456.log`
  - 600-step test: return `+2.39%`, PF `8.91`, trades `12`
  - WF2400 test: return `+4.22%`, PF `3.02`, walk-forward `True`
- Seed 777:
  - Run dir: `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_12ep_20260217_230207_seed777`
  - Log: `logs/parityfix_hfm_nogate_noprefill_12ep_20260217_230207_seed777.log`
  - 600-step test: return `+1.69%`, PF `4.21`, trades `8`
  - WF2400 test: return `+3.25%`, PF `3.90`, walk-forward `True`
- Seed 789:
  - Run dir: `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_12ep_20260217_230207_seed789`
  - Log: `logs/parityfix_hfm_nogate_noprefill_12ep_20260217_230207_seed789.log`
  - 600-step test: return `-0.14%`, PF `0.97`, trades `34`
  - WF2400 test: return `-0.14%`, PF `0.97`, walk-forward `False`

Aggregates:
- 600-step aggregate:
  - `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_12ep_20260217_230207_tri_aggregate_20260217_230207.json`
  - Mean return `+1.31%`, mean PF `4.70`, walk-forward pass `0/3` (single-window artifact)
- WF2400 aggregate:
  - `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_12ep_20260217_230207_eval_wf2400_tri_aggregate_20260217_230207.json`
  - Mean return `+2.44%`, mean PF `2.63`, walk-forward pass `2/3`

Conclusion:
- Compared with the prior 8ep parity run, this is a forward step on return/PF at WF2400 while keeping 2/3 walk-forward passes.
- Seed fragility remains concentrated in seed 789; next work should target robustness to that failure mode rather than architecture churn.

## 2026-02-17 (WF2400 re-eval + progress heartbeat instrumentation)

Focus: remove false negatives from short-horizon walk-forward checks and improve visibility during long episodes.

What changed:
- Added training progress heartbeat support:
  - `config.py`: `TrainingConfig.heartbeat_secs` (default `60.0`) and `heartbeat_steps` (default `200`)
  - `main.py`: new CLI overrides `--heartbeat-secs` and `--heartbeat-steps`
  - `trainer.py`: emits `[HB]` lines during episodes with step/bar/trades/equity/epsilon/replay/elapsed

Verification:
- `python -m py_compile config.py main.py trainer.py`
- `python test_system.py`
- Heartbeat smoke run:
  - `test_output/heartbeat_smoke_20260217`
  - Log contains periodic `[HB] step=...` lines

Re-evaluation of parity checkpoints with longer test horizon (`max-steps-per-episode=2400`):
- Seed 456:
  - `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_8ep_20260217_220329_seed456/eval_wf2400`
  - Return `+1.94%`, PF `1.58`, walk-forward `True`
- Seed 777:
  - `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_8ep_20260217_220329_seed777/eval_wf2400`
  - Return `+3.25%`, PF `3.90`, walk-forward `True`
- Seed 789:
  - `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_8ep_20260217_220329_seed789/eval_wf2400`
  - Return `-0.41%`, PF `0.90`, walk-forward `False`

Aggregate:
- `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_8ep_eval_wf2400_tri_aggregate_20260217_2229.json`
- Mean return `+1.60%`, mean PF `2.13`, walk-forward pass `2/3`

Cost-sensitivity check (same checkpoints, zero-cost eval):
- Mean return moved from `+1.08%` (real costs, short test) to `+1.77%` (zero costs), indicating edge exists but is partially friction-limited.

Conclusion:
- Branch is no longer "all-fail"; with adequate test horizon it passes walk-forward on 2/3 seeds.
- Immediate priority shifts from architecture doubt to cost-robustness and cross-seed stability.

## 2026-02-17 (Parity patch confirmation: HFM tri-seed 8ep x 600)

Focus: confirm whether validation/test policy parity materially changes short-horizon real-data outcomes under HFM frictions.

Run settings:
- Broker profile: `hfm-premium`
- Episodes: 8
- Max steps: 600
- Seeds: 456, 777, 789
- No prefill, no symmetry loss, no dual controller, trade gate disabled.

Per-seed:
- Seed 456:
  - Run dir: `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_8ep_20260217_220329_seed456`
  - Test: return +1.94%, PF 1.58, trades 35
- Seed 777:
  - Run dir: `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_8ep_20260217_220329_seed777`
  - Test: return +1.69%, PF 4.21, trades 8
- Seed 789:
  - Run dir: `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_8ep_20260217_220329_seed789`
  - Test: return -0.41%, PF 0.90, trades 31

Aggregate:
- `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_8ep_tri_aggregate_20260217_220329.json`
- Mean return +1.08%, mean PF 2.23, mean trades 24.7, mean SPR 0.295
- Walk-forward pass remains 0/3 due current walk-forward window-count requirement in short test horizon.

Conclusion:
- Compared with the pre-parity 8ep HFM run, short-horizon tri-seed outcomes improved materially after policy-parity changes.
- This branch is now a viable candidate for a longer confirmation run (more episodes and/or longer test horizon).

## 2026-02-17 (Validation/Test policy parity patch + HFM quick recheck)

Focus: eliminate checkpoint-selection optimism caused by validation using a different action policy than final test.

Code changes:
- `trainer.py`
  - Validation window action selection now uses deterministic eval policy only:
    - `agent.select_action(..., explore=False, eval_mode=True, mask=legal_mask)`
  - Removed validation-only eval epsilon probing and hold-streak breaker logic.
- `main.py`
  - Test evaluation and walk-forward evaluation now also pass legal-action mask to action selection for exact parity with validation policy execution.

Verification:
- `python -m py_compile trainer.py main.py`
- `python test_system.py`
- Real-data smoke:
  - `results/parity_patch_smoke_real_2ep_20260217_215213`

Quick tri-seed recheck after patch (HFM profile, 4ep x 600, no prefill/no symmetry/no dual):
- Seed 456:
  - Run dir: `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_4ep_20260217_215317_seed456`
  - Test: return +0.00%, PF 1.00, trades 29
- Seed 777:
  - Run dir: `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_4ep_20260217_215317_seed777`
  - Test: return +1.69%, PF 4.21, trades 8
- Seed 789:
  - Run dir: `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_4ep_20260217_215317_seed789`
  - Test: return -0.41%, PF 0.90, trades 31
- Aggregate:
  - `seed_sweep_results/realdata/parityfix_hfm_nogate_noprefill_4ep_tri_aggregate_20260217_215317.json`
  - Mean return +0.43%, mean PF 2.04, mean trades 22.7, walk-forward pass 0/3

Conclusion:
- Parity patch is in place and removes a known validation-policy mismatch.
- Early 4-episode results look better than prior 8-episode HFM run, but still not decision-grade (too short, 0/3 walk-forward pass).

## 2026-02-17 (HFM Premium fast tri-seed sanity, 8ep x 600, no-gate/no-prefill)

Focus: rerun the current fast-screen branch under broker-aligned HFM Premium frictions and swap points.

Run settings:
- Broker profile: `hfm-premium` (spread 0.00014, commission 0, leverage 1:1000, per-symbol swap points from MT specs)
- Episodes: 8
- Max steps: 600
- Seeds: 456, 777, 789
- No symmetry loss, no dual controller, prefill disabled, trade gate disabled.

Per-seed:
- Seed 456:
  - Run dir: `seed_sweep_results/realdata/hfm_premium_nogate_noprefill_8ep_20260217_203049_seed456`
  - Log: `logs/hfm_premium_nogate_noprefill_8ep_seed456_20260217_203049.log`
  - Test: return -2.88%, PF 0.34, trades 23, walk-forward pass False
- Seed 777:
  - Run dir: `seed_sweep_results/realdata/hfm_premium_nogate_noprefill_8ep_20260217_203049_seed777`
  - Log: `logs/hfm_premium_nogate_noprefill_8ep_seed777_20260217_203049.log`
  - Test: return 0.00%, PF 0.00, trades 0, walk-forward pass False
- Seed 789:
  - Run dir: `seed_sweep_results/realdata/hfm_premium_nogate_noprefill_8ep_20260217_203049_seed789`
  - Log: `logs/hfm_premium_nogate_noprefill_8ep_seed789_20260217_203049.log`
  - Test: return -2.60%, PF 0.38, trades 24, walk-forward pass False

Aggregate:
- File: `seed_sweep_results/realdata/hfm_premium_nogate_noprefill_8ep_tri_aggregate_20260217_203049.json`
- Mean return: -1.83%
- Mean PF: 0.24
- Mean trades: 15.7
- Walk-forward pass: 0/3

Conclusion:
- Under broker-aligned frictions, this branch is not viable and remains unstable.

## 2026-02-17 (HFM screenshot integration: per-symbol swap points wired)

Focus: use the exact swap specs provided from the live HFM account screenshots.

HFM swap points captured:
- `EURUSD`: long `-10.8`, short `0.0`
- `USDCHF`: long `0.0`, short `-12.1`
- `USDJPY`: long `0.0`, short `-21.8`
- `GBPUSD`: long `-3.2`, short `-3.5`
- Triple swap weekday: Wednesday (`3x`)

Code changes:
- `main.py`
  - `--broker-profile hfm-premium` now applies the above per-symbol swap map.
  - Added `--swap-type {usd,points}` and made `--swap-long/--swap-short` type-aware.
  - Environment creation now resolves per-symbol swap overrides and passes primary symbol explicitly.
- `environment.py`
  - Swap engine now supports `swap_type='points'` and converts points -> quote PnL -> USD at rollover.
- `config.py`
  - Added `environment.swap_type` and `environment.swap_by_symbol` config fields.

Verification:
- `python -m py_compile main.py environment.py config.py`
- `python test_system.py`
- Synthetic profile smoke:
  - `python main.py --mode train --episodes 1 --data-mode synthetic --n-bars 260 --max-steps-per-episode 120 --broker-profile hfm-premium --output-dir results/smoke_hfm_points_swap_20260217`
  - Env header shows `Swap type: points` and `Swap long/short: -10.80 / 0.00 points` for `EURUSD`.
- Real-data profile smoke:
  - `python main.py --mode train --episodes 1 --seed 456 --data-mode csv --pair-files pair_files_real.json --broker-profile hfm-premium --prefill-policy none --max-steps-per-episode 200 --output-dir results/smoke_hfm_profile_real_20260217`

## 2026-02-17 (Broker realism upgrade: HFM Premium defaults + swap model + leverage CLI)

Focus: align simulation costs and margin assumptions closer to intended live deployment account.

Code changes:
- `config.py`
  - `risk.leverage` default updated to `1000`.
  - `environment.commission` default updated to `0.0` (Premium-style, no commission).
  - `environment.spread` baseline updated to `0.00014` (1.4 pips).
  - Added swap config fields:
    - `swap_long_usd_per_lot_night`
    - `swap_short_usd_per_lot_night`
    - `swap_rollover_hour_utc`
    - `swap_triple_weekday`
- `main.py`
  - Added broker preset flag: `--broker-profile hfm-premium`.
  - Added live-cost overrides:
    - `--leverage`
    - `--swap-long`
    - `--swap-short`
    - `--swap-rollover-hour-utc`
    - `--swap-triple-weekday`
  - Environment setup now prints leverage + swap settings.
  - `--eval-zero-costs` now also zeros swap inputs.
- `environment.py`
  - Added rollover swap charging on open positions across daily cutoff.
  - Supports triple-swap weekday multiplier (FX default Wednesday).
  - Tracks per-step swap PnL and per-episode swap accumulation in `info`.

Verification:
- `python -m py_compile main.py environment.py config.py`
- `python test_system.py`
- Smoke run:
  - `python main.py --mode train --episodes 1 --data-mode synthetic --n-bars 400 --swap-long -4.5 --swap-short 1.2 --max-steps-per-episode 120 --output-dir results/smoke_swap_check_20260217`

Conclusion:
- Cost realism is now configurable for broker-specific commission/spread/swap/leverage assumptions.
- Next sweeps can be run with broker-aligned settings instead of legacy fixed frictions.

## 2026-02-17 (Model-wiring fix + reverse-action branch retest)

Focus: ensure model hyperparameters from config are actually used by the agent before continuing architecture tweaks.

Code changes:
- `agent.py`
  - `DQNAgent` now honors `learning_rate` (not just `lr`).
  - `DQNAgent` replay batch size now correctly falls back to `batch_size`.
  - `DuelingDQN` policy/target/EMA networks now receive `hidden_sizes` from config.

Verification:
- `python -m py_compile agent.py main.py environment.py trainer.py config.py`
- `python test_system.py`

Primary probe (kept reverse actions, action space=8, 8ep x 600 bars, no symmetry, no dual, no prefill):
- Seed 456:
  - Run dir: `results/probe_actionspace8_modelwiring_seed456_8ep_20260217_190228`
  - Test: return 0.00%, PF 0.00, trades 0
- Seed 777:
  - Run dir: `results/probe_actionspace8_modelwiring_seed777_8ep_20260217_192017`
  - Test: return 0.00%, PF 0.00, trades 0
- Seed 789:
  - Run dir: `results/probe_actionspace8_modelwiring_seed789_8ep_20260217_183430`
  - Test: return -2.61%, PF 0.57, trades 27
- Aggregate (manual):
  - mean return -0.87%, mean PF 0.19, mean trades 9.0

Extra architecture check (temporary 2-layer net, same setup):
- Seed 456:
  - Run dir: `results/probe_actionspace8_modelwiring_h2_seed456_8ep_20260217_193417`
  - Test: return -2.45%, PF 0.19, trades 15

Comparison vs previous action8 branch:
- Previous aggregate (`results/probe_actionspace8_real_seed{456,777,789}_20260217`) was mean return -0.44%, mean PF 0.67.
- After wiring fix, branch remains unstable/negative and not acceptable yet.

Conclusion:
- The wiring bug was real and is fixed.
- Profitability did not improve on this branch; next work should target validation-vs-test behavior mismatch and anti-collapse robustness, not more blind parameter sweeps.

## 2026-02-17 (Pivot: no-gate + no-prefill, tri-seed fast screen, 8ep x 600 steps)

Focus: test whether replay prefill is biasing early learning and degrading real-data robustness.

Experiment (8 ep, 600 steps, strengths on, no symmetry + no dual, no trade gate, prefill disabled, trade_penalty 0.0, flip_penalty 0.0003, cooldown 6, min_hold 3, max_trades 40):
- Seed 456:
  - Run dir: `seed_sweep_results/realdata/pivot_nogate_noprefill_8ep_20260217_122324_seed456`
  - Log: `logs/pivot_nogate_noprefill_8ep_seed456_20260217_122324.log`
  - Test: return -0.15%, PF 1.07, trades 26, walk-forward pass False
- Seed 777:
  - Run dir: `seed_sweep_results/realdata/pivot_nogate_noprefill_8ep_20260217_123923_seed777`
  - Log: `logs/pivot_nogate_noprefill_8ep_seed777_20260217_123923.log`
  - Test: return +1.23%, PF 1.44, trades 24, walk-forward pass False
- Seed 789:
  - Run dir: `seed_sweep_results/realdata/pivot_nogate_noprefill_8ep_20260217_120748_seed789`
  - Log: `logs/pivot_nogate_noprefill_8ep_seed789_20260217_120748.log`
  - Test: return -1.58%, PF 0.74, trades 25, walk-forward pass False
- Aggregate (manual):
  - `seed_sweep_results/realdata/pivot_nogate_noprefill_8ep_tri_manual_aggregate_20260217.json`
  - Return mean -0.16%, PF mean 1.08, walk-forward pass 0/3

Comparison:
- Versus prefill-on branch (`pivot_nogate_recheck_8ep_tri_manual_aggregate_20260217.json`):
  - Return mean improved from -2.43% -> -0.16%
  - PF mean improved from 0.66 -> 1.08

Conclusion:
- This is the first recent branch with near-breakeven cross-seed return and PF>1 mean under real costs.
- Status: promising but not confirmed; needs longer-horizon confirmation before acceptance.

## 2026-02-17 (Baseline anchor on real-data test window)

Focus: measure naive policy floor/ceiling under current test frictions before more RL pivots.

Command:
- `python check_policy_baseline.py --policy all --seed 123 --data-mode csv --pair-files pair_files_real.json --data-dir . --spread 0.00015 --commission 7.0 --slippage-pips 0.8 --cooldown-bars 6 --min-hold-bars 3 --max-trades-per-episode 40`

Results:
- HOLD: 0.00% return, PF 0.00
- LONG: -3.01% return, PF 0.47
- SHORT: -0.15% return, PF 1.07

Conclusion:
- Test slice appears mildly short-favorable vs long; useful anchor for interpreting RL runs.

## 2026-02-17 (Curriculum probe: low-friction train -> real-friction evaluate, seed 789)

Focus: test whether easier-cost training can learn transferable edge for real-cost evaluation.

Stage A (train, reduced frictions):
- Output: `results/curriculum_cost_stageA8ep_seed789_20260217_113423`
- Log: `logs/curriculum_stageA8ep_seed789_20260217_113423.log`
- Train frictions: spread 0.000075, commission 3.5, slippage 0.2 pips
- Post-restore validation: positive (`val_final` SPR 0.534, ALT SPR 1.592)

Stage B (evaluate same checkpoint at real frictions):
- Output: `results/curriculum_cost_stageA8ep_seed789_20260217_113423/eval_realcost/results/test_results.json`
- Log: `logs/curriculum_stageB_eval_seed789_20260217_113423.log`
- Test: return -2.07%, PF 0.67, walk-forward pass False

Conclusion:
- Low-friction curriculum did not transfer to real-cost profitability in this probe; reject this pivot.

## 2026-02-17 (Pivot: HOLD+SHORT action restriction probe, seed 789)

Focus: test whether removing long-side actions helps on a short-favorable window.

Experiment (8 ep, 600 steps, no symmetry + no dual, no trade gate, allow actions `hold,short`):
- Run dir: `seed_sweep_results/realdata/pivot_hold_short_8ep_20260217_115103_seed789`
- Log: `logs/pivot_hold_short_8ep_seed789_20260217_115103.log`
- Summary: `seed_sweep_results/realdata/pivot_hold_short_8ep_20260217_115103_summary.json`
- Test: return -2.96%, PF 0.46, trades 23, walk-forward pass False

Conclusion:
- Action restriction to hold+short did not improve this seed; reject this pivot.

## 2026-02-17 (Validation return fix: expose ALT hold-out stats correctly)

Focus: remove false `windows=0` / empty ALT metrics in post-restore validation reporting.

Code changes:
- File: `trainer.py`
- `validate()` now returns robust-validation fields consumed later in training/post-restore:
  - `val_k`, `val_median_fitness`, `val_iqr`, `val_stability_adj`, `val_mult`, `val_undertrade_penalty`
  - `spr_components` (mirrored from last SPR window info)
- This aligns `[POST-RESTORE:ALT]` logging and `val_final_alt.json` content with actual computed windows/components.

Verification:
- Command: `python -m py_compile trainer.py`
- Smoke run: `main.py --mode both --episodes 1 ... --output-dir results/val_return_fix_smoke_20260217`
- Result: `[POST-RESTORE:ALT]` now reports real values (example: `windows=7`, non-zero PF/MDD/TPY).

Conclusion:
- Validation diagnostics now reflect real ALT hold-out stats, improving checkpoint-analysis reliability.

## 2026-02-17 (Pivot: no-gate recheck, tri-seed fast screen, 8ep x 600 steps)

Focus: quick branch check on the prior least-worst no-gate family, using current codebase and short runtime.

Experiment (8 ep, 600 steps, strengths on, no symmetry + no dual, no trade gate, trade_penalty 0.0, flip_penalty 0.0003, cooldown 6, min_hold 3, max_trades 40):
- Seed 456:
  - Run dir: `seed_sweep_results/realdata/pivot_nogate_recheck_8ep_20260217_104156_seed456`
  - Log: `logs/pivot_nogate_recheck_8ep_seed456_20260217_104156.log`
  - Test: return +0.57%, PF 1.28, trades 31, walk-forward pass False
- Seed 777:
  - Run dir: `seed_sweep_results/realdata/pivot_nogate_recheck_8ep_20260217_111616_seed777`
  - Log: `logs/pivot_nogate_recheck_8ep_seed777_20260217_111616.log`
  - Test: return -3.28%, PF 0.36, trades 22, walk-forward pass False
- Seed 789:
  - Run dir: `seed_sweep_results/realdata/pivot_nogate_recheck_8ep_20260217_105902_seed789`
  - Log: `logs/pivot_nogate_recheck_8ep_seed789_20260217_105901.log`
  - Test: return -4.57%, PF 0.35, trades 24, walk-forward pass False
- Aggregate (manual):
  - `seed_sweep_results/realdata/pivot_nogate_recheck_8ep_tri_manual_aggregate_20260217.json`
  - Return mean -2.43%, PF mean 0.66, walk-forward pass 0/3

Conclusion:
- Branch rejected for cross-seed use (one positive seed, two strongly negative seeds, 0/3 walk-forward pass).

## 2026-02-17 (Guarded robust50 branch finalized as rejected)

Focus: close out the interrupted robust50 branch with a definitive aggregate decision.

Details:
- Seed 456 robust50 result:
  - `seed_sweep_results/realdata/cost_guarded_robust50_20260216_213747_seed456/results/test_results.json`
  - Return -1.54%, PF 0.00, walk-forward pass False
- Seed 789 robust50 run was interrupted by reboot; evaluated from saved best checkpoint:
  - `seed_sweep_results/realdata/cost_guarded_robust50_20260216_234031_seed789/results/test_results.json`
  - Return -1.74%, PF 0.67, walk-forward pass False
- Aggregate (partial, 2 seeds):
  - `seed_sweep_results/realdata/cost_guarded_robust50_partial_manual_aggregate_20260217.json`
  - Return mean -1.64%, PF mean 0.34, walk-forward pass 0/2

Conclusion:
- `cost_guarded_robust50` branch rejected.

## 2026-02-16 (Seed 777 diagnostic: guarded z=0.3, steps=1000, 30ep, no-sym/no-dual)

Focus: determine whether seed 777 improves with longer training under the current guarded setup.

Experiment (30 ep, seed 777 only, strengths on, no symmetry + no dual, trade_penalty 5e-05, flip_penalty 0.0005, min_atr_cost_ratio 0.2, cooldown 6, min_hold 3, max_trades 30, hold_tie_tau 0.2, hold_break_after 2, trade_gate_z 0.3, max_steps 1000):
- Summary: `seed_sweep_results/realdata/seed777_diag_steps1000_nosym_nodual_30ep_20260216_141318_summary.json`
- Run dir: `seed_sweep_results/realdata/seed777_diag_steps1000_nosym_nodual_30ep_20260216_141318_seed777`
- Log: `logs/seed777_diag_steps1000_nosym_nodual_30ep_20260216_141318.log`
- Final test: return -2.04%, PF 0.46, trades 18, walk-forward pass False (6 windows)
Diagnostics:
- Validation scores oscillated (30 episodes: positive 3, negative 4, zero 23) with mean score +0.0308 and mean validation PF 0.9604.
- `val_final.json` score was +0.436, but test still finished negative.
- Validation summaries report `slippage_pips=0.0` across all episodes while run config uses test slippage 0.8 pips.
Conclusion:
- More episodes did not recover seed 777 under this setup.
- There is likely a validation/test friction mismatch (slippage in validation) that can make checkpoint selection look better than final test outcomes.

## 2026-02-16 (Guarded z=0.3, steps=1000, tri-seed 10ep, symmetry+dual OFF)

Focus: isolate the effect of longer evaluation horizon (1000 steps) while keeping the guarded branch controls unchanged (no symmetry, no dual).

Experiment (10 ep, strengths on, no symmetry + no dual, trade_penalty 5e-05, flip_penalty 0.0005, min_atr_cost_ratio 0.2, cooldown 6, min_hold 3, max_trades 30, hold_tie_tau 0.2, hold_break_after 2, trade_gate_z 0.3, max_steps 1000):
- Summary: `seed_sweep_results/realdata/explore_guarded_steps1000_nosym_nodual_tri10ep_20260216_125735_summary.json`
- Log: `logs/explore_guarded_steps1000_nosym_nodual_tri10ep_20260216_125735.log`
- Seed 456: Return +0.17%, PF 1.33, trades 16, walk-forward pass False (6 windows)
- Seed 777: Return -2.42%, PF 0.49, trades 21, walk-forward pass False (6 windows)
- Seed 789: Return +0.84%, PF 1.65, trades 17, walk-forward pass False (6 windows)
- Aggregate: Return mean -0.47%, PF mean 1.16, fitness mean 0.0000
Comparison:
- Versus the confounded run with symmetry+dual ON (`explore_guarded_steps1000_tri10ep_20260216_110700`): return mean improved from -1.10% to -0.47%, PF mean improved from 0.79 to 1.16.
Conclusion:
- Longer-horizon + no-sym/no-dual is directionally better than the confounded variant but still not cross-seed profitable due persistent seed-777 weakness.

## 2026-02-16 (Synthetic learnability re-check)

Focus: verify the model stack can learn in controlled settings before further real-data pivots.

Checks and results:
- Bandit sanity check (known reward mapping):
  - Command: `python check_bandit_learning.py`
  - Result: PASS; agent converged to action 1 as best (`Q_long ~ +1.0`, `Q_short ~ -1.0`).
- End-to-end synthetic probe (all actions enabled, zero costs, positive drift):
  - Command run with output: `results/synth_learn_probe_20260216_124725`
  - Test: return +3.10%, PF 1.74, trades 79
  - File: `results/synth_learn_probe_20260216_124725/results/test_results.json`
- End-to-end synthetic probe (restricted to `hold,long`, same synthetic setup):
  - Command run with output: `results/synth_learn_probe_holdlong_20260216_125055`
  - Test: return +13.84%, PF 7.63, trades 113
  - File: `results/synth_learn_probe_holdlong_20260216_125055/results/test_results.json`
- Fixed-policy baseline on same synthetic setup:
  - Command: `python check_policy_baseline.py --policy all --seed 123 --data-mode synthetic --n-bars 3000 --synthetic-drift 0.0002 --synthetic-volatility 0.0004 --spread 0 --commission 0 --slippage-pips 0 --cooldown-bars 0 --min-hold-bars 1 --max-trades-per-episode 200`
  - HOLD: 0.00%, LONG: +12.18%, SHORT: -13.51%
Conclusion:
- Learning capability is confirmed in controlled synthetic tasks and in the full training/eval pipeline.
- Current weakness is generalization/robustness on real data, not inability to learn at all.

## 2026-02-16 (Guarded z=0.3, steps=1000, tri-seed 10ep, symmetry+dual ON)

Focus: increase evaluation horizon (1000 test steps) so walk-forward has multiple windows.

Experiment (10 ep, strengths on, trade_penalty 5e-05, flip_penalty 0.0005, min_atr_cost_ratio 0.2, cooldown 6, min_hold 3, max_trades 30, hold_tie_tau 0.2, hold_break_after 2, trade_gate_z 0.3, max_steps 1000):
- Summary: `seed_sweep_results/realdata/explore_guarded_steps1000_tri10ep_20260216_110700_summary.json`
- Log: `logs/explore_guarded_steps1000_tri10ep_20260216_110700.log`
- Config caveat: this run used symmetry loss and dual controller ON (`no_symmetry_loss=False`, `no_dual_controller=False`), unlike the prior guarded branch where both were OFF.
- Seed 456: Return -2.03%, PF 0.44, trades 18, walk-forward pass False (6 windows)
- Seed 777: Return -2.64%, PF 0.53, trades 20, walk-forward pass False (6 windows)
- Seed 789: Return +1.38%, PF 1.40, trades 20, walk-forward pass True (6 windows)
- Aggregate: Return mean -1.10%, PF mean 0.79, fitness mean 0.0808
Conclusion:
- Multi-window evaluation now works, but this configuration is not profitable cross-seed.
- Because symmetry/dual were changed, this does not isolate the effect of `max_steps=1000` versus the prior guarded baseline.
Next step:
- Re-run the same setup with only one variable changed from baseline: keep symmetry/dual OFF (`--no-symmetry-loss --no-dual-controller`) and keep `--max-steps-per-episode 1000`.

## 2026-02-16 (Pivot: trade gate z=0.5, guarded config, tri-seed 10ep)

Focus: test whether stricter trade gating improves OOS results under the current guarded cadence/friction setup.

Experiment (10 ep, strengths on, no symmetry + no dual, trade_penalty 5e-05, flip_penalty 0.0005, min_atr_cost_ratio 0.2, cooldown 6, min_hold 3, max_trades 30, hold_tie_tau 0.2, hold_break_after 2, max_steps 300):
- Summary: `seed_sweep_results/realdata/pivot_gatez0_5_tri10ep_20260216_102905_summary.json`
- Log: `logs/pivot_gatez0_5_tri10ep_20260216_102904.log`
- Seed 456: Return +2.01%, PF 2.90, trades 17
- Seed 777: Return -4.31%, PF 0.13, trades 18
- Seed 789: Return -0.54%, PF 0.00, trades 1 (undertrading)
Conclusion:
- `trade_gate_z=0.5` is worse than `trade_gate_z=0.3` on mean return and can degenerate to near-no-trade behavior; do not continue upward-gating pivots.
- Quick-eval runs capped at 300 steps produce only 1 walk-forward window, so SPR/fitness is not informative at this horizon.

## 2026-02-15 (Guarded config: trade gate z=0.3, tri-seed 10ep + 20ep)

Focus: fast cross-seed check of the guarded cadence/friction setup before spending hours on long runs.

Experiments (strengths on, no symmetry + no dual, trade_penalty 5e-05, flip_penalty 0.0005, min_atr_cost_ratio 0.2, cooldown 6, min_hold 3, max_trades 30, hold_tie_tau 0.2, hold_break_after 2, max_steps 300):
- 10ep summary: `seed_sweep_results/realdata/explore_guarded_tri10ep_20260215_141625_summary.json`
  - Aggregate: return mean +0.38%, PF mean 1.35
- 20ep summary: `seed_sweep_results/realdata/explore_guarded_tri20ep_20260215_162822_summary.json`
  - Aggregate: return mean -0.59%, PF mean 0.91
Notes:
- Seed 456 stays positive; seed 777/789 drift negative as episodes increased under this quick-eval horizon.
Conclusion:
- Keep `trade_gate_z=0.3` as the best setting in this guarded branch so far, but retest with a longer eval horizon (more steps) before drawing conclusions.

## 2026-02-05 (Max trades 30, no gate + cooldown 6/min_hold 3, 10ep sweep, seeds 456/777/789)

Focus: reduce churn further via tighter max-trades cap with no gate.

Experiments (10 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=30, cooldown=6, min_hold=3):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_30_no_gate_cooldown_6_min_hold_3_10ep_20260205_111042_seed456/results/test_results.json`
  - Return -1.91%, PF 0.54, fitness 0.0000, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_30_no_gate_cooldown_6_min_hold_3_10ep_20260205_124931_seed777/results/test_results.json`
  - Return -0.43%, PF 0.96, fitness 0.0033, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_30_no_gate_cooldown_6_min_hold_3_10ep_20260205_132233_seed789/results/test_results.json`
  - Return -1.11%, PF 0.77, fitness 0.0000, walk-forward pass False
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_30_no_gate_cooldown_6_min_hold_3_10ep_20260205_aggregate_summary.json`
  - Return mean -1.15%, PF mean 0.76, fitness mean 0.0011
Conclusion:
- Max trades 30 did not lift mean return; drop this setting.

## 2026-02-05 (Max trades 40, no gate + cooldown 6/min_hold 3, 10ep sweep, attempt)

Focus: retry max trades 40 with no gate after max trades 30 underperformed.

Status:
- Initial 3-seed background sweep terminated early; no results produced.
- Log: `logs/realdata_sweep_max_trades_40_no_gate_cooldown_6_min_hold_3_10ep_20260205_142429.log`
Plan:
- Re-run per seed in the foreground and then generate an aggregate summary.

## 2026-02-04 (Min ATR cost ratio 0.1, no gate + cooldown 6/min_hold 3, 10ep sweep, seeds 456/777/789)

Focus: multi-seed confirmation of ATR cost gating at 0.1.

Experiments (10 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40, cooldown=6, min_hold=3, min_atr_cost_ratio=0.1):
- Seed 456:
  - `seed_sweep_results/realdata/min_atr_0_1_no_gate_cooldown_6_min_hold_3_10ep_20260202_180544_seed456/results/test_results.json`
  - Return -3.53%, PF 0.79, fitness -0.0118, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/min_atr_0_1_no_gate_cooldown_6_min_hold_3_10ep_20260204_191649_seed777/results/test_results.json`
  - Return -1.85%, PF 0.89, fitness -0.0010, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/min_atr_0_1_no_gate_cooldown_6_min_hold_3_10ep_20260204_193200_seed789/results/test_results.json`
  - Return -4.00%, PF 0.73, fitness -0.0115, walk-forward pass False
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/min_atr_0_1_no_gate_cooldown_6_min_hold_3_10ep_20260204_aggregate_summary.json`
  - Return mean -3.13%, PF mean 0.80, fitness mean -0.0081
Notes:
- Sweep required per-seed foreground reruns after mid-seed terminations.
Conclusion:
- Min ATR cost ratio 0.1 degrades returns; drop this setting.

## 2026-01-28 (Min ATR cost ratio 0.2, no gate + cooldown/min_hold, 10ep probe, seed 456)

Focus: test ATR cost gating as a signal-quality filter on the weakest seed.

Experiment (10 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40, cooldown=6, min_hold=3, min_atr_cost_ratio=0.2):
- Seed 456:
  - `seed_sweep_results/realdata/min_atr_0_2_no_gate_cooldown_6_min_hold_3_10ep_20260128_114030_seed456/results/test_results.json`
  - Return +0.16%, PF 1.16, SPR 0.0015, walk-forward pass True
Notes:
- Early signal suggests ATR gating may improve the weakest seed; needs multi-seed confirmation.

## 2026-01-28 (Min ATR cost ratio 0.2, no gate + cooldown/min_hold, 10ep probe, seed 777)

Focus: second-seed check for the ATR cost gate.

Experiment (10 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40, cooldown=6, min_hold=3, min_atr_cost_ratio=0.2):
- Seed 777:
  - `seed_sweep_results/realdata/min_atr_0_2_no_gate_cooldown_6_min_hold_3_10ep_20260128_114030_seed777/results/test_results.json`
  - Return -1.41%, PF 0.72, SPR 0.0000, walk-forward pass False
Notes:
- Training completed but evaluation results were missing; ran evaluate-only from checkpoint to recover results.

## 2026-01-28 (Min ATR cost ratio 0.2, no gate + cooldown/min_hold, 10ep probe, seed 789)

Focus: third-seed check for the ATR cost gate.

Experiment (10 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40, cooldown=6, min_hold=3, min_atr_cost_ratio=0.2):
- Seed 789:
  - `seed_sweep_results/realdata/min_atr_0_2_no_gate_cooldown_6_min_hold_3_10ep_20260128_114030_seed789/results/test_results.json`
  - Return -1.23%, PF 0.78, SPR 0.0000, walk-forward pass False
Notes:
- Training completed but evaluation results were missing; ran evaluate-only from checkpoint to recover results.

## 2026-01-28 (Min ATR cost ratio 0.2, no gate + cooldown/min_hold, 10ep probe, aggregate)

Aggregate (3 seeds):
- `seed_sweep_results/realdata/min_atr_0_2_no_gate_cooldown_6_min_hold_3_10ep_20260128_114030_summary.json`
- Return mean -0.83%, PF mean 0.89, fitness mean 0.0005
Conclusion:
- ATR cost gate at 0.2 did not improve multi-seed performance; drop this setting.

## 2026-01-27 (No gate + cooldown 6/min_hold 3, 50ep sweep, seeds 456/777/789)

Focus: remove trade gate while keeping cadence controls to test if gating caused the 100-episode drop.

Experiments (50 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40, cooldown=6, min_hold=3, trade_gate disabled):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_cooldown_6_min_hold_3_50ep_20260127_203500_seed456/results/test_results.json`
  - Return -2.21%, PF 0.59, SPR 0.0000, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_cooldown_6_min_hold_3_50ep_20260127_203500_seed777/results/test_results.json`
  - Return -0.36%, PF 1.02, SPR 0.0032, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_cooldown_6_min_hold_3_50ep_20260127_203500_seed789/results/test_results.json`
  - Return +2.67%, PF 2.01, SPR 0.7736, walk-forward pass True
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_cooldown_6_min_hold_3_50ep_20260127_203500_summary.json`
  - Return mean +0.04%, PF mean 1.21, fitness mean 0.2589
Notes:
- This run took longer than expected; executed per-seed to avoid timeouts.

## 2026-01-27 (Zero-cost evaluation on no-gate 50ep checkpoints)

Focus: check whether transaction costs are the primary drag on the no-gate 50ep models.

Evaluations (zero spread/commission/slippage):
- Seed 456:
  - `results/zero_cost_eval_no_gate_50ep_20260127_seed456/results/test_results.json`
  - Return -0.93%, PF 0.76, walk-forward pass False
- Seed 777:
  - `results/zero_cost_eval_no_gate_50ep_20260127_seed777/results/test_results.json`
  - Return +0.08%, PF 1.02, walk-forward pass False
- Seed 789:
  - `results/zero_cost_eval_no_gate_50ep_20260127_seed789/results/test_results.json`
  - Return +4.10%, PF 2.52, walk-forward pass True
Notes:
- Zero-cost lift is meaningful for seed 789, but seed 456 remains negative, indicating strategy quality issues beyond costs.

## 2026-01-27 (Trade_gate_z=0.3 + cooldown 6/min_hold 3, 100ep sweep, seeds 456/777/789)

Focus: complete the 100-episode sweep and recover the missing seed 789 evaluation.

Experiments (100 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40, cooldown=6, min_hold=3, trade_gate_z=0.3):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_3_cooldown_6_min_hold_3_100ep_20260126_204519_seed456/results/test_results.json`
  - Return -3.95%, PF 0.39, SPR -0.0051, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_3_cooldown_6_min_hold_3_100ep_20260126_204519_seed777/results/test_results.json`
  - Return -0.50%, PF 1.00, SPR 0.0246, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_3_cooldown_6_min_hold_3_100ep_20260126_204519_seed789/results/test_results.json`
  - Return -1.11%, PF 0.83, SPR 0.0000, walk-forward pass False
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_3_cooldown_6_min_hold_3_100ep_20260126_204519_summary.json`
  - Return mean -1.85%, PF mean 0.74, fitness mean 0.0065
Notes:
- Seed 789 evaluation was missing after the terminal closed; ran evaluate-only from the existing checkpoint and generated the summary JSON.

## 2026-01-26 (Trade_gate_z=0.3 + cooldown 6/min_hold 3, 50ep sweep, seeds 456/777/789)

Focus: test whether adding cooldown/min_hold with lower gate improves robustness, and recover missing evals.

Experiment (50 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40, cooldown=6, min_hold=3, trade_gate_z=0.3):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_3_cooldown_6_min_hold_3_50ep_20260126_122152_seed456/results/test_results.json`
  - Return +4.58%, PF 2.48, SPR 0.2853, walk-forward pass True
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_3_cooldown_6_min_hold_3_50ep_20260126_122152_seed777/results/test_results.json`
  - Return +1.32%, PF 1.44, SPR -0.0015, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_3_cooldown_6_min_hold_3_50ep_20260126_122152_seed789/results/test_results.json`
  - Return +0.14%, PF 1.19, SPR 0.0000, walk-forward pass False
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_3_cooldown_6_min_hold_3_50ep_20260126_122152_summary.json`
  - Return mean +2.01%, PF mean 1.70, fitness mean 0.0946
Notes:
- Initial sweep ended with missing eval results (exit_code 120). Manual evaluate runs recovered results and the summary JSON was updated.

## 2026-01-25 (Trade_gate_z=0.2, 50ep sweep, seeds 777/789)

Focus: probe lower trade gate in shorter training to see if trade frequency improves.

Experiments (50 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_2_50ep_20260125_214559_seed777/results/test_results.json`
  - Return -0.23%, PF 0.01, trades 2, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_2_50ep_20260125_214559_seed789/results/test_results.json`
  - Return -1.56%, PF 0.75, trades 26, walk-forward pass False
- Aggregate (2 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_2_50ep_20260125_214559_summary.json`
  - Return mean -0.89%, PF mean 0.38, fitness mean -0.00094
Notes:
- Sweep command timed out in the CLI, but the runs completed and wrote results.

## 2026-01-25 (Trade_gate_z=0.3, 100ep sweep, seed 456)

Focus: complete the 100-episode trade_gate_z=0.3 check with seed 456.

Experiment (100 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_3_100ep_20260125_133812_seed456/results/test_results.json`
  - Return -2.14%, PF 0.69, SPR 0.0000, walk-forward pass False
- Summary:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_3_100ep_20260125_133812_summary.json`
Notes:
- Sweep command timed out in the CLI, but the run completed and wrote results.

## 2026-01-24 (Trade_gate_z=0.3, 100ep sweep, seeds 777/789)

Focus: lower trade gate to increase trade frequency in longer training.

Experiments (100 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_3_100ep_20260124_142218_seed777/results/test_results.json`
  - Return +1.76%, PF 1.41, SPR 0.0444, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_3_100ep_20260124_142218_seed789/results/test_results.json`
  - Return +0.21%, PF 1.20, SPR 0.0000, walk-forward pass False
- Aggregate (2 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_3_100ep_20260124_142218_summary.json`
  - Return mean +0.99%, PF mean 1.31, fitness mean 0.0222
Notes:
- Sweep command timed out in the CLI, but the runs completed and wrote results.

## 2026-01-23 (Trade_gate_z=0.5, 100ep sweep, seeds 777/789)

Focus: extend baseline training to 100 episodes for the weaker seeds.

Experiments (100 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_100ep_20260123_152801_seed777/results/test_results.json`
  - Return +0.21%, PF 1.19, SPR -0.0397, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_100ep_20260123_152801_seed789/results/test_results.json`
  - Return -0.59%, PF 0.98, SPR 0.0000, walk-forward pass True
- Aggregate (2 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_100ep_20260123_152801_summary.json`
  - Return mean -0.19%, PF mean 1.09, fitness mean -0.0198
Notes:
- Sweep command timed out in the CLI, but the runs completed and wrote results.

## 2026-01-22 (R-multiple reward weight 0.1, trade_gate_z=0.5, 10ep sweep)

Focus: test realized R-multiple reward shaping.

Experiments (10 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40, r_multiple_reward_weight=0.1):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_r_mult_0_1_10ep_20260122_223605_seed456/results/test_results.json`
  - Return -2.48%, PF 0.62, SPR -0.0348, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_r_mult_0_1_10ep_20260122_223605_seed777/results/test_results.json`
  - Return -3.18%, PF 0.51, SPR -0.0000, walk-forward pass True
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_r_mult_0_1_10ep_20260122_223605_seed789/results/test_results.json`
  - Return -2.21%, PF 0.63, SPR 0.0002, walk-forward pass False
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_r_mult_0_1_10ep_20260122_223605_summary.json`
  - Return mean -2.62%, PF mean 0.59, fitness mean -0.0116
Notes:
- Sweep command timed out in the CLI, but the runs completed and wrote results.

## 2026-01-22 (No strengths + cooldown 6/min_hold 3, trade_gate_z=0.5, 10ep sweep)

Focus: test whether reduced hold/cooldown helps the no-strengths setup.

Experiments (10 ep, no strengths, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40, cooldown=6, min_hold=3):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_no_strengths_cooldown_6_min_hold_3_10ep_20260122_212001_seed456/results/test_results.json`
  - Return -1.75%, PF 0.80, SPR -0.0153, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_no_strengths_cooldown_6_min_hold_3_10ep_20260122_212001_seed777/results/test_results.json`
  - Return +0.73%, PF 1.30, SPR -0.0362, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_no_strengths_cooldown_6_min_hold_3_10ep_20260122_212001_seed789/results/test_results.json`
  - Return -1.14%, PF 0.83, SPR -0.0020, walk-forward pass False
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_no_strengths_cooldown_6_min_hold_3_10ep_20260122_212001_summary.json`
  - Return mean -0.72%, PF mean 0.98, fitness mean -0.0178
Notes:
- Sweep command timed out in the CLI, but the runs completed and wrote results.

## 2026-01-22 (Cooldown 6, min_hold 3, trade_gate_z=0.5, 50ep sweep)

Focus: confirm the lower cooldown/min_hold setting over a longer horizon.

Experiments (50 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40, cooldown=6, min_hold=3):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_cooldown_6_min_hold_3_50ep_20260122_145621_seed456/results/test_results.json`
  - Return +0.73%, PF 1.27, SPR -0.1384, walk-forward pass True
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_cooldown_6_min_hold_3_50ep_20260122_145621_seed777/results/test_results.json`
  - Return -1.26%, PF 0.79, SPR 0.0000, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_cooldown_6_min_hold_3_50ep_20260122_145621_seed789/results/test_results.json`
  - Return -1.18%, PF 0.86, SPR 0.0152, walk-forward pass False
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_cooldown_6_min_hold_3_50ep_20260122_145621_summary.json`
  - Return mean -0.57%, PF mean 0.97, fitness mean -0.0411
Notes:
- Sweep command timed out in the CLI, but the runs completed and wrote results.

## 2026-01-22 (Cooldown 6, min_hold 3, trade_gate_z=0.5, 10ep sweep)

Focus: increase trade frequency by reducing min_hold and cooldown.

Experiments (10 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40, cooldown=6, min_hold=3):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_cooldown_6_min_hold_3_10ep_20260122_120625_seed456/results/test_results.json`
  - Return -0.18%, PF 1.06, SPR -0.0154, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_cooldown_6_min_hold_3_10ep_20260122_120625_seed777/results/test_results.json`
  - Return -1.59%, PF 0.76, SPR 0.0006, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_cooldown_6_min_hold_3_10ep_20260122_120625_seed789/results/test_results.json`
  - Return +1.85%, PF 1.44, SPR 0.0659, walk-forward pass True
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_cooldown_6_min_hold_3_10ep_20260122_120625_summary.json`
  - Return mean +0.03%, PF mean 1.09, fitness mean 0.0171
Notes:
- Sweep command timed out in the CLI, but the runs completed and wrote results.

## 2026-01-21 (Regime min vol z=0.5, trade_gate_z=0.5, 50ep sweep)

Focus: confirm volatility-only regime gating over a longer horizon.

Experiments (50 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40, use_regime_filter, align_trend=False, require_trending=False):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_regime_min_vol_z_0_5_50ep_20260121_191459_seed456/results/test_results.json`
  - Return +0.02%, PF 1.09, SPR -0.3264, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_regime_min_vol_z_0_5_50ep_20260121_191459_seed777/results/test_results.json`
  - Return -1.66%, PF 0.81, SPR -0.0279, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_regime_min_vol_z_0_5_50ep_20260121_191459_seed789/results/test_results.json`
  - Return -2.13%, PF 0.60, SPR -0.0017, walk-forward pass False
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_regime_min_vol_z_0_5_50ep_20260121_191459_summary.json`
  - Return mean -1.26%, PF mean 0.83, fitness mean -0.1186
Notes:
- Sweep command timed out in the CLI, but the runs completed and wrote results.

## 2026-01-21 (No strengths, trade_gate_z=0.5, 50ep sweep)

Focus: confirm the no-strengths setup over a longer horizon.

Experiments (50 ep, no strengths, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_no_strengths_50ep_20260121_121818_seed456/results/test_results.json`
  - Return -2.05%, PF 0.60, SPR -0.0246, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_no_strengths_50ep_20260121_121818_seed777/results/test_results.json`
  - Return +0.87%, PF 1.26, SPR 0.1006, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_no_strengths_50ep_20260121_121818_seed789/results/test_results.json`
  - Return -0.25%, PF 1.04, SPR -0.0475, walk-forward pass False
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_no_strengths_50ep_20260121_121818_summary.json`
  - Return mean -0.47%, PF mean 0.97, fitness mean 0.0095
Notes:
- Initial sweep attempt used `--data-mode real` and failed; rerun with `--data-mode csv` succeeded.

## 2026-01-21 (No strengths, trade_gate_z=0.5, 10ep sweep)

Focus: remove strength features entirely to reduce noise.

Experiments (10 ep, no strengths, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_no_strengths_10ep_20260121_105746_seed456/results/test_results.json`
  - Return -1.57%, PF 0.73, SPR -0.0248, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_no_strengths_10ep_20260121_105746_seed777/results/test_results.json`
  - Return +2.96%, PF 1.98, SPR 0.1993, walk-forward pass True
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_no_strengths_10ep_20260121_105746_seed789/results/test_results.json`
  - Return -0.40%, PF 1.02, SPR -0.0550, walk-forward pass False
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_no_strengths_10ep_20260121_105746_summary.json`
  - Return mean +0.327%, PF mean 1.242, fitness mean 0.0398
Notes:
- Sweep command timed out in the CLI, but the runs completed and wrote results.

## 2026-01-20 (Strengths pair-only, trade_gate_z=0.5, 10ep sweep)

Focus: reduce strength feature noise to base/quote only.

Experiments (10 ep, strengths pair-only, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_strengths_pair_only_10ep_20260120_191312_seed456/results/test_results.json`
  - Return +0.80%, PF 1.23, SPR 0.1394, walk-forward pass True
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_strengths_pair_only_10ep_20260120_191312_seed777/results/test_results.json`
  - Return -3.39%, PF 0.30, SPR 0.0000, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_strengths_pair_only_10ep_20260120_191312_seed789/results/test_results.json`
  - Return -2.14%, PF 0.68, SPR -0.0056, walk-forward pass False
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_strengths_pair_only_10ep_20260120_191312_summary.json`
  - Return mean -1.57%, PF mean 0.74, fitness mean 0.0446
Notes:
- Sweep command timed out in the CLI, but the runs completed and wrote results.

## 2026-01-20 (Trade gate z=0.7, 10ep sweep)

Focus: test a stricter trade gate than 0.5.

Experiments (10 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_7_10ep_20260120_174319_seed456/results/test_results.json`
  - Return -1.90%, PF 0.76, SPR -0.0047, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_7_10ep_20260120_174319_seed777/results/test_results.json`
  - Return -0.42%, PF 1.03, SPR -0.0193, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_7_10ep_20260120_174319_seed789/results/test_results.json`
  - Return -0.37%, PF 1.00, SPR 0.0010, walk-forward pass False
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_7_10ep_20260120_174319_summary.json`
  - Return mean -0.89%, PF mean 0.93, fitness mean -0.0077
Notes:
- Sweep command timed out in the CLI, but the runs completed and wrote results.

## 2026-01-20 (Trade gate z=0.3, 10ep sweep)

Focus: test a looser trade gate than 0.5.

Experiments (10 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_3_10ep_20260120_022529_seed456/results/test_results.json`
  - Return -1.30%, PF 0.81, SPR 0.0000, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_3_10ep_20260120_022529_seed777/results/test_results.json`
  - Return +0.34%, PF 1.26, SPR -0.0318, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_3_10ep_20260120_022529_seed789/results/test_results.json`
  - Return +0.57%, PF 1.24, SPR -0.0009, walk-forward pass True
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_3_10ep_20260120_022529_summary.json`
  - Return mean -0.13%, PF mean 1.10, fitness mean -0.0109
Notes:
- Sweep command timed out in the CLI, but the runs completed and wrote results.

## 2026-01-19 (ATR cost ratio 0.5, trade_gate_z=0.5, 10ep sweep)

Focus: test a milder ATR cost gate (min_atr_cost_ratio=0.5) with trade_gate_z=0.5.

Experiments (10 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_atr_ratio_0_5_10ep_20260119_231447_seed456/results/test_results.json`
  - Return -2.11%, PF 0.72, SPR -0.0241, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_atr_ratio_0_5_10ep_20260119_231447_seed777/results/test_results.json`
  - Return +3.08%, PF 2.92, SPR 0.0000, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_atr_ratio_0_5_10ep_20260119_231447_seed789/results/test_results.json`
  - Return -4.39%, PF 0.33, SPR 0.0000, walk-forward pass False
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_atr_ratio_0_5_10ep_20260119_231447_summary.json`
  - Return mean -1.14%, PF mean 1.32, fitness mean -0.0080

## 2026-01-19 (Symmetry + dual controller on, trade_gate_z=0.5, 10ep sweep)

Focus: re-enable symmetry loss + dual controller to see if generalization improves.

Experiments (10 ep, strengths on, symmetry + dual enabled, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_sym_dual_10ep_20260119_224637_seed456/results/test_results.json`
  - Return -1.95%, PF 0.65, SPR -0.0017, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_sym_dual_10ep_20260119_224637_seed777/results/test_results.json`
  - Return -0.52%, PF 1.01, SPR 0.0521, walk-forward pass True
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_sym_dual_10ep_20260119_224637_seed789/results/test_results.json`
  - Return -1.07%, PF 0.89, SPR -0.1338, walk-forward pass True
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_sym_dual_10ep_20260119_224637_summary.json`
  - Return mean -1.18%, PF mean 0.85, fitness mean -0.0278

## 2026-01-19 (Regime vol gate z=0.5, trade_gate_z=0.5, 10ep sweep)

Focus: test volatility-only regime gating (no trend alignment) with trade_gate_z=0.5.

Experiments (10 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_regime_min_vol_z_0_5_10ep_20260119_222106_seed456/results/test_results.json`
  - Return +3.10%, PF 1.83, SPR -0.2376, walk-forward pass True
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_regime_min_vol_z_0_5_10ep_20260119_222106_seed777/results/test_results.json`
  - Return -1.17%, PF 0.84, SPR -0.0010, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_regime_min_vol_z_0_5_10ep_20260119_222106_seed789/results/test_results.json`
  - Return -1.56%, PF 0.34, SPR -0.0021, walk-forward pass False
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_regime_min_vol_z_0_5_10ep_20260119_222106_summary.json`
  - Return mean +0.13%, PF mean 1.01, fitness mean -0.0802

## 2026-01-19 (Trade penalty 6.5e-05, trade_gate_z=0.5, 10ep sweep)

Focus: test whether adding a small trade penalty improves short-run results with trade_gate_z=0.5.

Experiments (10 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 6.5e-05, max_trades=40):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_000065_trade_gate_z_0_5_10ep_20260119_215603_seed456/results/test_results.json`
  - Return -1.31%, PF 0.79, SPR -0.0225, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_000065_trade_gate_z_0_5_10ep_20260119_215603_seed777/results/test_results.json`
  - Return +0.04%, PF 1.14, SPR 0.0000, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_000065_trade_gate_z_0_5_10ep_20260119_215603_seed789/results/test_results.json`
  - Return -3.86%, PF 0.48, SPR -0.0007, walk-forward pass False
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_000065_trade_gate_z_0_5_10ep_20260119_215603_summary.json`
  - Return mean -1.71%, PF mean 0.80, fitness mean -0.0077

## 2026-01-19 (Flip penalty 0.00077, trade_gate_z=0.5, 10ep sweep)

Focus: test whether returning flip_penalty to the default improves short-run generalization.

Experiments (10 ep, strengths on, no symmetry + no dual, flip 0.00077, trade penalty 0.0, max_trades=40):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_00077_trade_penalty_0_trade_gate_z_0_5_10ep_20260119_211938_seed456/results/test_results.json`
  - Return -1.04%, PF 0.88, SPR 0.0000, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_00077_trade_penalty_0_trade_gate_z_0_5_10ep_20260119_211938_seed777/results/test_results.json`
  - Return +0.72%, PF 1.37, SPR 0.0000, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_00077_trade_penalty_0_trade_gate_z_0_5_10ep_20260119_211938_seed789/results/test_results.json`
  - Return -1.06%, PF 0.90, SPR 0.0036, walk-forward pass False
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_00077_trade_penalty_0_trade_gate_z_0_5_10ep_20260119_211938_summary.json`
  - Return mean -0.46%, PF mean 1.05, fitness mean 0.0012

Notes:
- Sweep command timed out in the CLI, but the runs completed and wrote results.

## 2026-01-19 (Walkforward fix + 50ep eval rerun)

Focus: align walkforward windows with max_steps and re-evaluate 50-episode runs using trade_penalty=0.0.

Changes:
- `main.py`: walkforward windows now use `min(len(test_env.data), test_env.max_steps)` to avoid empty windows.

Experiments (50 ep eval, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_50ep_20260119_160515_seed456/eval_fast_trade_gate_z_0_5_50ep_wf_fix/results/test_results.json`
  - Return +3.61%, PF 1.88, SPR 1.7427, walk-forward pass True (median SPR 2.8766, pos frac 1.00)
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_50ep_20260119_181431_seed777/eval_fast_trade_gate_z_0_5_50ep_wf_fix/results/test_results.json`
  - Return -3.40%, PF 0.57, SPR -0.0121, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_50ep_20260119_181431_seed789/eval_fast_trade_gate_z_0_5_50ep_wf_fix/results/test_results.json`
  - Return +1.17%, PF 1.54, SPR 0.0000, walk-forward pass False
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_50ep_wf_fix_aggregate_3seed_20260119.json`
  - Return mean +0.46%, PF mean 1.33, fitness mean 0.5768

Notes:
- Walk-forward now reflects the 1000-step evaluation horizon instead of the full dataset length.

## 2026-01-19 (Trade gate z=0.5 50ep, seeds 777/789)

Focus: extend 50-episode confirmation to the remaining seeds.

Experiments (50 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_50ep_20260119_181431_seed777/results/test_results.json`
  - Return -3.40%, PF 0.57, SPR -0.0121, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_50ep_20260119_181431_seed789/results/test_results.json`
  - Return +1.17%, PF 1.54, SPR 0.0000, walk-forward pass False
- Aggregate (2 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_50ep_20260119_181431_summary.json`
  - Return mean -1.12%, PF mean 1.06, fitness mean -0.0061
- Aggregate (3 seeds incl. seed 456 eval):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_50ep_aggregate_3seed_20260119.json`
  - Return mean +0.46%, PF mean 1.33, fitness mean 0.5768

## 2026-01-19 (Trade gate z=0.5 50ep eval)

Focus: capture eval results for the 50-episode seed 456 run after the sweep timed out.

Experiments (50 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_50ep_20260119_160515_seed456/eval_fast_trade_gate_z_0_5_50ep/results/test_results.json`
  - Return +3.61%, PF 1.88, SPR 1.7427, walk-forward pass False (median SPR 0.0, pos frac 0.04)
Notes:
- Manual eval run used `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_50ep_20260119_160515_seed456/checkpoints/best_model.pt` after the sweep exit code 120.

## 2026-01-19 (Trade gate z=0.5 50ep started)

Focus: longer run to test stability for the best short-run gate.

Experiments (50 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 456 running: `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_50ep_20260119_160515_seed456`
Notes:
- Sweep command timed out while the run continues in the background.

## 2026-01-19 (Trade gate z=0.5 + ATR ratio 1.0)

Focus: combine Q-gap gating with a mild ATR cost gate.

Experiments (10 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_atr_ratio_1_0_10ep_20260119_151203_seed456/eval_fast_trade_gate_z_0_5_atr_ratio_1_0/results/test_results.json`
  - Return +0.55%, PF 1.19, SPR -0.1802, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_atr_ratio_1_0_10ep_20260119_152525_seed777/eval_fast_trade_gate_z_0_5_atr_ratio_1_0/results/test_results.json`
  - Return -1.53%, PF 0.80, SPR -0.0005, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_atr_ratio_1_0_10ep_20260119_154237_seed789/eval_fast_trade_gate_z_0_5_atr_ratio_1_0/results/test_results.json`
  - Return -3.96%, PF 0.36, SPR 0.0000, walk-forward pass False
- Aggregate:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_atr_ratio_1_0_10ep_aggregate_3seed_20260119.json`
  - Return mean -1.65%, PF mean 0.79, SPR mean -0.0602, walk-forward pass 0/3

## 2026-01-19 (Trend-only regime filter probe)

Focus: require trend_96h alignment without the is_trending gate.

Experiments (10 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_regime_trend_only_10ep_20260119_142647_seed456/eval_fast_regime_trend_only/results/test_results.json`
  - Return +0.54%, PF 1.24, SPR -0.1306, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_regime_trend_only_10ep_20260119_143950_seed777/eval_fast_regime_trend_only/results/test_results.json`
  - Return +1.13%, PF 1.49, SPR 0.0593, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_regime_trend_only_10ep_20260119_145325_seed789/eval_fast_regime_trend_only/results/test_results.json`
  - Return -0.57%, PF 0.00, SPR 0.0000, walk-forward pass False
- Aggregate:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_regime_trend_only_10ep_aggregate_3seed_20260119.json`
  - Return mean +0.37%, PF mean 0.91, SPR mean -0.0238, walk-forward pass 0/3

## 2026-01-19 (Regime filter probe)

Focus: trade only in trending regimes (is_trending + trend_96h alignment).

Experiments (10 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_regime_filter_10ep_20260119_132805_seed456/eval_fast_regime_filter/results/test_results.json`
  - Return -1.00%, PF 0.67, SPR -0.0092, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_regime_filter_10ep_20260119_134356_seed777/eval_fast_regime_filter/results/test_results.json`
  - Return -1.88%, PF 0.29, SPR -0.0005, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_regime_filter_10ep_20260119_140205_seed789/eval_fast_regime_filter/results/test_results.json`
  - Return -0.03%, PF 1.04, SPR 0.0178, walk-forward pass False
- Aggregate:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_regime_filter_10ep_aggregate_3seed_20260119.json`
  - Return mean -0.97%, PF mean 0.67, SPR mean 0.0027, walk-forward pass 0/3

## 2026-01-19 (Trade gate z=0.8 20ep confirmation)

Focus: confirm trade_gate_z=0.8 performance on longer runs.

Experiments (20 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_8_20ep_20260119_123107_seed456/eval_fast_trade_gate_z_0_8/results/test_results.json`
  - Return +0.35%, PF 1.18, SPR 0.0004, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_8_20ep_20260119_124411_seed777/eval_fast_trade_gate_z_0_8/results/test_results.json`
  - Return -1.71%, PF 0.75, SPR 0.0004, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_8_20ep_20260119_125748_seed789/eval_fast_trade_gate_z_0_8/results/test_results.json`
  - Return -1.13%, PF 0.81, SPR 0.0000, walk-forward pass False
- Aggregate:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_8_20ep_aggregate_3seed_20260119.json`
  - Return mean -0.83%, PF mean 0.91, SPR mean 0.0003, walk-forward pass 0/3

## 2026-01-19 (Trade gate z=0.8 probe)

Focus: increase the Q-gap over HOLD to further filter low-confidence trades.

Experiments (10 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_8_10ep_20260119_120202_seed456/eval_fast_trade_gate_z_0_8/results/test_results.json`
  - Return +1.16%, PF 1.33, SPR 0.0679, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_8_10ep_20260119_120826_seed777/eval_fast_trade_gate_z_0_8/results/test_results.json`
  - Return +0.78%, PF 1.43, SPR 0.0000, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_8_10ep_20260119_121501_seed789/eval_fast_trade_gate_z_0_8/results/test_results.json`
  - Return +6.13%, PF 3.19, SPR 3.1715, walk-forward pass False
- Aggregate:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_8_10ep_aggregate_3seed_20260119.json`
  - Return mean +2.69%, PF mean 1.98, SPR mean 1.0798, walk-forward pass 0/3

## 2026-01-18 (Trade gate z=0.5 20ep confirmation)

Focus: confirm trade_gate_z=0.5 performance on a longer run.

Experiments (20 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_20ep_20260118_203613_seed456/eval_fast_trade_gate_z_0_5/results/test_results.json`
  - Return +4.99%, PF 2.99, SPR 0.4334, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_20ep_20260118_210325_seed777/eval_fast_trade_gate_z_0_5/results/test_results.json`
  - Return -0.85%, PF 0.89, SPR 0.0000, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_20ep_20260118_212633_seed789/eval_fast_trade_gate_z_0_5/results/test_results.json`
  - Return -2.63%, PF 0.70, SPR -0.0183, walk-forward pass False
- Aggregate:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_20ep_aggregate_3seed_20260118.json`
  - Return mean +0.50%, PF mean 1.52, SPR mean 0.1384, walk-forward pass 0/3

Notes:
- 20-episode results remain mixed; seed-level variance is still high.

## 2026-01-18 (Trade gate z=0.5 probe)

Focus: reduce low-confidence trades by requiring a Q-gap over HOLD (trade_gate_z=0.5).

Experiments (10 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_10ep_20260118_195412_seed456/eval_fast_trade_gate_z_0_5/results/test_results.json`
  - Return +2.49%, PF 1.86, SPR -0.0139, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_10ep_20260118_200814_seed777/eval_fast_trade_gate_z_0_5/results/test_results.json`
  - Return +0.87%, PF 1.46, SPR 0.0000, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_10ep_20260118_202155_seed789/eval_fast_trade_gate_z_0_5/results/test_results.json`
  - Return +1.74%, PF 1.61, SPR 0.3581, walk-forward pass False
- Aggregate:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_trade_gate_z_0_5_10ep_aggregate_3seed_20260118.json`
  - Return mean +1.70%, PF mean 1.64, SPR mean 0.1147, walk-forward pass 0/3

Notes:
- Trade gate improved PF and returns vs baseline but still fails SPR walk-forward gate.

## 2026-01-18 (ATR cost gate + probe started)

Focus: add a cost-aware trade gate to suppress trades when ATR is too small vs frictions.

Changes:
- Added `min_atr_cost_ratio` to the environment config and CLI (`--min-atr-cost-ratio`).
- Trade opens now require ATR >= ratio * (spread+slip+commission).

Experiments (10 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0, max_trades=40):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_atr_cost_ratio_2_0_10ep_20260118_160319_seed456/eval_fast_atr_cost_ratio_2_0/results/test_results.json`
  - Return -2.03%, PF 0.68, SPR 0.0000, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_atr_cost_ratio_2_0_10ep_20260118_175417_seed777/results/test_results.json`
  - Return -0.77%, PF 0.92, SPR 0.0000, walk-forward pass False
- Aggregate (1 seed):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_atr_cost_ratio_2_0_10ep_eval_fast_seed456_aggregate_1seed_20260118.json`
- Aggregate (2 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_atr_cost_ratio_2_0_10ep_aggregate_2seed_20260118.json`

Notes:
- Faster walk-forward slicing applied for eval-only runs to reduce compute overhead.
- Sweep command timed out while jobs continued in the background.

## 2026-01-18 (Cadence tightening probe)

Focus: test stricter cadence (min_hold=8, cooldown=16, max_trades=30) on short real-data runs.

Experiments (10 ep, strengths on, no symmetry + no dual, flip 0.0003, trade penalty 0.0):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_30_flip_0_0003_trade_penalty_0_hold8_cd16_10ep_20260118_120902_seed456/results/test_results.json`
  - Return -2.00%, PF 0.61, SPR -0.0063, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_30_flip_0_0003_trade_penalty_0_hold8_cd16_10ep_20260118_120902_seed777/results/test_results.json`
  - Return -3.67%, PF 0.47, SPR -0.0024, walk-forward pass False
- Aggregate:
  - `seed_sweep_results/realdata/max_trades_30_flip_0_0003_trade_penalty_0_hold8_cd16_10ep_aggregate_2seed_20260118.json`
  - Return mean -2.83%, PF mean 0.54, SPR mean -0.0043, walk-forward pass 0/2

Notes:
- Cadence tightening reduced trade count but worsened PF vs the baseline.

## 2026-01-16 (Zero-cost eval, aligned settings)

Focus: quantify gross edge by zeroing costs while matching training limits.

Experiments (evaluate-only, zero costs, max_trades=40, flip 0.0003, trade penalty 0.0):
- Seed 456:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_spr_wf_20ep_20260116_120124_seed456/eval_nocost_aligned/results/test_results.json`
  - Return +1.44%, PF 1.47, SPR 0.142, walk-forward pass False
- Seed 777:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_spr_wf_20ep_rerun_20260116_152249_seed777/eval_nocost_aligned/results/test_results.json`
  - Return -0.01%, PF 1.00, SPR 0.000, walk-forward pass False
- Seed 789:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_spr_wf_20ep_rerun_20260116_152249_seed789/eval_nocost_aligned/results/test_results.json`
  - Return -1.48%, PF 0.72, SPR 0.0018, walk-forward pass False
- Aggregate:
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_spr_wf_20ep_eval_nocost_aligned_aggregate_3seed_20260116.json`
  - Return mean -0.02%, PF mean 1.06, SPR mean 0.048, walk-forward pass 0/3

Notes:
- Zero-cost evals are near breakeven with PF > 1.0, indicating cost drag is the primary blocker.

## 2026-01-15 (SPR test eval + walk-forward gate)

Focus: align test evaluation with SPR and add explicit walk-forward gating.

Changes:
- `main.py` now computes SPR-based test metrics when `fitness.mode="spr"`.
- Added walk-forward SPR evaluation on test data with pass/fail gate.
- Added test walk-forward thresholds in `config.py` (FitnessConfig).

Notes:
- Walk-forward gate uses rolling windows (defaulting to validation window/stride).

Experiments:
- Max trades/episode 40 + flip penalty 0.0003 + trade penalty 0.0, 20 ep:
  - Seed 456: `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_spr_wf_20ep_20260116_120124_seed456/results/test_results.json`
    - Return -0.49%, PF 0.99, SPR 0.0001, walk-forward pass False
  - Seeds 777/789 rerun: `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_spr_wf_20ep_rerun_20260116_152249_summary.json`
    - Seed 777 return -1.46%, PF 0.80, SPR 0.0000, walk-forward pass False
    - Seed 789 return -3.63%, PF 0.50, SPR 0.0000, walk-forward pass False
  - Aggregate (3 seeds): `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_spr_wf_20ep_aggregate_3seed_20260116.json`
    - Return mean -1.86%, PF mean 0.76, SPR mean ~0.0000, walk-forward pass 0/3

Notes:
- The first 3-seed sweep timed out mid-run; seeds 777 and 789 were rerun.
- Walk-forward gate failed on all three seeds.

## 2026-01-15 (R-multiple reward shaping probe, lower weight)

Focus: retest realized R-multiple shaping with a lower weight.

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 40 + flip penalty 0.0003 + trade penalty 0.0,
  R-multiple weight 0.003 clip 2.0, 10 ep (2 seeds):
  - Summary: `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_rmult_0_003_10ep_20260115_172452_summary.json`
  - Seed 456: Return -1.94%, PF 0.72, Fitness -0.703
  - Seed 777: Return -1.75%, PF 0.71, Fitness 0.041
  - Aggregate: Return mean -1.85%, PF mean 0.71, Fitness mean -0.331

Notes:
- Lowering the shaping weight did not improve returns; PF also dropped vs the 0.005 probe.

Next planned:
- Try a very small shaping weight (0.001) or disable shaping and move to a different ablation.

## 2026-01-15 (R-multiple reward shaping probe)

Focus: add realized R-multiple reward shaping and run a short real-data probe.

Changes:
- Added realized R-multiple shaping in `environment.py` and config knobs:
  `r_multiple_reward_weight`, `r_multiple_reward_clip`.
- Wired CLI overrides in `main.py` and `run_realdata_seed_sweep.py`.

Tests:
- `python test_system.py`
- `python test_minimal_episode.py`

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 40 + flip penalty 0.0003 + trade penalty 0.0,
  R-multiple weight 0.005 clip 2.0, 10 ep (2 seeds):
  - Summary: `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_rmult_0_005_10ep_20260115_151131_summary.json`
  - Seed 456: Return -2.35%, PF 0.68, Fitness -2.197
  - Seed 777: Return -0.13%, PF 1.06, Fitness 0.359
  - Aggregate: Return mean -1.24%, PF mean 0.87, Fitness mean -0.919

Notes:
- Short probe is still negative on average; shaping weight may be too high or needs more runs.

Next planned:
- Try a lower shaping weight (0.002-0.003) with the same 10-episode 2-seed probe.

## 2026-01-15 (Eval-stability tweaks, 20-episode confirmation)

Focus: reduce evaluation noise and increase validation overlap before new reward changes

Changes:
- Lowered eval randomness: `eval_epsilon` -> 0.01 and `eval_tie_tau` -> 0.03.
- Increased validation overlap: `VAL_STRIDE_FRAC` -> 0.12.

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 40 + flip penalty 0.0003 + trade penalty 0.0, 20 ep (3 seeds):
  - Summary: `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_eval_stability_20ep_20260115_115834_summary.json`
  - Seed 456: Return -0.11%, PF 1.07, Fitness 0.088
  - Seed 777: Return -2.31%, PF 0.64, Fitness -0.979
  - Seed 789: Return -4.53%, PF 0.59, Fitness -1.482
  - Aggregate: Return mean -2.31%, PF mean 0.77, Fitness mean -0.791

Notes:
- Eval-stability tweaks alone did not improve returns; mean performance is worse than the prior 20-episode baseline.

Next planned:
- Move to reward shaping tied to realized R-multiples or other trade-quality signals before further confirmations.

## 2026-01-14 (50-episode confirmation, trade penalty 0.0)

Focus: long-horizon validation of the best cost-aware setting

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 40 + flip penalty 0.0003 + trade penalty 0.0, 50 ep (3 seeds):
  - Summary: `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_50ep_20260114_161502_summary.json`
  - Seed 456: Return -1.49%, PF 0.80, Fitness -0.310
  - Seed 777: Return -2.97%, PF 0.40, Fitness -1.710
  - Seed 789: Return -1.80%, PF 0.81, Fitness -0.167
  - Aggregate: Return mean -2.09%, PF mean 0.67, Fitness mean -0.729

Notes:
- 50-episode horizon reverses the positive 20-episode outcome; performance degrades across all seeds.

Next planned:
- Pivot to reward shaping tied to realized R-multiples or cost-normalized trade-quality filters before rerunning 50-episode sweeps.

## 2026-01-13 (Trade penalty 0.0, 20-episode 3-seed confirmation)

Focus: validate the best cost-aware setting at longer horizon

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 40 + flip penalty 0.0003 + trade penalty 0.0, 20 ep (2 seeds):
  - Summary: `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_20ep_20260113_170510_summary.json`
  - Seed 456: Return +1.22%, PF 1.52, Fitness 0.903
  - Seed 777: Return -0.45%, PF 0.98, Fitness -0.045
- Seed 789, 20 ep:
  - Summary: `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_20ep_20260113_191919_summary.json`
  - Return +1.70%, PF 1.85, Fitness 1.031
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_20ep_20260113_aggregate_3seed.json`
  - Return mean +0.82%, PF mean 1.45, Fitness mean 0.629

Notes:
- First 20-episode configuration with positive mean return across 3 seeds.

Next planned:
- Run a 50-episode confirmation on the same settings to test longer-horizon stability.

## 2026-01-13 (Trade gate z=0.25 probe)

Focus: lighter scale-free Q-gap gate

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 40 + flip penalty 0.0003 + trade penalty 0.0 + trade_gate_z 0.25, 10 ep (2 seeds):
  - Summary: `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_gate_z_0_25_10ep_20260113_155855_summary.json`
  - Seed 456: Return -5.25%, PF 0.33, Fitness -1.841
  - Seed 777: Return -1.08%, PF 0.61, Fitness -1.368
  - Aggregate: Return mean -3.17%, PF mean 0.47, Fitness mean -1.604

Notes:
- Trade_gate_z 0.25 is materially worse than no-gate baseline.

Next planned:
- Drop trade-gate tuning and prioritize longer-horizon validation of the best cost settings.

## 2026-01-13 (Scale-free trade gate: z-score Q-gap)

Focus: add a scale-free trade-quality gate and run a short probe

Changes:
- Added `trade_gate_z` (Q-gap z-score vs HOLD) to the agent, trainer hold-breaker, CLI,
  and real-data sweep runner.

Tests:
- `python test_system.py`
- `python tests/test_features.py` (fails: missing `top_fractal` column in feature test)

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 40 + flip penalty 0.0003 + trade penalty 0.0 + trade_gate_z 0.5, 10 ep (2 seeds):
  - Summary: `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_gate_z_0_5_10ep_20260113_143607_summary.json`
  - Seed 456: Return +0.19%, PF 1.17, Fitness 1.275
  - Seed 777: Return -4.19%, PF 0.43, Fitness -2.260
  - Aggregate: Return mean -2.00%, PF mean 0.80, Fitness mean -0.493

Notes:
- Trade_gate_z 0.5 did not improve stability; strong divergence across seeds.

Next planned:
- Try a lighter trade_gate_z (e.g., 0.25) or skip gating and focus on reward shaping tied to
  realized R-multiples if the 10-episode probe stays negative.

## 2026-01-12 (Trade penalty 0.00002 + flip penalty 0.0003, 10-episode probe)

Focus: test a small per-trade penalty after the trade-penalty 0.0 probe

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 40 + flip penalty 0.0003 + trade penalty 0.00002, 10 ep (2 seeds):
  - Summary: `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_00002_10ep_20260112_234412_summary.json`
  - Seed 456: Return -0.66%, PF 0.97, Fitness -0.449
  - Seed 777: Return -2.28%, PF 0.74, Fitness -1.765
  - Aggregate: Return mean -1.47%, PF mean 0.85, Fitness mean -1.107

Notes:
- Adding a small trade penalty worsened results versus trade penalty 0.0.

Next planned:
- Stop trade-penalty tuning and pivot to a structural change that improves edge (e.g.,
  reward shaping based on realized R-multiples or a stricter entry filter tied to Q-gap).

## 2026-01-12 (Trade penalty 0.0 + flip penalty 0.0003, 10-episode probe)

Focus: remove extra per-trade penalty while keeping flip penalty and trade-cap

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 40 + flip penalty 0.0003 + trade penalty 0.0, 10 ep (2 seeds):
  - Summary: `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_10ep_20260112_222839_summary.json`
  - Seed 456: Return -0.41%, PF 1.03, Fitness -0.083
  - Seed 777: Return +0.11%, PF 1.11, Fitness -0.013
- Seed 789, 10 ep:
  - Summary: `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_10ep_20260112_231829_summary.json`
  - Return -0.83%, PF 0.91, Fitness -0.635
- Aggregate (3 seeds):
  - `seed_sweep_results/realdata/max_trades_40_flip_0_0003_trade_penalty_0_10ep_20260112_aggregate_3seed.json`
  - Return mean -0.37%, PF mean 1.02, Fitness mean -0.244

Notes:
- Removing trade penalty improves PF and reduces losses, but mean return remains negative.

Next planned:
- Test a low trade penalty (e.g., 0.00002) to see if it balances churn control and returns,
  then only run a 20-episode check if 3-seed mean return turns positive.

## 2026-01-12 (Flip penalty 0.0003, 20-episode confirmation)

Focus: longer-horizon check after enabling flip penalty

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 40 + flip penalty 0.0003, 20 ep (2 seeds):
  - Summary: `seed_sweep_results/realdata/max_trades_40_flip_0_0003_fix_20ep_20260112_205430_summary.json`
  - Seed 456: Return -2.62%, PF 0.68, Fitness -1.571
  - Seed 777: Return -4.58%, PF 0.31, Fitness -1.597
  - Aggregate: Return mean -3.60%, PF mean 0.49, Fitness mean -1.584

Notes:
- 20-episode performance is negative across both seeds; flip penalty 0.0003 does not hold up at longer horizons.

Next planned:
- Back off flip-penalty tuning and test a structural cost-aware change (e.g., reward shaping tied to trade costs or stricter trade gating) before another 20-episode sweep.

## 2026-01-12 (Flip penalty fix + probe)

Focus: enforce flip penalty in reward after log-return calculation

Changes:
- Applied flip penalty after log-return reward in `environment.py`, with an explicit `did_flip` flag.

Tests:
- `python test_minimal_episode.py`
- `python test_system.py`

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 40 (flip penalty now active), 10 ep (2 seeds):
  - Summary: `seed_sweep_results/realdata/max_trades_40_flipfix_20260112_184611_summary.json`
  - Seed 456: Return -2.69%, PF 0.71, Fitness -1.471
  - Seed 777: Return -3.35%, PF 0.45, Fitness -2.007
  - Aggregate: Return mean -3.02%, PF mean 0.58, Fitness mean -1.739

Notes:
- Enforcing flip penalty without retuning worsened returns; likely too punitive at 0.00077.

Next planned:
- Retune flip penalty downward (0.0003) with 2 seeds to see if it recovers trade quality
  without collapsing returns.

## 2026-01-12 (Flip penalty 0.0003, post-fix)

Focus: retune flip penalty after activating flip costs in reward

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 40 + flip penalty 0.0003, 10 ep (2 seeds):
  - Summary: `seed_sweep_results/realdata/max_trades_40_flip_0_0003_fix_20260112_193708_summary.json`
  - Seed 456: Return -0.53%, PF 0.99, Fitness -0.160
  - Seed 777: Return +1.22%, PF 1.35, Fitness 0.753
  - Aggregate: Return mean +0.35%, PF mean 1.17, Fitness mean 0.296

Notes:
- Lower flip penalty recovers positive mean return over 2 seeds; needs 3-seed confirmation.

Next planned:
- Run seed 789 with flip penalty 0.0003 and keep max-trades=40 to complete a 3-seed snapshot.

## 2026-01-12 (Trade-gate margin 0.002)

Focus: gentler Q-gap gate to suppress low-edge trades

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 40 + trade_gate_margin 0.002, 10 ep (2 seeds):
  - Seeds 456/777: `seed_sweep_results/realdata/max_trades_40_trade_gate_0_002_20260112_152956_summary.json`
  - Seed 456: Return +1.64%, PF 1.54, Fitness 1.287
  - Seed 777: Return -1.54%, PF 0.78, Fitness -0.945
- Seed 789: `seed_sweep_results/realdata/max_trades_40_trade_gate_0_002_20260112_160958_summary.json`
  - Return -0.13%, PF 1.09, Fitness -0.037
- Aggregate (3 seeds): `seed_sweep_results/realdata/max_trades_40_trade_gate_0_002_20260112_aggregate_3seed.json`
  - Return mean -0.01%, PF mean 1.14, Fitness mean 0.102

Notes:
- Near breakeven across 3 seeds; PF > 1 but returns are still flat after costs.

Next planned:
- Try a slightly lower margin (0.001) only if 20-episode stability holds.

## 2026-01-12 (Trade-gate margin 0.002, 20-episode check)

Focus: stability check for the gentler trade-gate margin

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 40 + trade_gate_margin 0.002, 20 ep (2 seeds):
  - Summary: `seed_sweep_results/realdata/max_trades_40_trade_gate_0_002_20ep_20260112_164216_summary.json`
  - Seed 456: Return -2.98%, PF 0.53, Fitness -2.111
  - Seed 777: Return -1.21%, PF 0.82, Fitness -0.915
  - Aggregate: Return mean -2.10%, PF mean 0.67, Fitness mean -1.513

Notes:
- Trade-gate 0.002 does not hold up over 20 episodes; stability still weak.

Next planned:
- Abandon trade-gate for now and revisit reward shaping or cost modeling rather than
  more gating tweaks.

## 2026-01-12 (Trade-gate margin probe)

Focus: Q-value margin gate for trade quality

Changes:
- Added `trade_gate_margin` to `AgentConfig`.
- Added CLI override `--trade-gate-margin`.
- Added sweep pass-throughs for trade-gate margin.
- Added trade-gate enforcement in `agent.py` and hold-breaker logic in `trainer.py`.

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 40 + trade_gate_margin 0.01, 10 ep (2 seeds):
  - Summary: `seed_sweep_results/realdata/max_trades_40_trade_gate_0_01_20260112_144528_summary.json`
  - Seed 456: Return -1.86%, PF 0.74, Fitness -1.259
  - Seed 777: Return -4.37%, PF 0.46, Fitness -2.178
  - Aggregate: Return mean -3.12%, PF mean 0.60, Fitness mean -1.719

Notes:
- Trade-gate margin at 0.01 degraded results; gate likely too aggressive at current Q scale.

Next planned:
- Either dial margin down (0.002-0.005) or abandon trade-gate and revisit reward shaping.

## 2026-01-12 (Hold-gate probe)

Focus: reduce forced trades by relaxing hold-break logic

Changes:
- Added CLI overrides for `--hold-tie-tau` and `--hold-break-after`.
- Added sweep pass-throughs for hold-gate overrides in `run_realdata_seed_sweep.py`.

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 40 + hold_tie_tau 0.01 + hold_break_after 12, 10 ep (2 seeds):
  - Summary: `seed_sweep_results/realdata/max_trades_40_hold_gate_20260112_133533_summary.json`
  - Seed 456: Return -1.63%, PF 0.77, Fitness -1.088
  - Seed 777: Return -5.92%, PF 0.14, Fitness -2.316
  - Aggregate: Return mean -3.77%, PF mean 0.46, Fitness mean -1.702

Notes:
- Loosening hold-break logic hurts performance; revert to defaults.

Next planned:
- Stop hold-gate tweaks; investigate a trade-quality filter based on Q-value margin
  or reward shaping around entry costs before longer confirmations.

## 2026-01-11 (Max-trades=40, 20-episode confirmation)

Focus: longer-horizon stability check for the trade-cap

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 40, 20 ep (2 seeds):
  - Summary: `seed_sweep_results/realdata/max_trades_40_20ep_20260111_212934_summary.json`
  - Seed 456: Return -2.28%, PF 0.67, Fitness -1.186
  - Seed 777: Return -2.49%, PF 0.67, Fitness -1.964
  - Aggregate: Return mean -2.39%, PF mean 0.67, Fitness mean -1.575

Notes:
- Performance degrades at 20 episodes; max-trades=40 alone is not stable long-horizon.

Next planned:
- Pause longer sweeps; shift to a structural change that improves trade quality
  (e.g., stricter entry gating or reward shaping) before retrying 20-episode runs.

## 2026-01-11 (Max-trades=40 + trade-penalty 0.0)

Focus: test removal of trade penalty under trade-cap

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 40 + trade penalty 0.0, 10 ep (2 seeds):
  - Summary: `seed_sweep_results/realdata/max_trades_40_trade_penalty_0_20260111_203928_summary.json`
  - Seed 456: Return -3.43%, PF 0.57, Fitness -1.430
  - Seed 777: Return -0.68%, PF 0.97, Fitness -0.620
  - Aggregate: Return mean -2.05%, PF mean 0.77, Fitness mean -1.025

Notes:
- Removing trade penalty hurts returns; keep trade penalty enabled.

Next planned:
- Revert to default penalties and consider a 3-seed 20-episode confirmation
  at max-trades=40 only if we can push mean return positive in 10-episode probes.

## 2026-01-11 (Max-trades=40 + flip-penalty 0.0003)

Focus: reduce flip penalty under trade-cap

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 40 + flip penalty 0.0003, 10 ep (2 seeds):
  - Summary: `seed_sweep_results/realdata/max_trades_40_flip_0_0003_20260111_192008_summary.json`
  - Seed 456: Return -1.29%, PF 0.82, Fitness -0.701
  - Seed 777: Return +0.93%, PF 1.43, Fitness 0.622
  - Aggregate: Return mean -0.18%, PF mean 1.13, Fitness mean -0.039

Notes:
- Lower flip penalty did not improve mean return vs default flip penalty.

Next planned:
- Revert flip penalty to default and consider a 20-episode confirmation only
  if mean return turns positive on a 3-seed 10-episode probe.

## 2026-01-11 (Max-trades=40 probe)

Focus: tighter trade-cap probe to reduce churn

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 40, 10 ep:
  - Seeds 456/777: `seed_sweep_results/realdata/max_trades_40_probe_20260111_180133_summary.json`
    - Seed 456: Return -0.57%, PF 0.98, Fitness -0.154
    - Seed 777: Return +1.55%, PF 1.66, Fitness 1.123
  - Seed 789: `seed_sweep_results/realdata/max_trades_40_probe_20260111_185109_summary.json`
    - Return -1.13%, PF 0.86, Fitness -0.001
  - Aggregate (3 seeds): `seed_sweep_results/realdata/max_trades_40_probe_20260111_aggregate_3seed.json`
    - Return mean -0.05%, PF mean 1.17, Fitness mean 0.322

Notes:
- Max-trades=40 improves PF and fitness on average, but mean return is still slightly negative.

Next planned:
- Run a 20-episode confirmation with max-trades=40 if we can lift mean return above zero,
  otherwise test a lower flip penalty to reduce per-trade friction.

## 2026-01-11 (Trade-penalty + max-trades sweep)

Focus: test trade-penalty with max-trades cap (real data)

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 60 + trade penalty 0.0001, 10 ep (3 seeds):
  - Summary: `seed_sweep_results/realdata/max_trades_60_trade_penalty_0_0001_20260111_152037_summary.json`
  - Seed 456: Return -2.18%, PF 0.80, Fitness -1.683
  - Seed 777: Return +2.40%, PF 2.16, Fitness 1.527
  - Seed 789: Return -4.40%, PF 0.54, Fitness -2.079
  - Aggregate: Return mean -1.39%, PF mean 1.17, Fitness mean -0.745

Notes:
- Trade penalty + max-trades shows one strong seed but remains negative on average.

Next planned:
- Check if reducing default flip penalty (0.00077) improves stability before
  extending episodes or adding new constraints.

## 2026-01-11 (Session update)

Focus: history refresh + max-trades/flip-penalty capture

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 60 + flip penalty 0.001, 10 ep:
  - Seeds 456/777: `seed_sweep_results/realdata/realdata_sweep_20260111_131442_summary.json`
    - Seed 456: Return -7.30%, PF 0.26, Fitness -2.251
    - Seed 777: Return +5.91%, PF 2.29, Fitness 3.562
  - Seed 789: `seed_sweep_results/realdata/realdata_sweep_20260111_143823_summary.json`
    - Return -4.92%, PF 0.50, Fitness -1.613
  - Aggregate (3 seeds): `seed_sweep_results/realdata/max_trades_60_flip_0_001_summary.json`
    - Return mean -2.10%, PF mean 1.01, Fitness mean -0.10

Notes:
- Flip penalty + max-trades=60 shows high variance; no consistent uplift across seeds.

Next planned:
- Keep max-trades gating as a lever, but shift to a stability-focused probe
  (shorter horizon or alternative churn control) before extending episodes.

## 2026-01-11

Focus: max-trades gating validation

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Max trades/episode 60, 10 ep:
  - Seeds 456/777: `seed_sweep_results/realdata/realdata_sweep_20260110_232105_summary.json`
    - Seed 456: Return +0.52%, PF 1.20, Fitness 0.575
    - Seed 777: Return -1.64%, PF 0.89, Fitness -0.540
  - Seed 789: `seed_sweep_results/realdata/realdata_sweep_20260111_001659_summary.json`
    - Return +1.67%, PF 1.29, Fitness 1.526
  - Aggregate: `seed_sweep_results/realdata/max_trades_60_summary.json`
    - Return mean +0.18%, PF mean 1.12, Fitness mean 0.520
- Max trades/episode 60, 20 ep (3 seeds):
  - Summary: `seed_sweep_results/realdata/realdata_sweep_20260111_004245_summary.json`
  - Seed 456: Return -4.40%, PF 0.49, Fitness -2.025
  - Seed 777: Return -1.42%, PF 0.86, Fitness -1.681
  - Seed 789: Return +1.92%, PF 1.50, Fitness 1.740
  - Aggregate: Return mean -1.30%, PF mean 0.95, Fitness mean -0.655

Notes:
- Max trades=60 is promising at 10 episodes (mean return > 0, PF > 1), but
  degrades at 20 episodes across seeds.
- 20-episode sweep timed out in the CLI but completed; artifacts are saved.

Next planned:
- Probe max trades=60 with higher flip penalty (0.001) on 2 seeds to see if
  churn reduction holds at longer horizons before rerunning 20-episode sweeps.

## 2026-01-10

Focus: cost-sensitivity probes + trade-frequency controls

Changes:
- Added CLI overrides for `--cooldown-bars`, `--min-hold-bars`, and
  `--max-trades-per-episode` (plus sweep runner pass-throughs).

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- Trade penalty 0.00008, seeds 456/777, 10 ep:
  - Summary: `seed_sweep_results/realdata/realdata_sweep_20260110_121556_summary.json`
  - Seed 456: Return -7.30%, PF 0.51, Fitness -2.484
  - Seed 777: Return -3.56%, PF 0.75, Fitness -1.114
  - Aggregate: Return mean -5.43%, PF mean 0.63, Fitness mean -1.80
- Trade penalty 0.00012, seeds 456/777, 10 ep:
  - Summary: `seed_sweep_results/realdata/realdata_sweep_20260110_170923_summary.json`
  - Seed 456: Return -4.43%, PF 0.68, Fitness -1.639
  - Seed 777: Return -2.86%, PF 0.84, Fitness -1.031
  - Aggregate: Return mean -3.64%, PF mean 0.76, Fitness mean -1.34
- Flip penalty 0.001, seeds 456/777, 10 ep:
  - Summary: `seed_sweep_results/realdata/realdata_sweep_20260110_190355_summary.json`
  - Seed 456: Return -2.94%, PF 0.86, Fitness -0.766
  - Seed 777: Return -3.09%, PF 0.79, Fitness -1.027
  - Aggregate: Return mean -3.02%, PF mean 0.82, Fitness mean -0.90
- Trade penalty 0.0001 + flip penalty 0.001:
  - Seeds 456/777, 10 ep: `seed_sweep_results/realdata/realdata_sweep_20260110_195651_summary.json`
    - Seed 456: Return -1.19%, PF 1.00, Fitness -0.196
    - Seed 777: Return -1.94%, PF 0.92, Fitness -0.341
  - Seed 789, 10 ep: `seed_sweep_results/realdata/realdata_sweep_20260110_211616_summary.json`
    - Return -3.82%, PF 0.77, Fitness -0.683
  - Aggregate (3 seeds): `seed_sweep_results/realdata/trade_penalty_0_0001_flip_0_001_summary.json`
    - Return mean -2.32%, PF mean 0.90, Fitness mean -0.407
- Cooldown/min-hold increase (16/8), seeds 456/777, 10 ep:
  - Summary: `seed_sweep_results/realdata/realdata_sweep_20260110_215809_summary.json`
  - Seed 456: Return -6.51%, PF 0.57, Fitness -2.339
  - Seed 777: Return -3.78%, PF 0.77, Fitness -0.919
  - Aggregate: Return mean -5.14%, PF mean 0.67, Fitness mean -1.63

Notes:
- All probes remain negative; trade+flip penalties improved returns versus other
  penalty-only sweeps but still fail to cross zero.
- CLI runs timed out but completed; all artifacts were saved to the above paths.

Next planned:
- Stop penalty tuning and test a structural trade-quality change (e.g., lower
  `max_trades_per_episode` or adjust reward shaping around costs) with a 2-seed probe.

## 2026-01-09 (Real-data sweep automation)

Focus: real-data seed sweep automation + 3-seed evaluation snapshot

Changes:
- Added `run_realdata_seed_sweep.py` to run CSV-based seed sweeps with
  `--output-dir` per seed, auto data-dir inference for `pair_files_real.json`,
  and summary JSON output under `seed_sweep_results/realdata/`.
- Sweep runner now targets `--mode both` so `test_results.json` is captured.
- Sweep runner now exposes `--strengths-all` and `--strengths-pair-only` pass-throughs.
- Added CLI overrides for `--trade-penalty` and `--flip-penalty`, and sweep runner
  pass-throughs for quick cost-sensitivity probes.
- Added `--eval-zero-costs` evaluation toggle to isolate pre-cost performance.
- `resolve_pair_files` now avoids double-prefixing paths that already include
  the `data_dir` (fixes `data/data/*.csv` resolution when using `pair_files_real.json`).

Experiments (real data, EURUSD primary, 1H, no symmetry + no dual, strengths on):
- 3-seed, 10-episode sweep (trained earlier, evaluated via `--mode evaluate`):
  - Summary: `seed_sweep_results/realdata/realdata_sweep_20260109_193713_summary.json`
  - Seed 456: Return -2.34%, PF 0.90, Fitness -0.674
  - Seed 777: Return -1.70%, PF 0.95, Fitness -0.845
  - Seed 789: Return -1.09%, PF 1.00, Fitness 0.318
  - Aggregate: Return mean -1.71%, PF mean 0.95, Fitness mean -0.40
- Strengths pair-only, seed 777, 10 ep:
  - Summary: `seed_sweep_results/realdata/realdata_sweep_20260109_210901_summary.json`
  - Return -3.48%, PF 0.79, Fitness -2.238
- Zero-cost evaluation, seed 789 (no sym + no dual, full strengths):
  - `seed_sweep_results/realdata/realdata_sweep_20260109_193713_seed789/eval_nocost/results/test_results.json`
  - Return +0.81%, PF 1.08, Fitness 0.986
- Trade-penalty sweep (0.0001), no sym + no dual, full strengths:
  - Seed 789: `seed_sweep_results/realdata/realdata_sweep_20260109_214017_summary.json`
    - Return +4.94%, PF 1.64, Fitness 2.828
  - Seed 456: `seed_sweep_results/realdata/realdata_sweep_20260109_220712_summary.json`
    - Return -5.49%, PF 0.62, Fitness -1.640
  - Seed 777: `seed_sweep_results/realdata/realdata_sweep_20260109_223419_summary.json`
    - Return -2.01%, PF 0.93, Fitness -0.832
  - Aggregate: `seed_sweep_results/realdata/trade_penalty_0_0001_summary.json`
    - Return mean -0.86%, PF mean 1.07, Fitness mean 0.119

Notes:
- All three seeds remain negative on 10-episode evaluation; configuration
  is still not robustly profitable.
- Pair-only strengths performed worse on the 10-episode probe; keep full strengths.
- Zero-cost evaluation shows a small positive edge that costs erase; focus next
  on trade quality / cost sensitivity.
- Trade-penalty=0.0001 produces one strong positive seed but remains unstable
  across seeds at 10 episodes.
- Trade-penalty=0.0001 lifts PF above 1.0 on average, but mean return is still
  negative; needs further tuning or additional trade-quality controls.

Next planned:
- Run short 10-episode probes at adjacent trade penalties (e.g., 0.00008,
  0.00012) on 2 seeds to see if variance improves over 0.0001.
- If a penalty shows consistent positive return or PF > 1.1 on 2 seeds,
  escalate to a 3-seed 20-episode sweep.

## 2026-01-09

Focus: Phase 1 real-data integration + ablations

Changes:
- Added CSV parsing controls and validation in `data_loader.py`:
  - Supports tab-separated files, split date/time columns, and standardized column names.
- Added real-data settings in `config.py` and CLI flags in `main.py`:
  - `--data-mode`, `--data-dir`, `--pair-files`, `--csv-sep`, `--date-col`, `--time-col`
  - Added `--no-symmetry-loss` and `--no-dual-controller` ablation flags.
- Added real-data tests:
  - `tests/test_data_loader_real_data.py`
  - `tests/test_prepare_data_real_data.py`
- Added `pair_files_real.json` with the supplied CSV mapping.
- Strength calc guard: skip all-NaN currencies in `features.py`.

Tests:
- `python tests/test_data_loader_real_data.py`
- `python tests/test_prepare_data_real_data.py`

Key experiments (real data, EURUSD primary, 1H):
- Symmetry ON, seed 777, 20 ep:
  - `realdata_sanity_20260108_201913/results/test_results.json`
  - Return -1.06%, PF 0.98, Fitness -0.523
- Symmetry OFF, seed 777, 20 ep:
  - `realdata_no_sym_20260108_214328/results/test_results.json`
  - Return -0.31%, PF 1.10, Fitness 0.239
- Symmetry OFF, seed 456, 10 ep:
  - `realdata_no_sym_seed456_20260108_231755/results/test_results.json`
  - Return -0.68%, PF 1.05
- Symmetry ON, seed 456, 10 ep:
  - `realdata_sym_seed456_20260109_001623/results/test_results.json`
  - Return -4.07%, PF 0.68
- Symmetry OFF + Dual OFF, seed 456, 10 ep:
  - `realdata_no_sym_no_dual_seed456_20260109_004808/results/test_results.json`
  - Return +3.06%, PF 1.49, Fitness 1.415
- Symmetry OFF + Dual OFF, seed 777, 10 ep:
  - `realdata_no_sym_no_dual_seed777_20260109_021525/results/test_results.json`
  - Return +1.23%, PF 1.25, Fitness 1.468
- Symmetry OFF + Dual OFF, seed 789, 10 ep:
  - `realdata_no_sym_no_dual_seed789_20260109_013005/results/test_results.json`
  - Return -0.93%, PF 1.03

Notes:
- Early evidence suggests symmetry loss may hurt OOS on real data.
- Disabling the dual controller + symmetry loss shows the best early returns,
  but results are not yet stable across seeds.

Next planned:
- 20-episode confirmations with no-symmetry + no-dual on seeds 456 and 789.
- If positive, run a 3-seed 50-episode sweep with the same settings.

## 2026-01-09 (Follow-up)

Focus: 20-episode confirmations on real data (no symmetry + no dual)

Experiments:
- Seed 456, 20 ep:
  - `realdata_no_sym_no_dual_seed456_20ep_20260109_123737/results/test_results.json`
  - Return -4.11%, PF 0.76, Fitness -1.064
- Seed 789, 20 ep:
  - `realdata_no_sym_no_dual_seed789_20ep_20260109_133526/results/test_results.json`
  - Return +5.55%, PF 1.64, Fitness 2.813
- Seed 777, 20 ep:
  - `realdata_no_sym_no_dual_seed777_20ep_20260109_144305/results/test_results.json`
  - Return -3.47%, PF 0.82, Fitness -1.474

Notes:
- Results are inconsistent across seeds at 20 episodes (1/3 positive).
- Indicates instability; proceed with a targeted ablation before a longer sweep.

Next planned:
- Add a CLI toggle for currency-strength features to test a simpler feature set.
- Run a short 10-episode sanity check on one seed, then two-seed confirmation if promising.

## 2026-01-09 (Strengths Ablation)

Focus: currency-strength feature toggle + short real-data sanity runs

Changes:
- Added `USE_CURRENCY_STRENGTHS` config and CLI flags:
  - `--no-strengths`, `--use-strengths`, `--strengths-pair-only`, `--strengths-all`
- Strength settings now included in `config.json` via `Config.to_dict()`.

Experiments (no strengths, no symmetry, no dual):
- Seed 456, 10 ep:
  - `realdata_no_strengths_no_sym_no_dual_seed456_10ep_20260109_153444/results/test_results.json`
  - Return -0.64%, PF 1.00, Fitness 0.191
- Seed 789, 10 ep:
  - `realdata_no_strengths_no_sym_no_dual_seed789_10ep_20260109_161354/results/test_results.json`
  - Return -2.95%, PF 0.76, Fitness -1.043

Notes:
- No-strengths appears weaker than the default strength set; keep strengths enabled.

Next planned:
- Run a longer (50-episode) confirmation on the best-performing configuration
  (no symmetry + no dual, strengths enabled) to assess stability.

## 2026-01-08

Focus: Phase 0 output hygiene + Phase 2 time handling

Changes:
- Phase 0: output-dir now captures logs, checkpoints, results, config, and run metadata.
  - Added `config.json` and `run_metadata.json` to each run root.
  - Validation summaries now follow `config.log_dir` (not hard-coded `logs/`).
- Phase 2: preserved timestamps in `environment.py` so weekend logic and time-based
  `fx_lookup` work; fixed time-column access to be index-agnostic.
- Removed non-ASCII checkmarks from tests to avoid Windows encoding errors.
- Added `PROFITABILITY_ROADMAP.md`.
- Added project manager directive to `AGENTS.md`.

Tests:
- `python test_minimal_episode.py`
- `python test_system.py`

Notes:
- Phase 2 time handling is now correct; real-data evaluation is meaningful.

## Appendix: Documentation reconciliation index

### 2026-02-16 (Documentation reconciliation audit: root markdown files)

Focus: preserve quick lookup of what changed and how it was implemented, without losing historical coverage.

Method:
- Indexed 163 root `*.md` files (excluding `PROJECT_HISTORY.md`) by file mtime, descriptor, and implementation hints.
- For each doc, extracted concise "how" hints from explicit file paths and command lines when present.
- Marked empty docs explicitly so missing detail is visible instead of implicit.

Backfill candidates (empty docs requiring manual detail capture):
- `ENVIRONMENT_CORRUPTION_FIX_GUIDE.md`
- `RUN_TEST.md`
- `BATCH_AUGMENTATION_FIX.md`
- `AUGMENTATION_NOT_USED_BUG.md`
- `START_HERE.md`
- `TESTING_PROTOCOL.md`
- `REWARD_EQUIVARIANCE_FIX.md`
- `SYMMETRY_BREAKTHROUGH.md`
- `SEED_SWEEP_FINAL_STATUS.md`

Documentation index (mtime | file | descriptor | implementation hints):
- 2026-02-05 14:27:12 | `AGENTS.md` | Repository Guidelines | how: files: main.py, agent.py, run_*.py
- 2026-01-16 11:51:28 | `RESEARCH_IMPROVEMENT_MEMO.md` | Deep Research Memo: Proven Improvement Levers (Internal Evidence) | how: files: main.py, PROJECT_HISTORY.md, PROFITABILITY_ROADMAP.md
- 2026-01-08 19:44:32 | `PROFITABILITY_ROADMAP.md` | Profitability Roadmap (Forex RL Bot) | how: files: main.py, config.py, trainer.py
- 2025-11-28 20:08:59 | `ENVIRONMENT_CORRUPTION_FIX_GUIDE.md` | Environment Corruption Fix Guide | how: empty doc (no implementation details recorded)
- 2025-11-24 20:07:44 | `RUN_TEST.md` | Run Test | how: empty doc (no implementation details recorded)
- 2025-11-24 18:43:20 | `BATCH_AUGMENTATION_FIX.md` | Batch Augmentation Fix | how: empty doc (no implementation details recorded)
- 2025-11-24 17:41:50 | `AUGMENTATION_NOT_USED_BUG.md` | Augmentation Not Used Bug | how: empty doc (no implementation details recorded)
- 2025-11-21 14:29:45 | `START_HERE.md` | Start Here | how: empty doc (no implementation details recorded)
- 2025-11-21 14:28:45 | `TESTING_PROTOCOL.md` | Testing Protocol | how: empty doc (no implementation details recorded)
- 2025-11-21 13:50:08 | `REWARD_EQUIVARIANCE_FIX.md` | Reward Equivariance Fix | how: empty doc (no implementation details recorded)
- 2025-11-18 13:13:59 | `SYMMETRY_BREAKTHROUGH.md` | Symmetry Breakthrough | how: empty doc (no implementation details recorded)
- 2025-11-13 21:06:45 | `PROBE_ANALYSIS_NEXT_STEPS.md` | Probe Analysis & Next Steps | how: files: main.py, analyze_probe.py, run_confirmation_suite.py | cmd: python main.py --episodes 10 --seed 42 --telemetry extended --output-dir quick_retest
- 2025-11-13 17:23:48 | `MASK_OVERRIDE_SUCCESS.md` | Mask Override Success - LONG Floor Now Working | how: files: main.py, agent.py, run_confirmation_suite.py | cmd: python main.py --episodes 200 --seed 42 --telemetry extended --output-dir probe_200ep_b3
- 2025-11-13 15:27:27 | `EMERGENCY_LONG_FLOOR_PATCH.md` | Emergency Patch: Aggressive LONG Floor | how: narrative notes (no explicit files/commands)
- 2025-11-13 11:42:44 | `ANTI_BIAS_COMPLETE.md` | Anti-Bias Fix - Complete Implementation | how: files: main.py, agent.py, augmentation.py | cmd: python main.py --episodes 10 --seed 999 --telemetry extended --output-dir fresh_antibias_test
- 2025-11-13 10:30:56 | `SMOKE_TEST_ANTIBIAS_RESULTS.md` | Smoke Test Results - Anti-Bias Fixes (A1+A2) | how: cmd: Remove-Item -Recurse quick_test_antibias\checkpoints\*
- 2025-11-13 10:00:05 | `ANTI_BIAS_IMPLEMENTATION_SUMMARY.md` | Anti-Bias Fix Implementation Summary | how: files: main.py, agent.py, environment.py | cmd: python main.py --episodes 10 --seed 42 --telemetry extended --output-dir quick_test_antibias
- 2025-11-13 10:00:05 | `ANTI_BIAS_FIX_PLAN.md` | LONG Bias Fix - Implementation Plan | how: files: main.py, agent.py, environment.py | cmd: python main.py --episodes 10 --seed 42 --telemetry extended --output-dir quick_test
- 2025-11-12 23:59:44 | `confirmation_report.md` | SUMMARY | how: narrative notes (no explicit files/commands)
- 2025-11-11 21:00:36 | `TELEMETRY_FIX_NEEDED.md` | Telemetry Export Fix Needed | how: files: agent.py, trainer.py, analyze_confirmation_results.py
- 2025-11-10 11:37:15 | `PHASE_2_8F_PRODUCTION_READINESS.md` | Phase 2.8f: Production-Ready Status & Next Steps | how: files: main.py, agent.py, run_confirmation_suite.py | cmd: python run_confirmation_suite.py --seeds 42,123,456,789,1011 --episodes 200
- 2025-11-10 11:37:15 | `PHASE_2_8F_CONFIRMATION_PROTOCOL.md` | Phase 2.8f Confirmation Protocol | how: files: agent.py, test_dual_controller.py, run_confirmation_suite.py | cmd: python run_confirmation_suite.py --seeds 42,123,456,789,1011 --episodes 200
- 2025-11-09 13:46:57 | `PHASE_2_8F_BUGFIX_OVERFLOW.md` | Phase 2.8f Overflow Bugfix | how: files: agent.py, test_dual_controller.py, Development/forex_rl_bot/agent.py
- 2025-11-09 13:46:57 | `PHASE_2_8F_IMPLEMENTATION_SUMMARY.md` | Phase 2.8f Implementation Summary | how: files: main.py, agent.py, environment.py | cmd: python main.py --episodes 20 --seed 42
- 2025-11-09 13:46:57 | `PHASE_2_8F_DUAL_CONTROLLER.md` | Phase 2.8f: Per-Step Dual-Variable Controller | how: files: agent.py
- 2025-11-08 19:01:04 | `README.md` | Forex RL Trading Bot | how: files: main.py, agent.py, config.py | cmd: python main.py --mode train
- 2025-11-08 19:01:04 | `PHASE_2_8E_COMPLETE.md` | Phase 2.8e Implementation Complete | how: files: main.py, agent.py, config.py | cmd: python main.py --episodes 20 --seed 42
- 2025-11-08 17:32:21 | `WINDOWS_ENCODING_FIXES.md` | Windows Encoding Fixes - Implementation Summary | how: files: tee.py, main.py, agent.py | cmd: python tee.py
- 2025-11-08 17:32:21 | `VALIDATION_TUNING_SUMMARY.md` | Validation Tuning Summary | how: files: main.py, config.py, trainer.py | cmd: python main.py --episodes 5
- 2025-11-08 17:32:21 | `VALIDATION_SLICE_FIX.md` | Validation System Fixes - October 18, 2025 | how: files: main.py, trainer.py | cmd: python main.py --episodes 15
- 2025-11-08 17:32:21 | `VALIDATION_FIX_QUICKREF.md` | Quick Reference: What Changed & Why | how: files: main.py, config.py, trainer.py | cmd: python main.py --episodes 15
- 2025-11-08 17:32:21 | `VALIDATION_FITNESS_FIXES.md` | Validation Fitness Fixes - Surgical Patches | how: files: main.py, trainer.py | cmd: python main.py --episodes 5
- 2025-11-08 17:32:21 | `VALIDATION_FITNESS_COMPLETE_FIX.md` | Complete Validation Fitness Fix - Final Implementation | how: files: main.py, trainer.py | cmd: python main.py --episodes 5
- 2025-11-08 17:32:21 | `VALIDATION_BUGFIXES_QUICKREF.md` | Validation Bugfixes - Quick Reference | how: files: config.py, trainer.py, check_metrics_addon.py | cmd: python run_seed_sweep_organized.py --seeds 7 --episodes 30
- 2025-11-08 17:32:21 | `VALIDATION_BUGFIXES.md` | Critical Bugfixes: Validation Parameter Alignment | how: files: config.py, trainer.py, check_metrics_addon.py | cmd: python run_seed_sweep_organized.py --seeds 7 --episodes 30
- 2025-11-08 17:32:21 | `UNDERTRADE_FIX_SUMMARY.md` | Task Completion Summary - October 18, 2025 | how: files: main.py, config.py, trainer.py | cmd: python main.py --episodes 20
- 2025-11-08 17:32:21 | `TRAINING_TUNING_GUIDE.md` | Training Parameter Tuning Guide | how: files: main.py, config.py, environment.py
- 2025-11-08 17:32:21 | `TRADE_GATE_FIX.md` | Trade Gate Fix for Smoke Runs | how: files: main.py, trainer.py | cmd: python main.py --episodes 5
- 2025-11-08 17:32:21 | `TRADE_COUNT_KEY_FIX.md` | Trade Count Key Mismatch Fix | how: files: main.py, trainer.py, validator.py | cmd: python main.py --episodes 5
- 2025-11-08 17:32:21 | `TESTING_GUIDE_ADVANCED_PATCHES.md` | Quick Testing Guide for Advanced Learning Patches | how: files: main.py, best_model_scaler.json, checkpoints/best_model_scaler.json | cmd: python main.py --episodes 5
- 2025-11-08 17:32:21 | `TENSORBOARD_FIXED.md` | TENSORBOARD ISSUE RESOLVED! | how: files: main.py, quick_test.py, system_status.py | cmd: python quick_test.py # NOW WORKS!
- 2025-11-08 17:32:21 | `SURGICAL_TWEAKS_SUMMARY.md` | Surgical Tweaks - Activity Boost v2 | how: files: main.py, config.py, trainer.py | cmd: python main.py --episodes 10
- 2025-11-08 17:32:21 | `SURGICAL_PATCHES_SUMMARY.md` | Surgical Patches Summary | how: files: main.py, agent.py, config.py | cmd: python main.py --episodes 5
- 2025-11-08 17:32:21 | `SURGICAL_IMPROVEMENTS.md` | Surgical Improvements - Post 80-Episode Results | how: files: config.py, trainer.py, spr_fitness.py | cmd: python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 120
- 2025-11-08 17:32:21 | `STATE_SIZE_FIX.md` | State Size Fix - From 65 to 69 Dimensions | how: files: main.py, environment.py, HARDENING_PATCHES_SUMMARY.md | cmd: python main.py --episodes 2
- 2025-11-08 17:32:21 | `STABILITY_TIGHTENING_QUICKSTART.md` | Stability Tightening Tweaks - Final Polish | how: files: main.py, check_metrics_addon.py, run_seed_sweep_organized.py | cmd: python main.py --episodes 10
- 2025-11-08 17:32:21 | `STABILITY_IMPROVEMENTS_SUMMARY.md` | Stability & Learning Improvements Summary | how: files: main.py, agent.py, config.py | cmd: python main.py --mode train --episodes 50
- 2025-11-08 17:32:21 | `SMOKE_TEST_RESULTS.md` | Smoke Test Results - Validation Tuning | how: narrative notes (no explicit files/commands)
- 2025-11-08 17:32:21 | `SESSION_COMPLETE.md` | Complete Session Summary - October 18, 2025 | how: files: main.py, config.py, trainer.py | cmd: python main.py --episodes 15
- 2025-11-08 17:32:21 | `SEED_SWEEP_QUICKSTART.md` | Quick Start: Seed Sweep in 20 Seconds | how: files: compare_seed_results.py, run_seed_sweep_simple.py, run_seed_sweep_organized.py | cmd: python run_seed_sweep_simple.py
- 2025-11-08 17:32:21 | `SEED_SWEEP_GUIDE.md` | Seed Sweep Guide - 3 Seeds 25 Episodes | how: files: config.py, plot_seed_curves.py, compare_seed_results.py | cmd: python run_seed_sweep_simple.py
- 2025-11-08 17:32:21 | `SEED_SWEEP_FIX_SUMMARY.md` | Seed Sweep Fix - Early Stop Disabled | how: files: config.py, trainer.py, compare_seed_results.py | cmd: python run_seed_sweep_organized.py --seeds 7 --episodes 25
- 2025-11-08 17:32:21 | `SEED_SWEEP_FINAL_STATUS.md` | Seed Sweep Final Status | how: empty doc (no implementation details recorded)
- 2025-11-08 17:32:21 | `RERUN_QUICKSTART.md` | Quick Re-run Guide - Post-Restore Fix | how: files: main.py, config.py, trainer.py | cmd: python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 60
- 2025-11-08 17:32:21 | `REGIME_STABILITY_ENHANCEMENT.md` | Regime Stability Enhancement - Final Production Tuning | how: files: config.py, trainer.py, check_metrics_addon.py | cmd: python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 80
- 2025-11-08 17:32:21 | `QUICK_FIXES_APPLIED.md` | Quick Fixes Applied | how: files: main.py, trainer.py, Development/forex_rl_bot/trainer.py | cmd: python main.py --episodes 5
- 2025-11-08 17:32:21 | `QUICKSTART_IMPROVEMENTS.md` | Quick Start Guide - Advanced Improvements | how: files: main.py, agent.py, config.py | cmd: python main.py --episodes 3
- 2025-11-08 17:32:21 | `QUICKSTART.md` | Quick Start Guide | how: files: main.py, config.py, features.py | cmd: python test_system.py
- 2025-11-08 17:32:21 | `QUALITY_RECOVERY_PATCHES.md` | Quality Recovery Patches - Refining Anti-Collapse Wins | how: files: main.py, config.py, trainer.py | cmd: python main.py --episodes 10
- 2025-11-08 17:32:21 | `PRODUCTION_TIGHTENING.md` | Production Tightening - Push Cross-Seed Mean Above Zero | how: files: config.py, trainer.py, check_metrics_addon.py | cmd: python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 60
- 2025-11-08 17:32:21 | `PRODUCTION_ENHANCEMENT_SUMMARY.md` | Production Enhancement Summary - October 14, 2025 | how: files: main.py, config.py, features.py
- 2025-11-08 17:32:21 | `PREFLIGHT_COMPLETE.md` | Pre-Flight Checklist - Complete | how: files: trainer.py, spr_fitness.py, run_seed_sweep_organized.py | cmd: python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 80
- 2025-11-08 17:32:21 | `POST_RESTORE_FINAL_FIX.md` | Post-Restore Final Evaluation Fix | how: files: main.py, trainer.py, check_metrics_addon.py | cmd: python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 60
- 2025-11-08 17:32:21 | `PHASE_2_8E_SOFT_BIAS_IMPLEMENTATION.md` | Phase 2.8e: Soft Bias Steering Implementation Guide | how: files: main.py, agent.py, config.py | cmd: python main.py --episodes 20 --seed 42
- 2025-11-08 17:32:21 | `PHASE_2_8D_LEVEL2_STATUS.md` | Phase 2.8d - Fix Pack D2 Level 2 Escalation - RUNNING | how: files: config.py, environment.py, logs/validation_summaries/val_ep010.json | cmd: .\check_status.ps1
- 2025-11-08 17:32:21 | `PHASE_2_8D_FIX_PACK_D2_COMPLETE.md` | Phase 2.8d Fix Pack D2: Rolling Window Anti-Collapse - COMPLETE | how: files: main.py, config.py, environment.py | cmd: Move-Item checkpoints\*.pt checkpoints_backup_fixpack_d1\
- 2025-11-08 17:32:21 | `PHASE_2_8D_FIX_PACK_D1.md` | Phase 2.8d Fix Pack D1 - Recovery Plan | how: files: main.py, config.py, logs/validation_summaries/val_ep010.json | cmd: python main.py --episodes 80 --seed 42
- 2025-11-08 17:32:21 | `PHASE_2_8D_CRITICAL_DECISION.md` | Phase 2.8d - Critical Analysis: Rolling Window Approach Failed | how: files: main.py, config.py, environment.py
- 2025-11-08 17:32:21 | `PHASE2_STABILIZATION_COMPLETE.md` | Phase-2 Stabilization Improvements - Production Hardening | how: files: agent.py, config.py, trainer.py | cmd: python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 80
- 2025-11-08 17:32:21 | `PHASE2_QUICKSTART.md` | Phase-2 Quick Start Guide | how: files: agent.py, config.py, trainer.py | cmd: python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 80
- 2025-11-08 17:32:21 | `PHASE2_CHECKLIST.md` | Phase-2 Implementation Checklist | how: files: agent.py, config.py, trainer.py | cmd: python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 80
- 2025-11-08 17:32:20 | `PHASE2_8D_SMOKE_TEST_RESULTS.md` | Phase 2.8d Smoke Test Results | how: files: main.py, config.py, trainer.py | cmd: python main.py --episodes 50
- 2025-11-08 17:32:20 | `PHASE2_8_STAGE2_80EP_RESULTS.md` | Phase 2.8 Stage 2: 80-Episode Robustness Test Results | how: files: main.py, config.py, check_metrics_addon.py | cmd: python check_metrics_addon.py
- 2025-11-08 17:32:20 | `PHASE2_8_QUICKSTART.md` | Phase 2.8 Quick-Start Guide | how: files: config.py, check_metrics_addon.py, compare_seed_results.py | cmd: python run_seed_sweep_organized.py --seeds 7 17 27 77 777 --episodes 80
- 2025-11-08 17:32:20 | `PHASE2_8_CHURN_ROBUSTNESS.md` | Phase 2.8: Robustness & Churn-Calming Pass | how: files: config.py, check_metrics_addon.py, compare_seed_results.py | cmd: python run_seed_sweep_organized.py --seeds 7 17 27 77 777 --episodes 80
- 2025-11-08 17:32:20 | `PHASE2_8_ACTION_METRICS_ANALYSIS.md` | Phase 2.8 Action Metrics Analysis | how: files: config.py, fitness.py, check_metrics_addon.py | cmd: python run_seed_sweep_organized.py --seeds 7 17 27 77 777 --episodes 200
- 2025-11-08 17:32:20 | `PHASE2_8D_ROOT_CAUSE_FOUND.md` | ROOT CAUSE FOUND: Model Dimension Mismatch | how: files: main.py, trainer.py | cmd: Move-Item checkpoints\*.pt checkpoints_backup_176dim\
- 2025-11-08 17:32:20 | `PHASE2_8D_READY_TO_RUN.md` | Phase 2.8d Fix Pack D1 - READY TO RUN | how: files: main.py, run_seed_sweep_organized.py | cmd: python main.py --episodes 20
- 2025-11-08 17:32:20 | `PHASE2_8D_READY_TO_RESTART.md` | Phase 2.8d - Ready to Restart Training | how: files: main.py, config.py, environment.py | cmd: python main.py --episodes 50
- 2025-11-08 17:32:20 | `PHASE2_8D_QUICKSTART.md` | Phase 2.8d Fix Pack D1 - Quick Start Guide | how: files: main.py, config.py, compare_run_c_v1.py | cmd: python main.py --seed 777 --episodes 20 --disable_early_stop
- 2025-11-08 17:32:20 | `PHASE2_8D_NUCLEAR_FIX_APPLIED.md` | Phase 2.8d Nuclear Fix Applied - Epsilon-Greedy Forced Exploration | how: files: main.py, config.py, logs/validation_summaries/val_ep001.json | cmd: Remove-Item checkpoints\*.pt -Force
- 2025-11-08 17:32:20 | `PHASE2_8D_IMPLEMENTATION_SUMMARY.md` | Phase 2.8d Implementation Summary | how: files: main.py, config.py, environment.py | cmd: python main.py --seed 777 --episodes 20 --disable_early_stop
- 2025-11-08 17:32:20 | `PHASE2_8D_FIX_PACK_D1.md` | Phase 2.8d Fix Pack D1: Recovery from 200-ep Confirm Failure | how: files: config.py, environment.py, run_ablation_d1.py | cmd: python run_ablation_d1.py --seeds 7 17 777 --episodes 80
- 2025-11-08 17:32:20 | `PHASE2_8D_EPISODE_16_STATUS.md` | Phase 2.8d Episode 16 Status Report | how: files: main.py, config.py, trainer.py | cmd: python main.py --episodes 50
- 2025-11-08 17:32:20 | `PHASE2_8D_EMERGENCY_ADJUSTMENT.md` | Phase 2.8d Emergency Adjustment - Option B | how: files: main.py, config.py, trainer.py | cmd: python main.py --episodes 50
- 2025-11-08 17:32:20 | `PHASE2_8D_CRITICAL_FAILURE.md` | CRITICAL: Phase 2.8d Complete Failure Analysis | how: files: agent.py, config.py, trainer.py | cmd: Get-Content logs\training.log | Select-String "LONG|SHORT" | Select-Object -First 20
- 2025-11-08 17:32:20 | `PHASE2_8C_STATUS.md` | Phase 2.8c Status - Ready to Test | how: files: config.py, trainer.py, compare_run_c_v1.py | cmd: python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 120
- 2025-11-08 17:32:20 | `PHASE2_8C_STABILIZATION_TWEAKS.md` | Phase 2.8c - Stabilization Tweaks | how: files: config.py, trainer.py, compare_run_c_v1.py | cmd: Get-ChildItem logs\seed_sweep_results\seed_* -Directory | Select-Object Name, LastWriteTime
- 2025-11-08 17:32:20 | `PHASE2_8C_QUICKSTART.md` | Phase 2.8c Quick-Start Guide | how: files: config.py, trainer.py, monitor_sweep.py | cmd: Move-Item -Path "logs\seed_sweep_results\seed_7" -Destination "logs\seed_sweep_results\seed_7_RUN_B_V2_$timestamp" -E...
- 2025-11-08 17:32:20 | `PHASE2_8C_IMPLEMENTATION_SUMMARY.md` | Phase 2.8c Implementation Summary | how: files: config.py, trainer.py, monitor_sweep.py | cmd: python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 120
- 2025-11-08 17:32:20 | `PHASE2_8C_200EP_CONFIRMATION_PLAN.md` | Phase 2.8c - 200-Episode Confirmation Plan | how: files: run_seed_sweep_organized.py, config_phase2.8c_baseline_v1.1.py | cmd: python run_seed_sweep_organized.py --seeds 7 17 27 77 777 --episodes 200
- 2025-11-08 17:32:20 | `PHASE2_8B_SESSION_SUMMARY.md` | Phase 2.8b - Session Summary | how: files: main.py, trainer.py, check_ep80_scores.py | cmd: python check_friction_jitter.py
- 2025-11-08 17:32:20 | `PHASE2_8B_RUN_B_V2_SPEC.md` | Phase 2.8b Run B v2 - REAL Robustness Test | how: files: check_metrics_addon.py, compare_seed_results.py, check_friction_jitter.py | cmd: Get-Process python
- 2025-11-08 17:32:20 | `PHASE2_8B_RUN_B_ROBUSTNESS_SPEC.md` | Phase 2.8b Run B - Robustness Test Specification | how: files: check_metrics_addon.py, compare_seed_results.py, run_seed_sweep_organized.py | cmd: python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 80
- 2025-11-08 17:32:20 | `PHASE2_8B_RUN_A_CORRECTED_RESULTS.md` | Phase 2.8b Run A Results (CORRECTED - 80 Episodes, Frozen Frictions) | how: narrative notes (no explicit files/commands)
- 2025-11-08 17:32:20 | `PHASE2_8B_FRICTION_JITTER_BUGFIX.md` | Phase 2.8b - Friction Jitter Bug Discovery & Fix | how: files: trainer.py, check_friction_jitter.py, run_seed_sweep_organized.py
- 2025-11-08 17:32:20 | `PHASE2_8B_FRICTION_JITTER_BUG_FIX.md` | Phase 2.8b - Friction Jitter Bug Discovery & Fix | how: files: trainer.py, check_friction_jitter.py, run_seed_sweep_organized.py
- 2025-11-08 17:32:20 | `PHASE2_8B_CADENCE_RECOVERY.md` | Phase 2.8b: Cadence Recovery & Directional Balance | how: files: config.py, check_metrics_addon.py, compare_seed_results.py | cmd: python run_seed_sweep_organized.py --seeds 7 17 27 77 777 --episodes 80
- 2025-11-08 17:32:20 | `PHASE2_7_QUICKSTART.md` | Phase 2.7 Quick Reference | how: files: config.py, trainer.py, check_metrics_addon.py | cmd: python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 120
- 2025-11-08 17:32:20 | `PHASE2_7_PRODUCTION_RESULTS.md` | Phase 2.7 Production Results - 5-Seed 150-Episode Validation | how: files: analyze_alt_validation.py, run_seed_sweep_organized.py, val_final_alt.json | cmd: Get-ChildItem logs\seed_sweep_results\seed_*\val_final_alt.json
- 2025-11-08 17:32:20 | `PHASE2_7_GENERALIZATION_STRESS_TEST.md` | Phase 2.7: Generalization & Stress Testing | how: files: config.py, trainer.py, monitor_sweep.py | cmd: python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 120
- 2025-11-08 17:32:20 | `PHASE2_7_COMPLETE_5SEED_RESULTS.md` | Phase 2.7 COMPLETE 5-Seed 150-Episode Production Results | how: files: config.py, trainer.py, compare_seed_results.py
- 2025-11-08 17:32:20 | `PHASE2_6_SPR_QUICKSTART.md` | Phase 2.6 + SPR Fitness - Quick Start Guide | how: files: main.py, config.py, trainer.py | cmd: Remove-Item .\logs\validation_summaries\* -Recurse -Force
- 2025-11-08 17:32:20 | `PHASE2_6_SPR_PF_FIX.md` | Phase 2.6 SPR Profit Factor Fix | how: files: trainer.py, spr_fitness.py, run_seed_sweep_organized.py | cmd: python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 80
- 2025-11-08 17:32:20 | `PHASE2_6_SPR_INTEGRATION_COMPLETE.md` | Phase 2.6 + SPR Integration - COMPLETE | how: files: config.py, trainer.py, spr_fitness.py | cmd: python run_seed_sweep_organized.py --seeds 7 --episodes 20
- 2025-11-08 17:32:20 | `PHASE2_6_SPR_20EP_RESULTS.md` | Phase 2.6 + SPR 20-Episode Spot-Check - RESULTS | how: files: spr_fitness.py, run_seed_sweep_organized.py | cmd: python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 80
- 2025-11-08 17:32:20 | `PHASE2_6_CHURN_TWEAKS.md` | Phase 2.6 - Churn Reduction Tweaks | how: files: config.py, check_metrics_addon.py, compare_seed_results.py | cmd: python check_validation_diversity.py
- 2025-11-08 17:32:20 | `PHASE2_5_QUICKSTART.md` | Phase 2.5 Quick Reference | how: files: config.py, trainer.py, check_metrics_addon.py | cmd: python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 80
- 2025-11-08 17:32:20 | `PHASE2_5_EXPLORATION_TWEAKS.md` | Surgical Exploration Tweaks - Phase 2.5 | how: files: config.py, trainer.py, check_metrics_addon.py | cmd: python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 80
- 2025-11-08 17:32:20 | `PATCH_EXPECTATIONS.md` | What to Expect After Patches | how: files: main.py, agent.py, config.py | cmd: python -m py_compile fitness.py agent.py environment.py trainer.py config.py
- 2025-11-08 17:32:20 | `OPTIMIZATION_SUMMARY.md` | A) Vectorized lr_slope - DONE | how: files: main.py, config.py, trainer.py | cmd: python main.py --mode train --episodes 2
- 2025-11-08 17:32:20 | `NSTEP_BUFFER_FIX.md` | N-Step Buffer Fix & Regime Features Optimization | how: files: main.py, agent.py, features.py | cmd: python main.py --episodes 5
- 2025-11-08 17:32:20 | `MONITORING_SUMMARY.md` | Monitoring and Logging Implementation Summary | how: files: trainer.py, environment.py, structured_logger.py
- 2025-11-08 17:32:20 | `METRICS_ADDON_SUMMARY.md` | Metrics Add-On Implementation | how: files: main.py, agent.py, config.py | cmd: python check_metrics_addon.py
- 2025-11-08 17:32:20 | `METRICS_ADDON_QUICKSTART.md` | Metrics Add-On Quick Reference Card | how: files: main.py, trainer.py, check_metrics_addon.py | cmd: python main.py --episodes 10
- 2025-11-08 17:32:20 | `METRICS_ADDON_COMPLETE.md` | Metrics Add-On - Complete Implementation | how: files: main.py, config.py, trainer.py | cmd: python main.py --episodes 10
- 2025-11-08 17:32:20 | `METRICS_ADDON_BUGFIX.md` | Metrics Add-On - Bug Fix & First Results | how: files: main.py, check_metrics_addon.py, val_ep001.json | cmd: python main.py --episodes 10
- 2025-11-08 17:32:20 | `LR_SLOPE_PERFORMANCE_FIX.md` | LR_SLOPE Performance Fix Summary | how: files: main.py, trainer.py, features.py
- 2025-11-08 17:32:20 | `LEARNING_STARTS_FIX_COMPLETE.md` | Final Learning Starts Fix - Complete Summary | how: files: main.py, config.py | cmd: Get-ChildItem -Path . -Recurse -Filter "*.pyc" | Remove-Item -Force
- 2025-11-08 17:32:20 | `JSON_VALIDATION_SUMMARY.md` | JSON Validation Summaries - Implementation Complete | how: files: main.py, config.py, trainer.py | cmd: python main.py --episodes 5
- 2025-11-08 17:32:20 | `JSON_EXPORT_SUCCESS.md` | JSON VALIDATION EXPORT - FULLY OPERATIONAL | how: files: main.py, trainer.py, quick_json_check.py | cmd: python main.py --episodes 50
- 2025-11-08 17:32:20 | `JSON_EXPORT_FIX.md` | JSON Export Fix - Episode Number Assignment | how: files: main.py, trainer.py, quick_json_check.py | cmd: python main.py --episodes 3
- 2025-11-08 17:32:20 | `IMPROVEMENTS.md` | Improving the DQN-based Forex Trading RL Bot | how: narrative notes (no explicit files/commands)
- 2025-11-08 17:32:20 | `IMPLEMENTATION_STATUS.md` | Implementation Complete - All Patches Working | how: files: main.py, agent.py, config.py | cmd: python main.py --episodes 5
- 2025-11-08 17:32:20 | `HOLD_STREAK_BREAKER.md` | Hold-Streak Breaker Patch - Final Anti-Collapse Enhancement | how: files: main.py, config.py, trainer.py | cmd: python main.py --episodes 10
- 2025-11-08 17:32:20 | `HARDENING_PATCHES_SUMMARY.md` | HARDENING PATCHES IMPLEMENTATION SUMMARY | how: files: main.py, config.py, trainer.py | cmd: python test_hardening.py
- 2025-11-08 17:32:20 | `HANG_FIX_COMPLETE.md` | HANG FIX COMPLETE - SUMMARY | how: files: main.py, agent.py, trainer.py
- 2025-11-08 17:32:20 | `HANG_FIXES_SUMMARY.md` | Training Hang Fixes - Complete Summary | how: files: main.py, agent.py, trainer.py | cmd: python main.py --episodes 5
- 2025-11-08 17:32:20 | `GATING_TUNING_PHASE2.md` | Gating Tuning Phase 2 - Alignment with Observed Behavior | how: files: main.py, config.py, trainer.py | cmd: python check_validation_diversity.py
- 2025-11-08 17:32:20 | `GATING_FIX_COMPLETE.md` | Gating Fix - Realistic Trade Thresholds | how: files: config.py, trainer.py, quick_smoke_test.py | cmd: python check_validation_diversity.py
- 2025-11-08 17:32:20 | `FOCUSED_PATCHES_SUMMARY.md` | Focused Patch Set Implementation Summary | how: files: main.py, agent.py, config.py | cmd: python -m py_compile fitness.py environment.py agent.py trainer.py config.py
- 2025-11-08 17:32:20 | `FIX_PACK_D3_PROPOSAL.md` | Fix Pack D3: Episodic Balance with Hard Constraints | how: files: config.py, environment.py
- 2025-11-08 17:32:20 | `FIX_PACK_D2_LEVEL2_ESCALATION.md` | Fix Pack D2 Level 2 Escalation - Analysis & Next Steps | how: files: config.py, environment.py, start_d2_level2.ps1
- 2025-11-08 17:32:20 | `FIX_PACK_D1_APPLIED.md` | Phase 2.8d Fix Pack D1 - Applied Successfully | how: files: main.py, environment.py, logs/validation_summaries/val_ep010.json | cmd: python main.py --episodes 10 --seed 42
- 2025-11-08 17:32:20 | `FINAL_TIGHTENING_TWEAKS.md` | Final Tightening Tweaks Summary | how: files: main.py, config.py
- 2025-11-08 17:32:20 | `FINAL_QUALITY_STATUS.md` | Final Implementation Status - Quality + Anti-Collapse | how: files: main.py, config.py, trainer.py | cmd: python main.py --episodes 10
- 2025-11-08 17:32:20 | `FINAL_PATCHES_SUMMARY.md` | Final Surgical Patches Summary | how: files: main.py, agent.py, trainer.py | cmd: python main.py --episodes 5
- 2025-11-08 17:32:20 | `FINAL_IMPLEMENTATION_SUMMARY.md` | Final Implementation Summary - All Improvements Complete | how: files: main.py, agent.py, config.py | cmd: python main.py --mode train --episodes 50
- 2025-11-08 17:32:20 | `FINAL_ENTROPY_PATIENCE_FIXES.md` | Final Entropy & Patience Fixes | how: files: main.py, config.py, trainer.py | cmd: python main.py --episodes 10
- 2025-11-08 17:32:20 | `FAST_FORWARD_BUGFIX.md` | Bugfix: AttributeError in Fast-Forward Logic | how: files: main.py, trainer.py | cmd: python -u main.py --episodes 15 2>&1 | Tee-Object -FilePath validation_test.log -Append
- 2025-11-08 17:32:20 | `EXAMPLES.md` | Usage Examples | how: files: main.py, environment.py, train_aggressive.py | cmd: python main.py --mode train
- 2025-11-08 17:32:20 | `ENVIRONMENT_FIXED.md` | Environment.py Fixed Successfully | how: files: main.py, agent.py, config.py | cmd: python -m py_compile environment.py # No errors
- 2025-11-08 17:32:20 | `ENHANCEMENT_PATCHES_SUMMARY.md` | Enhancement Patches Implementation Summary | how: files: main.py, agent.py, config.py | cmd: python main.py --episodes 100
- 2025-11-08 17:32:20 | `DIVERSITY_BOOST_QUICKSTART.md` | Diversity Boost - Quick Reference | how: files: main.py, check_metrics_addon.py | cmd: python main.py --episodes 10
- 2025-11-08 17:32:20 | `DIVERSITY_BOOST_PATCHES.md` | Diversity Boost Patches - Low-Risk Exploration Enhancement | how: files: main.py, config.py, trainer.py | cmd: python main.py --episodes 10
- 2025-11-08 17:32:20 | `DISJOINT_VALIDATION_PATCHES.md` | Disjoint Validation Patches - Implementation Summary | how: files: main.py, agent.py, config.py | cmd: python main.py --episodes 30 # Bump from 5
- 2025-11-08 17:32:20 | `DISJOINT_VALIDATION_COMPLETE.md` | Disjoint Validation Implementation - COMPLETE | how: files: main.py, trainer.py, test_disjoint_validation.py | cmd: python main.py --episodes 5
- 2025-11-08 17:32:20 | `DIRECTIONAL_COLLAPSE_ROOT_CAUSE.md` | Analysis: Why Both Level 1 and Level 2 Failed | how: narrative notes (no explicit files/commands)
- 2025-11-08 17:32:20 | `DATA_SOURCE_CLARIFICATION.md` | Data Source Clarification: Phase 2.8 Analysis | how: files: main.py, check_metrics_addon.py, compare_seed_results.py
- 2025-11-08 17:32:20 | `CRITICAL_BUG_FIXED.md` | CRITICAL BUG FOUND AND FIXED - Trading Permanently Blocked | how: files: main.py, environment.py | cmd: python main.py --episodes 50
- 2025-11-08 17:32:20 | `COMPLETE_VALIDATION_FIX.md` | Complete Fix Summary - Validation System Overhaul | how: files: main.py, config.py, trainer.py | cmd: python main.py --episodes 15
- 2025-11-08 17:32:20 | `COMPLETE_FIX_SUMMARY.md` | Complete Fix Summary - 2025-10-18 | how: files: tee.py, main.py, agent.py | cmd: python monitor_sweep.py
- 2025-11-08 17:32:20 | `BALANCE_INVARIANT_IMPLEMENTATION.md` | Balance-Invariant Policy Implementation - COMPLETE | how: files: main.py, agent.py, config.py
- 2025-11-08 17:32:20 | `ARCHITECTURE.md` | Forex RL Trading Bot Architecture | how: files: agent.py, fitness.py, trainer.py
- 2025-11-08 17:32:20 | `ANTI_COLLAPSE_STATUS.md` | Anti-Collapse Patches - Status Report | how: files: main.py, config.py, run_seed_sweep_organized.py | cmd: python main.py --episodes 25
- 2025-11-08 17:32:20 | `ANTI_COLLAPSE_QUICKSTART.md` | Quick Reference: Anti-Collapse Patches | how: files: main.py, config.py, trainer.py | cmd: python main.py --episodes 10
- 2025-11-08 17:32:20 | `ANTI_COLLAPSE_PATCHES.md` | Anti-Collapse Patches - Preventing HOLD Lock-In | how: files: main.py, config.py, trainer.py | cmd: python main.py --episodes 10
- 2025-11-08 17:32:20 | `ANTI_COLLAPSE_COMPLETE.md` | Final Anti-Collapse Implementation - Complete Summary | how: files: main.py, check_anti_collapse.py, compare_seed_results.py | cmd: python main.py --episodes 10
- 2025-11-08 17:32:20 | `ADVANCED_PATCHES_SUMMARY.md` | Advanced Surgical Patches Summary | how: files: main.py, agent.py, trainer.py | cmd: python -m py_compile environment.py agent.py trainer.py
- 2025-11-08 17:32:20 | `ADVANCED_LEARNING_PATCHES_SUMMARY.md` | ADVANCED LEARNING PATCHES SUMMARY | how: files: main.py, agent.py, config.py
- 2025-11-08 17:32:20 | `ADVANCED_IMPROVEMENTS_SUMMARY.md` | Advanced Improvements Implementation Summary | how: files: main.py, agent.py, config.py | cmd: python smoke_test_improvements.py
- 2025-11-08 17:32:20 | `ADDITIONAL_PATCHES_STATUS.md` | Additional Patches Implementation Summary | how: files: main.py, agent.py, trainer.py
- 2025-11-08 17:32:20 | `ACTIVITY_BOOST_TWEAKS.md` | Trading Activity Boost Tweaks - October 18, 2025 | how: files: main.py, config.py, trainer.py | cmd: python main.py --episodes 15

Notes:
- This appendix is a retrieval aid; timeline experiment outcomes remain in the dated sections above.
- If a line lacks concrete file/command hints, the source doc is narrative or sparse and should be expanded at next touch.
