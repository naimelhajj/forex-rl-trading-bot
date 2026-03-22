# Project Progress

This file is the single source of truth for the project's progress percentage toward the ultimate goal.

Ultimate goal:
- Build a forex RL trading system that is profitable on real data, robust to overfitting under realistic costs/slippage, and credible enough for later paper/live shadow validation.

Important:
- This percentage is not code completion.
- This percentage is an evidence-based estimate of how far the project has progressed toward a real, robust trading outcome.

Last updated: 2026-03-22

Current progress: 69%

Why 69% now:
- The core system works end-to-end on real data.
- The current branch now has selected-checkpoint walk-forward passes on nine real-data seeds: `123`, `777`, `1011`, `2027`, `3141`, `5051`, `8087`, `9091`, and `10007`.
- The branch is no longer in the "does this architecture learn at all?" phase.
- `10007` is now automatically recovered by the selector patch in a full rerun.
- `9091` is now also automatically recovered in a full rerun via the validation-MMR selector patch (`candidate_ep001.pt`), so the known weak-seed triad now closes on selected checkpoints rather than manual alternates.
- `777` now passes on the selected checkpoint in a clean rerun after tightening temporal bias, so the branch has its first fresh unseen-seed selected-pass proof beyond the prior confirmed set.
- `3141` now also passes on the selected checkpoint in a fresh unseen-seed run, so the selector branch is holding beyond the earlier validation set.
- The project is near the top of the "hard seeds mostly solved" band, but broader unseen-seed and regime validation still has not been done.

Current status snapshot:
- Confirmed real-data pass set:
  - `seed_sweep_results/realdata/seed123_baseline17_selectorfix_slack_nodual_20260316_155223_seed123/results/test_results.json`
  - `seed_sweep_results/realdata/seed777_baseline17_selectorfix_slack_nodual_temporalguard_20260321_133004_seed777/results/test_results.json`
  - `seed_sweep_results/realdata/seed1011_baseline17_selectorfix_slack_nodual_20260315_171432_seed1011/results/test_results.json`
  - `seed_sweep_results/realdata/seed2027_baseline17_selectorfix_slack_nodual_20260316_123918_seed2027/results/test_results.json`
  - `seed_sweep_results/realdata/seed3141_baseline17_selectorfix_slack_nodual_temporalguard_20260321_173233_seed3141/results/test_results.json`
  - `seed_sweep_results/realdata/seed8087_baseline17_selectorfix_slack_nodual_20260315_120432_seed8087/results/test_results.json`
  - `seed_sweep_results/realdata/seed5051_baseline17_selectorfix_slack_nodual_20260317_153038_seed5051/results/test_results.json`
  - `seed_sweep_results/realdata/seed10007_baseline17_selectorfix_slack_nodual_temporal3_20260318_234559_seed10007/results/test_results.json`
  - `seed_sweep_results/realdata/seed9091_baseline17_selectorfix_slack_nodual_valmmr_20260320_133958_seed9091/results/test_results.json`
- Aggregate summary:
  - `seed_sweep_results/realdata/baseline17_selectorfix_slack_nodual_selected9_summary_20260322.json`

Milestone scale:
- 0-20%: pipeline exists but no reliable learning evidence
- 20-40%: learns in controlled/synthetic settings, but real-data robustness is weak
- 40-60%: viable real-data branch exists and passes on multiple seeds
- 60-75%: hard-seed behavior is mostly solved and selector behavior is reliable
- 75-90%: broader out-of-sample/regime validation and deployment-style checks are passing
- 90-100%: paper/live shadow evidence supports moving toward real execution

What would move this upward:
- Broader real-data validation beyond the current confirmed set
- Time-slice/regime validation that preserves the current `9/9` selected-pass evidence
- Deployment-style validation such as paper/live shadow testing

What would move this downward:
- Regressions on the confirmed pass set
- New evidence that current walk-forward passes are not stable across additional seeds/regimes
- Discovery of evaluation or selector bugs that invalidate the current evidence base

Update rules for agents:
- Update this file whenever a material milestone changes the real probability of reaching the ultimate goal.
- Update this file whenever the project percentage should move by about 2 points or more.
- Every percentage change must include a short reason grounded in evidence and should reference concrete result files.
- Keep `PROJECT_HISTORY.md` as the detailed log; keep this file concise and current.
