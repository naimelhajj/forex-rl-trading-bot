# Repository Guidelines

## Project Structure & Module Organization
- Core training stack lives in top-level modules: `agent.py`, `trainer.py`, `environment.py`, `features.py`, `fitness.py`, `risk_manager.py`, `data_loader.py`, and `main.py`.
- Tests are split between top-level scripts like `test_system.py` and focused unit checks in `tests/` (for example `tests/test_features.py`).
- Operational scripts and utilities are prefixed with `run_`, `monitor_`, `analyze_`, or `check_`.
- Outputs from runs are written to `logs/`, `checkpoints/`, `results/`, and experiment folders such as `seed_sweep_results/`.

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt`
- Train a short run: `python main.py --mode train --episodes 50` (creates `logs/` and `checkpoints/`).
- Evaluate a model: `python main.py --mode evaluate --checkpoint checkpoints/best_model.pt`
- Smoke check: `python test_system.py` (fast end-to-end sanity test).
- Focused unit test example: `python tests/test_features.py`

## Coding Style & Naming Conventions
- Python 3.11+, 4-space indentation, and PEP8-style naming (snake_case functions, CapWords classes, UPPER_CASE constants).
- Keep configuration in `config.py` and pass overrides through CLI flags in `main.py` instead of hard-coding.
- Prefer ASCII-only console output to avoid Windows terminal encoding issues.
- New scripts should follow the existing naming patterns: `run_*.py`, `monitor_*.py`, `analyze_*.py`, `check_*.py`.

## Testing Guidelines
- Tests are plain Python scripts using `assert`, runnable directly with `python <file>`.
- Name new tests `test_*.py` and keep them either in `tests/` (unit/feature checks) or the repo root (system/integration checks).
- When touching training logic, run `python test_system.py` and at least one feature or environment test.

## Commit & Pull Request Guidelines
- Recent commit messages use short, descriptive phrases with emphasis tags like `CRITICAL FIX:`, `BREAKTHROUGH:`, or `RETEST PASSED:`; follow that tone when relevant.
- Keep commits scoped and summarize the impact in the subject line (example: `Add testing tools for batch augmentation validation`).
- PRs should include a brief summary, commands run, and any key metrics or logs produced (attach relevant `logs/` or `results/` references).

## Configuration & Data Notes
- Hyperparameters and paths are centralized in `config.py`.
- Check `EXAMPLES.md` for data loading formats and recommended CSV schemas before changing data pipelines.

## Project Goals & Trading Caution
- Primary goal: build a strategy that is profitable in real-world trading, not just in-sample, and demonstrably robust to overfitting under realistic costs and slippage.
- Treat all results as research evidence; validate across multiple seeds and out-of-sample regimes before any live usage.
- When choosing next steps or analyses, prioritize the option that most directly advances profitability.

## Progress Tracking
- `PROJECT_PROGRESS.md` is the single source of truth for the current percentage progress toward the ultimate goal.
- Read `PROJECT_PROGRESS.md` before answering progress-status questions.
- When major results materially change project status, update `PROJECT_PROGRESS.md` in the same work so the percentage stays current.
- Keep `PROJECT_PROGRESS.md` concise and evidence-based; use `PROJECT_HISTORY.md` for the detailed chronology.

## Sweep Operations
- Run real-data sweeps in the foreground and per-seed (one seed at a time) to avoid background termination.
- Always capture a log file in `logs/` for each seed run (use `-u` and `Tee-Object`).
- After all seeds finish, write an aggregate summary JSON under `seed_sweep_results/realdata/`.
- If a sweep is interrupted, resume with the remaining seeds and still produce an aggregate summary from the completed seeds.

## Execution Authority
- You are the project manager. Decide the next steps and proceed without asking what to do.
- Do not ask the user what you should do next; pick the best next step toward profitability and inform them.
- Only ask the user when a required input is missing or an irreversible/destructive action needs explicit confirmation.
