"""
Real-data seed sweep runner.
Runs multiple seeds against CSV data and collects test metrics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_SEEDS = [456, 777, 789]
DEFAULT_EPISODES = 20
DEFAULT_PAIR_FILES = "pair_files_real.json"
DEFAULT_OUTPUT_ROOT = Path("seed_sweep_results/realdata")


@dataclass
class SweepConfig:
    seeds: List[int]
    episodes: int
    pair_files: str
    data_dir: str | None
    data_mode: str
    output_root: Path
    run_prefix: str
    csv_sep: str | None
    date_col: str | None
    time_col: str | None
    no_symmetry_loss: bool
    no_dual_controller: bool
    no_strengths: bool
    strengths_all: bool
    strengths_pair_only: bool
    trade_penalty: float | None
    flip_penalty: float | None
    min_atr_cost_ratio: float | None
    use_regime_filter: bool | None
    regime_min_vol_z: float | None
    regime_align_trend: bool | None
    regime_require_trending: bool | None
    cooldown_bars: int | None
    min_hold_bars: int | None
    max_trades_per_episode: int | None
    hold_tie_tau: float | None
    hold_break_after: int | None
    epsilon_start: float | None
    epsilon_end: float | None
    epsilon_decay: float | None
    trade_gate_margin: float | None
    trade_gate_z: float | None
    r_multiple_reward_weight: float | None
    r_multiple_reward_clip: float | None
    max_steps_per_episode: int | None
    episode_timeout_min: float | None
    progress_report_sec: int
    prefill_policy: str | None
    allow_actions: str | None
    atr_mult_sl: float | None
    tp_mult: float | None
    anti_regression_horizon_rescue_enabled: bool | None
    anti_regression_horizon_window_bars: int | None
    anti_regression_horizon_start_frac: float | None
    anti_regression_horizon_end_frac: float | None
    anti_regression_horizon_candidate_limit: int | None
    anti_regression_horizon_incumbent_return_max: float | None
    anti_regression_horizon_return_edge_min: float | None
    anti_regression_horizon_pf_edge_min: float | None
    anti_regression_horizon_challenger_base_return_max: float | None
    anti_regression_horizon_challenger_robust_return_min: float | None
    anti_regression_horizon_challenger_pf_min: float | None
    anti_regression_horizon_min_trades: float | None
    anti_regression_top_k: int | None
    anti_regression_selector_mode: str | None
    anti_regression_auto_rescue_enabled: bool | None
    anti_regression_rescue_winner_forward_return_max: float | None
    anti_regression_rescue_forward_return_edge_min: float | None
    anti_regression_rescue_forward_pf_edge_min: float | None
    anti_regression_rescue_challenger_base_return_max: float | None
    anti_regression_rescue_challenger_forward_pf_min: float | None
    anti_regression_alignment_probe_enabled: bool | None
    anti_regression_alignment_probe_top_k: int | None
    anti_regression_alignment_probe_window_bars: int | None
    anti_regression_alignment_probe_stride_frac: float | None
    anti_regression_alignment_probe_use_all_windows: bool | None
    anti_regression_alignment_probe_return_edge_min: float | None
    anti_regression_alignment_probe_pf_edge_min: float | None
    anti_regression_alignment_probe_min_trades: float | None
    anti_regression_alignment_probe_require_pass: bool | None


def build_run_dir(root: Path, prefix: str, timestamp: str, seed: int) -> Path:
    return root / f"{prefix}_{timestamp}_seed{seed}"


def infer_data_dir(pair_files: str, explicit_data_dir: str | None) -> str | None:
    if explicit_data_dir is not None:
        return explicit_data_dir
    path = Path(pair_files)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            mapping = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    for value in mapping.values():
        parent = Path(value).parent
        if parent != Path("."):
            return "."
    return None


def load_test_results(results_path: Path) -> Dict[str, Any]:
    if not results_path.exists():
        return {"status": "missing_results", "results_path": str(results_path)}
    try:
        with results_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["status"] = "ok"
        return data
    except json.JSONDecodeError:
        return {"status": "invalid_json", "results_path": str(results_path)}


def resolve_results_path(run_dir: Path) -> Path:
    primary = run_dir / "results" / "test_results.json"
    if primary.exists():
        return primary
    fallback = run_dir / "eval" / "results" / "test_results.json"
    return fallback


def read_last_event(event_path: Path) -> Dict[str, Any] | None:
    if not event_path.exists():
        return None
    try:
        with event_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192), 0)
            chunk = f.read().decode("utf-8", errors="ignore")
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        if not lines:
            return None
        return json.loads(lines[-1])
    except (OSError, json.JSONDecodeError):
        return None


def run_single_seed(seed: int, config: SweepConfig, timestamp: str) -> Dict[str, Any]:
    run_dir = build_run_dir(config.output_root, config.run_prefix, timestamp, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        "main.py",
        "--mode", "both",
        "--episodes", str(config.episodes),
        "--seed", str(seed),
        "--data-mode", config.data_mode,
        "--pair-files", config.pair_files,
        "--output-dir", str(run_dir),
    ]

    if config.data_dir:
        cmd.extend(["--data-dir", config.data_dir])
    if config.csv_sep is not None:
        cmd.extend(["--csv-sep", config.csv_sep])
    if config.date_col:
        cmd.extend(["--date-col", config.date_col])
    if config.time_col:
        cmd.extend(["--time-col", config.time_col])
    if config.no_symmetry_loss:
        cmd.append("--no-symmetry-loss")
    if config.no_dual_controller:
        cmd.append("--no-dual-controller")
    if config.no_strengths:
        cmd.append("--no-strengths")
    if config.strengths_all:
        cmd.append("--strengths-all")
    if config.strengths_pair_only:
        cmd.append("--strengths-pair-only")
    if config.trade_penalty is not None:
        cmd.extend(["--trade-penalty", str(config.trade_penalty)])
    if config.flip_penalty is not None:
        cmd.extend(["--flip-penalty", str(config.flip_penalty)])
    if config.min_atr_cost_ratio is not None:
        cmd.extend(["--min-atr-cost-ratio", str(config.min_atr_cost_ratio)])
    if config.use_regime_filter is True:
        cmd.append("--use-regime-filter")
    elif config.use_regime_filter is False:
        cmd.append("--no-regime-filter")
    if config.regime_min_vol_z is not None:
        cmd.extend(["--regime-min-vol-z", str(config.regime_min_vol_z)])
    if config.regime_align_trend is True:
        cmd.append("--regime-align-trend")
    elif config.regime_align_trend is False:
        cmd.append("--regime-no-align-trend")
    if config.regime_require_trending is True:
        cmd.append("--regime-require-trending")
    elif config.regime_require_trending is False:
        cmd.append("--regime-no-require-trending")
    if config.cooldown_bars is not None:
        cmd.extend(["--cooldown-bars", str(config.cooldown_bars)])
    if config.min_hold_bars is not None:
        cmd.extend(["--min-hold-bars", str(config.min_hold_bars)])
    if config.max_trades_per_episode is not None:
        cmd.extend(["--max-trades-per-episode", str(config.max_trades_per_episode)])
    if config.hold_tie_tau is not None:
        cmd.extend(["--hold-tie-tau", str(config.hold_tie_tau)])
    if config.hold_break_after is not None:
        cmd.extend(["--hold-break-after", str(config.hold_break_after)])
    if config.epsilon_start is not None:
        cmd.extend(["--epsilon-start", str(config.epsilon_start)])
    if config.epsilon_end is not None:
        cmd.extend(["--epsilon-end", str(config.epsilon_end)])
    if config.epsilon_decay is not None:
        cmd.extend(["--epsilon-decay", str(config.epsilon_decay)])
    if config.trade_gate_margin is not None:
        cmd.extend(["--trade-gate-margin", str(config.trade_gate_margin)])
    if config.trade_gate_z is not None:
        cmd.extend(["--trade-gate-z", str(config.trade_gate_z)])
    if config.r_multiple_reward_weight is not None:
        cmd.extend(["--r-multiple-reward-weight", str(config.r_multiple_reward_weight)])
    if config.r_multiple_reward_clip is not None:
        cmd.extend(["--r-multiple-reward-clip", str(config.r_multiple_reward_clip)])
    if config.max_steps_per_episode is not None:
        cmd.extend(["--max-steps-per-episode", str(config.max_steps_per_episode)])
    if config.episode_timeout_min is not None:
        cmd.extend(["--episode-timeout-min", str(config.episode_timeout_min)])
    if config.prefill_policy is not None:
        cmd.extend(["--prefill-policy", str(config.prefill_policy)])
    if config.allow_actions is not None:
        cmd.extend(["--allow-actions", str(config.allow_actions)])
    if config.atr_mult_sl is not None:
        cmd.extend(["--atr-mult-sl", str(config.atr_mult_sl)])
    if config.tp_mult is not None:
        cmd.extend(["--tp-mult", str(config.tp_mult)])
    if config.anti_regression_top_k is not None:
        cmd.extend(["--anti-regression-top-k", str(config.anti_regression_top_k)])
    if config.anti_regression_selector_mode is not None:
        cmd.extend(["--anti-regression-selector-mode", str(config.anti_regression_selector_mode)])
    if config.anti_regression_auto_rescue_enabled is True:
        cmd.append("--anti-regression-auto-rescue")
    elif config.anti_regression_auto_rescue_enabled is False:
        cmd.append("--anti-regression-no-auto-rescue")
    if config.anti_regression_rescue_winner_forward_return_max is not None:
        cmd.extend(["--anti-regression-rescue-winner-forward-return-max", str(config.anti_regression_rescue_winner_forward_return_max)])
    if config.anti_regression_rescue_forward_return_edge_min is not None:
        cmd.extend(["--anti-regression-rescue-forward-return-edge-min", str(config.anti_regression_rescue_forward_return_edge_min)])
    if config.anti_regression_rescue_forward_pf_edge_min is not None:
        cmd.extend(["--anti-regression-rescue-forward-pf-edge-min", str(config.anti_regression_rescue_forward_pf_edge_min)])
    if config.anti_regression_rescue_challenger_base_return_max is not None:
        cmd.extend(["--anti-regression-rescue-challenger-base-return-max", str(config.anti_regression_rescue_challenger_base_return_max)])
    if config.anti_regression_rescue_challenger_forward_pf_min is not None:
        cmd.extend(["--anti-regression-rescue-challenger-forward-pf-min", str(config.anti_regression_rescue_challenger_forward_pf_min)])
    if config.anti_regression_horizon_rescue_enabled is True:
        cmd.append("--anti-regression-horizon-rescue")
    elif config.anti_regression_horizon_rescue_enabled is False:
        cmd.append("--anti-regression-no-horizon-rescue")
    if config.anti_regression_horizon_window_bars is not None:
        cmd.extend(["--anti-regression-horizon-window-bars", str(config.anti_regression_horizon_window_bars)])
    if config.anti_regression_horizon_start_frac is not None:
        cmd.extend(["--anti-regression-horizon-start-frac", str(config.anti_regression_horizon_start_frac)])
    if config.anti_regression_horizon_end_frac is not None:
        cmd.extend(["--anti-regression-horizon-end-frac", str(config.anti_regression_horizon_end_frac)])
    if config.anti_regression_horizon_candidate_limit is not None:
        cmd.extend(["--anti-regression-horizon-candidate-limit", str(config.anti_regression_horizon_candidate_limit)])
    if config.anti_regression_horizon_incumbent_return_max is not None:
        cmd.extend(["--anti-regression-horizon-incumbent-return-max", str(config.anti_regression_horizon_incumbent_return_max)])
    if config.anti_regression_horizon_return_edge_min is not None:
        cmd.extend(["--anti-regression-horizon-return-edge-min", str(config.anti_regression_horizon_return_edge_min)])
    if config.anti_regression_horizon_pf_edge_min is not None:
        cmd.extend(["--anti-regression-horizon-pf-edge-min", str(config.anti_regression_horizon_pf_edge_min)])
    if config.anti_regression_horizon_challenger_base_return_max is not None:
        cmd.extend(["--anti-regression-horizon-challenger-base-return-max", str(config.anti_regression_horizon_challenger_base_return_max)])
    if config.anti_regression_horizon_challenger_robust_return_min is not None:
        cmd.extend(["--anti-regression-horizon-challenger-robust-return-min", str(config.anti_regression_horizon_challenger_robust_return_min)])
    if config.anti_regression_horizon_challenger_pf_min is not None:
        cmd.extend(["--anti-regression-horizon-challenger-pf-min", str(config.anti_regression_horizon_challenger_pf_min)])
    if config.anti_regression_horizon_min_trades is not None:
        cmd.extend(["--anti-regression-horizon-min-trades", str(config.anti_regression_horizon_min_trades)])
    if config.anti_regression_alignment_probe_enabled is True:
        cmd.append("--anti-regression-alignment-probe")
    elif config.anti_regression_alignment_probe_enabled is False:
        cmd.append("--anti-regression-no-alignment-probe")
    if config.anti_regression_alignment_probe_top_k is not None:
        cmd.extend(["--anti-regression-alignment-probe-top-k", str(config.anti_regression_alignment_probe_top_k)])
    if config.anti_regression_alignment_probe_window_bars is not None:
        cmd.extend(["--anti-regression-alignment-probe-window-bars", str(config.anti_regression_alignment_probe_window_bars)])
    if config.anti_regression_alignment_probe_stride_frac is not None:
        cmd.extend(["--anti-regression-alignment-probe-stride-frac", str(config.anti_regression_alignment_probe_stride_frac)])
    if config.anti_regression_alignment_probe_use_all_windows is True:
        cmd.append("--anti-regression-alignment-probe-all-windows")
    elif config.anti_regression_alignment_probe_use_all_windows is False:
        cmd.append("--anti-regression-alignment-probe-even-windows")
    if config.anti_regression_alignment_probe_return_edge_min is not None:
        cmd.extend(["--anti-regression-alignment-probe-return-edge-min", str(config.anti_regression_alignment_probe_return_edge_min)])
    if config.anti_regression_alignment_probe_pf_edge_min is not None:
        cmd.extend(["--anti-regression-alignment-probe-pf-edge-min", str(config.anti_regression_alignment_probe_pf_edge_min)])
    if config.anti_regression_alignment_probe_min_trades is not None:
        cmd.extend(["--anti-regression-alignment-probe-min-trades", str(config.anti_regression_alignment_probe_min_trades)])
    if config.anti_regression_alignment_probe_require_pass is True:
        cmd.append("--anti-regression-alignment-probe-require-pass")
    elif config.anti_regression_alignment_probe_require_pass is False:
        cmd.append("--anti-regression-alignment-probe-allow-no-pass")

    print("=" * 70)
    print(f"SEED {seed} | episodes={config.episodes} | output={run_dir}")
    print("CMD:", " ".join(cmd))
    print("=" * 70)

    started_at = time.monotonic()
    next_report_at = started_at + max(5, int(config.progress_report_sec))
    event_path = run_dir / "logs" / "episode_events.jsonl"
    proc = subprocess.Popen(cmd)
    while proc.poll() is None:
        now = time.monotonic()
        if now >= next_report_at:
            elapsed_min = (now - started_at) / 60.0
            msg = f"[PROGRESS] seed={seed} elapsed={elapsed_min:.1f}m"
            if event_path.exists():
                age_sec = max(0.0, time.time() - event_path.stat().st_mtime)
                event = read_last_event(event_path)
                if event:
                    et = event.get("event_type", "?")
                    ep = event.get("episode", "?")
                    msg += f" | last_event={et} ep={ep} age={age_sec:.0f}s"
                    if et == "episode_end":
                        steps = event.get("steps")
                        eq = event.get("final_equity")
                        trades = event.get("trades", event.get("total_trades"))
                        msg += f" steps={steps} eq={eq:.2f} trades={trades}" if isinstance(eq, (int, float)) else f" steps={steps} trades={trades}"
                    elif et == "validation":
                        fit = event.get("fitness")
                        trades = event.get("total_trades")
                        if fit is not None:
                            msg += f" fitness={fit:.4f}"
                        if trades is not None:
                            msg += f" val_trades={trades}"
                else:
                    msg += " | waiting for first episode event"
            else:
                msg += " | waiting for logs/episode_events.jsonl"
            print(msg, flush=True)
            next_report_at = now + max(5, int(config.progress_report_sec))
        time.sleep(2)

    result = proc
    results_path = resolve_results_path(run_dir)
    metrics = load_test_results(results_path)

    elapsed_min = (time.monotonic() - started_at) / 60.0
    print(f"[SEED DONE] seed={seed} exit={result.returncode} elapsed={elapsed_min:.1f}m", flush=True)

    return {
        "seed": seed,
        "exit_code": result.returncode,
        "run_dir": str(run_dir),
        "results": metrics,
    }


def summarize_results(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    returns = []
    profits = []
    fitness = []
    for entry in entries:
        data = entry.get("results", {})
        if data.get("status") != "ok":
            continue
        if "test_return" in data:
            returns.append(data["test_return"])
        if "test_profit_factor" in data:
            profits.append(data["test_profit_factor"])
        if "test_fitness" in data:
            fitness.append(data["test_fitness"])

    def mean_std(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0, "count": 0}
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        return {"mean": mean_val, "std": variance ** 0.5, "count": len(values)}

    return {
        "return_pct": mean_std(returns),
        "profit_factor": mean_std(profits),
        "fitness": mean_std(fitness),
    }


def parse_args() -> SweepConfig:
    parser = argparse.ArgumentParser(description="Run a real-data seed sweep.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--pair-files", type=str, default=DEFAULT_PAIR_FILES)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--data-mode", type=str, default="csv")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-prefix", type=str, default="realdata_sweep")
    parser.add_argument("--csv-sep", type=str, default=None)
    parser.add_argument("--date-col", type=str, default=None)
    parser.add_argument("--time-col", type=str, default=None)
    parser.add_argument("--no-symmetry-loss", action="store_true")
    parser.add_argument("--no-dual-controller", action="store_true")
    parser.add_argument("--no-strengths", action="store_true")
    strengths_group = parser.add_mutually_exclusive_group()
    strengths_group.add_argument("--strengths-all", action="store_true")
    strengths_group.add_argument("--strengths-pair-only", action="store_true")
    parser.add_argument("--trade-penalty", type=float, default=None)
    parser.add_argument("--flip-penalty", type=float, default=None)
    parser.add_argument("--min-atr-cost-ratio", type=float, default=None)
    parser.add_argument("--use-regime-filter", dest="use_regime_filter", action="store_true")
    parser.add_argument("--no-regime-filter", dest="use_regime_filter", action="store_false")
    parser.set_defaults(use_regime_filter=None)
    parser.add_argument("--regime-min-vol-z", type=float, default=None)
    parser.add_argument("--regime-align-trend", dest="regime_align_trend", action="store_true")
    parser.add_argument("--regime-no-align-trend", dest="regime_align_trend", action="store_false")
    parser.set_defaults(regime_align_trend=None)
    parser.add_argument("--regime-require-trending", dest="regime_require_trending", action="store_true")
    parser.add_argument("--regime-no-require-trending", dest="regime_require_trending", action="store_false")
    parser.set_defaults(regime_require_trending=None)
    parser.add_argument("--cooldown-bars", type=int, default=None)
    parser.add_argument("--min-hold-bars", type=int, default=None)
    parser.add_argument("--max-trades-per-episode", type=int, default=None)
    parser.add_argument("--hold-tie-tau", type=float, default=None)
    parser.add_argument("--hold-break-after", type=int, default=None)
    parser.add_argument("--epsilon-start", type=float, default=None)
    parser.add_argument("--epsilon-end", type=float, default=None)
    parser.add_argument("--epsilon-decay", type=float, default=None)
    parser.add_argument("--trade-gate-margin", type=float, default=None)
    parser.add_argument("--trade-gate-z", type=float, default=None)
    parser.add_argument("--r-multiple-reward-weight", type=float, default=None)
    parser.add_argument("--r-multiple-reward-clip", type=float, default=None)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--episode-timeout-min", type=float, default=None)
    parser.add_argument("--progress-report-sec", type=int, default=30,
                        help="Print progress heartbeat every N seconds while a seed is running")
    parser.add_argument("--prefill-policy", type=str, default=None,
                        choices=["baseline", "random", "none"])
    parser.add_argument("--allow-actions", type=str, default=None,
                        help="Comma-separated actions: hold,long,short,move_sl or 0-3")
    parser.add_argument("--atr-mult-sl", type=float, default=None)
    parser.add_argument("--tp-mult", type=float, default=None)
    parser.add_argument("--anti-regression-top-k", type=int, default=None)
    parser.add_argument("--anti-regression-selector-mode", type=str, default=None,
                        choices=["tail_holdout", "future_first", "auto_rescue", "base_first"])
    parser.add_argument("--anti-regression-auto-rescue", dest="anti_regression_auto_rescue_enabled", action="store_true")
    parser.add_argument("--anti-regression-no-auto-rescue", dest="anti_regression_auto_rescue_enabled", action="store_false")
    parser.set_defaults(anti_regression_auto_rescue_enabled=None)
    parser.add_argument("--anti-regression-rescue-winner-forward-return-max", type=float, default=None)
    parser.add_argument("--anti-regression-rescue-forward-return-edge-min", type=float, default=None)
    parser.add_argument("--anti-regression-rescue-forward-pf-edge-min", type=float, default=None)
    parser.add_argument("--anti-regression-rescue-challenger-base-return-max", type=float, default=None)
    parser.add_argument("--anti-regression-rescue-challenger-forward-pf-min", type=float, default=None)
    parser.add_argument("--anti-regression-horizon-rescue", dest="anti_regression_horizon_rescue_enabled", action="store_true")
    parser.add_argument("--anti-regression-no-horizon-rescue", dest="anti_regression_horizon_rescue_enabled", action="store_false")
    parser.set_defaults(anti_regression_horizon_rescue_enabled=None)
    parser.add_argument("--anti-regression-horizon-window-bars", type=int, default=None)
    parser.add_argument("--anti-regression-horizon-start-frac", type=float, default=None)
    parser.add_argument("--anti-regression-horizon-end-frac", type=float, default=None)
    parser.add_argument("--anti-regression-horizon-candidate-limit", type=int, default=None)
    parser.add_argument("--anti-regression-horizon-incumbent-return-max", type=float, default=None)
    parser.add_argument("--anti-regression-horizon-return-edge-min", type=float, default=None)
    parser.add_argument("--anti-regression-horizon-pf-edge-min", type=float, default=None)
    parser.add_argument("--anti-regression-horizon-challenger-base-return-max", type=float, default=None)
    parser.add_argument("--anti-regression-horizon-challenger-robust-return-min", type=float, default=None)
    parser.add_argument("--anti-regression-horizon-challenger-pf-min", type=float, default=None)
    parser.add_argument("--anti-regression-horizon-min-trades", type=float, default=None)
    parser.add_argument("--anti-regression-alignment-probe", dest="anti_regression_alignment_probe_enabled", action="store_true")
    parser.add_argument("--anti-regression-no-alignment-probe", dest="anti_regression_alignment_probe_enabled", action="store_false")
    parser.set_defaults(anti_regression_alignment_probe_enabled=None)
    parser.add_argument("--anti-regression-alignment-probe-top-k", type=int, default=None)
    parser.add_argument("--anti-regression-alignment-probe-window-bars", type=int, default=None)
    parser.add_argument("--anti-regression-alignment-probe-stride-frac", type=float, default=None)
    parser.add_argument("--anti-regression-alignment-probe-all-windows", dest="anti_regression_alignment_probe_use_all_windows", action="store_true")
    parser.add_argument("--anti-regression-alignment-probe-even-windows", dest="anti_regression_alignment_probe_use_all_windows", action="store_false")
    parser.set_defaults(anti_regression_alignment_probe_use_all_windows=None)
    parser.add_argument("--anti-regression-alignment-probe-return-edge-min", type=float, default=None)
    parser.add_argument("--anti-regression-alignment-probe-pf-edge-min", type=float, default=None)
    parser.add_argument("--anti-regression-alignment-probe-min-trades", type=float, default=None)
    parser.add_argument("--anti-regression-alignment-probe-require-pass", dest="anti_regression_alignment_probe_require_pass", action="store_true")
    parser.add_argument("--anti-regression-alignment-probe-allow-no-pass", dest="anti_regression_alignment_probe_require_pass", action="store_false")
    parser.set_defaults(anti_regression_alignment_probe_require_pass=None)

    args = parser.parse_args()

    return SweepConfig(
        seeds=args.seeds,
        episodes=args.episodes,
        pair_files=args.pair_files,
        data_dir=args.data_dir,
        data_mode=args.data_mode,
        output_root=Path(args.output_root),
        run_prefix=args.run_prefix,
        csv_sep=args.csv_sep,
        date_col=args.date_col,
        time_col=args.time_col,
        no_symmetry_loss=args.no_symmetry_loss,
        no_dual_controller=args.no_dual_controller,
        no_strengths=args.no_strengths,
        strengths_all=args.strengths_all,
        strengths_pair_only=args.strengths_pair_only,
        trade_penalty=args.trade_penalty,
        flip_penalty=args.flip_penalty,
        min_atr_cost_ratio=args.min_atr_cost_ratio,
        use_regime_filter=args.use_regime_filter,
        regime_min_vol_z=args.regime_min_vol_z,
        regime_align_trend=args.regime_align_trend,
        regime_require_trending=args.regime_require_trending,
        cooldown_bars=args.cooldown_bars,
        min_hold_bars=args.min_hold_bars,
        max_trades_per_episode=args.max_trades_per_episode,
        hold_tie_tau=args.hold_tie_tau,
        hold_break_after=args.hold_break_after,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        trade_gate_margin=args.trade_gate_margin,
        trade_gate_z=args.trade_gate_z,
        r_multiple_reward_weight=args.r_multiple_reward_weight,
        r_multiple_reward_clip=args.r_multiple_reward_clip,
        max_steps_per_episode=args.max_steps_per_episode,
        episode_timeout_min=args.episode_timeout_min,
        progress_report_sec=max(5, int(args.progress_report_sec)),
        prefill_policy=args.prefill_policy,
        allow_actions=args.allow_actions,
        atr_mult_sl=args.atr_mult_sl,
        tp_mult=args.tp_mult,
        anti_regression_top_k=args.anti_regression_top_k,
        anti_regression_selector_mode=args.anti_regression_selector_mode,
        anti_regression_auto_rescue_enabled=args.anti_regression_auto_rescue_enabled,
        anti_regression_rescue_winner_forward_return_max=args.anti_regression_rescue_winner_forward_return_max,
        anti_regression_rescue_forward_return_edge_min=args.anti_regression_rescue_forward_return_edge_min,
        anti_regression_rescue_forward_pf_edge_min=args.anti_regression_rescue_forward_pf_edge_min,
        anti_regression_rescue_challenger_base_return_max=args.anti_regression_rescue_challenger_base_return_max,
        anti_regression_rescue_challenger_forward_pf_min=args.anti_regression_rescue_challenger_forward_pf_min,
        anti_regression_horizon_rescue_enabled=args.anti_regression_horizon_rescue_enabled,
        anti_regression_horizon_window_bars=args.anti_regression_horizon_window_bars,
        anti_regression_horizon_start_frac=args.anti_regression_horizon_start_frac,
        anti_regression_horizon_end_frac=args.anti_regression_horizon_end_frac,
        anti_regression_horizon_candidate_limit=args.anti_regression_horizon_candidate_limit,
        anti_regression_horizon_incumbent_return_max=args.anti_regression_horizon_incumbent_return_max,
        anti_regression_horizon_return_edge_min=args.anti_regression_horizon_return_edge_min,
        anti_regression_horizon_pf_edge_min=args.anti_regression_horizon_pf_edge_min,
        anti_regression_horizon_challenger_base_return_max=args.anti_regression_horizon_challenger_base_return_max,
        anti_regression_horizon_challenger_robust_return_min=args.anti_regression_horizon_challenger_robust_return_min,
        anti_regression_horizon_challenger_pf_min=args.anti_regression_horizon_challenger_pf_min,
        anti_regression_horizon_min_trades=args.anti_regression_horizon_min_trades,
        anti_regression_alignment_probe_enabled=args.anti_regression_alignment_probe_enabled,
        anti_regression_alignment_probe_top_k=args.anti_regression_alignment_probe_top_k,
        anti_regression_alignment_probe_window_bars=args.anti_regression_alignment_probe_window_bars,
        anti_regression_alignment_probe_stride_frac=args.anti_regression_alignment_probe_stride_frac,
        anti_regression_alignment_probe_use_all_windows=args.anti_regression_alignment_probe_use_all_windows,
        anti_regression_alignment_probe_return_edge_min=args.anti_regression_alignment_probe_return_edge_min,
        anti_regression_alignment_probe_pf_edge_min=args.anti_regression_alignment_probe_pf_edge_min,
        anti_regression_alignment_probe_min_trades=args.anti_regression_alignment_probe_min_trades,
        anti_regression_alignment_probe_require_pass=args.anti_regression_alignment_probe_require_pass,
    )


def main() -> int:
    config = parse_args()
    inferred_data_dir = infer_data_dir(config.pair_files, config.data_dir)
    if inferred_data_dir is not None:
        config.data_dir = inferred_data_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config.output_root.mkdir(parents=True, exist_ok=True)

    print("REAL-DATA SEED SWEEP")
    print(f"Seeds: {config.seeds}")
    print(f"Episodes: {config.episodes}")
    print(f"Pair files: {config.pair_files}")
    print(f"Data dir: {config.data_dir}")
    print(f"Output root: {config.output_root}")
    print(f"No symmetry loss: {config.no_symmetry_loss}")
    print(f"No dual controller: {config.no_dual_controller}")
    print(f"No strengths: {config.no_strengths}")
    print(f"Strengths all: {config.strengths_all}")
    print(f"Strengths pair-only: {config.strengths_pair_only}")
    print(f"Trade penalty: {config.trade_penalty}")
    print(f"Flip penalty: {config.flip_penalty}")
    print(f"Min ATR cost ratio: {config.min_atr_cost_ratio}")
    print(f"Use regime filter: {config.use_regime_filter}")
    print(f"Regime min vol z: {config.regime_min_vol_z}")
    print(f"Regime align trend: {config.regime_align_trend}")
    print(f"Regime require trending: {config.regime_require_trending}")
    print(f"Cooldown bars: {config.cooldown_bars}")
    print(f"Min hold bars: {config.min_hold_bars}")
    print(f"Max trades/episode: {config.max_trades_per_episode}")
    print(f"Hold tie tau: {config.hold_tie_tau}")
    print(f"Hold break after: {config.hold_break_after}")
    print(f"Trade gate margin: {config.trade_gate_margin}")
    print(f"Trade gate z: {config.trade_gate_z}")
    print(f"R-multiple reward weight: {config.r_multiple_reward_weight}")
    print(f"R-multiple reward clip: {config.r_multiple_reward_clip}")
    print(f"Selector top-k: {config.anti_regression_top_k}")
    print(f"Selector mode: {config.anti_regression_selector_mode}")
    print(f"Auto rescue enabled: {config.anti_regression_auto_rescue_enabled}")
    print(f"Horizon rescue enabled: {config.anti_regression_horizon_rescue_enabled}")
    print(f"Horizon window bars: {config.anti_regression_horizon_window_bars}")
    print(f"Horizon challenger robust return min: {config.anti_regression_horizon_challenger_robust_return_min}")
    print(f"Alignment probe enabled: {config.anti_regression_alignment_probe_enabled}")
    print(f"Alignment probe top-k: {config.anti_regression_alignment_probe_top_k}")
    print()

    results = []
    for seed in config.seeds:
        results.append(run_single_seed(seed, config, timestamp))

    summary = {
        "timestamp": timestamp,
        "config": {
            "seeds": config.seeds,
            "episodes": config.episodes,
            "pair_files": config.pair_files,
            "data_dir": config.data_dir,
            "data_mode": config.data_mode,
            "output_root": str(config.output_root),
            "run_prefix": config.run_prefix,
            "csv_sep": config.csv_sep,
            "date_col": config.date_col,
            "time_col": config.time_col,
            "no_symmetry_loss": config.no_symmetry_loss,
            "no_dual_controller": config.no_dual_controller,
            "no_strengths": config.no_strengths,
            "strengths_all": config.strengths_all,
            "strengths_pair_only": config.strengths_pair_only,
            "trade_penalty": config.trade_penalty,
            "flip_penalty": config.flip_penalty,
            "min_atr_cost_ratio": config.min_atr_cost_ratio,
            "use_regime_filter": config.use_regime_filter,
            "regime_min_vol_z": config.regime_min_vol_z,
            "regime_align_trend": config.regime_align_trend,
            "regime_require_trending": config.regime_require_trending,
            "cooldown_bars": config.cooldown_bars,
            "min_hold_bars": config.min_hold_bars,
            "max_trades_per_episode": config.max_trades_per_episode,
            "hold_tie_tau": config.hold_tie_tau,
            "hold_break_after": config.hold_break_after,
            "trade_gate_margin": config.trade_gate_margin,
            "trade_gate_z": config.trade_gate_z,
            "r_multiple_reward_weight": config.r_multiple_reward_weight,
            "r_multiple_reward_clip": config.r_multiple_reward_clip,
            "anti_regression_top_k": config.anti_regression_top_k,
            "anti_regression_selector_mode": config.anti_regression_selector_mode,
            "anti_regression_auto_rescue_enabled": config.anti_regression_auto_rescue_enabled,
            "anti_regression_rescue_winner_forward_return_max": config.anti_regression_rescue_winner_forward_return_max,
            "anti_regression_rescue_forward_return_edge_min": config.anti_regression_rescue_forward_return_edge_min,
            "anti_regression_rescue_forward_pf_edge_min": config.anti_regression_rescue_forward_pf_edge_min,
            "anti_regression_rescue_challenger_base_return_max": config.anti_regression_rescue_challenger_base_return_max,
            "anti_regression_rescue_challenger_forward_pf_min": config.anti_regression_rescue_challenger_forward_pf_min,
            "anti_regression_horizon_rescue_enabled": config.anti_regression_horizon_rescue_enabled,
            "anti_regression_horizon_window_bars": config.anti_regression_horizon_window_bars,
            "anti_regression_horizon_start_frac": config.anti_regression_horizon_start_frac,
            "anti_regression_horizon_end_frac": config.anti_regression_horizon_end_frac,
            "anti_regression_horizon_candidate_limit": config.anti_regression_horizon_candidate_limit,
            "anti_regression_horizon_incumbent_return_max": config.anti_regression_horizon_incumbent_return_max,
            "anti_regression_horizon_return_edge_min": config.anti_regression_horizon_return_edge_min,
            "anti_regression_horizon_pf_edge_min": config.anti_regression_horizon_pf_edge_min,
            "anti_regression_horizon_challenger_base_return_max": config.anti_regression_horizon_challenger_base_return_max,
            "anti_regression_horizon_challenger_robust_return_min": config.anti_regression_horizon_challenger_robust_return_min,
            "anti_regression_horizon_challenger_pf_min": config.anti_regression_horizon_challenger_pf_min,
            "anti_regression_horizon_min_trades": config.anti_regression_horizon_min_trades,
            "anti_regression_alignment_probe_enabled": config.anti_regression_alignment_probe_enabled,
            "anti_regression_alignment_probe_top_k": config.anti_regression_alignment_probe_top_k,
            "anti_regression_alignment_probe_window_bars": config.anti_regression_alignment_probe_window_bars,
            "anti_regression_alignment_probe_stride_frac": config.anti_regression_alignment_probe_stride_frac,
            "anti_regression_alignment_probe_use_all_windows": config.anti_regression_alignment_probe_use_all_windows,
            "anti_regression_alignment_probe_return_edge_min": config.anti_regression_alignment_probe_return_edge_min,
            "anti_regression_alignment_probe_pf_edge_min": config.anti_regression_alignment_probe_pf_edge_min,
            "anti_regression_alignment_probe_min_trades": config.anti_regression_alignment_probe_min_trades,
            "anti_regression_alignment_probe_require_pass": config.anti_regression_alignment_probe_require_pass,
        },
        "runs": results,
        "aggregate": summarize_results(results),
    }

    summary_path = config.output_root / f"{config.run_prefix}_{timestamp}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("SUMMARY")
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"Saved: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
