"""
Compare Phase 2.8c Run C v1 (stabilization tweaks) results.
Focus on jitter-averaged validation with trail-5 median finals.

PHASE-2.8c: Stabilization tweaks
- Jitter-averaged validation (K=3 draws per window)
- Tighter gating (VAL_EXP_TRADES_SCALE = 0.38)
- Enhanced reporting (trail-5 median for finals)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List

# Target seeds for Run C v1
RUN_C_V1_SEEDS = [7, 77, 777]  # Quick 3-seed test

def load_seed_jsons(seed: int, max_episodes: int = 150) -> List[Dict]:
    """Load validation JSONs for a seed."""
    seed_dir = Path(f"logs/seed_sweep_results/seed_{seed}")
    if not seed_dir.exists():
        return []
    
    jsons = []
    for ep in range(1, max_episodes + 1):
        json_file = seed_dir / f"val_ep{ep:03d}.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
                data['episode'] = ep
                jsons.append(data)
    return jsons

def compute_trail_5_median(jsons: List[Dict]) -> float:
    """Compute trailing-5 median of final scores."""
    if len(jsons) < 5:
        return 0.0
    
    # Get last 5 episodes
    last_5 = sorted(jsons, key=lambda x: x['episode'])[-5:]
    # Phase 2.8c uses 'score' field (jitter-averaged)
    finals = [j.get('score', j.get('spr_raw', j.get('fitness', j.get('median_fitness', 0.0)))) for j in last_5]
    return float(np.median(finals))

def compute_jitter_stability(jsons: List[Dict]) -> Dict:
    """Analyze friction jitter stability."""
    spreads = [j.get('spread', 0.0) for j in jsons]
    unique_spreads = len(set(spreads))
    
    spread_array = np.array(spreads)
    spread_std = float(np.std(spread_array)) if len(spread_array) > 0 else 0.0
    spread_range = (float(np.min(spread_array)), float(np.max(spread_array))) if len(spread_array) > 0 else (0.0, 0.0)
    
    return {
        'unique_spreads': unique_spreads,
        'spread_std': spread_std,
        'spread_range': spread_range,
        'jitter_working': unique_spreads > len(jsons) * 0.5  # At least 50% unique
    }

def analyze_seed(seed: int) -> Dict:
    """Analyze a single seed."""
    jsons = load_seed_jsons(seed, max_episodes=150)
    if not jsons:
        return None
    
    # Get SPR scores - Phase 2.8c uses 'score' field (jitter-averaged)
    sprs = [j.get('score', j.get('spr_raw', j.get('fitness', j.get('median_fitness', 0.0)))) for j in jsons]
    
    # Compute mean and final
    mean_spr = float(np.mean(sprs))
    last_spr = sprs[-1] if sprs else 0.0
    trail_5_median = compute_trail_5_median(jsons)
    
    # Compute trade stats
    trades_per_ep = [j.get('trades', 0) for j in jsons]
    mean_trades = float(np.mean(trades_per_ep))
    
    # Compute penalty rate
    penalty_episodes = sum(1 for j in jsons if j.get('penalty_applied', False))
    penalty_rate = penalty_episodes / len(jsons) if jsons else 0.0
    
    # Compute action metrics (from last episode)
    last_json = jsons[-1]
    entropy = last_json.get('action_entropy', 0.0)
    switch_rate = last_json.get('action_switch_rate', 0.0)
    
    # Analyze friction jitter
    jitter_stats = compute_jitter_stability(jsons)
    
    return {
        'seed': seed,
        'episodes': len(jsons),
        'mean_spr': mean_spr,
        'final_spr': last_spr,
        'trail_5_median': trail_5_median,
        'mean_trades': mean_trades,
        'penalty_rate': penalty_rate,
        'entropy': entropy,
        'switch_rate': switch_rate,
        **jitter_stats
    }

def compare_to_baseline(run_c_results: List[Dict], baseline_mean: float = -0.004) -> Dict:
    """Compare to Run A baseline (frozen frictions)."""
    # Compute cross-seed stats for Run C
    means = [r['mean_spr'] for r in run_c_results if r]
    trail_5s = [r['trail_5_median'] for r in run_c_results if r]
    trades = [r['mean_trades'] for r in run_c_results if r]
    
    cross_mean = float(np.mean(means))
    cross_trail_5 = float(np.mean(trail_5s))
    cross_trades = float(np.mean(trades))
    
    mean_std = float(np.std(means))
    
    # Compare to baseline
    mean_delta = cross_mean - baseline_mean
    variance_ratio = mean_std / 0.021 if 0.021 > 0 else 1.0  # Run A had ±0.021
    
    # Count positive finals
    positive_finals = sum(1 for r in run_c_results if r and r['trail_5_median'] > 0)
    
    return {
        'cross_mean': cross_mean,
        'cross_trail_5': cross_trail_5,
        'cross_trades': cross_trades,
        'mean_std': mean_std,
        'mean_delta': mean_delta,
        'variance_ratio': variance_ratio,
        'positive_finals': positive_finals,
        'total_seeds': len(run_c_results)
    }

def generate_verdict(comparison: Dict) -> str:
    """Generate GREEN/YELLOW/RED verdict."""
    mean = comparison['cross_mean']
    mean_delta = comparison['mean_delta']
    variance_ratio = comparison['variance_ratio']
    positive_finals = comparison['positive_finals']
    total_seeds = comparison['total_seeds']
    cross_trail_5 = comparison.get('cross_trail_5', 0.0)
    
    # GREEN: Mean ≥ +0.03, improvement vs baseline, trail-5 decent
    if mean >= 0.03 and mean_delta > 0.02:
        if cross_trail_5 >= 0.20 or positive_finals >= total_seeds * 0.33:  # At least 1/3 seeds positive
            return "🟢 GREEN - Agent is robust! Proceed to 200-episode confirmation."
    
    # YELLOW: Mean +0.02-0.03, or good mean but weak finals
    if mean >= 0.02 or (mean >= 0.01 and variance_ratio <= 3.0):
        return "🟡 YELLOW - Good mean, but some seeds weak. Expand to 5 seeds or fine-tune."
    
    # RED: Mean < +0.02, degradation, high variance
    return "🔴 RED - Performance degraded. Revert to Phase 2.8b or adjust penalties."

def main():
    """Main comparison."""
    print("=" * 80)
    print("PHASE 2.8c RUN C v1 - STABILIZATION TWEAKS RESULTS")
    print("=" * 80)
    print()
    
    # Analyze each seed
    results = []
    for seed in RUN_C_V1_SEEDS:
        print(f"Analyzing seed {seed}...")
        result = analyze_seed(seed)
        if result:
            results.append(result)
            print(f"  Mean SPR: {result['mean_spr']:+.3f}")
            print(f"  Final SPR: {result['final_spr']:+.3f}")
            print(f"  Trail-5 median: {result['trail_5_median']:+.3f}")
            print(f"  Trades/ep: {result['mean_trades']:.1f}")
            print(f"  Penalty rate: {result['penalty_rate']:.1%}")
            print(f"  Jitter working: {result['jitter_working']}")
            print()
        else:
            print(f"  No data found for seed {seed}")
            print()
    
    if not results:
        print("No results found!")
        return
    
    # Cross-seed summary
    print("=" * 80)
    print("RUN C v1 CROSS-SEED SUMMARY:")
    print("=" * 80)
    
    means = [r['mean_spr'] for r in results]
    trail_5s = [r['trail_5_median'] for r in results]
    trades = [r['mean_trades'] for r in results]
    penalty_rates = [r['penalty_rate'] for r in results]
    
    print(f"Cross-seed mean SPR:  {np.mean(means):+.3f} ± {np.std(means):.3f}")
    print(f"Cross-seed trail-5:   {np.mean(trail_5s):+.3f} ± {np.std(trail_5s):.3f}")
    print(f"Cross-seed trades/ep: {np.mean(trades):.1f} ± {np.std(trades):.1f}")
    print(f"Cross-seed penalty:   {np.mean(penalty_rates):.1%}")
    print(f"Positive trail-5: {sum(1 for t in trail_5s if t > 0)}/{len(results)} seeds")
    
    # Find best seed
    best_idx = np.argmax(means)
    best_seed = results[best_idx]['seed']
    print(f"Best seed (by mean): {best_seed}")
    print()
    
    # Jitter stability
    print("FRICTION JITTER STABILITY:")
    print("=" * 80)
    for r in results:
        print(f"Seed {r['seed']}: {r['unique_spreads']} unique spreads, "
              f"range {r['spread_range'][0]:.6f}-{r['spread_range'][1]:.6f}, "
              f"std {r['spread_std']:.6f}")
    print()
    
    # Compare to baseline
    print("COMPARISON TO RUN A BASELINE:")
    print("=" * 80)
    print("Run A (frozen frictions, 5 seeds):")
    print("  Cross-seed mean: -0.004 ± 0.021")
    print("  Cross-seed final: +0.451 ± 0.384")
    print()
    
    comparison = compare_to_baseline(results, baseline_mean=-0.004)
    print(f"Run C v1 (jitter-avg K=3, VAL_EXP_TRADES_SCALE=0.38, 3 seeds):")
    print(f"  Cross-seed mean: {comparison['cross_mean']:+.3f} ± {comparison['mean_std']:.3f}")
    print(f"  Cross-seed trail-5: {comparison['cross_trail_5']:+.3f}")
    print()
    print(f"Mean change: {comparison['mean_delta']:+.3f} (vs Run A baseline)")
    print(f"Variance change: {comparison['variance_ratio']:.1f}x (vs Run A ±0.021)")
    print(f"Positive finals: {comparison['positive_finals']}/{comparison['total_seeds']} seeds")
    print()
    
    # Verdict
    print("=" * 80)
    print("VERDICT:")
    print("=" * 80)
    verdict = generate_verdict(comparison)
    print(verdict)
    print()
    
    # Expected improvements
    print("EXPECTED IMPROVEMENTS (vs Run B v2):")
    print("=" * 80)
    print("Run B v2: Mean +0.018, Final +0.147, Variance ±0.037")
    print("Expected Run C v1 (with stabilization tweaks):")
    print("  - Mean: +0.03 to +0.06 (jitter-averaged validation)")
    print("  - Trail-5 median: +0.30 to +0.50 (reduced final-episode luck)")
    print("  - Variance: ±0.025 to ±0.035 (tighter gating)")
    print("  - Penalty rate: <10% (improved trade-count gating)")
    print()

if __name__ == "__main__":
    main()
