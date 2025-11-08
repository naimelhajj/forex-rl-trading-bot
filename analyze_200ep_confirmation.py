"""
Phase 2.8c - 200-Episode Confirmation Analysis
Implements full acceptance gates and behavioral metrics monitoring.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Target seeds for 200-episode confirmation
CONFIRMATION_SEEDS = [7, 17, 27, 77, 777]

# Acceptance gates (GO/NO-GO)
GATES = {
    'mean_spr_min': 0.04,
    'trail_5_min': 0.25,
    'std_max': 0.035,
    'penalty_max': 0.10,
    'positive_seeds_min': 3,  # out of 5
}

# Behavioral metric ranges
BEHAVIORAL_RANGES = {
    'entropy': (0.90, 1.10),
    'switch_rate': (0.14, 0.20),
    'hold_rate': (0.65, 0.80),
    'long_ratio_min': 0.40,
    'long_ratio_max': 0.60,
}

def load_seed_jsons(seed: int, max_episodes: int = 250) -> List[Dict]:
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
    """Compute trailing-5 median of scores."""
    if len(jsons) < 5:
        return 0.0
    
    last_5 = sorted(jsons, key=lambda x: x['episode'])[-5:]
    # Phase 2.8c uses 'score' field (jitter-averaged)
    finals = [j.get('score', j.get('median_fitness', 0.0)) for j in last_5]
    return float(np.median(finals))

def analyze_behavioral_metrics(jsons: List[Dict]) -> Dict:
    """Analyze behavioral metrics across episodes."""
    if not jsons:
        return {}
    
    # Last episode metrics
    last = jsons[-1]
    
    # Extract behavioral data
    entropy = last.get('action_entropy_bits', 0.0)
    switch_rate = last.get('switch_rate', 0.0)
    hold_rate = last.get('hold_rate', 0.0)
    
    # Long/Short balance
    ls_data = last.get('long_short', {})
    long_ratio = ls_data.get('long_ratio', 0.5)
    
    # Average over last 20 episodes for stability
    if len(jsons) >= 20:
        last_20 = jsons[-20:]
        avg_entropy = np.mean([j.get('action_entropy_bits', 0.0) for j in last_20])
        avg_switch = np.mean([j.get('switch_rate', 0.0) for j in last_20])
        avg_hold = np.mean([j.get('hold_rate', 0.0) for j in last_20])
        avg_long_ratio = np.mean([j.get('long_short', {}).get('long_ratio', 0.5) for j in last_20])
    else:
        avg_entropy = entropy
        avg_switch = switch_rate
        avg_hold = hold_rate
        avg_long_ratio = long_ratio
    
    return {
        'final_entropy': entropy,
        'final_switch_rate': switch_rate,
        'final_hold_rate': hold_rate,
        'final_long_ratio': long_ratio,
        'avg_entropy': avg_entropy,
        'avg_switch_rate': avg_switch,
        'avg_hold_rate': avg_hold,
        'avg_long_ratio': avg_long_ratio,
    }

def check_behavioral_gates(metrics: Dict) -> Tuple[bool, List[str]]:
    """Check if behavioral metrics are in acceptable ranges."""
    issues = []
    
    # Use averages for stability
    entropy = metrics.get('avg_entropy', 0.0)
    switch = metrics.get('avg_switch_rate', 0.0)
    hold = metrics.get('avg_hold_rate', 0.0)
    long_ratio = metrics.get('avg_long_ratio', 0.5)
    
    ent_min, ent_max = BEHAVIORAL_RANGES['entropy']
    if not (ent_min <= entropy <= ent_max):
        issues.append(f"Entropy {entropy:.3f} outside [{ent_min}, {ent_max}]")
    
    sw_min, sw_max = BEHAVIORAL_RANGES['switch_rate']
    if not (sw_min <= switch <= sw_max):
        issues.append(f"Switch rate {switch:.3f} outside [{sw_min}, {sw_max}]")
    
    hold_min, hold_max = BEHAVIORAL_RANGES['hold_rate']
    if not (hold_min <= hold <= hold_max):
        issues.append(f"Hold rate {hold:.3f} outside [{hold_min}, {hold_max}]")
    
    long_min = BEHAVIORAL_RANGES['long_ratio_min']
    long_max = BEHAVIORAL_RANGES['long_ratio_max']
    if not (long_min <= long_ratio <= long_max):
        issues.append(f"Long ratio {long_ratio:.3f} outside [{long_min}, {long_max}]")
    
    return len(issues) == 0, issues

def analyze_seed(seed: int, max_episodes: int = 250) -> Dict:
    """Comprehensive seed analysis."""
    jsons = load_seed_jsons(seed, max_episodes)
    if not jsons:
        return None
    
    # Get SPR scores
    sprs = [j.get('score', j.get('median_fitness', 0.0)) for j in jsons]
    
    # Compute statistics
    mean_spr = float(np.mean(sprs))
    final_spr = sprs[-1] if sprs else 0.0
    trail_5 = compute_trail_5_median(jsons)
    
    # Trade stats
    trades_per_ep = [j.get('trades', 0.0) for j in jsons]
    mean_trades = float(np.mean(trades_per_ep))
    
    # Penalty rate
    penalty_episodes = sum(1 for j in jsons if j.get('penalty_applied', False))
    penalty_rate = penalty_episodes / len(jsons) if jsons else 0.0
    
    # Behavioral metrics
    behavioral = analyze_behavioral_metrics(jsons)
    behavioral_ok, behavioral_issues = check_behavioral_gates(behavioral)
    
    # Friction jitter check
    spreads = [j.get('spread', 0.0) for j in jsons]
    unique_spreads = len(set(spreads))
    jitter_working = unique_spreads > len(jsons) * 0.5
    
    return {
        'seed': seed,
        'episodes': len(jsons),
        'mean_spr': mean_spr,
        'final_spr': final_spr,
        'trail_5_median': trail_5,
        'mean_trades': mean_trades,
        'penalty_rate': penalty_rate,
        'unique_spreads': unique_spreads,
        'jitter_working': jitter_working,
        'behavioral': behavioral,
        'behavioral_ok': behavioral_ok,
        'behavioral_issues': behavioral_issues,
    }

def check_acceptance_gates(results: List[Dict]) -> Tuple[str, Dict, List[str]]:
    """Check all acceptance gates and return verdict."""
    if not results:
        return "RED", {}, ["No results to analyze"]
    
    # Compute cross-seed statistics
    means = [r['mean_spr'] for r in results]
    trail_5s = [r['trail_5_median'] for r in results]
    
    cross_mean = float(np.mean(means))
    cross_trail_5 = float(np.mean(trail_5s))
    std_means = float(np.std(means))
    
    # Penalty rate
    penalty_rates = [r['penalty_rate'] for r in results]
    cross_penalty = float(np.mean(penalty_rates))
    
    # Positive trail-5 count
    positive_trail_5_count = sum(1 for t in trail_5s if t > 0)
    
    # Gate checks
    gates_passed = []
    gates_failed = []
    
    # Gate 1: Mean SPR
    if cross_mean >= GATES['mean_spr_min']:
        gates_passed.append(f"✅ Mean SPR {cross_mean:+.3f} ≥ {GATES['mean_spr_min']:+.3f}")
    else:
        gates_failed.append(f"❌ Mean SPR {cross_mean:+.3f} < {GATES['mean_spr_min']:+.3f}")
    
    # Gate 2: Trail-5 median
    if cross_trail_5 >= GATES['trail_5_min']:
        gates_passed.append(f"✅ Trail-5 {cross_trail_5:+.3f} ≥ {GATES['trail_5_min']:+.3f}")
    else:
        gates_failed.append(f"❌ Trail-5 {cross_trail_5:+.3f} < {GATES['trail_5_min']:+.3f}")
    
    # Gate 3: Std of means
    if std_means <= GATES['std_max']:
        gates_passed.append(f"✅ Std of means {std_means:.3f} ≤ {GATES['std_max']:.3f}")
    else:
        gates_failed.append(f"❌ Std of means {std_means:.3f} > {GATES['std_max']:.3f}")
    
    # Gate 4: Penalty rate
    if cross_penalty <= GATES['penalty_max']:
        gates_passed.append(f"✅ Penalty rate {cross_penalty:.1%} ≤ {GATES['penalty_max']:.1%}")
    else:
        gates_failed.append(f"❌ Penalty rate {cross_penalty:.1%} > {GATES['penalty_max']:.1%}")
    
    # Gate 5: Positive seeds
    if positive_trail_5_count >= GATES['positive_seeds_min']:
        gates_passed.append(f"✅ Positive seeds {positive_trail_5_count}/{len(results)} ≥ {GATES['positive_seeds_min']}")
    else:
        gates_failed.append(f"❌ Positive seeds {positive_trail_5_count}/{len(results)} < {GATES['positive_seeds_min']}")
    
    # Behavioral checks
    behavioral_issues_count = sum(1 for r in results if not r['behavioral_ok'])
    if behavioral_issues_count == 0:
        gates_passed.append(f"✅ All seeds pass behavioral metrics")
    else:
        gates_failed.append(f"⚠️ {behavioral_issues_count}/{len(results)} seeds have behavioral issues")
    
    # Verdict
    if len(gates_failed) == 0:
        verdict = "GREEN"
    elif len(gates_failed) <= 2 and cross_mean >= 0.03:
        verdict = "YELLOW"
    else:
        verdict = "RED"
    
    stats = {
        'cross_mean': cross_mean,
        'cross_trail_5': cross_trail_5,
        'std_means': std_means,
        'cross_penalty': cross_penalty,
        'positive_count': positive_trail_5_count,
        'total_seeds': len(results),
    }
    
    return verdict, stats, gates_passed + gates_failed

def main():
    """Main analysis."""
    print("=" * 80)
    print("PHASE 2.8c - 200-EPISODE CONFIRMATION ANALYSIS")
    print("=" * 80)
    print()
    
    # Analyze each seed
    results = []
    for seed in CONFIRMATION_SEEDS:
        print(f"Analyzing seed {seed}...")
        result = analyze_seed(seed, max_episodes=250)
        if result:
            results.append(result)
            print(f"  Episodes: {result['episodes']}")
            print(f"  Mean SPR: {result['mean_spr']:+.3f}")
            print(f"  Trail-5 median: {result['trail_5_median']:+.3f}")
            print(f"  Trades/ep: {result['mean_trades']:.1f}")
            print(f"  Penalty rate: {result['penalty_rate']:.1%}")
            print(f"  Friction jitter: {'✅' if result['jitter_working'] else '❌'}")
            print(f"  Behavioral: {'✅' if result['behavioral_ok'] else '⚠️'}")
            if result['behavioral_issues']:
                for issue in result['behavioral_issues']:
                    print(f"    - {issue}")
            print()
        else:
            print(f"  ❌ No data found for seed {seed}")
            print()
    
    if not results:
        print("❌ No results found! Run not completed.")
        return
    
    # Cross-seed summary
    print("=" * 80)
    print("CROSS-SEED SUMMARY:")
    print("=" * 80)
    
    means = [r['mean_spr'] for r in results]
    trail_5s = [r['trail_5_median'] for r in results]
    trades = [r['mean_trades'] for r in results]
    
    print(f"Cross-seed mean SPR:  {np.mean(means):+.3f} ± {np.std(means):.3f}")
    print(f"Cross-seed trail-5:   {np.mean(trail_5s):+.3f} ± {np.std(trail_5s):.3f}")
    print(f"Cross-seed trades/ep: {np.mean(trades):.1f} ± {np.std(trades):.1f}")
    print(f"Best seed (by mean): {results[np.argmax(means)]['seed']}")
    print(f"Best seed (by trail-5): {results[np.argmax(trail_5s)]['seed']}")
    print()
    
    # Acceptance gates
    print("=" * 80)
    print("ACCEPTANCE GATES:")
    print("=" * 80)
    
    verdict, stats, gates = check_acceptance_gates(results)
    
    for gate in gates:
        print(gate)
    print()
    
    # Behavioral summary
    print("=" * 80)
    print("BEHAVIORAL METRICS SUMMARY:")
    print("=" * 80)
    
    for result in results:
        b = result['behavioral']
        print(f"Seed {result['seed']}:")
        print(f"  Entropy: {b.get('avg_entropy', 0.0):.3f} (target: 0.90-1.10)")
        print(f"  Switch rate: {b.get('avg_switch_rate', 0.0):.3f} (target: 0.14-0.20)")
        print(f"  Hold rate: {b.get('avg_hold_rate', 0.0):.3f} (target: 0.65-0.80)")
        print(f"  Long ratio: {b.get('avg_long_ratio', 0.5):.3f} (target: 0.40-0.60)")
        print()
    
    # Final verdict
    print("=" * 80)
    print("FINAL VERDICT:")
    print("=" * 80)
    
    if verdict == "GREEN":
        print("🟢 GREEN - ALL GATES PASSED!")
        print()
        print("✅ Production-ready! Lock Phase 2.8c as SPR Baseline v1.1")
        print()
        print("Next steps:")
        print("1. Archive config as config_phase2.8c_baseline_v1.1.py")
        print(f"2. Select production seed (recommend: {results[np.argmax(means)]['seed']})")
        print("3. Create comprehensive documentation")
        print("4. Paper trading integration (1-week test)")
        print("5. Live trading preparation")
    elif verdict == "YELLOW":
        print("🟡 YELLOW - Some gates marginal")
        print()
        print("⚠️ Review failed gates and consider micro-tunes:")
        for gate in gates:
            if gate.startswith("❌") or gate.startswith("⚠️"):
                print(f"  {gate}")
        print()
        print("Consider:")
        print("- If finals too luck-heavy: VAL_TRIM_FRACTION 0.20 → 0.25")
        print("- If trade counts high: VAL_EXP_TRADES_SCALE 0.38 → 0.40")
        print("- If whipsaw high: hold_tie_tau 0.035 → 0.038-0.040")
    else:
        print("🔴 RED - Multiple gates failed")
        print()
        print("❌ Performance degraded. Action required:")
        for gate in gates:
            if gate.startswith("❌"):
                print(f"  {gate}")
        print()
        print("Recommend: Revert to Phase 2.8b and analyze degradation")
    
    print()

if __name__ == "__main__":
    main()
