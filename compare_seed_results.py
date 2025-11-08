"""
Compare Seed Sweep Results
Analyzes JSON files from multiple seed runs
"""

import json
from pathlib import Path
import numpy as np

RESULTS_BASE = Path("logs/seed_sweep_results")

# Auto-detect available seeds from directory structure
def get_available_seeds():
    """Discover all seed directories and extract seed numbers."""
    if not RESULTS_BASE.exists():
        return []
    seeds = []
    for seed_dir in RESULTS_BASE.iterdir():
        if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
            try:
                seed_num = int(seed_dir.name.replace("seed_", ""))
                seeds.append(seed_num)
            except ValueError:
                pass
    return sorted(seeds)

SEEDS = get_available_seeds()

def load_final_score(seed_dir: Path):
    """
    Load final score with preference for post-restore evaluation.
    Returns: (score, source_type)
    """
    # Prefer explicit post-restore eval if present
    final_path = seed_dir / "val_final.json"
    if final_path.exists():
        try:
            js = json.load(open(final_path, "r"))
            return js.get("score", float("nan")), "post-restore"
        except Exception:
            pass
    
    # Else: last chronological episode (not lexicographic)
    episodes = []
    for p in sorted(seed_dir.glob("val_ep*.json")):
        try:
            ep = json.load(open(p, "r"))
            idx = ep.get("episode", None)
            if isinstance(idx, int):
                episodes.append((idx, ep.get("score", float("nan"))))
        except Exception:
            pass
    
    if episodes:
        episodes.sort(key=lambda t: t[0])
        return episodes[-1][1], "last-episode"
    
    return float("nan"), "none"

def load_seed_data(seed: int):
    """Load all JSON files for a seed"""
    seed_dir = RESULTS_BASE / f"seed_{seed}"
    if not seed_dir.exists():
        print(f"⚠️  No results found for seed {seed} in {seed_dir}")
        return []
    
    json_files = sorted(seed_dir.glob("val_ep*.json"))
    data = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data.append(json.load(f))
        except Exception as e:
            print(f"⚠️  Error reading {json_file}: {e}")
    
    return data

def analyze_seed(seed: int, data: list):
    """Compute statistics for one seed"""
    if not data:
        return None
    
    scores = [d['score'] for d in data]
    trades = [d['trades'] for d in data]
    penalties = [d['penalty'] for d in data]
    
    # Get final score using post-restore if available
    seed_dir = RESULTS_BASE / f"seed_{seed}"
    final_score, final_source = load_final_score(seed_dir)
    
    return {
        'seed': seed,
        'episodes': len(data),
        'score_mean': np.mean(scores),
        'score_std': np.std(scores),
        'score_final': final_score,
        'final_source': final_source,
        'score_best': max(scores),
        'score_range': (min(scores), max(scores)),
        'trades_mean': np.mean(trades),
        'trades_final': trades[-1] if trades else 0.0,
        'penalty_rate': sum(1 for p in penalties if p > 0.01) / len(penalties) * 100,
        'zero_trade_rate': sum(1 for t in trades if t < 1.0) / len(trades) * 100,
    }

def main():
    print("="*70)
    print("SEED SWEEP COMPARISON")
    print("="*70)
    print()
    
    all_stats = []
    
    for seed in SEEDS:
        print(f"Loading seed {seed}...")
        data = load_seed_data(seed)
        
        if not data:
            print(f"  ⚠️  No data found\n")
            continue
        
        stats = analyze_seed(seed, data)
        all_stats.append(stats)
        
        print(f"  ✓ Loaded {stats['episodes']} episodes\n")
    
    if not all_stats:
        print("❌ No data to compare!")
        return
    
    print("="*70)
    print("SUMMARY BY SEED")
    print("="*70)
    print()
    
    # Header
    print(f"{'Seed':<8} {'Episodes':<10} {'Score Mean':<12} {'Score Final':<12} {'Score Best':<12} {'Trades':<10}")
    print("-"*70)
    
    # Data rows
    for stats in all_stats:
        print(f"{stats['seed']:<8} "
              f"{stats['episodes']:<10} "
              f"{stats['score_mean']:>+11.3f} "
              f"{stats['score_final']:>+11.3f} "
              f"{stats['score_best']:>+11.3f} "
              f"{stats['trades_mean']:>9.1f}")
    
    print()
    print("="*70)
    print("DETAILED STATISTICS")
    print("="*70)
    print()
    
    for stats in all_stats:
        print(f"Seed {stats['seed']}:")
        print(f"  Episodes:        {stats['episodes']}")
        print(f"  Score Mean:      {stats['score_mean']:+.3f} ± {stats['score_std']:.3f}")
        print(f"  Score Final:     {stats['score_final']:+.3f} ({stats['final_source']})")
        print(f"  Score Best:      {stats['score_best']:+.3f}")
        print(f"  Score Range:     {stats['score_range'][0]:+.3f} to {stats['score_range'][1]:+.3f}")
        print(f"  Trades Mean:     {stats['trades_mean']:.1f}")
        print(f"  Trades Final:    {stats['trades_final']:.1f}")
        print(f"  Penalty Rate:    {stats['penalty_rate']:.1f}% of validations")
        print(f"  Zero-Trade Rate: {stats['zero_trade_rate']:.1f}% of validations")
        print()
    
    # Cross-seed comparison
    print("="*70)
    print("CROSS-SEED COMPARISON")
    print("="*70)
    print()
    
    score_means = [s['score_mean'] for s in all_stats]
    score_finals = [s['score_final'] for s in all_stats]
    
    print(f"Score Mean across seeds:    {np.mean(score_means):+.3f} ± {np.std(score_means):.3f}")
    print(f"Score Final across seeds:   {np.mean(score_finals):+.3f} ± {np.std(score_finals):.3f}")
    print(f"Best seed (by mean):        {all_stats[np.argmax(score_means)]['seed']}")
    print(f"Best seed (by final):       {all_stats[np.argmax(score_finals)]['seed']}")
    print()
    
    # Consistency check
    score_std_mean = np.mean([s['score_std'] for s in all_stats])
    score_cross_std = np.std(score_means)
    
    print("Consistency:")
    if score_cross_std < score_std_mean:
        print(f"  ✓ CONSISTENT - Seed variation ({score_cross_std:.3f}) < within-seed variation ({score_std_mean:.3f})")
        print(f"    → Learning is stable across seeds")
    else:
        print(f"  ⚠️  INCONSISTENT - Seed variation ({score_cross_std:.3f}) > within-seed variation ({score_std_mean:.3f})")
        print(f"    → Results depend heavily on seed choice")
    print()

if __name__ == "__main__":
    main()
