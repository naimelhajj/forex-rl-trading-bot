"""
Quick sanity check for episode 80 scores across seeds.
Verifies if the +0.051 "final score" bug is in raw data or reporting logic.
"""

import json
from pathlib import Path

print("=" * 70)
print("Episode 80 Score Sanity Check (Phase 2.8b Run A)")
print("=" * 70)

seeds = [7, 17, 27, 77, 777]
scores_found = []

for seed in seeds:
    fn = Path(f"logs/seed_sweep_results/seed_{seed}/val_ep080.json")
    if fn.exists():
        with open(fn, 'r') as f:
            d = json.load(f)
        
        score = d.get('score', 'N/A')
        trades = d.get('trades', 'N/A')
        spr_comp = d.get('spr_components', {})
        
        if score != 'N/A':
            scores_found.append(score)
        
        print(f"Seed {seed:>3}: score={score if score=='N/A' else f'{score:+.5f}':>10}, "
              f"trades={trades:>4}, spr_trades={spr_comp.get('trades', 'N/A'):>4}")
    else:
        print(f"Seed {seed:>3}: Episode 80 file not found")

print("=" * 70)

if scores_found:
    import numpy as np
    scores_arr = np.array(scores_found)
    print(f"\nScore Statistics:")
    print(f"  Min:    {scores_arr.min():+.5f}")
    print(f"  Max:    {scores_arr.max():+.5f}")
    print(f"  Mean:   {scores_arr.mean():+.5f}")
    print(f"  Std:    {scores_arr.std():.5f}")
    print(f"  Unique: {len(set(scores_found))} different values")
    
    if len(set(scores_found)) == 1:
        print("\n⚠️  WARNING: All seeds have IDENTICAL scores!")
        print("    This confirms a bug in either:")
        print("    1. Score calculation/aggregation in validation code")
        print("    2. JSON export logic (all writing same value)")
    else:
        print("\n✓  Scores differ - bug is in compare_seed_results.py reporting")
else:
    print("\n❌ No valid scores found!")
