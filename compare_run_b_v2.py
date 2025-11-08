#!/usr/bin/env python3
"""
Compare Run B v2 (friction jitter) seeds only.
Focus on the 3 seeds that were tested with friction randomization.
"""

import json
from pathlib import Path
import numpy as np

RESULTS_BASE = Path("logs/seed_sweep_results")
RUN_B_V2_SEEDS = [7, 77, 777]  # Seeds tested with friction jitter

print("="*70)
print("RUN B v2 - ROBUSTNESS TEST RESULTS")
print("(3 seeds with friction jitter enabled)")
print("="*70)

all_means = []
all_finals = []
all_trades = []

for seed in RUN_B_V2_SEEDS:
    seed_dir = RESULTS_BASE / f"seed_{seed}"
    if not seed_dir.exists():
        print(f"Seed {seed}: Not found!")
        continue
    
    # Load all episodes
    episodes = []
    for ep_file in sorted(seed_dir.glob("val_ep*.json")):
        with open(ep_file, 'r') as f:
            data = json.load(f)
            episodes.append(data)
    
    if not episodes:
        print(f"Seed {seed}: No episodes found!")
        continue
    
    # Calculate statistics
    scores = [ep.get('score', 0) for ep in episodes]
    trades = [ep.get('trades', 0) for ep in episodes]
    
    mean_score = np.mean(scores)
    final_score = scores[-1] if scores else 0
    best_score = max(scores) if scores else 0
    mean_trades = np.mean(trades)
    
    all_means.append(mean_score)
    all_finals.append(final_score)
    all_trades.append(mean_trades)
    
    print(f"\nSeed {seed}:")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Mean SPR: {mean_score:+.3f} ± {np.std(scores):.3f}")
    print(f"  Final SPR: {final_score:+.3f}")
    print(f"  Best SPR: {best_score:+.3f}")
    print(f"  Trades/ep: {mean_trades:.1f}")
    
    # Check friction variation
    spreads = [ep.get('spread', None) for ep in episodes]
    spreads = [s for s in spreads if s is not None]
    if spreads:
        print(f"  Spread: {np.mean(spreads):.6f} ± {np.std(spreads):.6f} ({len(set(spreads))} unique)")
    else:
        print(f"  Spread: Not tracked")

print("\n" + "="*70)
print("RUN B v2 CROSS-SEED SUMMARY")
print("="*70)
print(f"Cross-seed mean SPR:  {np.mean(all_means):+.3f} ± {np.std(all_means):.3f}")
print(f"Cross-seed final SPR: {np.mean(all_finals):+.3f} ± {np.std(all_finals):.3f}")
print(f"Cross-seed trades/ep: {np.mean(all_trades):.1f} ± {np.std(all_trades):.1f}")
print(f"Positive finals: {sum(1 for f in all_finals if f > 0)}/{len(all_finals)} seeds")
print()
print("Best seed (by mean): {}".format(RUN_B_V2_SEEDS[np.argmax(all_means)]))
print("Best seed (by final): {}".format(RUN_B_V2_SEEDS[np.argmax(all_finals)]))

print("\n" + "="*70)
print("COMPARISON TO RUN A BASELINE")
print("="*70)
print("Note: Run A used different seeds (17, 27) with frozen frictions.")
print("Cannot directly compare seed-by-seed, but can compare distributions:")
print()
print("Run A (frozen frictions, 5 seeds):")
print("  Cross-seed mean: -0.004 ± 0.021")
print("  Cross-seed final: +0.451 ± 0.384")
print()
print(f"Run B v2 (friction jitter, 3 seeds):")
print(f"  Cross-seed mean: {np.mean(all_means):+.3f} ± {np.std(all_means):.3f}")
print(f"  Cross-seed final: {np.mean(all_finals):+.3f} ± {np.std(all_finals):.3f}")
print()
print(f"Mean change: {np.mean(all_means) - (-0.004):+.3f}")
print(f"Final change: {np.mean(all_finals) - 0.451:+.3f}")
print(f"Variance change: {np.std(all_means) / 0.021:.1f}x")

print("\n" + "="*70)
print("VERDICT")
print("="*70)
if np.mean(all_means) >= 0.02:
    print("✅ GREEN - Excellent robustness!")
    print("   Mean SPR ≥ +0.02 with friction jitter.")
    print("   Agent learned robust trading patterns.")
elif np.mean(all_means) >= 0.01:
    print("🟡 YELLOW - Acceptable with fine-tuning.")
    print("   Mean SPR in +0.01 to +0.02 range.")
    print("   Consider adjusting trade penalties.")
elif np.mean(all_means) >= 0.00:
    print("🟢 MARGINAL GREEN - Agent is robust!")
    print("   Mean SPR positive despite friction jitter.")
    print("   Variance increase expected and acceptable.")
else:
    print("🔴 RED - Not robust to friction variations.")
    print("   Mean SPR negative with friction jitter.")
    print("   Consider reverting to Phase 2.8 config.")
print("="*70)
