#!/usr/bin/env python3
"""
Verify friction randomization in validation results.
Check if spread/slippage values vary across episodes.
"""

import json
from pathlib import Path
import numpy as np

RESULTS_BASE = Path("logs/seed_sweep_results")

def check_friction_variation(seed: int, max_episodes: int = 80):
    """Check if friction parameters vary across validation episodes."""
    seed_dir = RESULTS_BASE / f"seed_{seed}"
    if not seed_dir.exists():
        print(f"Seed {seed}: Directory not found")
        return
    
    spreads = []
    slippages = []
    
    for ep in range(1, max_episodes + 1):
        json_file = seed_dir / f"val_ep{ep:03d}.json"
        if not json_file.exists():
            continue
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Check if friction params are stored
        spread = data.get('spread', None)
        slippage = data.get('slippage_pips', None)
        
        if spread is not None:
            spreads.append(spread)
        if slippage is not None:
            slippages.append(slippage)
    
    print(f"\nSeed {seed}:")
    print(f"  Episodes checked: {max_episodes}")
    
    if spreads:
        print(f"  Spread values found: {len(spreads)}")
        print(f"    Min: {min(spreads):.6f}")
        print(f"    Max: {max(spreads):.6f}")
        print(f"    Mean: {np.mean(spreads):.6f}")
        print(f"    Std: {np.std(spreads):.6f}")
        print(f"    Unique values: {len(set(spreads))}")
    else:
        print(f"  ❌ No spread values found in JSON files!")
    
    if slippages:
        print(f"  Slippage values found: {len(slippages)}")
        print(f"    Min: {min(slippages):.6f}")
        print(f"    Max: {max(slippages):.6f}")
        print(f"    Mean: {np.mean(slippages):.6f}")
        print(f"    Std: {np.std(slippages):.6f}")
        print(f"    Unique values: {len(set(slippages))}")
    else:
        print(f"  ❌ No slippage values found in JSON files!")
    
    return spreads, slippages

if __name__ == "__main__":
    print("="*70)
    print("FRICTION RANDOMIZATION VERIFICATION")
    print("="*70)
    
    # Check Run B seeds (should have friction jitter)
    print("\n" + "="*70)
    print("RUN B SEEDS (Friction Jitter ENABLED)")
    print("="*70)
    
    for seed in [7, 77, 777]:
        check_friction_variation(seed, max_episodes=80)
    
    # Compare to Run A seeds (should have frozen frictions)
    print("\n" + "="*70)
    print("RUN A SEEDS (Friction Frozen - for comparison)")
    print("="*70)
    
    for seed in [17, 27]:
        check_friction_variation(seed, max_episodes=80)
    
    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("="*70)
    print("If friction jitter is working:")
    print("  ✓ Run B seeds should show Std > 0 and many unique values")
    print("  ✓ Run A seeds should show Std = 0 and 1 unique value")
    print("\nIf friction values are NOT in JSON files:")
    print("  → Need to check if fitness.py saves these parameters")
    print("="*70)
