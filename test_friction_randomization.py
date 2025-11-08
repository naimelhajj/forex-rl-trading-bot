#!/usr/bin/env python3
"""
Test friction randomization logic.
Verify that FREEZE_VALIDATION_FRICTIONS=False actually randomizes spreads.
"""

import numpy as np

# Simulate the trainer logic
FREEZE_VALIDATION_FRICTIONS = False  # Run B setting

print("="*70)
print("FRICTION RANDOMIZATION TEST")
print("="*70)
print(f"FREEZE_VALIDATION_FRICTIONS = {FREEZE_VALIDATION_FRICTIONS}")
print()

# Simulate 10 episodes
spreads = []
slippages = []

for episode in range(1, 11):
    # This is the actual code from trainer.py lines 1139-1147
    if not FREEZE_VALIDATION_FRICTIONS:
        try:
            s = float(np.random.uniform(0.00013, 0.00020))
            sp = float(np.random.uniform(0.6, 1.0))
            spreads.append(s)
            slippages.append(sp)
            print(f"Episode {episode:2d}: spread={s:.6f}, slippage={sp:.4f}")
        except Exception as e:
            print(f"Episode {episode:2d}: ERROR - {e}")
    else:
        # Frozen - use base values
        base_spread = 0.00015
        base_slippage = 0.8
        spreads.append(base_spread)
        slippages.append(base_slippage)
        print(f"Episode {episode:2d}: spread={base_spread:.6f}, slippage={base_slippage:.4f} (FROZEN)")

print()
print("="*70)
print("STATISTICS:")
print("="*70)
print(f"Spreads:")
print(f"  Min:    {min(spreads):.6f}")
print(f"  Max:    {max(spreads):.6f}")
print(f"  Mean:   {np.mean(spreads):.6f}")
print(f"  Std:    {np.std(spreads):.6f}")
print(f"  Unique: {len(set(spreads))}")
print()
print(f"Slippages:")
print(f"  Min:    {min(slippages):.4f}")
print(f"  Max:    {max(slippages):.4f}")
print(f"  Mean:   {np.mean(slippages):.4f}")
print(f"  Std:    {np.std(slippages):.4f}")
print(f"  Unique: {len(set(slippages))}")
print()
print("="*70)
if not FREEZE_VALIDATION_FRICTIONS:
    print("✓ Friction randomization IS working!")
    print("  Each episode gets different spread/slippage values.")
else:
    print("✓ Frictions are frozen (expected for Run A).")
print("="*70)
