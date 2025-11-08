"""
Quick check of validation diversity using JSON summaries
Reads from logs/validation_summaries/*.json files
"""

import json
import os
from pathlib import Path

summary_dir = Path("logs/validation_summaries")

if not summary_dir.exists():
    print("Validation summaries directory not found.")
    print("Run training first: python main.py --episodes 5")
    exit(1)

# Find all JSON files
json_files = sorted(summary_dir.glob("val_ep*.json"))

if not json_files:
    print("No validation JSON files found yet. Training may still be running.")
    exit(0)

print("="*70)
print("VALIDATION SLICE DIVERSITY CHECK")
print("="*70)
print(f"\nFound {len(json_files)} validation summaries\n")

# Read and display each validation
for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ep = data.get('episode', '?')
    trades = data.get('trades', 0.0)
    mult = data.get('mult', 0.0)
    pen = data.get('penalty', 0.0)
    score = data.get('score', 0.0)
    k = data.get('k', 0)
    
    status = "✓" if trades >= 5 else "⚠"
    
    print(f"Ep {ep:3d}: K={k} | trades={trades:4.1f} | mult={mult:.2f} | "
          f"pen={pen:.3f} | score={score:+8.5f} {status}")

print("\n" + "="*70)
print("LEGEND:")
print("  ✓ = Healthy activity (5+ trades)")
print("  ⚠ = Low/zero trades (potential issue)")
print("\nKEY CHECKS:")
print("  1. Trade counts should vary (not all 0 or all the same)")
print("  2. Penalty (pen) should be 0.000 when trades >= 6-7")
print("  3. Multiplier (mult) should reach 0.75-1.00 for good episodes")
print("  4. K should be consistent (typically 6-7 windows per validation)")
print("="*70)

# Summary statistics
if json_files:
    all_trades = [json.load(open(f))['trades'] for f in json_files]
    all_scores = [json.load(open(f))['score'] for f in json_files]
    
    print("\nSUMMARY STATISTICS:")
    print(f"  Trade count range: {min(all_trades):.1f} - {max(all_trades):.1f}")
    print(f"  Score range: {min(all_scores):+.5f} - {max(all_scores):+.5f}")
    print(f"  Episodes with 0 trades: {sum(1 for t in all_trades if t == 0)}")
    print(f"  Episodes with 5+ trades: {sum(1 for t in all_trades if t >= 5)}")
    print("="*70)
