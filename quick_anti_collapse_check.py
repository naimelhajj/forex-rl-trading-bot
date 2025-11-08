"""
Quick check for anti-collapse patches
Simple version without complex formatting
"""

import json
from pathlib import Path

json_dir = Path("logs/validation_summaries")
jsons = sorted(json_dir.glob("val_ep*.json"))

if not jsons:
    print("ERROR: No JSON files found")
    exit(1)

print(f"Found {len(jsons)} JSON files")

# Check first file for new fields
first = json.load(open(jsons[0]))
has_actions = 'actions' in first
has_hold_rate = 'hold_rate' in first

print(f"\nNew fields present: actions={has_actions}, hold_rate={has_hold_rate}")

if has_actions and has_hold_rate:
    print("\nPATCH C: SUCCESS - New fields detected")
    
    # Count episodes with new fields
    with_fields = []
    for f in jsons:
        data = json.load(open(f))
        if 'actions' in data and 'hold_rate' in data:
            with_fields.append(data)
    
    print(f"Episodes with new fields: {len(with_fields)}/{len(jsons)}")
    
    if with_fields:
        # Calculate stats
        hold_rates = [d['hold_rate'] for d in with_fields]
        trades = [d['trades'] for d in with_fields]
        
        avg_hold = sum(hold_rates) / len(hold_rates)
        min_hold = min(hold_rates)
        max_hold = max(hold_rates)
        
        avg_trades = sum(trades) / len(trades)
        
        print(f"\nHOLD Rate: avg={avg_hold:.3f}, range={min_hold:.3f}-{max_hold:.3f}")
        print(f"Trades: avg={avg_trades:.1f}")
        
        # Check for collapse
        collapsed = sum(1 for d in with_fields if d['hold_rate'] > 0.95 and d['trades'] < 3)
        print(f"\nPolicy collapse (>95% HOLD + <3 trades): {collapsed}/{len(with_fields)}")
        
        if collapsed == 0:
            print("STATUS: No policy collapse detected")
        elif collapsed < len(with_fields) * 0.1:
            print("STATUS: Low collapse rate (<10%)")
        else:
            print("STATUS: High collapse rate - may need tuning")
        
        # Show first 3 examples
        print(f"\nFirst 3 episodes with new fields:")
        for d in with_fields[:3]:
            a = d['actions']
            print(f"  Ep {d['episode']:02d}: trades={d['trades']:.1f}, hold_rate={d['hold_rate']:.3f}, penalty={d['penalty']:.3f}")
            print(f"    Actions: HOLD={a['hold']}, LONG={a['long']}, SHORT={a['short']}, FLAT={a['flat']}")
else:
    print("\nWARNING: Old JSON format - run training to generate new format")
