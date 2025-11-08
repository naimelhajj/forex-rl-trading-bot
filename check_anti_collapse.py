"""
Quick verification of anti-collapse patches
Checks validation JSONs for new fields and policy collapse indicators
"""

import json
from pathlib import Path
import numpy as np

def analyze_validation_jsons():
    """Analyze validation JSONs for anti-collapse patch effects"""
    
    json_dir = Path("logs/validation_summaries")
    if not json_dir.exists():
        print(f"[ERROR] No validation summaries found at {json_dir}")
        return
    
    jsons = sorted(json_dir.glob("val_ep*.json"))
    if not jsons:
        print("[ERROR] No JSON files found")
        return
    
    print("="*60)
    print("ANTI-COLLAPSE PATCH VERIFICATION")
    print("="*60)
    
    # Load all data
    data = []
    for f in jsons:
        try:
            data.append(json.load(open(f)))
        except Exception as e:
            print(f"[WARN] Could not load {f.name}: {e}")
    
    if not data:
        print("[ERROR] No valid JSON data loaded")
        return
    
    print(f"\n[OK] Loaded {len(data)} validation summaries\n")
    
    # Check for new fields (Patch C)
    print("="*60)
    print("PATCH C VERIFICATION: Action Histogram Fields")
    print("="*60)
    first = data[0]
    has_actions = 'actions' in first
    has_hold_rate = 'hold_rate' in first
    
    if has_actions and has_hold_rate:
        print("✅ Both 'actions' and 'hold_rate' fields present")
        print(f"\nSample from {jsons[0].name}:")
        print(f"  actions: {first['actions']}")
        print(f"  hold_rate: {first['hold_rate']:.3f}")
    else:
        print("❌ Missing fields:")
        if not has_actions:
            print("  - 'actions' field not found")
        if not has_hold_rate:
            print("  - 'hold_rate' field not found")
    
    # Analyze HOLD rates
    if has_hold_rate:
        print("\n" + "="*60)
        print("HOLD RATE ANALYSIS (Policy Collapse Detection)")
        print("="*60)
        
        # Filter to only episodes with the new fields (handle mixed old/new JSONs)
        data_with_fields = [d for d in data if 'hold_rate' in d and 'actions' in d]
        if not data_with_fields:
            print("[WARN] No episodes have hold_rate field yet - run training first")
            return
        
        if len(data_with_fields) < len(data):
            print(f"[INFO] Analyzing {len(data_with_fields)}/{len(data)} episodes with new fields\n")
        
        hold_rates = [d['hold_rate'] for d in data_with_fields]
        trades = [d['trades'] for d in data_with_fields]
        
        print(f"\nHOLD Rate Statistics:")
        print(f"  Median: {np.median(hold_rates):.3f}")
        print(f"  Mean:   {np.mean(hold_rates):.3f}")
        print(f"  Std:    {np.std(hold_rates):.3f}")
        print(f"  Range:  {min(hold_rates):.3f} - {max(hold_rates):.3f}")
        
        # Policy collapse detection (>95% HOLD + <3 trades)
        collapsed = sum(1 for d in data_with_fields 
                       if d['hold_rate'] > 0.95 and d['trades'] < 3)
        collapse_rate = collapsed / len(data_with_fields) * 100
        
        print(f"\nPolicy Collapse Detection:")
        print(f"  Episodes with >95% HOLD and <3 trades: {collapsed}/{len(data_with_fields)} ({collapse_rate:.1f}%)")
        
        if collapse_rate > 15:
            print("  ⚠️  WARNING: High collapse rate (>15%)")
            print("     Consider increasing eval_epsilon to 0.03")
        elif collapse_rate > 5:
            print("  ⚠️  CAUTION: Moderate collapse rate (5-15%)")
            print("     Monitor closely, may need tuning")
        else:
            print("  ✅ HEALTHY: Low collapse rate (<5%)")
        
        # Group by HOLD rate
        print(f"\nEpisode Distribution by HOLD Rate:")
        high_hold = [d for d in data_with_fields if d['hold_rate'] > 0.90]
        mid_hold = [d for d in data_with_fields if 0.70 <= d['hold_rate'] <= 0.90]
        low_hold = [d for d in data_with_fields if d['hold_rate'] < 0.70]
        
        print(f"  High HOLD (>90%):  {len(high_hold):2d} episodes", end="")
        if high_hold:
            avg_trades = np.mean([d['trades'] for d in high_hold])
            print(f" (avg {avg_trades:.1f} trades)")
        else:
            print()
        
        print(f"  Mid HOLD (70-90%): {len(mid_hold):2d} episodes", end="")
        if mid_hold:
            avg_trades = np.mean([d['trades'] for d in mid_hold])
            print(f" (avg {avg_trades:.1f} trades)")
        else:
            print()
        
        print(f"  Low HOLD (<70%):   {len(low_hold):2d} episodes", end="")
        if low_hold:
            avg_trades = np.mean([d['trades'] for d in low_hold])
            print(f" (avg {avg_trades:.1f} trades)")
        else:
            print()
    
    # Analyze opportunity-scaled penalties (Patch B)
    print("\n" + "="*60)
    print("PATCH B VERIFICATION: Opportunity-Scaled Penalties")
    print("="*60)
    
    penalties = [d['penalty'] for d in data if d['penalty'] > 0]
    if penalties:
        print(f"\nPenalty Statistics (when applied):")
        print(f"  Mean:   {np.mean(penalties):.3f}")
        print(f"  Median: {np.median(penalties):.3f}")
        print(f"  Range:  {min(penalties):.3f} - {max(penalties):.3f}")
        
        # Check if penalties vary (not always 0.25)
        penalty_std = np.std(penalties)
        if penalty_std < 0.01:
            print(f"  ⚠️  WARNING: Very low variation (std={penalty_std:.4f})")
            print("     Penalties may not be scaling properly")
        else:
            print(f"  ✅ HEALTHY: Penalties varying (std={penalty_std:.3f})")
        
        # Show episodes with reduced penalties
        reduced = [d for d in data if 0 < d['penalty'] < 0.20]
        if reduced:
            print(f"\n  Episodes with reduced penalty (<0.20): {len(reduced)}/{len([d for d in data if d['penalty'] > 0])}")
            print(f"    Avg trades in reduced-penalty eps: {np.mean([d['trades'] for d in reduced]):.1f}")
        else:
            print(f"\n  ⚠️  No episodes with reduced penalty (<0.20)")
            print("     Opportunity scaling may not be active")
    else:
        print("\n✅ No penalties applied (all validations met trade threshold)")
    
    # Show sample episodes
    print("\n" + "="*60)
    print("SAMPLE EPISODES (First 5 with new fields)")
    print("="*60)
    
    # Get episodes with new fields for sampling
    data_with_fields = [d for d in data if 'hold_rate' in d and 'actions' in d]
    sample_data = data_with_fields[:5] if data_with_fields else data[:5]
    
    for d in sample_data:
        if has_actions and has_hold_rate and 'actions' in d:
            a = d['actions']
            total = sum(a.values())
            non_hold_pct = (total - a['hold']) / max(1, total) * 100
            print(f"\nEpisode {d['episode']:02d}:")
            print(f"  Trades: {d['trades']:.1f} | HOLD rate: {d['hold_rate']:.3f} | Penalty: {d['penalty']:.3f}")
            print(f"  Actions: HOLD={a['hold']}, LONG={a['long']}, SHORT={a['short']}, FLAT={a['flat']}")
            print(f"  Non-HOLD: {non_hold_pct:.1f}% (expect ~2% from eval_epsilon)")
        else:
            print(f"\nEpisode {d['episode']:02d}:")
            print(f"  Trades: {d['trades']:.1f} | Penalty: {d['penalty']:.3f}")
            print(f"  [Old format - no action histogram]")
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    analyze_validation_jsons()
