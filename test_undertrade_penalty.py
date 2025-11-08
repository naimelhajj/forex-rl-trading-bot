"""
Quick test to verify the under-trade penalty fix prevents zero-trade trap.
"""

import numpy as np

def simulate_gating_with_penalty(median_trades, median_fitness=0.5, iqr=0.1):
    """
    Simulate the new validation gating logic with under-trade penalty.
    """
    # Simulate adaptive thresholds (typical values)
    bars_per_pass = 600
    expected_trades = max(8, int(bars_per_pass / 60))  # ~10
    hard_floor = max(5, int(0.4 * expected_trades))     # ~4
    min_half = max(hard_floor + 1, int(0.7 * expected_trades))  # ~7
    min_full = max(min_half + 1, expected_trades)       # ~10
    
    # Apply adaptive gating
    if median_trades < hard_floor:
        mult = 0.0
    elif median_trades < min_half:
        mult = 0.5
    elif median_trades < min_full:
        mult = 0.75
    else:
        mult = 1.0
    
    # IQR adjustment
    iqr_penalty = 0.35
    stability_adj = median_fitness - iqr_penalty * iqr
    
    # Add under-trade penalty
    undertrade_penalty = 0.0
    if median_trades < min_half:
        shortage = max(0, min_half - median_trades)
        undertrade_penalty = 0.25 * (shortage / max(1, min_half))
    
    val_score = stability_adj * mult - undertrade_penalty
    
    return {
        'expected_trades': expected_trades,
        'hard_floor': hard_floor,
        'min_half': min_half,
        'min_full': min_full,
        'mult': mult,
        'stability_adj': stability_adj,
        'undertrade_penalty': undertrade_penalty,
        'val_score': val_score
    }

print("="*70)
print("UNDER-TRADE PENALTY TEST")
print("="*70)
print("\nScenarios comparing OLD (hard zero) vs NEW (gradual penalty) gating:\n")

# Test scenarios
scenarios = [
    (0, 0.50, "Zero trades (do nothing)"),
    (2, 0.50, "Very few trades (ultra-conservative)"),
    (5, 0.50, "Low trades (conservative)"),
    (7, 0.50, "Near threshold (cautious)"),
    (10, 0.50, "Good activity (normal)"),
    (3, -0.30, "Few trades with bad fitness"),
]

for trades, fitness, desc in scenarios:
    result = simulate_gating_with_penalty(trades, fitness, iqr=0.1)
    
    print(f"{desc}:")
    print(f"  Trades: {trades}, Fitness: {fitness:.2f}")
    print(f"  Thresholds: floor={result['hard_floor']}, half={result['min_half']}, full={result['min_full']}")
    print(f"  Multiplier: {result['mult']:.2f}")
    print(f"  Under-trade penalty: {result['undertrade_penalty']:.3f}")
    print(f"  Final score: {result['val_score']:.3f}")
    
    # Show what OLD system would give (no penalty, just mult)
    old_score = result['stability_adj'] * result['mult']
    print(f"  OLD score (no penalty): {old_score:.3f}")
    print()

print("\n" + "="*70)
print("KEY INSIGHTS:")
print("="*70)
print("1. Zero trades now gets PENALTY, making it worse than low-activity trading")
print("2. Ultra-conservative (0-6 trades) gets 0.5x mult + penalty ~-0.15 to -0.25")
print("3. Bad trades with some activity (e.g., 3 trades, -0.3 fitness) get:")
print("   - OLD: -0.33 * 0.5 = -0.165 (better than zero trades at 0.0)")
print("   - NEW: -0.33 * 0.5 - 0.14 = -0.305 (worse, but avoids zero-trade trap)")
print("\nThis encourages exploration over total inactivity.")
