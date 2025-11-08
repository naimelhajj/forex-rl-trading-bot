"""
Quick test to verify metrics add-on integration.
Run this to confirm the helper function works correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the helper function
from trainer import compute_policy_metrics

def test_basic():
    """Test basic functionality."""
    # Simulate action sequence: mostly HOLD with some trades
    actions = [0]*50 + [1]*5 + [0]*20 + [2]*5 + [0]*20  # 50 HOLD, 5 LONG, 20 HOLD, 5 SHORT, 20 HOLD
    
    metrics = compute_policy_metrics(actions)
    
    print("Testing metrics computation...")
    print(f"  Total actions: {len(actions)}")
    print(f"  Action counts: {metrics['actions']}")
    print(f"  Hold rate: {metrics['hold_rate']:.3f}")
    print(f"  Entropy: {metrics['action_entropy_bits']:.3f} bits")
    print(f"  Max hold streak: {metrics['hold_streak_max']}")
    print(f"  Avg hold length: {metrics['avg_hold_length']:.2f}")
    print(f"  Switch rate: {metrics['switch_rate']:.3f}")
    print(f"  Long/Short: {metrics['long_short']}")
    
    # Verify calculations
    assert metrics['actions']['HOLD'] == 90, "HOLD count mismatch"
    assert metrics['actions']['LONG'] == 5, "LONG count mismatch"
    assert metrics['actions']['SHORT'] == 5, "SHORT count mismatch"
    assert 0.85 < metrics['hold_rate'] < 0.95, "Hold rate out of range"
    assert metrics['hold_streak_max'] == 50, f"Max streak should be 50, got {metrics['hold_streak_max']}"
    assert metrics['long_short']['long'] == 5, "Long count mismatch"
    assert metrics['long_short']['short'] == 5, "Short count mismatch"
    assert abs(metrics['long_short']['long_ratio'] - 0.5) < 0.01, "Long ratio should be 0.5"
    
    print("\n✅ All assertions passed!")

def test_edge_cases():
    """Test edge cases."""
    print("\nTesting edge cases...")
    
    # Empty sequence
    metrics = compute_policy_metrics([])
    assert metrics['hold_rate'] == 0.0
    assert metrics['action_entropy_bits'] == 0.0
    print("  ✅ Empty sequence handled")
    
    # All HOLD
    metrics = compute_policy_metrics([0]*100)
    assert metrics['hold_rate'] == 1.0
    assert metrics['hold_streak_max'] == 100
    assert metrics['switch_rate'] == 0.0
    print("  ✅ All HOLD handled")
    
    # No HOLD (alternating)
    metrics = compute_policy_metrics([1, 2, 1, 2, 1, 2])
    assert metrics['hold_rate'] == 0.0
    assert metrics['switch_rate'] == 1.0  # Every step switches
    print("  ✅ No HOLD handled")
    
    # Balanced distribution
    metrics = compute_policy_metrics([0]*25 + [1]*25 + [2]*25 + [3]*25)
    assert abs(metrics['action_entropy_bits'] - 2.0) < 0.01, "Balanced should be 2.0 bits"
    print("  ✅ Balanced distribution handled")
    
    print("\n✅ All edge cases passed!")

def test_realistic():
    """Test with realistic validation-like sequence."""
    print("\nTesting realistic sequence...")
    
    # Simulate ~600 bars (10 hours) with realistic policy
    # 70% HOLD, 15% LONG, 15% SHORT, occasional switches
    import random
    random.seed(42)
    
    actions = []
    current_action = 0  # Start with HOLD
    for _ in range(600):
        # Hold for a while
        hold_duration = random.randint(8, 25)
        actions.extend([0] * hold_duration)
        
        # Then maybe trade
        if random.random() < 0.3:  # 30% chance of non-HOLD
            trade_action = random.choice([1, 2])  # LONG or SHORT
            trade_duration = random.randint(5, 15)
            actions.extend([trade_action] * trade_duration)
    
    # Trim to 600
    actions = actions[:600]
    
    metrics = compute_policy_metrics(actions)
    
    print(f"  Total steps: {len(actions)}")
    print(f"  Hold rate: {metrics['hold_rate']:.3f}")
    print(f"  Entropy: {metrics['action_entropy_bits']:.3f} bits")
    print(f"  Max hold streak: {metrics['hold_streak_max']}")
    print(f"  Avg hold length: {metrics['avg_hold_length']:.2f}")
    print(f"  Switch rate: {metrics['switch_rate']:.3f}")
    print(f"  Long ratio: {metrics['long_short']['long_ratio']:.3f}")
    print(f"  Short ratio: {metrics['long_short']['short_ratio']:.3f}")
    
    # Sanity checks
    assert 0.5 < metrics['hold_rate'] < 0.9, "Hold rate should be realistic"
    assert 0.5 < metrics['action_entropy_bits'] < 2.5, "Entropy should be reasonable (0.5-2.5 bits)"
    assert metrics['hold_streak_max'] < 150, "Max streak should be reasonable"
    assert 0.01 < metrics['switch_rate'] < 0.5, "Switch rate should be realistic"
    
    print("\n✅ Realistic sequence passed!")

if __name__ == "__main__":
    print("=" * 60)
    print("METRICS ADD-ON VERIFICATION TEST")
    print("=" * 60)
    
    try:
        test_basic()
        test_edge_cases()
        test_realistic()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - METRICS ADD-ON WORKING!")
        print("=" * 60)
        print("\nReady to use! Run training and check with:")
        print("  python check_metrics_addon.py")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
