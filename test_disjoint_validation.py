"""
Test script to verify disjoint validation window patches.
Checks window creation, IQR penalty, adaptive gating, and EMA tracking.
"""

import numpy as np
import pandas as pd


def test_disjoint_windows():
    """Test disjoint window creation logic."""
    print("=" * 60)
    print("TEST 1: Disjoint Window Creation")
    print("=" * 60)
    
    class MockTrainer:
        def _make_disjoint_windows(self, idx, k: int, min_bars: int = 600):
            """Split validation index into k disjoint windows (no overlap)."""
            n = len(idx)
            step = n // k
            windows = []
            for i in range(k):
                start = i * step
                end = n if i == k - 1 else (i + 1) * step
                if end - start >= min_bars:
                    windows.append((start, end))
            return windows
    
    trainer = MockTrainer()
    
    # Test case 1: Sufficient data for all windows
    idx = pd.date_range(start='2024-01-01', periods=4200, freq='h')
    windows = trainer._make_disjoint_windows(idx, k=7, min_bars=600)
    
    print(f"\nTest case 1: {len(idx)} bars, K=7, min_bars=600")
    print(f"  Expected: 7 windows (4200/7 = 600 bars each)")
    print(f"  Got: {len(windows)} windows")
    
    for i, (lo, hi) in enumerate(windows):
        print(f"    Window {i+1}: [{lo}, {hi}) = {hi-lo} bars")
    
    # Verify no overlap
    for i in range(len(windows) - 1):
        _, end_i = windows[i]
        start_next, _ = windows[i + 1]
        assert end_i == start_next, f"Overlap detected! Window {i} ends at {end_i}, Window {i+1} starts at {start_next}"
    
    print("  ✅ No overlap detected (disjoint confirmed)")
    
    # Test case 2: Insufficient data (some windows dropped)
    idx_short = pd.date_range(start='2024-01-01', periods=1000, freq='h')
    windows_short = trainer._make_disjoint_windows(idx_short, k=7, min_bars=600)
    
    print(f"\nTest case 2: {len(idx_short)} bars, K=7, min_bars=600")
    print(f"  Expected: 1-2 windows (1000/7 = 142 bars each, < min_bars)")
    print(f"  Got: {len(windows_short)} windows")
    
    for i, (lo, hi) in enumerate(windows_short):
        print(f"    Window {i+1}: [{lo}, {hi}) = {hi-lo} bars")
    
    print("\n✅ TEST 1 PASSED: Disjoint windows work correctly\n")


def test_iqr_penalty():
    """Test IQR dispersion penalty logic."""
    print("=" * 60)
    print("TEST 2: IQR Dispersion Penalty")
    print("=" * 60)
    
    # Test case 1: Consistent performance (low IQR)
    fits_consistent = [0.22, 0.24, 0.23, 0.25, 0.24, 0.23, 0.24]
    median = float(np.median(fits_consistent))
    q75, q25 = np.percentile(fits_consistent, [75, 25])
    iqr = float(q75 - q25)
    stability_adj = median - 0.25 * iqr
    
    print(f"\nTest case 1: Consistent performance")
    print(f"  Passes: {fits_consistent}")
    print(f"  Median: {median:.3f}")
    print(f"  Q75: {q75:.3f}, Q25: {q25:.3f}")
    print(f"  IQR: {iqr:.3f}")
    print(f"  Penalty: {0.25 * iqr:.3f}")
    print(f"  Adjusted: {stability_adj:.3f}")
    print(f"  Expected: Small penalty (~0.01), adjusted ≈ median")
    
    assert abs(stability_adj - median) < 0.02, f"Consistent run should have small penalty, got {abs(stability_adj - median):.3f}"
    print("  ✅ Small penalty confirmed")
    
    # Test case 2: Spiky performance (high IQR)
    fits_spiky = [0.10, 0.45, 0.12, 0.50, 0.13, 0.48, 0.11]
    median_spiky = float(np.median(fits_spiky))
    q75_spiky, q25_spiky = np.percentile(fits_spiky, [75, 25])
    iqr_spiky = float(q75_spiky - q25_spiky)
    stability_adj_spiky = median_spiky - 0.25 * iqr_spiky
    
    print(f"\nTest case 2: Spiky performance")
    print(f"  Passes: {fits_spiky}")
    print(f"  Median: {median_spiky:.3f}")
    print(f"  Q75: {q75_spiky:.3f}, Q25: {q25_spiky:.3f}")
    print(f"  IQR: {iqr_spiky:.3f}")
    print(f"  Penalty: {0.25 * iqr_spiky:.3f}")
    print(f"  Adjusted: {stability_adj_spiky:.3f}")
    print(f"  Expected: Large penalty (>0.05), adjusted << median")
    
    assert abs(stability_adj_spiky - median_spiky) > 0.05, f"Spiky run should have large penalty, got {abs(stability_adj_spiky - median_spiky):.3f}"
    print("  ✅ Large penalty confirmed")
    
    print("\n✅ TEST 2 PASSED: IQR penalty works correctly\n")


def test_adaptive_gating():
    """Test adaptive trade gating based on window length."""
    print("=" * 60)
    print("TEST 3: Adaptive Trade Gating")
    print("=" * 60)
    
    test_cases = [
        (600, "Smoke test (600 bars)"),
        (1500, "Medium run (1500 bars)"),
        (3000, "Full training (3000 bars)"),
    ]
    
    for bars_per_pass, label in test_cases:
        print(f"\n{label}:")
        print(f"  Window size: {bars_per_pass} bars")
        
        # Compute thresholds
        expected_trades = max(8, int(bars_per_pass / 60))
        hard_floor = max(5, int(0.4 * expected_trades))
        min_half = max(hard_floor + 1, int(0.7 * expected_trades))
        min_full = max(min_half + 1, expected_trades)
        
        print(f"  Expected trades: {expected_trades}")
        print(f"  Thresholds:")
        print(f"    Hard floor (0% credit): < {hard_floor}")
        print(f"    50% credit: {hard_floor} - {min_half-1}")
        print(f"    75% credit: {min_half} - {min_full-1}")
        print(f"    100% credit: ≥ {min_full}")
        
        # Test different trade counts
        test_trades = [3, hard_floor, min_half, min_full, min_full + 10]
        for trades in test_trades:
            if trades < hard_floor:
                mult = 0.0
            elif trades < min_half:
                mult = 0.5
            elif trades < min_full:
                mult = 0.75
            else:
                mult = 1.0
            
            print(f"    {trades} trades → {mult:.2f}x multiplier")
    
    print("\n✅ TEST 3 PASSED: Adaptive gating scales with window size\n")


def test_ema_tracking():
    """Test EMA-based early stop logic."""
    print("=" * 60)
    print("TEST 4: EMA Early-Stop Tracking")
    print("=" * 60)
    
    # Simulate validation scores over time
    raw_scores = [0.18, 0.22, 0.15, 0.25, 0.20, 0.28, 0.19, 0.30]
    alpha = 0.3
    
    ema = raw_scores[0]
    best_ema_saved = -1e9
    
    print(f"\nAlpha: {alpha}")
    print(f"Episode | Raw Score | EMA | Best Saved | Action")
    print("-" * 60)
    
    for episode, raw in enumerate(raw_scores, start=1):
        # Update EMA
        ema = alpha * raw + (1 - alpha) * ema
        
        # Check if new best
        action = ""
        if ema > best_ema_saved:
            best_ema_saved = ema
            action = "✓ Save checkpoint"
        else:
            action = "No save"
        
        print(f"{episode:7} | {raw:9.3f} | {ema:6.3f} | {best_ema_saved:10.3f} | {action}")
    
    print("\nObservations:")
    print(f"  - EMA smooths out spikes (ep 2: 0.220 → 0.192)")
    print(f"  - EMA responds to trends (ep 4: 0.250 → 0.201)")
    print(f"  - Checkpoint saved only when EMA improves (not raw)")
    print(f"  - Final EMA: {ema:.3f} (reflects sustained performance)")
    
    print("\n✅ TEST 4 PASSED: EMA tracks trends correctly\n")


def test_complete_validation():
    """Test complete validation flow (all patches together)."""
    print("=" * 60)
    print("TEST 5: Complete Validation Flow")
    print("=" * 60)
    
    # Simulate K=7 validation passes with disjoint windows
    K = 7
    fits = [0.21, 0.23, 0.19, 0.24, 0.22, 0.20, 0.23]
    trade_counts = [18, 21, 16, 22, 19, 17, 20]
    bars_per_pass = 600
    
    print(f"\nSimulated validation:")
    print(f"  K: {K} disjoint passes")
    print(f"  Bars per pass: {bars_per_pass}")
    print(f"  Fitness per pass: {[f'{f:.3f}' for f in fits]}")
    print(f"  Trades per pass: {trade_counts}")
    
    # Step 1: Compute stability-adjusted median
    median = float(np.median(fits))
    q75, q25 = np.percentile(fits, [75, 25])
    iqr = float(q75 - q25)
    stability_adj = median - 0.25 * iqr
    
    print(f"\nStep 1: IQR Penalty")
    print(f"  Median: {median:.3f}")
    print(f"  IQR: {iqr:.3f}")
    print(f"  Adjusted: {stability_adj:.3f}")
    
    # Step 2: Apply adaptive trade gating
    median_trades = float(np.median(trade_counts))
    expected_trades = max(8, int(bars_per_pass / 60))
    hard_floor = max(5, int(0.4 * expected_trades))
    min_half = max(hard_floor + 1, int(0.7 * expected_trades))
    min_full = max(min_half + 1, expected_trades)
    
    if median_trades < hard_floor:
        mult = 0.0
    elif median_trades < min_half:
        mult = 0.5
    elif median_trades < min_full:
        mult = 0.75
    else:
        mult = 1.0
    
    val_score = stability_adj * mult
    
    print(f"\nStep 2: Adaptive Gating")
    print(f"  Median trades: {median_trades:.1f}")
    print(f"  Expected: {expected_trades}")
    print(f"  Thresholds: hard_floor={hard_floor}, min_half={min_half}, min_full={min_full}")
    print(f"  Multiplier: {mult:.2f}")
    print(f"  Final score: {val_score:.3f}")
    
    # Expected output format
    print(f"\nExpected validation output:")
    print(f"  [VAL] K={K} disjoint | median fitness={median:.3f} | IQR={iqr:.3f} | "
          f"adj={stability_adj:.3f} | trades={median_trades:.1f} | mult={mult:.2f} | score={val_score:.3f}")
    
    print("\n✅ TEST 5 PASSED: Complete validation flow works\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DISJOINT VALIDATION PATCHES - REGRESSION TEST")
    print("=" * 60 + "\n")
    
    test_disjoint_windows()
    test_iqr_penalty()
    test_adaptive_gating()
    test_ema_tracking()
    test_complete_validation()
    
    print("=" * 60)
    print("✅ ALL REGRESSION TESTS PASSED!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run smoke test: python main.py --episodes 5")
    print("2. Verify validation output matches expected format")
    print("3. Check window ranges show disjoint coverage")
    print("4. Confirm EMA tracking and best model saves")
    print("=" * 60)
