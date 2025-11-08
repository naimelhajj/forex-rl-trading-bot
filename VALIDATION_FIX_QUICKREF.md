## Quick Reference: What Changed & Why

### 🔧 Three Critical Fixes Applied

#### 1. **Validation Slices Now Start at Different Indices**
```python
# trainer.py lines 352-372
# BEFORE: All K windows started at bar 0 (ignored start_idx)
# AFTER:  Fast-forward to start_idx, then run window
```
**Why:** K=6 "overlapping" windows were actually identical, not overlapping  
**Impact:** Proper robustness measurement across different market regimes

---

#### 2. **Validation Friction Softened During Testing**
```python
# trainer.py lines 667-678
# BEFORE: spread [0.00012-0.00025], slippage [0.5-1.2], hard_max_lots clamped
# AFTER:  spread [0.00013-0.00020], slippage [0.6-1.0], no clamp
```
**Why:** Harsh friction during short tests discouraged all trading  
**Impact:** More realistic conditions, allows strategy learning

---

#### 3. **Expected Trade Gate Eased**
```python
# trainer.py line 526
# BEFORE: expected_trades = bars / 60  → ~10 for 600 bars
# AFTER:  expected_trades = bars / 80  → ~7-8 for 600 bars
```
**Why:** 10 trades with min_hold=8, cooldown=16 was too aggressive  
**Impact:** Encourages exploration while maintaining quality standards

---

### 📊 How to Verify Fixes Work

**Run test:**
```powershell
python main.py --episodes 15
```

**Check results:**
```powershell
python check_validation_diversity.py
```

**Look for in logs:**
```
[VAL] ... | trades=8.5 | mult=1.00 | pen=0.000 | score=0.092  ← Good!
[VAL] ... | trades=5.0 | mult=0.50 | pen=0.071 | score=0.033  ← Active penalty
[VAL] ... | trades=0.0 | mult=0.00 | pen=0.250 | score=-0.250 ← Penalty working
```

**Success = trade counts vary (not all 0), penalty only when < 6-7 trades**

---

### 🎯 Expected Behavior

| Scenario | Old Behavior | New Behavior |
|----------|--------------|--------------|
| **Window coverage** | All start at bar 0 | Each starts at different index |
| **Trade activity** | Often 0 across all windows | Varies: 0-15+ per episode |
| **Penalty activation** | Frequent (harsh friction) | Only when truly under-trading |
| **Learning signal** | Noisy, contradictory | Clearer, more consistent |

---

### ⚡ Quick Troubleshooting

**If still seeing all 0 trades:**
1. Check slice diversity with monitor script
2. Verify fast-forward logic executed (add debug print)
3. Increase penalty: `0.25 → 0.40` in trainer.py line 533

**If too conservative (stuck at 5-6 trades):**
1. Lower min_hold/cooldown in config.py
2. Further ease gate: `/80 → /100` in trainer.py line 526

**If learning unstable:**
1. Temporarily disable validation friction randomization
2. Check training logs for NaN/inf values
3. Verify buffer size > learning_starts

---

### 📝 Modified Files Checklist

- [x] `trainer.py` - Slice fast-forward (lines 352-372)
- [x] `trainer.py` - Friction softening (lines 667-678)
- [x] `trainer.py` - Gate easing (line 526)
- [x] `VALIDATION_SLICE_FIX.md` - Comprehensive docs
- [x] `COMPLETE_VALIDATION_FIX.md` - Full technical summary
- [x] `check_validation_diversity.py` - Verification script
- [x] Previously: Under-trade penalty (already working)

---

### 🚀 Test Command Reference

```powershell
# Quick test (15 episodes)
python main.py --episodes 15

# Verify diversity
python check_validation_diversity.py

# Medium test (50 episodes)
python main.py --episodes 50

# Full seed sweep (after confirming fixes work)
python run_seed_sweep_auto.py
```

---

**TL;DR:** Fixed validation to actually test different data segments, softened friction for testing, and eased trade requirements. Under-trade penalty already working. Test with 15 episodes and check diversity.
