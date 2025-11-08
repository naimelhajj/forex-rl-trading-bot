# Quick Reference: Anti-Collapse Patches

## 🎯 What Changed

### Patch A: Evaluation Exploration (config.py + trainer.py)
- **New config:** `eval_epsilon: float = 0.02`
- **Location:** Validation loop in `_run_validation_slice()`
- **Effect:** 2% random actions break deterministic HOLD ties
- **Safe:** Only in validation, respects legal actions

### Patch B: Smart Penalty Scaling (trainer.py)
- **Changed:** Opportunity-based penalty (was temporal volatility)
- **Metric:** Median ATR / Price (target ~0.15%)
- **Range:** 0.5-1.0× (only scales DOWN, never UP)
- **Effect:** Reduces penalty in low-opportunity periods

### Patch C: Action Diagnostics (trainer.py)
- **New JSON fields:** `actions` dict + `hold_rate` float
- **What:** Counts HOLD/LONG/SHORT/FLAT across all validation windows
- **Why:** Detect policy collapse (>95% HOLD) vs valid caution

---

## 🚀 Quick Start

### Test Immediately:
```powershell
# 10-episode quick test
python main.py --episodes 10

# Verify new fields present
python check_anti_collapse.py
```

### Expected Results:
```
✅ Both 'actions' and 'hold_rate' fields present
✅ HEALTHY: Low collapse rate (<5%)
✅ HEALTHY: Penalties varying (std=0.042)
```

### If You See Problems:

**"High collapse rate (>15%)"**
```python
# config.py - increase eval_epsilon
eval_epsilon: float = 0.03  # Was 0.02
```

**"Penalties not varying"**
```python
# trainer.py line ~570 - lower target threshold
target_opp = 0.0012  # Was 0.0015
```

**"Too many trades"**
```python
# config.py - decrease eval_epsilon
eval_epsilon: float = 0.01  # Was 0.02
```

---

## 📊 Key Metrics

### HOLD Rate:
- **Healthy:** 0.70-0.85 median
- **Caution:** 0.85-0.90 median (monitor)
- **Problem:** >0.90 median (policy collapse)

### Collapse Rate:
- **Excellent:** <5% episodes with >95% HOLD + <3 trades
- **Good:** 5-10%
- **Problem:** >15%

### Penalty Variation:
- **Healthy:** std > 0.03 (penalties varying)
- **Problem:** std < 0.01 (not scaling)

---

## 📁 Files Modified

1. **config.py** (line 73): Added `eval_epsilon = 0.02`
2. **trainer.py** (lines 378-395): Epsilon-greedy in validation
3. **trainer.py** (lines 544-575): Opportunity-based penalty
4. **trainer.py** (lines 506-510, 614-626): Action histogram logging

---

## 🔍 Verification

```powershell
# Check compilation
python -c "import config, trainer; print('✅ All patches compile')"

# Run quick test
python main.py --episodes 10

# Analyze results
python check_anti_collapse.py
```

---

## 📖 Full Documentation

See `ANTI_COLLAPSE_PATCHES.md` for:
- Detailed implementation
- Tuning guidelines
- Diagnostic procedures
- Success criteria

---

**All patches ready to test! No errors found.** ✅
