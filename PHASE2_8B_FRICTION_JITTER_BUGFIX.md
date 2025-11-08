# Phase 2.8b - Friction Jitter Bug Discovery & Fix

**Date:** 2025-10-30  
**Status:** ✅ FIXED - Ready for Run B v2

## 🔍 Bug Discovery

During investigation of why Run B (friction jitter enabled) produced **identical** results to Run A (frozen frictions), we discovered a critical bug in the `validate()` function.

### The Problem

**Expected behavior:**
- Run A: `FREEZE_VALIDATION_FRICTIONS = True` → Use fixed spread (0.00015)
- Run B: `FREEZE_VALIDATION_FRICTIONS = False` → Randomize spread each episode (0.00013-0.00020)

**Actual behavior:**
- Both runs produced identical results (cross-seed mean: -0.004 ± 0.021)

### Root Cause

**File:** `trainer.py`, `validate()` function  
**Lines:** 714-720, 829-830

```python
# Lines 714-720 (BUG: Caching first friction value)
base_spread = getattr(self.val_env, '_base_spread', self.val_env.spread)
base_commission = getattr(self.val_env, '_base_commission', self.val_env.commission)

if not hasattr(self.val_env, '_base_spread'):
    self.val_env._base_spread = self.val_env.spread  # ← Saves FIRST value forever!
    self.val_env._base_commission = self.val_env.commission

# Lines 829-830 (BUG: Always restores to cached value)
self.val_env.spread = base_spread  # ← Always restores to FIRST value!
self.val_env.commission = base_commission
```

**Execution flow:**
1. **Episode 1**: Friction randomization sets spread = 0.000189 (random)
2. **validate()**: Saves 0.000189 as `_base_spread` ✅
3. **Episode 2**: Friction randomization sets spread = 0.000170 (different)
4. **validate()**: Reads `_base_spread` = 0.000189 (IGNORES new value!) ❌
5. **Episodes 3-80**: Always uses 0.000189 (first random value, then frozen)

**Impact:**
- Run B appeared to have friction jitter enabled, but actually used a **fixed** spread (the first random value from episode 1)
- This is why Run A and Run B had identical results - both were effectively frozen!

## ✅ The Fix

**Changed:** `validate()` to capture CURRENT friction values instead of caching a "base"

### Before (BUGGY):
```python
# Save base values once (WRONG - caches forever!)
base_spread = getattr(self.val_env, '_base_spread', self.val_env.spread)
base_commission = getattr(self.val_env, '_base_commission', self.val_env.commission)

if not hasattr(self.val_env, '_base_spread'):
    self.val_env._base_spread = self.val_env.spread
    self.val_env._base_commission = self.val_env.commission

# ... validation logic ...

# Restore to cached base (WRONG - ignores per-episode randomization!)
self.val_env.spread = base_spread
self.val_env.commission = base_commission
```

### After (FIXED):
```python
# PHASE-2.8b: Capture CURRENT friction values (may be randomized per episode)
# Don't restore to a "base" - use whatever was set before this validation!
current_spread = self.val_env.spread
current_commission = self.val_env.commission

# ... validation logic ...

# Restore original values after all validation passes
# PHASE-2.8b: Restore to current values (which may be randomized)
self.val_env.spread = current_spread
self.val_env.commission = current_commission
```

## 🔧 Additional Improvements

### 1. Friction Parameter Tracking
Added spread/slippage to validation JSON summaries for verification:

```python
# trainer.py, line ~1006
"spread": float(getattr(self.val_env, 'spread', 0)),
"slippage_pips": float(getattr(self.val_env.risk_manager, 'slippage_pips', 0) 
                       if hasattr(self.val_env, 'risk_manager') else 0),
```

### 2. Script Cleanup Fix
Fixed `run_seed_sweep_organized.py` to clear old validation files before each seed:

```python
def clean_validation_summaries():
    """Clean validation_summaries directory before each run"""
    source_dir = Path("logs/validation_summaries")
    if source_dir.exists():
        for json_file in source_dir.glob("val_ep*.json"):
            json_file.unlink()
        print(f"[OK] Cleaned validation_summaries directory")

# Call before each seed run
for seed in seeds:
    clean_validation_summaries()  # ← Prevents 150-file bug
    update_config_seed(seed)
    # ... rest of loop
```

## 📊 Verification Tests

### Test 1: Friction Randomization Logic ✅
```python
# test_friction_randomization.py
FREEZE_VALIDATION_FRICTIONS = False
for episode in range(1, 11):
    s = float(np.random.uniform(0.00013, 0.00020))
    sp = float(np.random.uniform(0.6, 1.0))
    print(f"Episode {episode}: spread={s:.6f}, slippage={sp:.4f}")

# Result: 10 unique values - randomization works! ✅
```

### Test 2: Friction Parameter Storage ✅
```python
# check_friction_jitter.py
# Checked if spread/slippage saved in validation JSONs
# Result: NOT saved in old runs (expected) ✅
# New runs will include these fields ✅
```

## 🚀 Next Steps

### Run B v2 - Real Robustness Test
With the bug fixed, we can now run a **genuine** robustness test:

**Configuration:**
- Seeds: 7, 77, 777 (worst, mid, best from Run A)
- Episodes: 80
- Friction jitter: **ENABLED** (will actually work now!)
- Spread range: 0.00013 - 0.00020 (±33% from base 0.00015)
- Slippage range: 0.6 - 1.0 pips (±25% from base 0.8)

**Expected Outcomes:**

**✅ GREEN (Agent is robust):**
- Cross-seed mean: +0.02 to +0.04
- Degradation from Run A: ≤ 0.03
- Variance: ±0.025-0.035 (slightly higher than Run A)
- At least 2/3 positive finals

**🟡 YELLOW (Minor degradation, fine-tune):**
- Cross-seed mean: +0.01 to +0.02
- Degradation: 0.03-0.05
- Excessive penalties (>10%)
- **Action:** Reduce trade_penalty, increase cooldown

**🔴 RED (Not robust, rollback):**
- Cross-seed mean: < +0.01
- Degradation: > 0.05
- Multiple seed collapses
- **Action:** Revert to Phase 2.8 config

## 📝 Files Modified

1. **trainer.py** (lines 714-720, 829-830, 1006-1008)
   - Fixed friction caching bug
   - Added spread/slippage tracking

2. **run_seed_sweep_organized.py** (lines 56-76, 130-150)
   - Added `clean_validation_summaries()` function
   - Call cleanup before each seed run

3. **New verification scripts:**
   - `check_friction_jitter.py` - Verify friction tracking in JSONs
   - `test_friction_randomization.py` - Test randomization logic

## 🎯 Success Criteria

Run B v2 will be considered successful if:
1. Friction parameters actually vary across episodes (verify in JSONs)
2. Cross-seed mean ≥ +0.02 (slight degradation acceptable)
3. No catastrophic collapses (all seeds finish)
4. Variance remains stable (±0.025-0.040)

---

**Status:** Ready to start Run B v2! 🚀
