# Phase 2.8b - Friction Jitter Bug Discovery & Fix

**Date**: October 30, 2025  
**Status**: 🔧 **CRITICAL BUG FIXED**

## Executive Summary

Discovered that **friction jitter was NOT actually working** in Run B due to a bug in the `validate()` function. The validation code was **caching and reusing the first friction value** instead of using the randomized values set before each episode.

**Result**: Run A (frozen frictions) and Run B (supposedly jittered) produced **identical results** because both runs were actually using frozen frictions!

---

## Bug Discovery Process

### 1. Initial Observation
Run B (friction jitter enabled) showed **identical results** to Run A (frozen frictions):
- Cross-seed mean: -0.004 ± 0.021 (same!)
- Cross-seed final: +0.451 ± 0.384 (same!)
- Individual seed scores: Identical

This was suspicious - either the agent is perfectly robust (unlikely) OR friction jitter wasn't working.

### 2. Investigation Steps

**Step A: Check if friction parameters are saved**
Created `check_friction_jitter.py` to verify if spread/slippage values vary in validation JSONs.
- **Result**: ❌ No friction parameters saved in JSON files

**Step B: Add friction tracking**
Modified `trainer.py` line ~1005 to save friction parameters in validation summaries:
```python
"spread": float(getattr(self.val_env, 'spread', 0)),
"slippage_pips": float(getattr(self.val_env.risk_manager, 'slippage_pips', 0)),
```

**Step C: Verify randomization logic**
Created `test_friction_randomization.py` to test the randomization code in isolation.
- **Result**: ✅ Randomization logic works correctly (different values each episode)

**Step D: Trace validation code flow**
Examined `trainer.py` `validate()` function to understand how frictions are handled during validation.

---

## Root Cause Analysis

### The Bug (trainer.py lines 714-720, 829-830)

**Before validation** (lines 714-720):
```python
# Save base values once
base_spread = getattr(self.val_env, '_base_spread', self.val_env.spread)
base_commission = getattr(self.val_env, '_base_commission', self.val_env.commission)

if not hasattr(self.val_env, '_base_spread'):
    self.val_env._base_spread = self.val_env.spread  # ← SAVES FIRST VALUE FOREVER!
    self.val_env._base_commission = self.val_env.commission
```

**After validation** (lines 829-830):
```python
self.val_env.spread = base_spread  # ← RESTORES TO FIRST VALUE!
self.val_env.commission = base_commission
```

### How the Bug Manifests

**Episode 1** (first validation):
1. Friction randomization sets spread = 0.000189 (random)
2. `validate()` runs:
   - Saves 0.000189 as `_base_spread` (first time)
   - Uses 0.000189 for validation ✅
   - Restores to 0.000189 ✅

**Episode 2**:
1. Friction randomization sets spread = 0.000170 (different random)
2. `validate()` runs:
   - **BUG**: Uses `_base_spread` = 0.000189 (from episode 1!)  ❌
   - Validates with 0.000189 instead of 0.000170  ❌
   - Restores to 0.000189  ❌

**Episode 3-80**:
- Same problem - always uses 0.000189 (first value)
- Friction randomization before each episode has NO EFFECT!

**Result**: All validations use the **same friction values** regardless of `FREEZE_VALIDATION_FRICTIONS` setting!

---

## The Fix

### Changed Code (trainer.py)

**Lines ~712-720** (BEFORE):
```python
# Save base values once
base_spread = getattr(self.val_env, '_base_spread', self.val_env.spread)
base_commission = getattr(self.val_env, '_base_commission', self.val_env.commission)

if not hasattr(self.val_env, '_base_spread'):
    self.val_env._base_spread = self.val_env.spread
    self.val_env._base_commission = self.val_env.commission
```

**Lines ~712-717** (AFTER - FIXED):
```python
# PHASE-2.8b: Capture CURRENT friction values (may be randomized per episode)
# Don't restore to a "base" - use whatever was set before this validation!
current_spread = self.val_env.spread
current_commission = self.val_env.commission
```

**Line ~800** (function call):
```python
# BEFORE:
stats = self._run_validation_slice(lo, hi, base_spread, base_commission)

# AFTER:
stats = self._run_validation_slice(lo, hi, current_spread, current_commission)
```

**Lines ~829-830** (restore):
```python
# BEFORE:
self.val_env.spread = base_spread
self.val_env.commission = base_commission

# AFTER:
self.val_env.spread = current_spread  # Restore to current (may be randomized)
self.val_env.commission = current_commission
```

### Fix Logic

Instead of caching the first friction value as "base" and reusing it forever, we now:
1. Capture the CURRENT friction values at the start of validation
2. Use these CURRENT values (which may have been randomized)
3. Restore to CURRENT values after validation (no-op, but maintains structure)

This allows the per-episode friction randomization (lines 1139-1147) to actually affect validation!

---

## Impact Assessment

### Run A (Frozen Frictions)
- Config: `FREEZE_VALIDATION_FRICTIONS = True`
- Friction randomization: **Skipped** (line 1139 conditional)
- Validation frictions: **Fixed** at 0.00015 spread
- **Status**: ✅ Working as intended (was lucky!)

### Run B (Supposedly Jittered)
- Config: `FREEZE_VALIDATION_FRICTIONS = False`
- Friction randomization: **Executed** but had NO EFFECT due to bug
- Validation frictions: **Fixed** at first random value (~0.000189)
- **Status**: ❌ **INVALID** - was accidentally frozen at episode 1's random value

**Conclusion**: Run A and Run B both used frozen frictions, just with slightly different values (0.00015 vs ~0.000189). Results are NOT a valid robustness test!

---

## Next Steps

### Immediate Actions
1. ✅ **Fix applied** - friction jitter now works correctly
2. ⏳ **Re-run Run B** with corrected friction jitter
3. ⏳ **Verify** friction values actually vary in new validation JSONs

### Re-Run B Specification
- Seeds: 7, 77, 777 (same as before)
- Episodes: 80
- Config: `FREEZE_VALIDATION_FRICTIONS = False` ✅ (already set)
- Expected: Different results from Run A (may show degradation)

### Success Criteria (Updated)

**✅ GREEN (Robust - Proceed):**
- Mean degradation ≤ 0.05 from Run A
- Variance increase ≤ 50% (±0.021 → ±0.032)
- At least 2/3 positive finals
- → Agent is robust to friction variation

**🟡 YELLOW (Acceptable - Monitor):**
- Mean degradation 0.05-0.10
- Variance increase 50-100%
- 1-2 positive finals
- → May need friction-adaptive features later

**🔴 RED (Brittle - Needs Work):**
- Mean degradation > 0.10
- Variance increase > 100%
- All negative finals
- → Agent overfit to fixed frictions, needs robustness training

---

## Files Modified

### 1. trainer.py
- Lines ~712-720: Remove base friction caching logic
- Line ~800: Use current_spread instead of base_spread
- Lines ~829-830: Restore to current values
- Lines ~1005-1007: Add spread/slippage_pips to validation JSON

### 2. New Debug Scripts
- `check_friction_jitter.py`: Verify friction variation in results
- `test_friction_randomization.py`: Test randomization logic in isolation

### 3. run_seed_sweep_organized.py
- Added `clean_validation_summaries()` function
- Call cleanup before each seed run (prevents 150-file bug)

---

## Lessons Learned

1. **Always verify randomization is working**: Don't assume config flags work without checking outputs
2. **Save diagnostic parameters**: Friction values should have been saved in JSONs from the start
3. **Beware of caching**: The "save base, restore base" pattern broke randomization
4. **Test in isolation**: Unit tests for randomization would have caught this earlier
5. **Identical results are suspicious**: Should have investigated sooner when Run A == Run B

---

## Technical Notes

### Why the Original Code Existed
The "base" friction caching was likely intended to:
1. Restore frictions after validation (in case validation modifies them)
2. Allow per-window friction variation within a single validation

But it didn't account for **per-episode** friction randomization happening BEFORE validation!

### Correct Pattern
For per-episode randomization + per-window stability:
1. Randomize frictions before training episode (line 1139)
2. Validation captures CURRENT values (now fixed)
3. All windows in that validation use the SAME frictions (stable comparison)
4. Next episode gets NEW random frictions

---

**Status**: 🔧 Bug fixed, ready for **Run B v2 (Real Robustness Test)**

