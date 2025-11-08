# Validation Bugfixes - Quick Reference

## What Changed (2025-10-25)

### 🐛 Bug 1: Wrong Config Section
**Problem:** Validation read eval params from `config.training` (wrong!) instead of `config.agent`

**Your tuned values were IGNORED:**
- ❌ `eval_epsilon: 0.05` → was using `0.0`
- ❌ `hold_break_after: 7` → was using `20`
- ❌ `eval_tie_tau: 0.05` → was using `0.03`

**Fixed:** `trainer.py` line ~517
```python
agent_cfg = getattr(self.config, 'agent', None)
eval_epsilon = getattr(agent_cfg, 'eval_epsilon', 0.05)  # NOW CORRECT
hold_break_after = getattr(agent_cfg, 'hold_break_after', 7)  # NOW CORRECT
```

---

### 🐛 Bug 2: Hidden Validation Noise
**Problem:** Spread/slippage randomized EVERY episode (±30-40% swings!)

**Impact:**
- ✗ Non-stationary validation signal
- ✗ Cross-seed comparisons invalid
- ✗ Seed 77's 16.7% penalty rate likely from unlucky draws

**Fixed:** `config.py` + `trainer.py`
```python
# config.py line ~18
FREEZE_VALIDATION_FRICTIONS: bool = True  # Freeze spread/slippage

# trainer.py line ~1133 (wrapped randomization)
if self.val_env is not None and not getattr(self.config, 'FREEZE_VALIDATION_FRICTIONS', False):
    # Only randomize if flag is False
```

---

## Expected Improvements

**Before Fixes (120×3 run):**
- Seed 77 penalty rate: **16.7%** ⚠️
- Score variance: **High** (hidden friction noise)
- Hold recovery: **Slow** (20-bar breaker)

**After Fixes (expected):**
- Seed 77 penalty rate: **~3-5%** ✅
- Score variance: **Lower** (no noise) ✅
- Hold recovery: **Fast** (7-bar breaker) ✅
- Cross-seed consistency: **Tight** (same conditions) ✅

---

## Testing Commands

### Smoke Test (30 episodes, ~90 min):
```powershell
python run_seed_sweep_organized.py --seeds 7 --episodes 30
python check_validation_diversity.py
```

### Production Run (150 episodes × 5 seeds, ~30-35 hours):
```powershell
python run_seed_sweep_organized.py --seeds 7 17 27 77 777 --episodes 150
python compare_seed_results.py
python check_validation_diversity.py
python check_metrics_addon.py
```

---

## What to Check

### ✅ Success Indicators:
1. Validation logs show `eval_epsilon=0.05`, `hold_break_after=7`
2. Seed 77 penalty rate drops to ≤5%
3. Cross-seed StdDev ≤ 0.03 (tight clustering)
4. Mean SPR: +0.02 to +0.08 (all seeds)
5. Score distribution tighter (StdDev 0.10-0.15)

### ⚠️ Watch For:
- If seed 77 still >10% penalties → investigate hold-breaker
- If mean drops >30% → frozen frictions removed "lucky" draws (good!)
- If variance increases → may need to tune eval_tie_tau

---

## Files Modified

1. **config.py** (3 changes):
   - Line ~18: `FREEZE_VALIDATION_FRICTIONS = True`
   - Line ~192: `VAL_SPREAD_JITTER = (0.95, 1.05)` (was ±30%)
   - Line ~193: `VAL_COMMISSION_JITTER = (0.95, 1.05)` (was ±20%)

2. **trainer.py** (2 changes):
   - Line ~517: Use `config.agent` (not `config.training`)
   - Line ~1133: Freeze frictions (wrapped randomization)

---

## Next Steps

1. ⏳ **Let current 120×3 run finish** (serves as baseline)
2. ✅ **Smoke test** (30 episodes, seed 7)
3. 🚀 **Production run** (150 episodes × 5 seeds)

---

**Status:** ✅ Implemented and validated (syntax)
**Impact:** 🎯 High - Removes hidden noise, ensures config alignment
**Risk:** 🟢 Low - Conservative fixes

**Key Win:** Validation now reflects your tuned configuration! 🎉
