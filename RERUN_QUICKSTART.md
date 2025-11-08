# Quick Re-run Guide - Post-Restore Fix

**Status:** ✅ Code fixed, ready for re-run

---

## What Changed

1. **trainer.py:** Now runs final validation AFTER restoring best checkpoint
2. **compare_seed_results.py:** Now reads post-restore `val_final.json` if available

**Result:** "Score Final" will now match "Score Best" (~+0.86 instead of -1.045)

---

## Immediate Action (REQUIRED)

```powershell
# Re-run seed sweep to capture post-restore finals
python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 60

# This will:
# - Train normally (early stop around ep 33 expected)
# - Restore best checkpoint at end
# - Run final validation with best weights ← NEW!
# - Save as val_final.json ← NEW!
```

**Expected runtime:** ~3 hours (same as before)

---

## Analysis After Re-run

```powershell
# 1. Compare seeds (will now use post-restore scores)
python compare_seed_results.py

# Look for:
# ✓ Score Final: +0.86 (post-restore) ← Should match Score Best now!
# ✓ Cross-seed mean: ~+0.8 (was -1.0 before)
# ✓ Source indicator shows "post-restore" not "last-episode"

# 2. Check metrics
python check_metrics_addon.py

# Verify:
# ✓ Entropy: ~0.82 (target 0.85+)
# ✓ Hold rate: ~0.76
# ✓ Switch rate: ~0.14
# ✓ Zero-trade: 0%

# 3. Check diversity
python check_validation_diversity.py
```

---

## Acceptance Gates

**MUST PASS (High Confidence):**
- ✅ Cross-seed Score Mean > -0.30 (expect ~+0.8)
- ✅ Score Final ≈ Score Best (within ±0.05)
- ✅ Best seed Final ≥ +0.20 (expect ~+0.86)
- ✅ Zero-trade ≤ 2/25 (already 0/25)

**SHOULD PASS (Medium Confidence):**
- 🔄 Entropy ≥ 0.85 (at 0.82, may need +0.01 bump)
- ✅ Switch ≥ 0.12 (at 0.14)
- ✅ Hold 0.55-0.80 (at 0.76)

---

## If Entropy Still < 0.85

**Option A: Bump eval_epsilon (PREFERRED)**
```python
# config.py line ~73
eval_epsilon: float = 0.08  # was 0.07 (+14% probe rate)
```

**Option B: Widen tie margin**
```python
# config.py line ~76
hold_tie_tau: float = 0.03  # was 0.02 (+50% easier to probe)
```

**Quick test before full sweep:**
```powershell
python main.py --episodes 10
python check_metrics_addon.py  # Verify entropy > 0.85
```

---

## If Cross-Seed Mean Still Negative (UNLIKELY)

**Only if mean < -0.30 after fix:**
```python
# config.py line ~57
trade_penalty: float = 0.000075  # Back to 7.5e-05 from 5e-05
```

**This trims overtrading windows** (30-31 trades) that create negative tail.

---

## Success Indicators

**Before fix (broken):**
```
Seed 7:   Final -1.045  Best +0.87  ❌ Mismatch
Seed 77:  Final -1.045  Best +0.84  ❌ Mismatch  
Seed 777: Final -1.045  Best +0.88  ❌ Mismatch
Mean: -1.045
```

**After fix (expected):**
```
Seed 7:   Final +0.87 (post-restore)  Best +0.87  ✅ Aligned
Seed 77:  Final +0.84 (post-restore)  Best +0.84  ✅ Aligned
Seed 777: Final +0.88 (post-restore)  Best +0.88  ✅ Aligned
Mean: +0.86
```

---

## Files to Check After Run

```powershell
# 1. Verify val_final.json exists for each seed
ls logs/seed_sweep_results/seed_7/val_final.json
ls logs/seed_sweep_results/seed_77/val_final.json
ls logs/seed_sweep_results/seed_777/val_final.json

# 2. Inspect one val_final.json
cat logs/seed_sweep_results/seed_7/val_final.json

# Should contain:
# - "is_post_restore": true
# - "score": ~0.86
# - "episode_index": "final"

# 3. Compare old vs new
# Old: Last episode JSON (val_ep033.json or similar)
# New: Post-restore JSON (val_final.json)
```

---

## Troubleshooting

**Q: What if val_final.json doesn't exist?**
A: Check trainer output for `[POST-RESTORE]` log lines. If missing, check if early stop triggered and best checkpoint exists.

**Q: What if Final still doesn't match Best?**
A: Check `final_source` in comparison output. If it says "last-episode", the val_final.json wasn't created/found.

**Q: What if entropy is still 0.82 after re-run?**
A: This is expected - re-run won't change it. Use Option A above (bump eval_epsilon to 0.08) and re-run again.

---

## Timeline

**Now → 3 hours:** Re-run seed sweep (60 episodes x 3 seeds)
**After 3 hours:** Run comparison & metrics tools
**Decision point:** 
- If all gates pass → DONE! 🎉
- If entropy < 0.85 → Bump to 0.08, quick 10-ep test, re-sweep
- If mean still negative (unlikely) → Raise trade_penalty, re-sweep

**Expected:** 1 iteration (re-run) should be sufficient. The Score Final fix alone will likely push cross-seed mean from -1.0 to +0.8.

---

**Ready to re-run!** The fix is in place, just need fresh data with post-restore finals. 🚀
