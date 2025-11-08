# Phase 2.7 Quick Reference

## What Changed

### 1. ✅ Hold-Out Validation (Generalization Test)
**File:** `trainer.py` (lines ~1367-1420)
- Runs **alternate validation regime** after post-restore
- Wider stride (120 bars vs 90), shifted start (+50 bars)
- Saves to `logs/validation_summaries/val_final_alt.json`
- Console: `[POST-RESTORE:ALT] windows=X | SPR=Y.YYY ...`

### 2. ✅ Trade Pacing Stress Test
**File:** `config.py` (line ~57)
- `max_trades_per_episode: 100` (was 120, -17%)
- Tests if profitability from quality vs quantity

### 3. ✅ Stricter SPR Bounds
**File:** `config.py` (FitnessConfig, lines ~100-102)
- `spr_pf_cap: 6.0` (was 10.0) - reduce outlier influence
- `spr_dd_floor_pct: 1.0` (was 0.05) - prevent tiny DD inflation
- Expect tighter score distribution, fewer +1.5+ spikes

---

## Run Commands

### Test 1: With All Improvements
```powershell
python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 120
```

### Analysis After Run
```powershell
python compare_seed_results.py
python check_validation_diversity.py
python check_metrics_addon.py

# Check hold-out validation results
cat logs\validation_summaries\val_final_alt.json
```

---

## What to Look For

### 🟢 Success Indicators:
1. **Primary SPR:** Cross-seed mean ≥ +0.01 (maintained)
2. **Alt SPR:** Positive for seeds 7 & 777 (generalization!)
3. **Score distribution:** Tighter (fewer +1.5+ outliers)
4. **Penalty rate:** ≤ 8% (maintained or improved)
5. **Late-episode positives:** More +0.3 to +0.7 scores

### Console Output Example:
```
[POST-RESTORE] Final score: 0.543 (saved to ...)
[POST-RESTORE:ALT] Running hold-out validation (shifted/strided)...
[POST-RESTORE:ALT] windows=5 | SPR=0.324 | PF=2.15 | MDD=3.42% | 
                   MMR=1.23% | TPY=18.5 | SIG=0.78
```

### 🟡 Watch For:
- Alt SPR negative → May be overfitting to 600/90 regime
- Penalty rate >15% → Trade ceiling too tight
- Mean SPR drops >50% → Bounds too strict

---

## Files to Check After Run

1. `logs/validation_summaries/val_final.json` - Primary validation
2. `logs/validation_summaries/val_final_alt.json` - **NEW: Alt hold-out**
3. `results/seed_comparison_*.json` - Cross-seed analysis

---

## Reverting Changes (if needed)

### If Alt Validation Shows Overfitting:
*No revert needed* - add more validation diversity instead

### If Trade Pacing Too Tight (penalty >15%):
```python
# config.py line ~57
max_trades_per_episode: int = 110  # Relax slightly
```

### If SPR Bounds Too Strict (mean drops >50%):
```python
# config.py FitnessConfig
spr_pf_cap: float = 8.0        # Relax from 6.0
spr_dd_floor_pct: float = 0.5  # Relax from 1.0
```

---

## Expected Runtime
- **120 episodes × 3 seeds:** ~18-20 hours
- **Analysis:** ~30 minutes
- **Total:** ~1 day per test cycle

---

## Next Actions

**Immediate:**
1. Launch 120-episode run
2. Monitor console for `[POST-RESTORE:ALT]` output
3. Wait for completion (~18-20 hours)

**After Completion:**
1. Check alt validation scores (positive = good!)
2. Compare score distribution (tighter = good!)
3. Verify penalty rate ≤ 8%
4. Decide: success → extend to 200 episodes, or adjust → rerun

---

**Key Achievement Target:** 
Moving from **"it learns"** → **"it generalizes and scales"** 🎯
