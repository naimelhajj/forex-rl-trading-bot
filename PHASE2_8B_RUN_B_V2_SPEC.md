# Phase 2.8b Run B v2 - REAL Robustness Test

**Date:** 2025-10-30  
**Status:** 🔄 RUNNING  
**Bug Fix:** ✅ Friction jitter now working correctly!

## 🎯 Test Objective

Test agent robustness with **actual** friction randomization (previous Run B had a bug that froze frictions).

## ⚙️ Configuration

**Seeds:** 7, 77, 777 (worst, mid, best from Run A)  
**Episodes:** 80 per seed  
**Duration:** ~4-5 hours (3 seeds × 80 episodes)

**Friction Randomization:**
- `FREEZE_VALIDATION_FRICTIONS = False` ✅
- Spread: 0.00013 - 0.00020 (±33% from base 0.00015)
- Slippage: 0.6 - 1.0 pips (±25% from base 0.8)
- **NEW:** Each episode gets different friction values!

**Phase 2.8b Config:**
- cooldown_bars: 11 (was 12 in Phase 2.8)
- trade_penalty: 0.000065 (was 0.00007)
- flip_penalty: 0.0007
- min_hold_bars: 6

## 📊 Baseline (Run A - Frozen Frictions)

| Metric | Value |
|--------|-------|
| Cross-seed mean | -0.004 ± 0.021 |
| Cross-seed final | +0.451 ± 0.384 |
| Positive finals | 4/5 seeds (80%) |
| Elite seeds | 27 (+0.907), 777 (+0.873) |
| Variance | ±0.021 (very stable!) |

## 🎯 Success Criteria

### ✅ GREEN (Excellent Robustness)
- Cross-seed mean: ≥ +0.02
- Degradation from Run A: ≤ 0.03
- Variance: ±0.025-0.035 (slight increase acceptable)
- Positive finals: ≥ 2/3 seeds (67%)
- **Decision:** Proceed to 200-episode confirmation!

### 🟡 YELLOW (Acceptable with Fine-Tuning)
- Cross-seed mean: +0.01 to +0.02
- Degradation: 0.03-0.05
- Excessive penalties: >10% of validations
- **Decision:** Adjust penalties, re-run
- **Tweaks:** 
  - trade_penalty: 0.000065 → 0.000070
  - cooldown_bars: 11 → 12

### 🔴 RED (Not Robust - Rollback)
- Cross-seed mean: < +0.01
- Degradation: > 0.05
- Multiple seed collapses or zero-trade episodes
- **Decision:** Revert to Phase 2.8 config

## 🔍 What to Check

1. **Friction variation in JSONs:**
   - Verify spread/slippage actually vary across episodes
   - Check with: `python check_friction_jitter.py`

2. **Performance degradation:**
   - Compare Run B v2 to Run A baseline
   - Acceptable degradation: ≤ 0.03 in mean SPR

3. **Stability:**
   - Cross-seed variance should be ±0.025-0.040
   - Higher than Run A is expected (friction jitter adds noise)

4. **Trade activity:**
   - Trades/ep: 24-32 (similar to Run A)
   - Penalty rate: <10% (watch seeds 7 & 77)

## 📈 Expected Timeline

```
[16:30] Seed 7 starts   → ~17:50 complete  (1h 20m)
[17:50] Seed 77 starts  → ~19:10 complete  (1h 20m)
[19:10] Seed 777 starts → ~20:30 complete  (1h 20m)
```

**Total:** ~4 hours

## 🔧 Monitoring Commands

```bash
# Check if still running
Get-Process python

# Check recent episode (seed 7 example)
Get-Content logs\training_history.json | python -m json.tool | Select-Object -Last 50

# Monitor validation summaries
Get-ChildItem logs\validation_summaries\val_ep*.json | Measure-Object

# Check friction variation (after completion)
python check_friction_jitter.py
```

## 📝 Key Differences from Buggy Run B

| Aspect | Buggy Run B | Run B v2 (Fixed) |
|--------|-------------|------------------|
| Friction jitter | ❌ Broken (cached first value) | ✅ Working (varies each episode) |
| Spread values | Fixed ~0.000189 | Random 0.00013-0.00020 |
| Results | Identical to Run A | **TBD** (real test!) |
| Validation JSONs | No spread/slippage fields | ✅ Tracking added |

## 🎯 Post-Run Analysis

After completion, run:

```bash
# 1. Verify friction jitter worked
python check_friction_jitter.py

# 2. Compare to Run A baseline  
python compare_seed_results.py

# 3. Check action metrics
python check_metrics_addon.py

# 4. Analyze individual seeds
python -c "import json; print(json.load(open('logs/seed_sweep_results/seed_7/val_ep080.json')))"
```

---

**Started:** 2025-10-30 16:30  
**Expected completion:** 2025-10-30 ~20:30  
**Status:** 🔄 Running seed 7/3...
