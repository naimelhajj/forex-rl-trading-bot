# Phase 2.7: Generalization & Stress Testing

## Date: 2025-10-24

Based on **strongest results to date** from 3×120 run showing **cross-seed mean +0.013** with finals **+0.54 ± 0.14** 🎉

## Results Summary (3×120 Baseline)

**Achieved Metrics:**
- Cross-seed mean SPR: **+0.013** (positive!)
- Final episodes: **+0.54 ± 0.14** (strong convergence)
- Median trades: **~22** per validation window
- Behavioral health:
  - Entropy: **~0.79-0.80 bits** (healthy diversity)
  - Switch rate: **~0.12** (no mode collapse)
- Penalty observations:
  - Seeds 7 & 777: **~2-3%** (excellent!)
  - Seed 77: **~17% mid-run cluster** (recovered later)

**Key Finding:** Solid learning signal with healthy exploration and controlled trade activity.

---

## Three Generalization & Stress Tests Implemented

### ✅ 1. Hold-Out Validation (Shifted/Strided Regime)

**Purpose:** Verify robustness beyond the primary 600-bar/90-stride training regime.

**Implementation:** Added alternate validation in `trainer.py` post-restore block.

**Changes (`trainer.py` lines ~1367-1420):**
```python
# --- ALT HOLD-OUT VALIDATION (shifted/strided) ---
# Alternate regime: wider stride (~120 bars = 20%), shifted start
alt_stride = int(600 * 0.20)   # ~120 bars
alt_start = 50                 # shift start by 50 bars
alt_windows = build_overlapping_windows(
    series_len=len(val_prices),
    window=600,
    stride=alt_stride,
    start=alt_start,
    target_k=self.cfg.VAL_K,
    min_k=self.cfg.VAL_MIN_K,
)

# Evaluate with SPR fitness
alt_score, alt_components, alt_details = evaluate_windows_SPR(
    env=self.val_env,
    policy=self.eval_policy,
    scaler=scaler,
    windows=alt_windows,
    iqr_cap=self.cfg.VAL_IQR_PENALTY,
    gating_cfg=self.cfg,
)

# Save as val_final_alt.json
save_val_summary(
    path="logs/validation_summaries/val_final_alt.json",
    episode="final_alt",
    score=alt_score,
    ...
    extra={"regime": "alt_600x120_shift50"}
)
```

**Output:**
- Console: `[POST-RESTORE:ALT] windows=X | SPR=Y.YYY | PF=... | TPY=... | SIG=...`
- File: `logs/validation_summaries/val_final_alt.json`

**Success Criteria:**
- Primary SPR (600/90): **≥ 0** (maintained)
- Alt SPR (600/120+shift): **> 0** for seeds 7 & 777 (positive generalization)
- Seed 77: Positive alt SPR acceptable even if smaller

**Interpretation:**
- ✅ **Both positive:** Policy generalizes well, not overfitting
- ⚠️ **Alt negative:** May be overfitting to specific window boundaries
- 🟢 **Alt > Primary:** Policy robust across multiple regimes

---

### ✅ 2. Trade Pacing Stress Test (Quality > Quantity)

**Purpose:** Ensure profitability isn't from "spraying trades" - validate quality over quantity.

**Implementation:** Reduced trade ceiling in `config.py`.

**Changes (`config.py` line ~57):**
```python
max_trades_per_episode: int = 100  # STRESS-TEST: Lower from 120 (-17%)
```

**Impact:**
- **Before:** Up to 120 trades per episode (~30 trades per 600-bar window)
- **After:** Up to 100 trades per episode (~25 trades per 600-bar window)
- **Expected:** Late-run positives persist, penalty rate stays low

**Success Criteria:**
- ✓ Late-episode positives **maintained** (≥ +0.10 in episodes 80-120)
- ✓ Median trades: **~18-24** (slight decrease from ~22)
- ✓ Penalty rate: **≤ 8%** (no increase from trade scarcity)
- ✓ Trade quality: **PF improves** or stays same (fewer, better trades)

**Yellow Flags:**
- ⚠️ Penalty rate spikes to >15% → Trade ceiling too tight, revert to 110-115
- ⚠️ Late positives disappear → Agent needs more trade budget

---

### ✅ 3. Stricter SPR Formula (Tighter Bounds)

**Purpose:** Reduce outlier influence and prevent tiny drawdowns from inflating scores.

**Implementation:** Tightened PF cap and raised DD floor in `config.py`.

**Changes (`config.py` FitnessConfig lines ~100-102):**
```python
spr_pf_cap: float = 6.0      # STRICTER: Cap at 6 (was 10) - reduce outlier wins
spr_dd_floor_pct: float = 1.0  # STRICTER: Floor MDD at 1% (was 0.05) - prevent tiny DD inflation
```

**Impact:**

**PF Cap (10.0 → 6.0):**
- **Before:** PF = min(10.0, gross_profit / gross_loss)
- **After:** PF = min(6.0, gross_profit / gross_loss)
- **Effect:** De-emphasizes strategies with outlier wins (e.g., 1 huge win, 20 small losses)
- **Preserves:** Consistent profitable strategies (PF 1.5-4.0 range)

**DD Floor (0.05% → 1.0%):**
- **Before:** mdd_pct = max(0.05, actual_mdd_pct)
- **After:** mdd_pct = max(1.0, actual_mdd_pct)
- **Effect:** Prevents strategies with tiny drawdowns (<1%) from getting inflated PF/MDD ratios
- **Reality check:** All real strategies have >1% drawdown

**SPR Formula:**
```
SPR = (PF / MDD%) × MMR% × Significance × Stagnation

Where:
- PF: min(6.0, gross_profit / gross_loss)  ← CAPPED TIGHTER
- MDD%: max(1.0, actual_mdd_pct)           ← FLOORED HIGHER
- MMR%: Mean monthly return as % of balance
- Significance: (min(1, TPY / 100))²
- Stagnation: 1 - (days_since_peak / test_days)
```

**Expected Outcomes:**
- Cross-seed mean: **Similar or +0.01 to +0.03 higher** (better signal quality)
- Score distribution: **Tighter** (fewer noisy +1.0+ spikes, more mid-positive 0.1-0.6)
- Episode-to-episode variance: **Lower** (more stable rankings)
- Outlier episodes: **Reduced** (no more PF=9.8 with MDD=0.2% anomalies)

**Success Criteria:**
- ✓ Cross-seed mean: **≥ +0.01** (maintained or improved)
- ✓ Score StdDev: **Decreases by 10-20%** (tighter distribution)
- ✓ Late-episode high scores: **More +0.3 to +0.7**, fewer +1.5+ spikes
- ✓ Ranking consistency: Same episodes still perform well (no reordering)

---

## Seed 77 Penalty Cluster Analysis

**Observation:** Seed 77 showed **~17% penalty rate** mid-run (episodes 40-70), then recovered.

**Root Cause:** Grace logic working correctly:
1. Episode has **1-3 trades** (ultra-low)
2. First offense: Grace applied (`mult=0.00, pen=0.00` → warning)
3. **Second consecutive** ultra-low episode: Penalty applied (`pen=0.03-0.08`)
4. Seed 77 had a **sequence** of such episodes → penalties accumulated

**Is This a Problem?** 
- ✅ **No** - Grace logic is **working as designed**
- ✅ Agent recovered after ~20-30 episodes (learned to increase trade activity)
- ✅ Seeds 7 & 777 showed **2-3% penalty rate** (excellent gating)

**Optional Adjustments (if seed 77 pattern persists):**

**Option A: Softer Penalty Cap**
```python
# config.py
VAL_PENALTY_MAX: float = 0.07  # Down from 0.08 (12.5% reduction)
```

**Option B: Degenerate Slice Down-Weight**
```python
# In trainer.py gating logic (lines ~892-905):
# Treat ≤2 trades as degenerate (don't penalize, just down-weight)
if median_trades <= 2:
    # Degenerate slice - apply 50% down-weight instead of penalty
    mult = 0.50
    undertrade_penalty = 0.0
elif median_trades < min_half:
    # Normal low-trade logic with grace counter
    if self.last_val_was_low_trade:
        # Second consecutive - penalize
        ...
```

**Recommendation:** 
- **Run all 3 improvements first** before adjusting penalties
- If seed 77 still shows >15% penalty rate after trade pacing stress test, apply Option B
- Keep Option A in reserve (last resort)

---

## Testing Plan

### Test 1: Hold-Out Validation Check (Immediate)

**Command:**
```powershell
python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 120
```

**Duration:** ~18-20 hours (6-7 hours per seed)

**Check After Completion:**
```powershell
python compare_seed_results.py
python check_validation_diversity.py
python check_metrics_addon.py
```

**Expected Outputs:**
- `logs/validation_summaries/val_final.json` (primary 600/90 regime)
- `logs/validation_summaries/val_final_alt.json` (alt 600/120+shift regime)

**Success Metrics:**
1. **Primary SPR:** Cross-seed mean ≥ +0.01 (maintained)
2. **Alt SPR:** Seeds 7 & 777 show positive alt scores
3. **Correlation:** Primary and alt scores positively correlated
4. **Late-episode positives:** More episodes with +0.3 to +0.7 (tighter PF/DD bounds working)

---

### Test 2: Trade Pacing Stress (After Test 1 Success)

**Already Applied:** `max_trades_per_episode = 100` (in config.py)

**Command:**
```powershell
python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 120
```

**Compare with Test 1 results:**

**Metrics to Watch:**
- Median trades per window: **~18-24** (down from ~22)
- Penalty rate: **≤ 8%** (should not increase)
- Late-episode positives: **Maintained** (≥ +0.10 in episodes 80-120)
- Trade quality (PF): **Same or higher** (fewer, better trades)

**Success:** Same or better SPR with fewer trades = quality confirmed ✅

**Yellow Flags:**
- Penalty rate >15% → Too tight, increase to `max_trades=110`
- Late positives disappear → Revert to `max_trades=120`

---

### Test 3: Continuous Monitoring (Ongoing)

**Tools:**
```powershell
# Real-time monitoring during runs
python monitor_sweep.py

# Post-run analysis
python check_validation_diversity.py
python check_metrics_addon.py
python compare_seed_results.py
```

**Score Distribution Analysis:**
```python
# In check_validation_diversity.py output
# Look for tighter distribution:
# - Mean: +0.01 to +0.05
# - StdDev: 0.10 to 0.15 (down from ~0.20-0.25)
# - Range: [-0.2, +0.8] (fewer +1.5+ outliers)
```

---

## Green/Yellow Flags

### 🟢 Green Flags (Success):

**Primary Validation:**
- ✅ Cross-seed mean SPR: **≥ +0.01** (maintained or improved)
- ✅ Late-episode positives: **≥15% of episodes with score > +0.10**
- ✅ Score distribution: **StdDev decreases 10-20%** (tighter)

**Alt Hold-Out:**
- ✅ **Alt SPR > 0** for seeds 7 & 777 (generalization confirmed)
- ✅ **Positive correlation** between primary and alt scores
- ✅ Alt score magnitude: **≥50% of primary score** (robust)

**Trade Pacing:**
- ✅ Median trades: **~18-24** (controlled reduction)
- ✅ Penalty rate: **≤ 8%** (no increase from scarcity)
- ✅ PF improves: **+0.2 to +0.5 higher** (quality confirmed)

**Behavioral Health:**
- ✅ Entropy: **≥ 0.77 bits** (maintained diversity)
- ✅ Switch rate: **~0.12** (no collapse)
- ✅ Hold rate: **0.70-0.85** (reasonable)

---

### 🟡 Yellow Flags (Watch/Adjust):

**Alt Hold-Out:**
- ⚠️ **Alt SPR negative** → May be overfitting to 600/90 windows
- ⚠️ **Low correlation** (r < 0.3) → Policy unstable across regimes
- Action: Add more validation diversity (different window sizes)

**Trade Pacing:**
- ⚠️ **Penalty rate >15%** → Trade ceiling too tight
- Action: Increase `max_trades_per_episode` to 110-115
- ⚠️ **Late positives disappear** → Agent needs more trade budget
- Action: Revert to `max_trades=120`

**Score Distribution:**
- ⚠️ **StdDev increases** → Stricter bounds causing instability
- Action: Relax PF cap to 7.0 or DD floor to 0.75%
- ⚠️ **Mean decreases >50%** → Bounds too harsh
- Action: Revert to `spr_pf_cap=10.0, spr_dd_floor_pct=0.05`

**Seed 77 Penalties:**
- ⚠️ **>15% penalty rate persists** → Apply degenerate slice down-weight (Option B)
- ⚠️ **>20% penalty rate** → Apply softer penalty cap (Option A)

---

## Files Modified

### 1. `config.py` (2 changes)

**Line ~57: Trade Pacing Stress Test**
```python
max_trades_per_episode: int = 100  # STRESS-TEST: Lower from 120 (-17%)
```

**Lines ~100-102: Stricter SPR Bounds**
```python
spr_pf_cap: float = 6.0      # STRICTER: Cap at 6 (was 10)
spr_dd_floor_pct: float = 1.0  # STRICTER: Floor MDD at 1% (was 0.05)
```

---

### 2. `trainer.py` (1 addition)

**Lines ~1367-1420: Hold-Out Validation**
```python
# --- ALT HOLD-OUT VALIDATION (shifted/strided) ---
alt_stride = int(600 * 0.20)   # ~120 bars
alt_start = 50                 # shift start
alt_windows = build_overlapping_windows(...)
alt_score, alt_components, alt_details = evaluate_windows_SPR(...)

# Save as val_final_alt.json
save_val_summary(
    path="logs/validation_summaries/val_final_alt.json",
    episode="final_alt",
    extra={"regime": "alt_600x120_shift50"}
)
```

---

## Expected Timeline

**Test 1 (Hold-Out + Stricter SPR):**
- Runtime: **18-20 hours** (3 seeds × 120 episodes)
- Analysis: **30 minutes** (compare primary vs alt validation)
- Decision point: **24 hours from start**

**Test 2 (Trade Pacing - already applied):**
- Runtime: **18-20 hours** (same 3×120 sweep)
- Analysis: **30 minutes** (compare with Test 1)
- Decision point: **48 hours from start**

**Total Testing Time:** ~2-3 days for comprehensive validation

---

## Success Definition

**Minimum Success (Test 1):**
- ✓ Primary cross-seed mean: **≥ +0.01**
- ✓ Alt validation: **≥2 seeds show positive alt SPR**
- ✓ Late-episode positives: **Maintained**

**Full Success (Tests 1+2):**
- ✓ Primary cross-seed mean: **+0.02 to +0.05**
- ✓ Alt validation: **All 3 seeds positive**
- ✓ Trade pacing: **Same SPR with 100 trades (quality confirmed)**
- ✓ Score distribution: **Tighter (StdDev -10-20%)**
- ✓ Behavioral health: **Maintained**

**Stretch Goal:**
- 🎯 Cross-seed mean: **≥ +0.08**
- 🎯 Late-episode spikes: **≥5% of episodes with score > +0.50**
- 🎯 Alt validation: **Alt SPR ≥ 75% of primary SPR**

---

## Next Steps After Success

1. **Hold-out regime becomes standard** - add to all training runs
2. **Trade pacing locked** - keep `max_trades=100` as production ceiling
3. **SPR bounds finalized** - PF cap=6, DD floor=1% as production config
4. **Extended run** - 200-episode × 3-seed confirmation
5. **Multi-pair validation** - test on GBPUSD, USDJPY (if data available)

---

**Status:** ✅ All improvements implemented and tested (syntax)
**Next Action:** Launch Test 1 (120-episode × 3-seed with hold-out validation)

**Command:**
```powershell
python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 120
```

**Monitor:** Console for `[POST-RESTORE:ALT]` output showing alt regime results

**Key Achievement:** Moving from "it learns" to **"it generalizes and scales"** 🎯🚀
