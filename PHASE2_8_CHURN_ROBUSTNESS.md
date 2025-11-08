# Phase 2.8: Robustness & Churn-Calming Pass

**Date:** October 27, 2025  
**Baseline:** Phase 2.7 (5-seed × 150-episode production validation)  
**Status:** 🔧 **IN PROGRESS - Ready for testing**

---

## Executive Summary

### **Context:**

Phase 2.7 delivered **outstanding production results**:
- ✅ Cross-seed mean: +0.037 ± 0.076
- ✅ 4/5 seeds positive final convergence (80%)
- ✅ 3/5 seeds zero penalties (60%)
- ✅ Penalty rate: 2.4% average (down from 16.7%)

### **Identified Issues:**

1. **Action entropy too high:** 1.086 bits (target: 0.80-0.95)
2. **Switch rate too punchy:** 19.3% (target: 14-17%)
3. **Slight short bias:** 57.4% short vs 42.6% long
4. **Frozen frictions:** All tests ran with `FREEZE_VALIDATION_FRICTIONS=True` (not realistic)

### **Phase 2.8 Goals:**

1. ✅ **Calm churn:** Reduce action entropy to 0.80-0.95 bits, switch rate to 14-17%
2. ✅ **Robustness test:** Enable friction jitter (±10%) to validate gains aren't a mirage
3. ✅ **Fitness calibration:** Match SPR to observed 30-36 trades/ep cadence
4. ✅ **Mild diversification:** Add 3 pairs for anti-overfit nudge

---

## Changes Implemented

### **A) Churn-Calming Tweaks**

**Goal:** Reduce action entropy from ~1.09 to 0.80-0.95 bits, switch rate from 19% to 14-17%

**EnvironmentConfig changes:**
```python
# BEFORE (Phase 2.7)          # AFTER (Phase 2.8)
min_hold_bars: 5         →    min_hold_bars: 6        (+1 bar, 20% increase)
cooldown_bars: 10        →    cooldown_bars: 12       (+2 bars, 20% increase)
flip_penalty: 0.0006     →    flip_penalty: 0.0007    (+17% penalty)
trade_penalty: 0.00006   →    trade_penalty: 0.00007  (+17% penalty)
```

**AgentConfig changes:**
```python
# BEFORE (Phase 2.7)          # AFTER (Phase 2.8)
eval_epsilon: 0.05       →    eval_epsilon: 0.03      (-40% eval probing)
hold_break_after: 7      →    hold_break_after: 8     (+14% patience)
hold_tie_tau: 0.032      →    hold_tie_tau: 0.035     (+9% hold tolerance)
noisy_sigma_init: 0.03   →    noisy_sigma_init: 0.02  (-33% parameter noise)
```

**Expected Impact:**
- Entropy: 1.09 → **0.85** bits (20% reduction)
- Switch rate: 19% → **15%** (4pp reduction)
- Hold length: 10.6 → **12-13 bars** (15% longer)
- Trades/ep: 31 → **28-32** (slight reduction, quality focus)

---

### **B) Robustness Pass (Enable Friction Jitter)**

**Goal:** Test if gains survive realistic cost variation (±10% spread/commission)

**Top-level config:**
```python
# BEFORE (Phase 2.7)                    # AFTER (Phase 2.8)
FREEZE_VALIDATION_FRICTIONS: True  →   FREEZE_VALIDATION_FRICTIONS: False
```

**Validation jitter ranges:**
```python
# BEFORE (Phase 2.7)                    # AFTER (Phase 2.8)
VAL_SPREAD_JITTER: (0.95, 1.05)    →   VAL_SPREAD_JITTER: (0.90, 1.10)    (±5% → ±10%)
VAL_COMMISSION_JITTER: (0.95, 1.05) →  VAL_COMMISSION_JITTER: (0.90, 1.10) (±5% → ±10%)
```

**What this means:**
- Spread varies: 1.35 pips to 1.65 pips (was fixed at 1.5)
- Commission varies: $6.30 to $7.70 per lot (was fixed at $7.00)
- Each validation episode gets random costs within range
- Tests if strategy adapts or collapses under cost uncertainty

**Expected Impact:**
- Cross-seed mean: +0.037 → **+0.010 to +0.030** (realistic degradation)
- Penalty rate: 2.4% → **3-5%** (slight increase acceptable)
- If mean drops below **-0.05**, frictions are too punitive (need recalibration)

---

### **C) Fitness Calibration**

**Goal:** Match SPR scoring to observed 30-36 trades/ep cadence

**FitnessConfig changes:**
```python
# BEFORE (Phase 2.7)                         # AFTER (Phase 2.8)
spr_pf_cap: 6.0                         →    spr_pf_cap: 5.0           (-17% cap)
spr_target_trades_per_year: 100.0       →    spr_target_trades_per_year: 120.0  (+20% target)
spr_dd_floor_pct: 1.0                   →    spr_dd_floor_pct: 1.5     (+50% floor)
```

**Rationale:**
1. **PF cap 6.0 → 5.0:** Observed PFs in 0.85-1.15 range; 6.0 cap was too loose
2. **Target TPY 100 → 120:** Late-run episodes show 30-36 trades/ep (≈120 TPY), not 100
3. **DD floor 1.0% → 1.5%:** Prevent tiny drawdowns from inflating scores artificially

**Expected Impact:**
- Scores with tiny DD (<1.5%) now penalized
- Higher trade cadence rewarded (120 TPY vs 100)
- Outlier PFs (>5.0) capped more aggressively
- **More realistic fitness landscape**

---

### **D) Pair Diversification**

**Goal:** Add anti-overfit nudge with different spread/volatility profiles

**DataConfig changes:**
```python
# BEFORE (Phase 2.7)
pairs: ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY']  (4 pairs)

# AFTER (Phase 2.8)
pairs: ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY',   (7 pairs)
        'USDCAD', 'AUDUSD', 'GBPJPY']
```

**Why these pairs:**
- **USDCAD:** Commodity-linked, different spread (2-3 pips)
- **AUDUSD:** Pacific session, risk-on proxy
- **GBPJPY:** High volatility cross (3-5 pips spread)

**Expected Impact:**
- More diverse training signal
- Tests adaptability to different spread/volatility regimes
- May slightly reduce mean SPR (diversification tax) but improve robustness
- **Worth 0.01-0.02 SPR cost** for generalization gain

---

## Two-Stage Testing Plan

### **Stage 1: Churn-Calmed + Frozen Frictions**

**Purpose:** Isolate churn reduction effects from cost randomness

**Config state:**
```python
FREEZE_VALIDATION_FRICTIONS: True  # Keep frozen for now
# All churn tweaks (A) + fitness calibration (C) + pairs (D) enabled
```

**Command:**
```powershell
python run_seed_sweep_organized.py --seeds 7 17 27 77 777 --episodes 80
python compare_seed_results.py
python check_validation_diversity.py
python check_metrics_addon.py
```

**Success Criteria:**
- ✅ Cross-seed mean: **≥ +0.02** (allow 0.01-0.02 drop from diversification)
- ✅ Action entropy: **0.80-0.95 bits** (down from 1.09)
- ✅ Switch rate: **14-17%** (down from 19%)
- ✅ Trades/ep: **28-34** (slight reduction acceptable)
- ✅ Penalty rate: **≤ 5%**
- ✅ Hold length: **12-14 bars** (up from 10.6)

**If fails:** Roll back most aggressive tweak first (min_hold_bars 6→5 or cooldown 12→11)

---

### **Stage 2: Full Robustness Test (Friction Jitter Enabled)**

**Purpose:** Validate gains survive realistic cost variation

**Config state:**
```python
FREEZE_VALIDATION_FRICTIONS: False  # Enable jitter
# All changes (A + B + C + D) enabled
```

**Command:**
```powershell
# Edit config.py: Set FREEZE_VALIDATION_FRICTIONS = False
python run_seed_sweep_organized.py --seeds 7 17 27 77 777 --episodes 80
python compare_seed_results.py
python check_validation_diversity.py
python check_metrics_addon.py
```

**Success Criteria:**
- ✅ Cross-seed mean: **≥ +0.01** (within 0.05 of Stage 1 results)
- ✅ Action entropy: **0.80-0.95 bits** (maintained)
- ✅ Switch rate: **14-17%** (maintained)
- ✅ Trades/ep: **28-34** (maintained)
- ✅ Penalty rate: **≤ 6%** (allow +1% for jitter)
- ✅ **No collapse:** Best seed still positive

**If degradation > 0.05:**
- Friction jitter too aggressive → reduce to ±5% (0.95, 1.05)
- SPR floor too high → lower to 1.0%
- Target TPY mismatch → adjust to observed cadence

---

## Expected Results Summary

| Metric | Phase 2.7 | **Stage 1 Target** | **Stage 2 Target** |
|--------|-----------|-------------------|-------------------|
| **Cross-seed mean** | +0.037 | **+0.020 to +0.035** | **+0.010 to +0.030** |
| **Action entropy** | 1.086 bits | **0.80-0.95 bits** | **0.80-0.95 bits** |
| **Switch rate** | 19.3% | **14-17%** | **14-17%** |
| **Hold length** | 10.6 bars | **12-14 bars** | **12-14 bars** |
| **Trades/ep** | 31.7 | **28-32** | **28-32** |
| **Penalty rate** | 2.4% | **≤ 5%** | **≤ 6%** |
| **Positive seeds** | 4/5 (80%) | **≥ 3/5 (60%)** | **≥ 3/5 (60%)** |

---

## Rollback Plan (If Stage 1 Fails)

**If entropy/switch rate don't improve:**
1. Reduce `noisy_sigma_init` 0.02 → 0.015 (less noise)
2. Increase `hold_tie_tau` 0.035 → 0.040 (more hold bias)
3. Increase `min_hold_bars` 6 → 7 (even stricter)

**If mean SPR drops too much (<+0.01):**
1. Reduce `flip_penalty` 0.0007 → 0.00065 (less harsh)
2. Reduce `trade_penalty` 0.00007 → 0.000065 (less harsh)
3. Reduce `spr_dd_floor_pct` 1.5 → 1.2 (less strict)

**If Stage 2 collapses (mean < -0.05):**
1. Reduce jitter: `(0.90, 1.10)` → `(0.95, 1.05)` (±5%)
2. Lower `spr_dd_floor_pct` 1.5 → 1.0 (original)
3. Check if spread/commission base values need adjustment

---

## Next Steps After Validation

### **If both stages succeed:**

**Promote to "SPR Baseline v1.0":**
```powershell
# Archive current config
cp config.py config_phase2.8_baseline_v1.0.py

# Run 5-seed × 200-episode confirmation sweep (robustness ON)
python run_seed_sweep_organized.py --seeds 7 17 27 77 777 --episodes 200
python compare_seed_results.py
python check_validation_diversity.py
```

**Expected 200-episode results:**
- Cross-seed mean: **+0.015 to +0.035**
- Penalty rate: **≤ 6%**
- Late-run excellence maintained (Ep 150-200 positive)
- Best seed candidate for production deployment

**Then proceed to:**
- Paper trading deployment (best seed)
- Multi-timeframe testing (4H, Daily)
- Live broker integration prep

---

### **If Stage 1 succeeds but Stage 2 fails:**

**Diagnosis:** Churn tweaks work, but friction sensitivity too high

**Action:**
1. Keep churn tweaks (A)
2. Reduce jitter to ±5%: `(0.95, 1.05)`
3. Re-run Stage 2 with gentler jitter
4. If still fails, increase spread/commission base or lower DD floor

---

### **If both stages fail:**

**Diagnosis:** Churn tweaks too aggressive or fitness miscalibrated

**Action:**
1. Roll back to Phase 2.7 config
2. Apply gentler tweaks:
   - `min_hold_bars: 5` (keep)
   - `cooldown_bars: 11` (split difference)
   - `flip_penalty: 0.00065` (gentler)
   - `eval_epsilon: 0.04` (gentler)
3. Re-test with conservative changes

---

## Files Modified

### **config.py changes:**

**Line ~18:**
```python
FREEZE_VALIDATION_FRICTIONS: bool = False  # PHASE-2.8: Enable for robustness test
```

**Lines ~58-62 (EnvironmentConfig):**
```python
cooldown_bars: int = 12       # Was 10
min_hold_bars: int = 6        # Was 5
trade_penalty: float = 0.00007  # Was 0.00006
flip_penalty: float = 0.0007    # Was 0.0006
```

**Lines ~74-77 (AgentConfig):**
```python
eval_epsilon: float = 0.03      # Was 0.05
hold_tie_tau: float = 0.035     # Was 0.032
hold_break_after: int = 8       # Was 7
noisy_sigma_init: float = 0.02  # Was 0.03
```

**Lines ~113-115 (FitnessConfig):**
```python
spr_pf_cap: float = 5.0                     # Was 6.0
spr_target_trades_per_year: float = 120.0   # Was 100.0
spr_dd_floor_pct: float = 1.5               # Was 1.0
```

**Lines ~148-149 (DataConfig):**
```python
pairs: ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY', 
        'USDCAD', 'AUDUSD', 'GBPJPY']  # Added 3 pairs
```

**Lines ~194-195:**
```python
VAL_SPREAD_JITTER: (0.90, 1.10)      # Was (0.95, 1.05)
VAL_COMMISSION_JITTER: (0.90, 1.10)  # Was (0.95, 1.05)
```

---

## Monitoring Checklist

### **During Stage 1 run:**
- [ ] Action entropy drops below 1.0 bits by episode 40
- [ ] Switch rate drops below 18% by episode 40
- [ ] Hold length increases above 11 bars by episode 40
- [ ] Trade count stays in 28-34 range
- [ ] Penalty rate ≤ 5% throughout
- [ ] No zero-trade episodes

### **During Stage 2 run:**
- [ ] Cross-seed mean within 0.05 of Stage 1
- [ ] No seed collapses to negative < -0.10
- [ ] Penalty rate ≤ 6% (allow +1% for jitter)
- [ ] Trade count remains stable (28-34)
- [ ] Best seed still achieves +0.05+ final score

### **Red flags:**
- ⚠️ Entropy increases or plateaus above 1.0
- ⚠️ Switch rate increases or stays above 18%
- ⚠️ Trade count drops below 20/ep (over-penalized)
- ⚠️ Penalty rate spikes above 8%
- ⚠️ All seeds negative in Stage 2
- ⚠️ Degradation > 0.10 from Stage 1 to Stage 2

---

**Phase 2.8 Status:** ✅ **CONFIG READY - Awaiting Stage 1 Test**  
**Next Action:** Run Stage 1 (churn-calmed + frozen frictions)  
**Duration:** ~6-8 hours (5 seeds × 80 episodes)

---

*Generated: October 27, 2025*  
*Config baseline: Phase 2.7 (5-seed production validation)*  
*Target: Churn reduction + robustness validation*
