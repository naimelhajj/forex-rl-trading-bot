# Phase 2.8 Quick-Start Guide

## 🚀 Quick Commands (Copy-Paste Ready)

### **IMPORTANT: Current Config State**

Your `config.py` is now set to:
- ✅ Churn-calming tweaks enabled (A)
- ✅ Robustness jitter **ENABLED** (B) - `FREEZE_VALIDATION_FRICTIONS = False`
- ✅ Fitness calibration enabled (C)
- ✅ Pair diversification enabled (D)

**This is Stage 2 configuration (full robustness test).**

---

## 🎯 Recommended Testing Sequence

### **Option 1: Run Stage 2 First (Robustness Test)**

Since config already has `FREEZE_VALIDATION_FRICTIONS = False`, you can run robustness test immediately:

```powershell
# Stage 2: Full robustness test (friction jitter enabled)
python run_seed_sweep_organized.py --seeds 7 17 27 77 777 --episodes 80

# Analyze results
python compare_seed_results.py
python check_validation_diversity.py
python check_metrics_addon.py
```

**Expected results:**
- Cross-seed mean: **+0.010 to +0.030**
- Action entropy: **0.80-0.95 bits**
- Switch rate: **14-17%**
- Penalty rate: **≤ 6%**

**Duration:** ~6-8 hours

---

### **Option 2: Run Stage 1 First (Isolated Churn Test)**

To isolate churn effects, first freeze frictions:

**Step 1: Edit config.py line ~18:**
```python
FREEZE_VALIDATION_FRICTIONS: bool = True  # Stage 1: Freeze for isolated test
```

**Step 2: Run Stage 1:**
```powershell
python run_seed_sweep_organized.py --seeds 7 17 27 77 777 --episodes 80
python compare_seed_results.py
python check_validation_diversity.py
python check_metrics_addon.py
```

**Expected results:**
- Cross-seed mean: **+0.020 to +0.035**
- Action entropy: **0.80-0.95 bits**
- Switch rate: **14-17%**

**Step 3: Edit config.py line ~18:**
```python
FREEZE_VALIDATION_FRICTIONS: bool = False  # Stage 2: Enable jitter
```

**Step 4: Run Stage 2:**
```powershell
python run_seed_sweep_organized.py --seeds 7 17 27 77 777 --episodes 80
python compare_seed_results.py
python check_validation_diversity.py
python check_metrics_addon.py
```

**Duration:** ~12-16 hours total (2 stages)

---

## 📊 Quick Success Check (After Each Stage)

### **Check 1: Cross-Seed Mean**
```powershell
python compare_seed_results.py
```

**Look for:** `Score Mean across seeds:` line  
✅ **PASS:** ≥ +0.01 (Stage 2) or ≥ +0.02 (Stage 1)  
⚠️ **FAIL:** < +0.01 (too much degradation)

---

### **Check 2: Action Metrics**
```powershell
python check_metrics_addon.py
```

**Look for:** `Averages:` section  
✅ **PASS:**  
  - `action_entropy: 0.80-0.95` (was 1.09)  
  - `switch_rate: 0.14-0.17` (was 0.19)  
  - `avg_hold_length: 12-14` (was 10.6)

⚠️ **FAIL:**  
  - Entropy still > 1.0  
  - Switch rate still > 0.18

---

### **Check 3: Trade Activity**
```powershell
python check_validation_diversity.py
```

**Look for:**  
✅ **PASS:**  
  - Trade counts: 20-35 range  
  - Penalty episodes: ≤ 5-6% (4-5 out of 80)  
  - No ⚠️ warnings (zero trades)

⚠️ **FAIL:**  
  - Many trade counts < 20 (over-penalized)  
  - Penalty rate > 8%  
  - Multiple zero-trade episodes

---

## 🔧 Quick Fixes (If Stage Fails)

### **If entropy/switch rate NOT reduced:**

**Option A: More aggressive churn reduction**
```python
# config.py
min_hold_bars: 6 → 7
cooldown_bars: 12 → 13
hold_tie_tau: 0.035 → 0.040
```

**Option B: Less parameter noise**
```python
# config.py
noisy_sigma_init: 0.02 → 0.015
```

---

### **If mean SPR drops too much (< +0.01):**

**Option A: Less penalty**
```python
# config.py
flip_penalty: 0.0007 → 0.00065
trade_penalty: 0.00007 → 0.000065
```

**Option B: Gentler fitness floor**
```python
# config.py
spr_dd_floor_pct: 1.5 → 1.2
```

---

### **If Stage 2 collapses (mean < -0.05):**

**Option A: Reduce friction jitter**
```python
# config.py
VAL_SPREAD_JITTER: (0.90, 1.10) → (0.95, 1.05)
VAL_COMMISSION_JITTER: (0.90, 1.10) → (0.95, 1.05)
```

**Option B: Go back to frozen (concede robustness)**
```python
# config.py
FREEZE_VALIDATION_FRICTIONS: bool = True
```

---

## 🎯 Decision Tree

```
Start → Run Stage 2 (robustness test)
           ↓
      Results good?
       ↙         ↘
     YES         NO
      ↓           ↓
   Promote!   Try Stage 1
              (frozen)
                 ↓
            Results good?
             ↙         ↘
           YES         NO
            ↓           ↓
       Jitter too   Roll back
       aggressive   to Phase 2.7
            ↓
      Reduce jitter
      to ±5%
```

---

## ✅ Success Path (If Both Stages Pass)

### **1. Archive current config:**
```powershell
cp config.py config_phase2.8_baseline_v1.0.py
```

### **2. Confirmation sweep (200 episodes, robustness ON):**
```powershell
python run_seed_sweep_organized.py --seeds 7 17 27 77 777 --episodes 200
```

**Duration:** ~15-20 hours

**Expected:**
- Cross-seed mean: **+0.015 to +0.035**
- Penalty rate: **≤ 6%**
- Late-run positives maintained
- Best seed: +0.10+ final score

### **3. Select production candidate:**

Check `compare_seed_results.py` output:
- **Seed 777:** Likely best mean (elite performer)
- **Seed 17:** Likely best final (convergence champion)
- **Seed 7:** Likely best peak (aggressive trader)

### **4. Deploy to paper trading:**
```powershell
# Model in: checkpoints/best_model.pt (from best seed)
# Ready for paper trading integration
```

---

## 📈 Monitoring During Run

### **Check progress every ~20 episodes:**
```powershell
# Quick peek at latest results
python check_validation_diversity.py

# Look for:
# - Trade counts stabilizing in 28-34 range
# - Entropy dropping below 1.0
# - Penalty rate ≤ 5-6%
```

### **If you see early warnings:**
- **Many penalties (>8%):** Stop run, reduce penalties
- **Zero trades:** Stop run, check min_hold_bars not too high
- **All negative scores:** Stop run, fitness miscalibrated

---

## 🚨 Emergency Rollback

**If everything fails, revert to Phase 2.7 config:**

```python
# config.py critical settings:
FREEZE_VALIDATION_FRICTIONS: bool = True
min_hold_bars: int = 5
cooldown_bars: int = 10
flip_penalty: float = 0.0006
trade_penalty: float = 0.00006
eval_epsilon: float = 0.05
hold_tie_tau: float = 0.032
hold_break_after: int = 7
noisy_sigma_init: float = 0.03
spr_pf_cap: float = 6.0
spr_target_trades_per_year: float = 100.0
spr_dd_floor_pct: float = 1.0
pairs: ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY']
VAL_SPREAD_JITTER: (0.95, 1.05)
VAL_COMMISSION_JITTER: (0.95, 1.05)
```

---

**Current Status:** ✅ **Ready to run Stage 2 (robustness test)**  
**Recommended:** Run Stage 2 first to save time, roll back to Stage 1 only if needed  
**Duration:** ~6-8 hours (80 episodes × 5 seeds)

---

*Quick-Start Guide - Phase 2.8*  
*Generated: October 27, 2025*
