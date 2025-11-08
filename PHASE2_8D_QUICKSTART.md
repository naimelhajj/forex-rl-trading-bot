# Phase 2.8d Fix Pack D1 - Quick Start Guide

**Status**: ✅ IMPLEMENTED - Ready for ablation testing  
**Date**: November 5, 2025

---

## 🎯 What Was Fixed

### Problem Diagnosis from 200-ep Confirm
- ❌ Entropy too high: 1.18–1.22 (target: 0.90–1.10)
- ❌ Hold rate too low: 0.58–0.62 (target: 0.65–0.80)
- ❌ Directional collapse: Long_ratio 0.065–0.934 (target: 0.40–0.60)
- ❌ Trade count high: ~29–30/ep with 0% penalty

### Implemented Fixes

| Fix | Parameter | Old | New | Impact |
|-----|-----------|-----|-----|--------|
| D1.1 | `entropy_beta` | ~0.020 | **0.014** | −30% exploration bonus → tighter policy |
| D1.2 | `hold_tie_tau` | 0.035 | **0.038** | +0.003 → stronger hold bias |
| D1.3 | `flip_penalty` | 0.0007 | **0.00077** | +10% churn cost |
| D1.4 | `VAL_EXP_TRADES_SCALE` | 0.38 | **0.42** | Tighter gating → suppress noise |
| D1.5 | `VAL_TRIM_FRACTION` | 0.20 | **0.25** | +5% trimming → remove outliers |
| D1.6 | `ls_balance_lambda` | N/A | **0.003** | NEW: L/S balance regularizer |

---

## 🚀 Quick Test Commands

### 1. Verify Configuration
```powershell
python -c "from config import Config; c = Config(); print(f'entropy_beta={c.environment.entropy_beta}, ls_lambda={c.environment.ls_balance_lambda}, hold_tau={c.agent.hold_tie_tau}, flip_pen={c.environment.flip_penalty}, exp_scale={c.VAL_EXP_TRADES_SCALE}, trim={c.VAL_TRIM_FRACTION}')"
```

**Expected Output**:
```
entropy_beta=0.014, ls_lambda=0.003, hold_tau=0.038, flip_pen=0.00077, exp_scale=0.42, trim=0.25
```

### 2. Run Single-Seed Smoke Test (20 episodes)
```powershell
python main.py --seed 777 --episodes 20 --disable_early_stop
```

**What to Check**:
- Training completes without errors
- Validation JSONs show `entropy`, `hold_frac`, `long_ratio` fields
- Entropy settles toward 0.90–1.10 range
- Hold fraction improves toward 0.65–0.80
- Long_ratio stays within 0.40–0.60

---

## 🧪 Ablation Protocol (Next Step)

### Purpose
Identify minimal fix set that restores GREEN performance

### Variants (Test Incrementally)
Run **3 seeds × 80 episodes** for each variant:

**A**: Base 2.8c + D1.2 (hold_tie_tau only)
```powershell
# Temporarily revert other changes, keep only hold_tie_tau=0.038
```

**B**: A + D1.1 (add entropy_beta)
**C**: B + D1.3 (add flip_penalty)
**D**: C + D1.4 (add exp_trades_scale)
**E**: D + D1.5 (add trim_fraction)
**F**: E + D1.6 (add ls_balance_lambda) ← **FULL FIX PACK**

### Acceptance Gate (Find First Variant That Passes)
- Mean SPR ≥ **+0.03**
- Trail-5 median ≥ **+0.20**
- Entropy **0.90–1.10**
- Hold **0.65–0.80**
- Long_ratio **0.40–0.60** on ≥2/3 seeds

### Run Full Fix Pack F (Recommended First)
```powershell
python run_seed_sweep_organized.py --seeds 7 17 777 --episodes 80
```

After completion (~3-4 hours):
```powershell
python compare_run_c_v1.py  # Quick analysis
python analyze_behavioral_metrics.py --seeds 7 17 777  # Detailed behavior check
```

---

## 📊 Full 150-ep Confirmation (If Ablation Passes)

Once you find the best variant:

```powershell
# Run 5 seeds × 150 episodes
python run_seed_sweep_organized.py --seeds 7 17 27 77 777 --episodes 150
```

Then analyze:
```powershell
python analyze_200ep_confirmation.py  # Use same script, works for 150-ep too
```

**Acceptance Gates** (Full):
1. Mean SPR ≥ +0.04
2. Trail-5 median ≥ +0.25
3. σ(means) ≤ 0.035
4. Penalty rate ≤ 10%
5. ≥3/5 seeds with trail-5 > 0
6. Behavioral metrics in bands

---

## 🔍 Diagnostic Scripts

### Direction PnL Split
```powershell
python -c "import json, glob; seeds=[7,17,777]; 
for s in seeds: 
    files=sorted(glob.glob(f'logs/seed_sweep_results/seed_{s}/val_ep*.json'))[-10:];
    longs=[json.load(open(f))['long_trades'] for f in files if 'long_trades' in json.load(open(f))];
    shorts=[json.load(open(f))['short_trades'] for f in files if 'short_trades' in json.load(open(f))];
    if longs and shorts: print(f'Seed {s}: L={sum(longs)} S={sum(shorts)} ratio={sum(longs)/(sum(longs)+sum(shorts)):.2f}');"
```

### Rolling Behavior (Manual Check)
```powershell
# Check last 10 episodes of each seed
foreach ($s in 7,17,777) {
    $files = Get-ChildItem "logs\seed_sweep_results\seed_$s\val_ep*.json" | Sort-Object Name | Select-Object -Last 10;
    $entropy = ($files | ForEach-Object { (Get-Content $_.FullName | ConvertFrom-Json).entropy } | Measure-Object -Average).Average;
    $hold = ($files | ForEach-Object { (Get-Content $_.FullName | ConvertFrom-Json).hold_frac } | Measure-Object -Average).Average;
    Write-Host "Seed $s : entropy=$($entropy.ToString('0.00')) hold=$($hold.ToString('0.00'))";
}
```

---

## ✅ Implementation Checklist

- [x] Update `hold_tie_tau` 0.035 → 0.038 in config.py
- [x] Update `flip_penalty` 0.0007 → 0.00077 in config.py
- [x] Update `VAL_EXP_TRADES_SCALE` 0.38 → 0.42 in config.py
- [x] Update `VAL_TRIM_FRACTION` 0.20 → 0.25 in config.py
- [x] Add `entropy_beta` = 0.014 to EnvironmentConfig
- [x] Add `ls_balance_lambda` = 0.003 to EnvironmentConfig
- [x] Add entropy_beta/ls_balance_lambda to environment __init__
- [x] Track `long_trades`, `short_trades`, `action_counts` in reset()
- [x] Update tracking when opening long/short positions
- [x] Implement entropy bonus in reward calculation
- [x] Implement L/S balance regularizer in reward calculation
- [ ] Run smoke test (20 episodes)
- [ ] Run ablation (3 seeds × 80 episodes per variant)
- [ ] Run full confirmation (5 seeds × 150 episodes with best variant)

---

## 🔴 If Still Red After D1

**Escalation Options** (apply incrementally):

1. **More aggressive hold**:
   ```python
   hold_tie_tau: 0.038 → 0.042  # +0.004 total from baseline
   ```

2. **Reduce entropy further**:
   ```python
   entropy_beta: 0.014 → 0.0126  # −10% more
   ```

3. **Stronger L/S regularizer**:
   ```python
   ls_balance_lambda: 0.003 → 0.006  # Double the penalty (temporary)
   ```

4. **Consider Phase 2.8b revert**:
   - If multiple escalations fail
   - Deep dive: jitter-averaging broken? Overfitting? Gating logic?

---

## 📝 Expected Outcomes

### If GREEN (All Gates Pass)
→ Lock Phase 2.8d as **SPR Baseline v1.2**  
→ Select production seed (likely 777)  
→ Proceed to paper trading integration

### If YELLOW (1-2 Gates Marginal)
→ Apply escalation options above  
→ Re-run 120-episode test with 5 seeds  
→ Micro-tune until GREEN

### If RED (Multiple Gates Failed)
→ Revert to Phase 2.8b (proven stable)  
→ Analyze degradation: gating? jitter? training dynamics?  
→ Consider alternative approaches (reduce jitter to ±8%, single-draw validation, etc.)

---

**Next Action**: Run smoke test to verify implementation, then proceed with ablation testing.
