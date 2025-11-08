# Phase 2.8c Status - Ready to Test

## ✅ Implementation Complete

### 🎯 Objective
Turn Run B v2's 🟡 YELLOW rating into 🟢 GREEN by stabilizing validation readings and reducing noise.

### 📋 Changes Implemented

#### 1. Configuration Updates (`config.py`)
- ✅ Added `VAL_JITTER_DRAWS = 3` (jitter-averaged validation)
- ✅ Updated `VAL_EXP_TRADES_SCALE = 0.38` (tighter gating, was 0.32)
- ✅ Maintained ±10% friction jitter (realistic robustness)

#### 2. Trainer Updates (`trainer.py`)
- ✅ Implemented jitter-averaged validation (K=3 draws per window)
- ✅ Each draw randomizes frictions within ±10% range
- ✅ Averages fitness, trades, and action metrics across draws
- ✅ Updated log messages to show "jitter-avg K=3"

#### 3. Analysis Tools (`compare_run_c_v1.py`)
- ✅ Trail-5 median computation for stable finals
- ✅ Friction jitter stability analysis
- ✅ GREEN/YELLOW/RED verdict generation
- ✅ Comparison to Run A & Run B v2 baselines

#### 4. Documentation
- ✅ `PHASE2_8C_STABILIZATION_TWEAKS.md` - Full technical docs
- ✅ `PHASE2_8C_QUICKSTART.md` - Quick-start guide
- ✅ `PHASE2_8C_IMPLEMENTATION_SUMMARY.md` - Implementation summary
- ✅ `PHASE2_8C_STATUS.md` - This status document

---

## 🚀 Ready to Run

### Run C v1 Configuration
```python
# Stabilization tweaks
VAL_JITTER_DRAWS = 3              # Jitter-averaged validation
VAL_EXP_TRADES_SCALE = 0.38       # Tighter gating
VAL_SPREAD_JITTER = (0.90, 1.10)  # ±10% realistic jitter
VAL_COMMISSION_JITTER = (0.90, 1.10)
FREEZE_VALIDATION_FRICTIONS = False  # Jitter enabled
```

### Test Protocol
- **Seeds:** 7, 77, 777 (quick 3-seed test)
- **Episodes:** 120 per seed
- **Duration:** ~6-8 hours
- **Friction:** ±10% jitter, K=3 averaged per validation

### Start Command
```powershell
python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 120
```

### Monitoring Commands
```powershell
# Check progress
Get-ChildItem logs\seed_sweep_results\seed_* -Directory | Select-Object Name, LastWriteTime

# Verify process running
Get-Process python -ErrorAction SilentlyContinue

# Watch logs (should see "jitter-avg K=3")
Get-Content logs\training.log -Tail 50 -Wait
```

### Analysis Command
```powershell
python compare_run_c_v1.py
```

---

## 🎯 Success Criteria

### 🟢 GREEN (Proceed to 200-episode)
- Cross-seed mean ≥ +0.03
- Cross-seed trail-5 ≥ +0.30
- Mean variance ≤ ±0.035
- Penalty rate <10%
- Positive trail-5: 3/3 seeds
- Friction jitter working

### 🟡 YELLOW (Fine-tune)
- Cross-seed mean +0.02-0.03
- Cross-seed trail-5 +0.20-0.30
- Mean variance ±0.035-0.045
- Penalty rate 10-15%
- Action: Run 120-episode test with 5 seeds, adjust gating

### 🔴 RED (Revert or debug)
- Cross-seed mean <+0.02
- Degradation vs Run B v2
- High variance (>±0.05)
- Penalty rate >15%
- Action: Revert to Phase 2.8b, analyze what broke

---

## 📊 Expected Results

### vs Run B v2 (Baseline)
```
Run B v2 (jitter, no averaging):
  Cross-seed mean: +0.018 ± 0.037
  Cross-seed final: +0.147 ± 0.196
  Trades/ep: 25.7 ± 0.7

Expected Run C v1 (jitter-averaged):
  Cross-seed mean: +0.040 ± 0.030  (+122% improvement)
  Cross-seed trail-5: +0.400 ± 0.100  (+171% improvement)
  Trades/ep: 25.5 ± 0.8  (stable)
  Penalty rate: <10%  (improved gating)
```

### Key Improvements
| Metric | Run B v2 | Run C v1 Target | Improvement |
|--------|----------|-----------------|-------------|
| Mean SPR | +0.018 | +0.040 | +122% |
| Mean variance | ±0.037 | ±0.030 | -19% |
| Trail-5 median | +0.147 | +0.400 | +171% |
| Finals variance | ±0.196 | ±0.100 | -49% |
| Penalty rate | ~15% | <10% | -33% |

---

## 🔍 Verification Before Start

### Config Verification
```powershell
# Should see:
# VAL_JITTER_DRAWS = 3
# VAL_EXP_TRADES_SCALE = 0.38
# FREEZE_VALIDATION_FRICTIONS = False
python -c "import config; c = config.Config(); print(f'VAL_JITTER_DRAWS = {c.VAL_JITTER_DRAWS}'); print(f'VAL_EXP_TRADES_SCALE = {c.VAL_EXP_TRADES_SCALE}')"
python -c "import config; print(f'FREEZE_VALIDATION_FRICTIONS = {config.FREEZE_VALIDATION_FRICTIONS}')"
```

### Trainer Verification
```powershell
# Should not error
python -c "import trainer; print('Trainer imports successfully')"
```

### Script Verification
```powershell
# Should exist
Test-Path compare_run_c_v1.py
Test-Path run_seed_sweep_organized.py
```

---

## 🎯 Next Steps

### Immediate
1. ✅ Run verification checks above
2. ⏳ Start Run C v1 (3 seeds × 120 episodes)
3. ⏳ Monitor progress (~6-8 hours)
4. ⏳ Analyze results with `compare_run_c_v1.py`
5. ⏳ Make GREEN/YELLOW/RED decision

### If GREEN
1. Lock Phase 2.8c as SPR Baseline v1.1
2. Archive config as `config_phase2.8c_baseline_v1.1.py`
3. Run 200-episode confirmation (5 seeds)
4. Select production seed (likely 777 or 27)
5. Paper trading integration

### If YELLOW
1. Run 120-episode test with 5 seeds
2. Adjust `VAL_EXP_TRADES_SCALE` to 0.36
3. Increase `VAL_JITTER_DRAWS` to 5
4. Re-evaluate

### If RED
1. Revert to Phase 2.8b
2. Analyze degradation cause
3. Consider alternative approaches

---

## 📝 Technical Details

### Jitter-Averaging Logic
```python
# For each window (lo, hi):
if jitter_draws > 1 and not freeze_frictions:
    jitter_fits = []
    for draw_i in range(jitter_draws):
        # Randomize frictions
        jitter_spread = current_spread * uniform(0.90, 1.10)
        jitter_commission = current_commission * uniform(0.90, 1.10)
        
        # Run validation slice
        stats = _run_validation_slice(lo, hi, jitter_spread, jitter_commission)
        jitter_fits.append(stats['fitness'])
    
    # Average over draws
    window_fit = mean(jitter_fits)
```

### Expected Variance Reduction
```
Single draw variance: σ = 0.037
Jitter-averaged (K=3): σ ≈ 0.037 / √3 ≈ 0.021

Actual Run C v1 target: σ ≈ 0.030 (conservative, includes window variance)
```

### Trail-5 Median
```python
# Get last 5 episodes
last_5 = sorted(jsons, key=lambda x: x['episode'])[-5:]
finals = [j.get('spr_raw', 0.0) for j in last_5]
trail_5_median = median(finals)

# Variance reduction:
# Single episode: σ = 0.196
# Trail-5 median: σ ≈ 0.088 (central limit theorem)
```

---

## 🎉 Why This Should Work

### Theoretical Basis
1. **Jitter-averaging:** Central limit theorem → variance ÷ √K
2. **Tighter gating:** Reduces under-trade spikes without penalizing normal behavior
3. **Trail-5 median:** Robust estimator, less sensitive to outliers

### Empirical Evidence
- Run B v2 showed positive mean (+0.018) despite noise
- Friction jitter verified working (80 unique spreads)
- Trade pacing stable (25.7 ± 0.7)
- Seed 777 excellent (+0.071 mean)

### Expected Impact
- **Mean improvement:** +0.018 → +0.040 (jitter-averaged validation stabilizes learning signal)
- **Variance reduction:** ±0.037 → ±0.030 (K=3 averaging)
- **Finals stability:** ±0.196 → ±0.100 (trail-5 median)
- **Penalty reduction:** ~15% → <10% (tighter gating)

---

## 📊 Monitoring Checklist

During run, verify:
- [ ] "jitter-avg K=3" appears in logs
- [ ] Friction values vary (unique spreads)
- [ ] Penalty episodes <10%
- [ ] No crashes or hangs
- [ ] Episodes complete every ~3-4 minutes

After run, verify:
- [ ] All seeds completed (120 episodes each)
- [ ] Friction jitter working (`check_friction_jitter.py`)
- [ ] Results analyzed (`compare_run_c_v1.py`)
- [ ] GREEN/YELLOW/RED verdict

---

## ✅ Status Summary

**Implementation:** COMPLETE ✅  
**Configuration:** VERIFIED ✅  
**Documentation:** COMPLETE ✅  
**Ready to test:** YES ✅

**Next action:** Start Run C v1 with:
```powershell
python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 120
```

**Expected outcome:** 🟢 GREEN rating → Proceed to 200-episode production validation! 🚀
