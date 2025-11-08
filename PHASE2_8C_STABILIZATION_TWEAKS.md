# Phase 2.8c - Stabilization Tweaks

## 🎯 Objective
Turn Run B v2's 🟡 YELLOW rating into 🟢 GREEN by stabilizing validation readings and reducing noise.

## 📊 Run B v2 Results (Baseline for 2.8c)
- **Cross-seed mean:** +0.018 ± 0.037
- **Cross-seed final:** +0.147 ± 0.196
- **Trades/ep:** 25.7 ± 0.7
- **Friction jitter:** ±10% spread/commission (WORKING ✅)
- **Seeds:** 7, 77, 777 (3 seeds × 80 episodes)
- **Issue:** Finals noisy (±0.196), last-episode luck dominates

## 🔧 Stabilization Tweaks

### 1. Jitter-Averaged Validation (K=3)
**Problem:** Single friction draw per validation → noisy readings  
**Solution:** Average each validation over K=3 friction draws

**Implementation:**
```python
# config.py
VAL_JITTER_DRAWS: int = 3  # Average over K=3 jitter draws per validation
```

**trainer.py changes:**
- Modified `validate()` to run K draws per window when `VAL_JITTER_DRAWS > 1`
- Each draw uses randomized frictions within ±10% range
- Fitness averaged across draws (mean of K SPR scores)
- Trade counts averaged across draws
- Action metrics aggregated across draws

**Expected impact:**
- Reduce final-episode luck (variance ÷ √3 ≈ 0.58x)
- More stable SPR readings across episodes
- Finals variance: ±0.196 → ±0.11 (target)

### 2. Tighter Trade-Count Gating
**Problem:** `VAL_EXP_TRADES_SCALE=0.32` too lenient → occasional under-trade spikes  
**Solution:** Raise to 0.38 for gentler floor

**Changes:**
```python
# config.py
VAL_EXP_TRADES_SCALE: float = 0.38  # Raised from 0.32
```

**Expected impact:**
- `min_full` rises from ~19 to ~22 trades (600-bar windows)
- Most episodes still get mult=1.00 (typical 25-27 trades)
- Filter occasional under-trade spikes without penalizing normal behavior

### 3. Enhanced Reporting (Trail-5 Median)
**Problem:** Reporting last-episode final score → sensitive to final-episode luck  
**Solution:** Report trailing-5 median alongside last-episode

**Implementation:**
- `compare_run_c_v1.py` computes trail-5 median from last 5 episodes
- More robust measure of final performance
- Reduces impact of single lucky/unlucky episode

**Expected impact:**
- Finals reporting: ±0.196 → ±0.10 (trail-5 median variance)
- Better signal for production seed selection

### 4. Keep Existing SPR Caps
**Decision:** Maintain Phase 2.8b SPR tweaks
- PF cap at 6.0 (suppress outlier wins)
- MDD soft-floor (emphasize consistency)
- IQR penalty cap at 0.6 (tighter stability)

## 📋 Run C v1 Protocol

### Configuration
```python
# Phase 2.8c: Stabilization tweaks
VAL_JITTER_DRAWS = 3              # Jitter-averaged validation
VAL_EXP_TRADES_SCALE = 0.38       # Tighter gating
VAL_SPREAD_JITTER = (0.90, 1.10)  # ±10% (realistic)
VAL_COMMISSION_JITTER = (0.90, 1.10)  # ±10%
FREEZE_VALIDATION_FRICTIONS = False  # Jitter enabled
```

### Test Setup
- **Seeds:** 7, 77, 777 (quick 3-seed test, can expand to 5)
- **Episodes:** 120 per seed (~6-8 hours total)
- **Friction:** ±10% jitter, K=3 averaged per validation
- **Gating:** VAL_EXP_TRADES_SCALE = 0.38

### Expected Results
✅ **GREEN criteria:**
- Cross-seed mean: +0.03 to +0.06
- Cross-seed trail-5: +0.30 to +0.50
- Mean variance: ±0.025 to ±0.035
- Penalty rate: <10%
- Positive trail-5: 3/3 seeds (100%)

🟡 **YELLOW criteria:**
- Cross-seed mean: +0.02 to +0.03
- Cross-seed trail-5: +0.20 to +0.30
- Mean variance: ±0.035 to ±0.045
- Penalty rate: 10-15%

🔴 **RED criteria:**
- Cross-seed mean: <+0.02
- Degradation vs Run B v2
- High variance (>±0.05)
- Penalty rate: >15%

### Monitoring Commands
```powershell
# Monitor training progress
Get-ChildItem logs\seed_sweep_results\seed_* -Directory | Select-Object Name, LastWriteTime

# Check if process running
Get-Process python -ErrorAction SilentlyContinue

# Analyze results after completion
python compare_run_c_v1.py
```

## 🔬 Technical Details

### Jitter-Averaged Validation Logic
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

### Trade-Count Gating Impact
```python
# 600-bar window, EXP_TRADES_SCALE = 0.38:
expected = 600 * 0.38 = 228 bars
avg_hold = 600 / 25 = 24 bars (for 25 trades)
min_full = expected / avg_hold ≈ 22 trades

# Multiplier calculation:
trades < 22 → mult = trades / 22 (penalty)
22 ≤ trades ≤ 24 → mult = 1.00 (full credit)
trades > 24 → mult = 1.00 (no penalty for activity)
```

### Trail-5 Median Computation
```python
# Get last 5 episodes
last_5 = sorted(jsons, key=lambda x: x['episode'])[-5:]
finals = [j.get('spr_raw', 0.0) for j in last_5]
trail_5_median = median(finals)

# Variance reduction:
# Single episode: σ = 0.196
# Trail-5 median: σ ≈ 0.196 / √5 ≈ 0.088 (estimated)
```

## 📈 Expected Improvements

### vs Run A (Frozen Frictions)
- Run A mean: -0.004 ± 0.021
- Expected Run C v1: +0.04 ± 0.030 (GREEN target)
- **Improvement:** +0.044 mean gain, stable under jitter

### vs Run B v2 (Jitter, No Averaging)
- Run B v2 mean: +0.018 ± 0.037
- Expected Run C v1: +0.04 ± 0.030
- **Improvement:** +0.022 mean gain, 25% variance reduction

### Key Metrics Targets
| Metric | Run B v2 | Run C v1 Target | Improvement |
|--------|----------|-----------------|-------------|
| Cross-seed mean | +0.018 | +0.04 | +122% |
| Mean variance | ±0.037 | ±0.030 | -19% |
| Cross-seed trail-5 | +0.147 | +0.40 | +171% |
| Finals variance | ±0.196 | ±0.11 | -44% |
| Penalty rate | ~15% | <10% | -33% |

## 🚀 Next Steps

### If GREEN (Mean ≥ +0.03, Trail-5 ≥ +0.30)
1. ✅ Lock Phase 2.8c as SPR Baseline v1.1
2. ✅ Archive config as `config_phase2.8c_baseline_v1.1.py`
3. ✅ Run 200-episode confirmation (5 seeds: 7, 17, 27, 77, 777)
4. ✅ Select production seed (likely 777 or 27)
5. ✅ Paper trading integration

### If YELLOW (Mean +0.02-0.03)
1. Run 120-episode test with 5 seeds (expand seed diversity)
2. Fine-tune penalties:
   - Lower `VAL_EXP_TRADES_SCALE` to 0.36 (gentler gating)
   - Increase `VAL_JITTER_DRAWS` to 5 (more averaging)
3. Re-evaluate with expanded test

### If RED (Mean <+0.02)
1. Revert to Phase 2.8b (proven stable)
2. Analyze what caused degradation:
   - Gating too tight? (check penalty episodes)
   - Jitter averaging broken? (verify K=3 logic)
   - Overfitting to frozen frictions?
3. Consider alternative approaches:
   - Reduce jitter range to ±8%
   - Keep single-draw validation
   - Adjust penalty tuning

## 📝 Files Changed

### config.py
- Added `VAL_JITTER_DRAWS = 3`
- Changed `VAL_EXP_TRADES_SCALE = 0.38` (from 0.32)
- Updated comments to reflect Phase 2.8c

### trainer.py
- Modified `validate()` to support jitter-averaged validation
- Added loop over K friction draws per window
- Averaged fitness, trades, and action metrics across draws
- Updated log messages to show jitter-averaging status

### New Scripts
- `compare_run_c_v1.py` - Enhanced comparison with trail-5 median
- `PHASE2_8C_STABILIZATION_TWEAKS.md` - This document

## 🎯 Success Criteria Summary

**Run C v1 is GREEN if:**
- ✅ Cross-seed mean ≥ +0.03
- ✅ Cross-seed trail-5 ≥ +0.30
- ✅ Mean variance ≤ ±0.035
- ✅ Penalty rate <10%
- ✅ Positive trail-5: 3/3 seeds
- ✅ Friction jitter working (80+ unique spreads per seed)

**Impact:** Agent robustly trades under varying market frictions, ready for production validation (200 episodes).
