# Phase 2.6 + SPR Fitness - Quick Start Guide

**Date:** 2025-01-22  
**Status:** ✅ READY - Config tweaks + SPR fitness implemented

---

## What's New (2 Major Changes)

### 1. ✅ **Phase 2.6 Churn Reduction** (Config Only)

**Environment:**
- `cooldown_bars: 8 → 10` (+2 bars)
- `min_hold_bars: 4 → 5` (+1 bar)
- `trade_penalty: 0.00005 → 0.00006` (+20%)
- `flip_penalty: 0.0005 → 0.0006` (+20%)

**Agent:**
- `eval_epsilon: 0.06 → 0.05` (-17% probing)
- `hold_tie_tau: 0.030 → 0.032` (+7% tolerance)
- `hold_break_after: 6 → 7` (+17% patience)

**Expected:** Trades 27 → 24 average, spikes 40-43 → max ~32

---

### 2. ✅ **SPR Fitness Metric** (New Module)

**Replaces:** Sharpe/CAGR composite  
**New formula:** `SPR = (PF / MDD%) × MMR% × Significance × Stagnation_Penalty`

**Components:**
- **PF** = Profit Factor (gross profit / gross loss, capped at 10)
- **MDD%** = Max Drawdown percentage (floored at 0.05%)
- **MMR%** = Mean Monthly Return as % of initial balance
- **Significance** = (min(1, trades_per_year / 100))² (guards low-sample luck)
- **Stagnation_Penalty** = 1 - (days_since_peak / test_days)

**Why SPR?**
- Sharpe can be inflated by serial correlation in returns
- CAGR doesn't account for drawdown severity
- SPR balances profitability, risk, consistency, and activity

---

## Files Modified

### 1. `config.py` - Fitness Config
```python
@dataclass
class FitnessConfig:
    # Fitness mode selection
    mode: str = "spr"  # "spr" or "legacy"
    
    # SPR parameters
    spr_pf_cap: float = 10.0
    spr_target_trades_per_year: float = 100.0
    spr_dd_floor_pct: float = 0.05
    spr_use_pandas: bool = True
    
    # Legacy weights (for mode="legacy")
    sharpe_weight: float = 1.0
    cagr_weight: float = 2.0
    # ... etc
```

### 2. `spr_fitness.py` - New Module
Contains:
- `compute_spr_fitness()` - Main function
- `_profit_factor()` - PF calculation
- `_max_drawdown_pct()` - MDD calculation
- `_monthly_return_pct_mean()` - MMR calculation
- `_stagnation_days()` - Days since peak
- Helper functions for timestamp conversion

### 3. `trainer.py` - Integration (PENDING)
Need to add SPR computation to validation loop.

---

## Integration Steps (trainer.py)

### Step 1: Import SPR Module

Add at top of `trainer.py`:
```python
from spr_fitness import compute_spr_fitness
```

### Step 2: Collect Trade-Level Data

In your validation loop, you'll need to collect:
```python
# Per validation window, track:
window_timestamps = []      # Bar timestamps
window_equity = []          # Equity curve
window_trade_pnls = []      # Realized P&L per trade
```

**Where to collect:**
- `window_timestamps`: Already have in validation env
- `window_equity`: Track balance after each step
- `window_trade_pnls`: Extract from env when trades close

### Step 3: Compute SPR Instead of Sharpe/CAGR

Replace your current fitness computation:

**Before (Sharpe/CAGR):**
```python
sharpe = compute_sharpe(returns, ...)
cagr = compute_cagr(equity, ...)
fitness = sharpe * sharpe_weight + cagr * cagr_weight
```

**After (SPR):**
```python
if self.config.fitness.mode == "spr":
    fitness_raw, spr_info = compute_spr_fitness(
        timestamps=window_timestamps,
        equity_curve=window_equity,
        trade_pnls=window_trade_pnls,
        initial_balance=self.config.environment.initial_balance,
        seconds_per_bar=3600,  # 1H bars = 3600 seconds
        pf_cap=self.config.fitness.spr_pf_cap,
        dd_floor_pct=self.config.fitness.spr_dd_floor_pct,
        target_trades_per_year=self.config.fitness.spr_target_trades_per_year,
        use_pandas=self.config.fitness.spr_use_pandas,
    )
    
    # Apply existing gating (mult, penalties)
    fitness_adj = fitness_raw * mult - undertrade_penalty - iqr_penalty
    
    # Store for logging
    summary["spr"] = round(fitness_raw, 6)
    summary["pf"] = round(spr_info["pf"], 3)
    summary["mdd_pct"] = round(spr_info["mdd_pct"], 3)
    summary["mmr_pct"] = round(spr_info["mmr_pct_mean"], 3)
    summary["significance"] = round(spr_info["significance"], 3)
    
else:
    # Legacy Sharpe/CAGR path
    fitness_raw, fitness_adj = compute_legacy_fitness(...)
```

### Step 4: Update Logging Output

Replace Sharpe/CAGR prints with SPR components:

**Before:**
```
Val - Reward: ... Sharpe: -3.46 | CAGR: -29.93%
```

**After:**
```
Val - Reward: ... SPR: 2.345 | PF: 1.85 | MDD: 8.2% | MMR: 1.5%
```

---

## Testing Workflow

### Quick Test (10 episodes)

```powershell
# Clear old validation JSONs
if (Test-Path .\logs\validation_summaries) { 
    Remove-Item .\logs\validation_summaries\* -Recurse -Force 
}

# Run smoke test
python main.py --episodes 10

# Check SPR values appear in logs
# Look for: "SPR: X.XXX | PF: X.XX | MDD: X.XX% | MMR: X.XX%"
```

**What to check:**
- ✅ SPR values are reasonable (0.1 to 100 typical range)
- ✅ PF between 0.5 and 10.0 (capped)
- ✅ MDD% between 0.05% and 50% (floored)
- ✅ MMR% makes sense (e.g., -2% to +5% monthly)
- ✅ No crashes or NaN values

### Full Test (20 episodes)

```powershell
# Run single seed with Phase 2.6 + SPR
python run_seed_sweep_organized.py --seeds 7 --episodes 20

# Check results
python check_validation_diversity.py
# Target: Trades 24-28, max <35

python check_metrics_addon.py
# Target: Entropy 0.78+

# NEW: Check SPR distribution
python -c "
import json
import glob
files = glob.glob('logs/validation_summaries/val_ep*.json')
sprs = [json.load(open(f))['spr'] for f in files[:20]]
print(f'SPR range: [{min(sprs):.3f}, {max(sprs):.3f}]')
print(f'SPR mean: {sum(sprs)/len(sprs):.3f}')
"
```

### Comparison Test (3 seeds × 80 episodes)

```powershell
# Full run with SPR
python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 80

# Compare against baseline (if you have Sharpe/CAGR runs)
python compare_seed_results.py
```

---

## SPR Tuning Knobs

### If SPR scores too low (all < 1.0):

**Option 1:** Lower drawdown floor (makes denominator smaller)
```python
spr_dd_floor_pct: 0.05 → 0.02
```

**Option 2:** Increase target trades (easier to hit significance)
```python
spr_target_trades_per_year: 100 → 150
```

**Option 3:** Increase PF cap (rewards very profitable policies)
```python
spr_pf_cap: 10.0 → 15.0
```

### If SPR scores too high (all > 1000):

**Option 1:** Raise drawdown floor
```python
spr_dd_floor_pct: 0.05 → 0.20
```

**Option 2:** Decrease target trades (stricter significance)
```python
spr_target_trades_per_year: 100 → 50
```

**Option 3:** Lower PF cap
```python
spr_pf_cap: 10.0 → 5.0
```

### If SPR variance too high:

**Option 1:** Use log-scale SPR for selection
```python
fitness_for_selection = np.log1p(spr_score)
```

**Option 2:** Clip extreme SPR values
```python
spr_clipped = np.clip(spr_score, 0.1, 100.0)
```

---

## Expected Results (Phase 2.6 + SPR)

### Trade Activity
```
Before (Phase 2.5):
- Trades/window: 22-43 (average 27.5)
- Spikes: 40-43 trades
- Entropy: ~0.74 bits

After (Phase 2.6):
- Trades/window: 20-28 (average 24)  ✅
- Spikes: max ~32 trades  ✅
- Entropy: ~0.78-0.82 bits  ✅
```

### Fitness Scores
```
Sharpe/CAGR (legacy):
- Range: -5.0 to +3.0
- Mean: ~-0.5 to +0.5
- Selection: Based on Sharpe × weight + CAGR × weight

SPR (new):
- Range: 0.1 to 50 (typical)
- Mean: ~5-15 (depends on tuning)
- Selection: Higher SPR = better
- Interpretable: SPR of 10 means "(PF/MDD%) × MMR% × guards = 10"
```

### Gating (Unchanged)
```
Penalty rate: ~10-20%  ✅
mult=1.00 rate: ~70-80%  ✅
Window sizing: 600 bars, K=7  ✅
Thresholds: min_full≈19  ✅
```

---

## Decision Tree After Testing

```
After 20-episode test:

SPR values reasonable (0.1-100)?
├─ YES → ✅ SPR working!
│         Check correlation with equity growth
│         Verify not rewarding churn
│
└─ NO (all < 0.1 or all > 1000) → Tune parameters
         Check: pf_cap, dd_floor_pct, target_trades
         See "SPR Tuning Knobs" above

Trades 24-28 per window?
├─ YES → ✅ Churn reduction working!
│
└─ NO (still 30+) → Further churn reduction
         Increase cooldown to 12
         OR increase flip_penalty to 0.0007

Entropy > 0.77?
├─ YES → ✅ Quality diversity!
│
└─ NO → Adjust exploration
         Keep eval_epsilon at 0.05
         Check tie-only logic working
```

---

## Integration Checklist

Before running full tests:

- [ ] `spr_fitness.py` created and tested
- [ ] `config.py` updated with fitness.mode and SPR params
- [ ] `trainer.py` imports `compute_spr_fitness`
- [ ] Validation loop collects: timestamps, equity, trade_pnls
- [ ] SPR computed when `fitness.mode == "spr"`
- [ ] Logging updated to show SPR components
- [ ] Quick 10-episode test passes
- [ ] SPR values in reasonable range
- [ ] No crashes or NaN values

Full testing:

- [ ] 20-episode spot-check completed
- [ ] Trades in 24-28 range
- [ ] Entropy > 0.77
- [ ] SPR correlates with equity growth
- [ ] 3-seed × 80-episode run scheduled

---

## Fallback Plan

If SPR causes issues:

1. **Switch back to legacy:**
   ```python
   # config.py
   fitness.mode = "legacy"
   ```

2. **Keep Phase 2.6 tweaks:**
   - Churn reduction still valuable
   - Independent of fitness metric

3. **Debug SPR separately:**
   - Test with synthetic data
   - Verify component calculations
   - Tune parameters offline

---

## Why This Combination Works

**Phase 2.6:**
- Reduces noise trades (40+ spikes)
- Tightens score distribution
- Makes agent behavior more predictable

**SPR Fitness:**
- Avoids Sharpe's serial correlation bias
- Balances profit, risk, and consistency
- Guards against low-sample luck
- Penalizes stagnation

**Together:**
- Phase 2.6 creates cleaner behavior for SPR to measure
- SPR rewards quality over quantity (matches churn reduction goal)
- Both align toward "selective, consistent profitability"

---

## Next Steps

1. **Integrate SPR into trainer.py** (see Step 2-4 above)
2. **Run 10-episode smoke test**
3. **Verify SPR values reasonable**
4. **Run 20-episode spot-check**
5. **If successful, proceed to 3-seed × 80**

**All config changes complete! Ready for trainer integration.** 🎯🚀

---

**Files ready:**
- ✅ `config.py` - SPR params added
- ✅ `spr_fitness.py` - Module complete
- ⏳ `trainer.py` - Integration pending (your next step)
