# Phase 2.8d Fix Pack D1: Recovery from 200-ep Confirm Failure

**Status**: Implementing surgical fixes to restore GREEN performance  
**Date**: November 5, 2025  
**Diagnosis**: Too-random policy, directional collapse, weak neutrality prior

---

## 🔴 200-ep Confirm Failure Analysis

### What Broke
- **Too-random policy**: Entropy ~1.18–1.22 (target: 0.90–1.10)
- **Hold rate too low**: 0.58–0.62 (target: 0.65–0.80)
- **Directional collapse**: Long_ratio extremes (0.065, 0.934) - seed-driven local minima
- **Gating too permissive**: 0% penalty with ~29–30 trades/ep (allows noisy high-turnover)

### Root Causes
1. Exploration too high → excessive randomness
2. Hold-tie threshold too loose → agent breaks position prematurely  
3. No L/S balance regularizer → seeds drift to directional extremes
4. Trade count gating too lenient → high-turnover noise survives

---

## ✅ Fix Pack D1 Implementation

### 1. **Reduce Exploration** (−30% entropy bonus)
```python
# config.py - NEW parameter
ENTROPY_BETA: float = 0.014  # Reduced from assumed 0.020 (−30%)
```

### 2. **Strengthen "Hold on Near-Ties"** (+0.003 to +0.005)
```python
# config.py - AgentConfig
hold_tie_tau: float = 0.038  # Raised from 0.035 (+0.003)
# Alternative: 0.040 for more aggressive hold bias
```

### 3. **Discourage Churn** (+10% switch cost)
```python
# config.py - EnvironmentConfig  
flip_penalty: float = 0.00077  # Raised from 0.0007 (+10%)
```

### 4. **Tighten Expected-Trades Gate**
```python
# config.py
VAL_EXP_TRADES_SCALE: float = 0.42  # Raised from 0.38
# Down-weights low-quality high-turnover regimes
```

### 5. **Trim Luck in Aggregation** (+0.05)
```python
# config.py
VAL_TRIM_FRACTION: float = 0.25  # Raised from 0.20 (+0.05)
# Removes top/bottom 25% to reduce outlier influence
```

### 6. **L/S Balance Regularizer** (NEW)
```python
# environment.py - Add to reward calculation
# Penalty ∝ |EMA(long_ratio) − 0.5|, weight λ = 0.003
LS_BALANCE_LAMBDA: float = 0.003  # Start conservative
```

---

## 🧪 Quick Ablation Protocol (3 seeds × 80 eps each)

**Purpose**: Identify minimal fix set that restores performance

**Variants** (cumulative):
- **A**: Base (2.8b) + #2 hold_tie_tau only
- **B**: A + #1 lower entropy  
- **C**: B + #3 switch_cost
- **D**: C + #4 exp_trades_scale
- **E**: D + #5 trim_fraction
- **F**: E + #6 L/S balance

**Acceptance Gate** (find first variant that passes):
- Mean SPR ≥ +0.03
- Trail-5 median ≥ +0.20
- Entropy 0.90–1.10
- Hold 0.65–0.80
- Long_ratio 0.40–0.60 on ≥2/3 seeds

**Command**:
```bash
python run_ablation_d1.py --seeds 7 17 777 --episodes 80
```

---

## 🎯 Full 150-ep Confirmation (Best Variant)

**After** finding best variant from ablation:

**Setup**:
- 5 seeds × 150 episodes
- Keep jitter-avg K=3 (eval)
- Robustness check: K=5 on last 10 episodes

**Acceptance Gates** (same as before):
1. Mean SPR ≥ +0.04
2. Trail-5 median ≥ +0.25  
3. σ(means) ≤ 0.035
4. Penalty rate ≤ 10%
5. ≥3/5 seeds with trail-5 > 0
6. Behavioral metrics in bands

**Command**:
```bash
python run_seed_sweep_organized.py --seeds 7 17 27 77 777 --episodes 150
```

---

## 📊 Diagnostics (Run Alongside)

### 1. **Direction PnL Split**
```bash
python analyze_direction_pnl.py --seeds 7 17 27 77 777
```
- Long vs short contribution per seed (fixed frictions)
- Verify balance (40-60% each direction)

### 2. **Rolling Behavior**  
```bash
python analyze_rolling_behavior.py --window 25
```
- 25-ep rolling: entropy, hold, switch, long_ratio
- Confirm settlement inside bands after ~50 eps

### 3. **Trade Quality vs Count**
```bash
python analyze_trade_quality.py
```
- Scatter: (trades/ep → SPR)  
- Verify gate (#4) suppresses over-active low-quality regimes

---

## 🔴 If Still Red After D1

**Escalation Options**:

1. **More aggressive hold bias**:
   - `hold_tie_tau: 0.040 → 0.042` (+0.002)
   
2. **Reduce entropy further**:
   - `ENTROPY_BETA: 0.014 → 0.0126` (−10%)
   
3. **Stronger L/S regularizer** (temporary):
   - `LS_BALANCE_LAMBDA: 0.003 → 0.006–0.008`
   - Back off if long_ratio stays ~0.5 for 100 eps

4. **Consider Phase 2.8b revert**:
   - If multiple fixes fail → analyze degradation deeper
   - Possible root cause: jitter-averaging broken? Overfitting?

---

## 📝 Implementation Checklist

- [ ] Add `ENTROPY_BETA` to config.py
- [ ] Implement entropy reward bonus in environment.py
- [ ] Update `hold_tie_tau` from 0.035 → 0.038
- [ ] Update `flip_penalty` from 0.0007 → 0.00077
- [ ] Update `VAL_EXP_TRADES_SCALE` from 0.38 → 0.42
- [ ] Update `VAL_TRIM_FRACTION` from 0.20 → 0.25
- [ ] Add L/S balance regularizer in environment.py
- [ ] Create `run_ablation_d1.py` script
- [ ] Create diagnostic scripts (direction_pnl, rolling_behavior, trade_quality)
- [ ] Document all changes in config/code
- [ ] Run ablation to find best variant
- [ ] Run full 150-ep confirmation
- [ ] Analyze results and update status

---

## 🎯 Expected Outcomes

**If GREEN** (all gates pass):
- Lock Phase 2.8d as SPR Baseline v1.2
- Select production seed (likely 777 or best performer)
- Proceed to paper trading integration

**If YELLOW** (1-2 gates marginal):
- Apply escalation options above
- Re-run 120-episode test with 5 seeds

**If RED** (multiple gates failed):
- Deep dive analysis: gating? jitter? overfitting?
- Consider revert to Phase 2.8b (proven stable)
- Explore alternative approaches

---

**Next Step**: Implement all 6 fixes, then run ablation to isolate minimal fix set.
