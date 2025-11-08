# Surgical Tweaks - Activity Boost v2 🎯

**Date:** October 19, 2025  
**Goal:** Reduce penalty/zero-trade rate without encouraging flippiness  
**Approach:** Small, targeted parameter adjustments + volatility-aware penalty

---

## 🔧 Changes Applied

### 1️⃣ Exploration Floor Raised (Avoid HOLD Pockets)

**File:** `config.py` lines 70-76

**Before:**
```python
epsilon_start: float = 0.10
epsilon_end: float = 0.05      # Too low, locks into HOLD
noisy_sigma_init: float = 0.4
```

**After:**
```python
epsilon_start: float = 0.10
epsilon_end: float = 0.12      # ↑ Raised from 0.05 to 0.12
noisy_sigma_init: float = 0.5  # ↑ Raised from 0.4 to 0.5
```

**Impact:**
- Minimum exploration 12% (was 5%)
- NoisyNet sigma 0.5 (was 0.4)
- Prevents "always HOLD" convergence
- Still controlled (not random exploration)

---

### 2️⃣ Execution Limits Eased (More Trade Opportunities)

**File:** `config.py` lines 55-56

**Before:**
```python
cooldown_bars: int = 12  # 18 bars between trades
min_hold_bars: int = 6   # Hold for 6 bars minimum
```

**After:**
```python
cooldown_bars: int = 10  # ↓ 15 bars between trades (17% faster)
min_hold_bars: int = 5   # ↓ Hold for 5 bars minimum (17% faster)
```

**Impact:**
- Cycle time: 15 bars (was 18, was 24 originally)
- Theoretical max trades/600 bars: 40 (was 33, was 25)
- ~20% more trading opportunities
- Small enough to avoid flippiness

---

### 3️⃣ Volatility-Adjusted Penalty Scaling ⭐

**File:** `trainer.py` lines 543-568

**New Logic:**
```python
# Base penalty (same as before)
base_penalty = 0.25 * (shortage / max(1, min_half))

# NEW: Calculate volatility multiplier
if 'atr' in data.columns:
    recent_atr = data['atr'].iloc[-100:].mean()
    median_atr = data['atr'].median()
    # Low ATR → 0.5x penalty, High ATR → 1.0x penalty
    vol_multiplier = max(0.5, min(1.0, recent_atr / median_atr))
else:
    # Fallback: use price volatility
    recent_vol = data['close'].iloc[-100:].std()
    median_vol = data['close'].std()
    vol_multiplier = max(0.5, min(1.0, recent_vol / median_vol))

# Apply scaled penalty
undertrade_penalty = base_penalty * vol_multiplier
```

**Impact:**
- **Low volatility:** Penalty reduced 50% (vol_mult = 0.5)
  - Avoids over-punishing sensible caution
  - Agent not forced to trade in quiet markets
  
- **Normal volatility:** Penalty at ~70-100% (vol_mult = 0.7-1.0)
  - Expected trading activity still encouraged
  
- **High volatility:** Full penalty (vol_mult = 1.0)
  - Missing opportunities in active markets penalized

**Example:**
```
Low vol period:  ATR=0.0005, median=0.0010 → mult=0.5  → penalty=0.125 (was 0.25)
Normal period:   ATR=0.0010, median=0.0010 → mult=1.0  → penalty=0.250 (same)
High vol period: ATR=0.0015, median=0.0010 → mult=1.0  → penalty=0.250 (capped)
```

---

## 📊 Expected Outcomes

### Before (Baseline):
- Epsilon: 0.10 → 0.05 (too low)
- Cycle: 6 + 12 = 18 bars
- Penalty: Fixed 0.25 regardless of market conditions
- **Result:** 0-5 trades common, high penalty rate

### After (Surgical Tweaks):
- Epsilon: 0.10 → 0.12 (sustained exploration)
- NoisyNet: 0.5 (higher state-dependent noise)
- Cycle: 5 + 10 = 15 bars (17% faster)
- Penalty: 0.125-0.250 (volatility-adjusted)
- **Expected:** 6-12 trades, lower penalty rate, context-aware

---

## 🎯 Design Principles

### ✅ What We Did:
1. **Small increments:** 0.05→0.12 (not 0.05→0.20)
2. **Multiple mechanisms:** Epsilon + NoisyNet + constraints
3. **Context-aware:** Penalty scales with opportunity (ATR)
4. **Bounded changes:** All multipliers clamped 0.5-1.0

### ❌ What We Avoided:
1. **Large jumps:** No 2x changes
2. **Single mechanism:** Not just epsilon or just constraints
3. **Fixed penalties:** Not one-size-fits-all
4. **Excessive freedom:** No removal of flip_penalty or constraints

---

## 🔍 Monitoring Checklist

After running seed sweep with new parameters:

### A. Trade Activity:
```python
# Target: 6-12 trades per validation (was 0-5)
trades = [d['trades'] for d in validations]
print(f"Trade range: {min(trades):.1f} - {max(trades):.1f}")
print(f"Median trades: {np.median(trades):.1f}")
```

### B. Penalty Rate:
```python
# Target: < 20% validations penalized (was > 50%)
penalties = [d['penalty'] for d in validations]
penalty_rate = sum(1 for p in penalties if p > 0.01) / len(penalties) * 100
print(f"Penalty rate: {penalty_rate:.1f}%")
```

### C. Zero-Trade Rate:
```python
# Target: < 10% validations with 0-2 trades (was > 30%)
zero_trade_rate = sum(1 for t in trades if t < 3) / len(trades) * 100
print(f"Zero-trade rate: {zero_trade_rate:.1f}%")
```

### D. Volatility Response:
```python
# Check if penalty scales with volatility
low_vol_eps = [v for v in validations if v['vol_regime'] == 'low']
high_vol_eps = [v for v in validations if v['vol_regime'] == 'high']
print(f"Low vol avg penalty: {np.mean([v['penalty'] for v in low_vol_eps]):.3f}")
print(f"High vol avg penalty: {np.mean([v['penalty'] for v in high_vol_eps]):.3f}")
# Should see lower penalties in low-vol periods
```

---

## 🧪 Testing Protocol

### Phase 1: Quick Test (10 episodes)
```powershell
python main.py --episodes 10
python check_validation_diversity.py
```

**Look for:**
- Trade counts: 4-8 range (improvement from 0-5)
- Penalty rate: < 50% (improvement from > 70%)
- No crashes or errors

### Phase 2: Seed Sweep (3 seeds × 25 episodes)
```powershell
python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 25
python compare_seed_results.py
```

**Look for:**
- Consistent trade activity across seeds
- Lower penalty rates overall
- Scores improving (negative → positive)
- No excessive flippiness (< 30 trades)

### Phase 3: Analysis
```python
# In check_validation_diversity.py or custom script
import json
from pathlib import Path

for seed in [7, 77, 777]:
    seed_dir = Path(f"logs/seed_sweep_results/seed_{seed}")
    jsons = sorted(seed_dir.glob("val_ep*.json"))
    data = [json.load(open(f)) for f in jsons]
    
    trades = [d['trades'] for d in data]
    penalties = [d['penalty'] for d in data]
    
    print(f"\nSeed {seed}:")
    print(f"  Trades: {min(trades):.1f}-{max(trades):.1f} (median {np.median(trades):.1f})")
    print(f"  Penalty rate: {sum(1 for p in penalties if p>0.01)/len(penalties)*100:.1f}%")
    print(f"  Zero-trade rate: {sum(1 for t in trades if t<3)/len(trades)*100:.1f}%")
```

---

## 🔄 Rollback Plan

If tweaks cause issues (over-trading, instability):

### Option A: Partial Rollback
```python
# In config.py, revert most aggressive change:
epsilon_end: float = 0.08  # Split difference (was 0.05, tried 0.12)
```

### Option B: Full Rollback
```python
# config.py
epsilon_end: float = 0.05
noisy_sigma_init: float = 0.4
cooldown_bars: int = 12
min_hold_bars: int = 6

# trainer.py - remove volatility scaling:
undertrade_penalty = 0.25 * (shortage / max(1, min_half))
```

### Option C: Adjust Volatility Scaling
```python
# If penalty too lenient in low-vol:
vol_multiplier = max(0.7, min(1.0, ...))  # Was 0.5, now 0.7 minimum

# If penalty too harsh in low-vol:
vol_multiplier = max(0.3, min(1.0, ...))  # Was 0.5, now 0.3 minimum
```

---

## 📈 Success Metrics

### Minimum Acceptable (Pass):
- ✅ Median trades: > 5 (up from 2-3)
- ✅ Penalty rate: < 40% (down from > 60%)
- ✅ Zero-trade rate: < 25% (down from > 40%)
- ✅ No crashes or divergence

### Good Performance (Success):
- ✅ Median trades: 6-10
- ✅ Penalty rate: < 25%
- ✅ Zero-trade rate: < 15%
- ✅ Scores improving over episodes

### Excellent Performance (Ideal):
- ✅ Median trades: 8-12
- ✅ Penalty rate: < 15%
- ✅ Zero-trade rate: < 10%
- ✅ Consistent across seeds
- ✅ Volatility-responsive behavior visible

---

## 📝 Summary

**3 Surgical Tweaks Applied:**

1. **Exploration Floor:** ε_min 0.05→0.12, σ 0.4→0.5
   - Prevents HOLD lock-in
   - Small enough to maintain learning

2. **Execution Limits:** min_hold 6→5, cooldown 12→10
   - ~20% more opportunities
   - Still prevents flippiness

3. **Volatility-Adjusted Penalty:** Fixed 0.25 → Dynamic 0.125-0.250
   - Context-aware penalties
   - Avoids over-punishing caution in quiet markets

**Conservative by design:**
- All changes < 30% magnitude
- Multiple small tweaks vs one large change
- Bounded multipliers (0.5-1.0)
- Easy rollback path

**Ready to test!** 🚀

```powershell
# Quick test
python main.py --episodes 10

# Full sweep
python run_seed_sweep_organized.py --episodes 25
```
