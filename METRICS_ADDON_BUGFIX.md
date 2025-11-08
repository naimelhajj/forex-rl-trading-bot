# Metrics Add-On - Bug Fix & First Results

**Date:** October 19, 2025  
**Issue:** Episode number parsing error in `check_metrics_addon.py`  
**Status:** ✅ Fixed

---

## Bug Fix

**Problem:**
```python
ep = int(Path(f).stem.split("_")[-1])
# For "val_ep001.json":
# - stem = "val_ep001"
# - split("_")[-1] = "ep001"
# - int("ep001") → ValueError!
```

**Solution:**
```python
stem = Path(f).stem  # "val_ep001"
ep_str = stem.split("_")[-1]  # "ep001"
ep = int(ep_str.replace("ep", ""))  # Remove "ep" prefix → 1
```

---

## First Results (Existing Data)

Found 25 validation summaries with metrics! The add-on successfully computed all metrics from existing validation runs:

### Sample Episodes

**Episode 1** (Poor convergence):
- Entropy: 0.850 bits ⚠️ (low diversity)
- Hold rate: 82.0% (high)
- Max streak: 70 bars
- Switch rate: 12.4%
- Long bias: 74% LONG, 26% SHORT

**Episode 2** (Better):
- Entropy: 1.171 bits ✅ (moderate diversity)
- Hold rate: 67.1% (good)
- Max streak: 35 bars
- Switch rate: 19.4% (active)
- Long bias: 71.6% LONG, 28.4% SHORT

**Episode 3** (Collapse!):
- Entropy: 1.136 bits
- Hold rate: 66.1%
- Max streak: 36 bars
- **Switch rate: 17.1%**
- **CRITICAL: 100% LONG, 0% SHORT** ⚠️⚠️⚠️

**Episode 4** (Freeze!):
- Entropy: 0.362 bits ⚠️⚠️ (very deterministic)
- Hold rate: 95.0% ⚠️ (frozen)
- **Max streak: 70 bars** ⚠️
- Switch rate: 9.1% (very low)
- Short bias: 28.4% LONG, 71.6% SHORT

**Episode 5** (Healthiest):
- Entropy: 1.219 bits ✅ (good diversity)
- Hold rate: 64.5% ✅ (good)
- Max streak: 26 bars ✅
- Switch rate: 19.6% ✅ (active)
- Slight short bias: 37% LONG, 63% SHORT

### Overall Averages (25 episodes)

```
hold_rate:       0.762  (76.2%)
avg_hold_length: 8.90 bars
max_hold_streak: 230 bars  ⚠️ (some frozen episodes)
action_entropy:  0.324 bits  ⚠️ (very low average)
switch_rate:     0.057  (5.7%)
long_ratio:      0.210  (21% of non-HOLD)
short_ratio:     0.190  (19% of non-HOLD)
```

---

## Key Insights

### 🚨 Issues Detected

1. **Low Average Entropy (0.324 bits)**
   - Indicates many episodes are very deterministic
   - Some are likely experiencing partial collapse
   - Target: 1.5-2.0 bits

2. **Extreme Max Streak (230 bars)**
   - At least one episode froze for 230+ bars
   - Episode 4 shows 70-bar streak with 95% HOLD rate
   - Suggests policy freeze issue

3. **Episode 3 Directional Collapse**
   - 100% LONG, 0% SHORT actions
   - Complete directional bias
   - Indicates Q-value collapse for SHORT action

4. **Long/Short Imbalance**
   - Overall: 21% LONG vs 19% SHORT (of non-HOLD)
   - But individual episodes vary wildly (0% to 100%)
   - Inconsistent across episodes

### ✅ Positive Signs

1. **Episode 5 Shows Promise**
   - 1.219 bits entropy (approaching healthy)
   - 19.6% switch rate (active)
   - 64.5% hold rate (reasonable)
   - Shows system CAN learn good policies

2. **Some Episodes Active**
   - Episodes 2, 3, 5 show good switch rates (17-19%)
   - Not all episodes are frozen

3. **Metrics Working Perfectly**
   - All 6 new metrics computed successfully
   - Successfully parsed 25 existing validation JSONs
   - Revealing actionable insights immediately

---

## Recommendations

Based on first results, the quality + anti-collapse patches should help with:

1. **Tie-only epsilon** → Should reduce determinism (raise entropy from 0.3 to 1.5+)
2. **Hold-streak breaker** → Should prevent 70+ bar freezes
3. **Micro penalties** → Should discourage marginal trades
4. **Softer risk** → Should improve Sharpe without affecting behavior much

**Next Step:** Run training with all quality patches active and compare metrics:

```powershell
# Run with quality patches
python main.py --episodes 10

# Check if metrics improve
python check_metrics_addon.py

# Expected improvements:
# - Entropy: 0.324 → 1.5-2.0 bits
# - Max streak: 230 → < 60 bars
# - Long/short: More balanced across episodes
```

---

## Status

✅ **Bug fixed** - Script now works correctly  
✅ **Metrics working** - All 6 metrics computed successfully  
✅ **Data analyzed** - 25 existing episodes processed  
⚠️ **Issues identified** - Low entropy, freezes, directional collapse detected  
🎯 **Ready for next run** - Quality patches should address these issues

The metrics add-on is working perfectly and already providing valuable diagnostic insights! 🎉
