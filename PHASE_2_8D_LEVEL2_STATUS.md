# Phase 2.8d - Fix Pack D2 Level 2 Escalation - RUNNING

**Status**: Training started with 8-10x stronger penalties  
**Date**: 2025-01-15  
**Seed**: 42  
**Episodes**: 80 (early stop possible)

---

## Executive Summary

### What Happened (Level 1 Failure)
- **Episode 10**: 0.022 (2% long - SHORT collapse)
- **Episode 20**: 0.573 (57% long - temporarily balanced)
- **Episode 50**: 0.926 (93% long - LONG collapse)

**Root cause**: Penalties were **1000x too weak** compared to trading rewards.

### Level 2 Solution (NOW RUNNING)
- **8-10x stronger penalties** that agent cannot ignore
- **Tighter margin**: 40-60% (was 35-65%)
- **Expected outcome**: Long ratio stays 0.40-0.60 throughout training

---

## Parameter Changes

### Level 1 (FAILED) → Level 2 (CURRENT)

| Parameter | Level 1 | Level 2 | Change |
|-----------|---------|---------|--------|
| **ls_balance_lambda** | 0.006 | **0.050** | 8.3x |
| **hold_balance_lambda** | 0.002 | **0.020** | 10x |
| **Margin tolerance** | 0.15 | **0.10** | Tighter |
| **Allowed range** | 35-65% | **40-60%** | Narrower |

---

## Penalty Calculations

### Level 1 (Too Weak)
**At 93% long collapse**:
```
Imbalance = |0.93 - 0.5| - 0.15 = 0.276
Penalty = 0.006 × (0.276)² = 0.000457 per step
Over 500 bars = 0.228 reward units
Vs episode reward = +10 to +50
Impact = <1% (NEGLIGIBLE - agent ignores)
```

### Level 2 (Significant)
**At 93% long collapse**:
```
Imbalance = |0.93 - 0.5| - 0.10 = 0.33
Penalty = 0.050 × (0.33)² = 0.00545 per step
Over 500 bars = 2.73 reward units
Vs episode reward = +10 to +50
Impact = 5-30% (SIGNIFICANT - agent must respond)
```

**At 60% long (edge of allowed range)**:
```
Imbalance = |0.60 - 0.5| - 0.10 = 0.0
Penalty = 0.0 (no penalty within margin)
```

**At 65% long (slightly out of range)**:
```
Imbalance = |0.65 - 0.5| - 0.10 = 0.05
Penalty = 0.050 × (0.05)² = 0.000125 per step
Over 500 bars = 0.063 (small nudge back)
```

---

## Monitoring Plan

### Episode 10 Checkpoint (~30 minutes)

**Command**:
```powershell
$val = Get-Content logs\validation_summaries\val_ep010.json | ConvertFrom-Json
[math]::Round($val.long_short.long_ratio, 3)
```

**Decision matrix**:
- 🟢 **0.40-0.60**: Perfect! Level 2 working
- 🟡 **0.30-0.70**: Good progress, continue monitoring
- 🔴 **<0.30 or >0.70**: Escalate to Level 3
- 🛑 **<0.10 or >0.90**: STOP immediately

### Episode 20 Checkpoint (~1 hour)

**Must show**:
- Long ratio staying 0.40-0.60
- No gradual drift (e.g., 0.45 → 0.50 → 0.55 → 0.60 → 0.70)
- If drifting beyond 0.65: STOP and escalate

### Episode 40-50 (Mid-point)

**Watch for**:
- Consistent balance throughout
- Early stopping likely if converged
- Compare to Level 1 Episode 50 (was 0.926)

---

## Escalation Paths

### If Level 2 Still Fails (>0.70 or <0.30 at Episode 10)

**Level 3: 20x penalties**
```python
# config.py
ls_balance_lambda: 0.120    # 20x original (0.006)
hold_balance_lambda: 0.040  # 20x original (0.002)
margin: 0.08                # 42-58% range
```

**Penalty at 93% long**:
```
Imbalance = |0.93 - 0.5| - 0.08 = 0.35
Penalty = 0.120 × (0.35)² = 0.0147 per step
Over 500 bars = 7.35 reward units (15-75% of episode!)
```

### If Level 3 Fails

**Level 4: Episodic hard penalty**
```python
# environment.py - in episode_done()
episode_long_ratio = self.trades_long / (self.trades_long + self.trades_short)
if episode_long_ratio > 0.65 or episode_long_ratio < 0.35:
    reward -= 10.0  # Large one-time penalty at episode end
```

### If All Penalties Fail

**Fundamental hypothesis to test**:
1. Is training data strongly directional? (Check raw price trend)
2. Are features biased toward LONG? (Check feature engineering)
3. Is 50/50 balance actually optimal? (May need 60/40 if data is bullish)

---

## Expected Timeline

| Time | Episode | Checkpoint |
|------|---------|------------|
| **Now** | 1 | Training started |
| **+30 min** | 10 | **CRITICAL** - First assessment |
| **+1 hour** | 20 | Early validation |
| **+2 hours** | 40 | Mid-point check |
| **+2.5 hours** | 50 | Likely early stop (if converged) |
| **+4 hours** | 80 | Full completion (if no early stop) |

---

## Success Criteria

### Level 2 GREEN (proceed to cross-validation)
- ✅ Episode 10: long_ratio 0.40-0.60
- ✅ Episode 20: long_ratio 0.40-0.60
- ✅ Episode 50-80: long_ratio 0.40-0.60
- ✅ No collapse events across any episode
- ✅ Entropy 0.95-1.05, hold 0.65-0.78

**Next steps**:
- Run seeds 123, 777 (80 episodes each)
- If ≥2/3 seeds pass → Full confirmation (5 seeds × 150 episodes)

### Level 2 YELLOW (needs tuning)
- ⚠️ Episode 10: long_ratio 0.30-0.70 (borderline)
- ⚠️ Gradual drift observed (0.45 → 0.55 → 0.65)
- ⚠️ Occasional excursions to 0.70-0.75

**Actions**:
- Increase to 12x penalties (ls=0.060, hold=0.024)
- Tighten margin to 0.08 (42-58% range)
- Restart training

### Level 2 RED (escalate immediately)
- ❌ Episode 10: long_ratio >0.70 or <0.30
- ❌ Episode 20: collapse continues (>0.80 or <0.20)

**Actions**:
- STOP training immediately
- Escalate to Level 3 (20x penalties)
- Consider alternative approaches (episodic penalties, multi-objective)

---

## Technical Deep Dive

### Why Quadratic Penalties?

**Linear penalty**: `reward -= lambda × imbalance`
- At 10% imbalance: penalty = 0.050 × 0.10 = 0.005
- At 30% imbalance: penalty = 0.050 × 0.30 = 0.015 (3x)

**Quadratic penalty**: `reward -= lambda × imbalance²`
- At 10% imbalance: penalty = 0.050 × 0.01 = 0.0005
- At 30% imbalance: penalty = 0.050 × 0.09 = 0.0045 (9x)

**Effect**: Small deviations lightly penalized, large deviations heavily penalized → Prevents extremes.

### Why Rolling Window?

**Episode-level** (OLD - doesn't work):
- Only checks at episode end
- Agent can collapse for 450 bars, take 2 opposite trades at end
- Episode long_ratio = 0.94 but episode-level check sees "somewhat balanced"

**Rolling window** (NEW - continuous feedback):
- Checks every step over last 500 bars
- Agent gets penalty signal immediately as drift happens
- Cannot "game" the system with end-of-episode corrections

### Why 500 Bars?

- Training episodes: ~500-600 bars
- 500 bars = full episode context
- Shorter (200): Too noisy, agent exploits local patterns
- Longer (1000): Too stale, slow response to regime shifts
- Sweet spot: Intra-episode behavioral control

---

## Files Modified

1. **config.py** (Line 65-69):
   - `ls_balance_lambda`: 0.006 → **0.050**
   - `hold_balance_lambda`: 0.002 → **0.020**

2. **environment.py** (Line 528):
   - `margin`: 0.15 → **0.10**

3. **Scripts created**:
   - `start_d2_level2.ps1` - Restart with Level 2
   - `FIX_PACK_D2_LEVEL2_ESCALATION.md` - This document
   - `check_status.ps1` - Quick status checker

---

## Monitoring Commands

**Quick check**:
```powershell
.\check_status.ps1
```

**Episode 10 detailed**:
```powershell
$val = Get-Content logs\validation_summaries\val_ep010.json | ConvertFrom-Json
Write-Host "Long ratio: $([math]::Round($val.long_short.long_ratio, 3))"
Write-Host "Entropy: $([math]::Round($val.action_entropy_bits, 2))"
Write-Host "Hold rate: $([math]::Round($val.hold_rate, 2))"
Write-Host "SPR: $([math]::Round($val.score, 3))"
```

**Compare Level 1 vs Level 2**:
```powershell
$old = Get-Content logs_backup_seed42_fixpack_d1\val_ep010.json | ConvertFrom-Json
$new = Get-Content logs\validation_summaries\val_ep010.json | ConvertFrom-Json
Write-Host "Level 1 (Episode 10): $([math]::Round($old.long_short.long_ratio, 3))"
Write-Host "Level 2 (Episode 10): $([math]::Round($new.long_short.long_ratio, 3))"
```

---

## Hypothesis Testing

### If Level 2 Works
**Conclusion**: Agent was learning valid patterns but penalties were too weak to enforce balance constraint.

### If Level 2 Fails
**Investigate**:
1. **Data regime**: Is training data strongly directional?
2. **Feature bias**: Do features favor one direction?
3. **Reward structure**: Is directional parking actually optimal?
4. **Exploration**: Is epsilon-greedy insufficient?

---

**Status**: ⏳ TRAINING IN PROGRESS  
**Next Checkpoint**: Episode 10 (~30 minutes)  
**Critical Metric**: long_ratio must be 0.40-0.60  
**If >0.70 or <0.30**: Escalate to Level 3 immediately
