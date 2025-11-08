# Fix Pack D2 Level 2 Escalation - Analysis & Next Steps

## Episode 50 Post-Mortem: Why Fix Pack D2 Failed

### Observed Collapse Pattern
- **Episode 10**: 0.022 (2% long - SHORT collapse!)
- **Episode 20**: 0.573 (57% long - temporarily balanced)
- **Episode 50**: 0.926 (93% long - LONG collapse!)

### Root Cause: Penalties Too Weak

**Math breakdown (Episode 50 - 93% long)**:
```
Imbalance = |0.926 - 0.5| - 0.15 = 0.276
Penalty per step = 0.006 × (0.276)² = 0.000457
Over 500 bars = 500 × 0.000457 = 0.228 reward units
```

**Meanwhile**:
- Typical trading rewards: **+1 to +5 per step**
- Per-episode cumulative reward: **+10 to +50**
- Penalty impact: **<1% of total reward**

**Conclusion**: Agent ignores penalty because it's **1000x smaller** than trading rewards.

---

## Level 2 Escalation: 8-10x Stronger Penalties

### Parameter Changes

**config.py**:
```python
ls_balance_lambda: 0.050    # Was 0.006 (8.3x increase)
hold_balance_lambda: 0.020  # Was 0.002 (10x increase)
```

**environment.py**:
```python
margin: 0.10  # Was 0.15 (tighter: 40-60% range, not 35-65%)
```

### New Penalty Math (93% long scenario)

```
Imbalance = |0.93 - 0.5| - 0.10 = 0.33
Penalty per step = 0.050 × (0.33)² = 0.00545
Over 500 bars = 500 × 0.00545 = 2.73 reward units
```

**Expected impact**:
- Per-episode cumulative reward: **+10 to +50**
- Directional penalty: **~2-3 units** (now **5-30% of reward**)
- Agent **must** pay attention (no longer negligible)

### Why 8-10x is the Right Scale

1. **Below 5x**: Still negligible vs trading rewards
2. **8-10x**: Enters "noticeable but not dominant" range
3. **Above 20x**: Risk over-constraining (may force 50/50 regardless of data)

---

## Success Criteria (Level 2)

### Episode 10 Checkpoint (~30 min)
- 🟢 **0.40-0.60**: Excellent - penalties working
- 🟡 **0.30-0.70**: Good - continue monitoring
- 🔴 **<0.30 or >0.70**: Still collapsing - go to Level 3

### Episode 20 Checkpoint (~1 hour)
- Must maintain 0.40-0.60 range
- If drifting >0.65 or <0.35 → STOP and escalate

### Episode 50-80 (Completion)
- Consistent 0.40-0.60 across all episodes
- No "slow drift" pattern (e.g., 0.45 → 0.50 → 0.55 → 0.65 → 0.75)

---

## If Level 2 Still Fails

### Level 3 Escalation (20x penalties)
```python
ls_balance_lambda: 0.120    # 20x original
hold_balance_lambda: 0.040  # 20x original
margin: 0.08                # 42-58% range
```

### Level 4: Episodic Hard Penalty
```python
# environment.py - in episode done
if episode_long_ratio > 0.65 or episode_long_ratio < 0.35:
    reward -= 10.0  # Large episodic penalty
```

### Level 5: Multi-Objective Optimization
Instead of single reward, optimize:
```
fitness = (SPR * 0.7) + (balance_score * 0.3)
where balance_score = 1.0 - |long_ratio - 0.5|
```

---

## Alternative Hypothesis

### Is LONG Actually Optimal in Training Data?

**Check data regime**:
```powershell
python -c "import pandas as pd; df=pd.read_csv('data/EURUSD_M30_train.csv'); print(f'Train period: {df.Date.iloc[0]} to {df.Date.iloc[-1]}'); print(f'Total return: {((df.Close.iloc[-1]/df.Close.iloc[0])-1)*100:.2f}%')"
```

**If training data is strongly bullish** (e.g., +20% LONG bias):
- Agent may be "correctly" learning LONG strategy
- Forcing 50/50 may be **counter-productive**
- Solution: Multi-objective (SPR + balance) or separate evaluation metric

---

## Next Actions

1. **START Level 2 run**: `.\start_d2_level2.ps1`
2. **Monitor Episode 10**: Check if long_ratio 0.40-0.60
3. **If still collapsed**: Go to Level 3 (20x penalties)
4. **If working**: Complete to Episode 80, then cross-validate seeds 123, 777

---

## Technical Notes

### Why Quadratic Penalties?
- **Linear**: Small imbalances ignored
- **Quadratic**: Exponential growth prevents extremes
- **At 93% long**: Quadratic penalty 27x larger than linear

### Why Rolling Window?
- Episode-level only checks at end (too late)
- Rolling window provides **continuous feedback**
- 500-bar window = ~1 full episode context

### Why 40-60% vs 35-65%?
- 35-65% too permissive (agent drifts to 65%, then 70%, then 80%...)
- 40-60% forces tighter clustering around 50/50
- Can relax later if too constraining

---

**Status**: Level 2 Escalation Ready  
**Estimated Time**: ~4 hours (80 episodes)  
**Next Checkpoint**: Episode 10 (~30 minutes)
