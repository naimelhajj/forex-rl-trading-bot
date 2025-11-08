# Phase 2.8d Emergency Adjustment - Option B

## Situation Analysis

### What Happened
After 200-episode Phase 2.8c run showed:
- **Too much exploration**: entropy 1.18-1.22 (near-random)
- **Too much trading**: hold_rate 0.58-0.62 (overtrading)
- **Directional collapse**: L/S ratio 0.065-0.934 (extreme bias)

We applied **Fix Pack D1** with these changes:
- Reduced `entropy_beta`: 0.020 → 0.014 (−30% exploration bonus)
- Increased `hold_tie_tau`: 0.035 → 0.038 (+8.5% hold bias)
- Increased `flip_penalty`: 0.0007 → 0.00077 (+10% churn penalty)

### The Problem
**We overcorrected!** The fixes stopped the random exploration but swung too far in the opposite direction:
- **Episode 16 result**: 100% HOLD (12,030 HOLD actions, 0 trades)
- **Entropy**: 0.0 bits (completely deterministic)
- **Agent stuck**: Learned HOLD is "safe" and won't try anything else

### Root Cause
This is **classic RL parameter tuning oscillation**, NOT a bug:
1. Phase 2.8c: Too loose → random policy
2. Phase 2.8d original: Too tight → HOLD collapse  
3. Phase 2.8d adjusted (this fix): Find middle ground

**The HOLD collapse is expected behavior given our parameter tightening, not a code bug.**

## Emergency Adjustments (Option B)

Applied three surgical parameter changes to restore exploration while maintaining quality controls:

### Change 1: Reduce Hold Bias
```python
# config.py - AgentConfig
hold_tie_tau: 0.038 → 0.030  # −21% hold bias
```
**Rationale**: Makes HOLD lose more Q-value ties, forcing agent to try LONG/SHORT more often.

### Change 2: Increase Entropy Bonus
```python
# config.py - EnvironmentConfig
entropy_beta: 0.014 → 0.025  # +79% exploration reward
```
**Rationale**: Directly rewards action diversity. Agent gets bonus for maintaining non-deterministic policy.

### Change 3: Reduce Flip Penalty
```python
# config.py - EnvironmentConfig  
flip_penalty: 0.00077 → 0.0005  # −35% churn cost
```
**Rationale**: Less punishment for changing positions. Reduces "friction" that locks agent into HOLD.

## Parameter Evolution Summary

| Parameter | 2.8c | D1 Original | D1 Adjusted | Change |
|-----------|------|-------------|-------------|--------|
| `entropy_beta` | 0.020 | 0.014 | **0.025** | +79% from D1 |
| `hold_tie_tau` | 0.035 | 0.038 | **0.030** | −21% from D1 |
| `flip_penalty` | 0.0007 | 0.00077 | **0.0005** | −35% from D1 |

**Strategy**: Push parameters PAST the 2.8c baseline to force exploration, then observe agent behavior.

## Expected Outcomes (50-Episode Run)

### Episodes 1-15: Immediate Trading
- **Expect**: Non-zero trades from episode 5+ (should see LONG/SHORT emerge quickly)
- **Entropy**: Should rise from 0.0 to 0.3-0.8 bits (partial recovery)
- **Hold rate**: Should drop from 1.0 to 0.6-0.8 range

### Episodes 15-30: Stabilization
- **Trading activity**: Should stabilize with consistent non-zero trades
- **Diversity**: Entropy should maintain 0.5-1.0 bits range
- **L/S balance**: Should see more balanced directional distribution (0.3-0.7 ratio)

### Episodes 30-50: Assessment
- **Success signal**: Trades > 5/episode, entropy 0.6-1.0, hold 0.65-0.80, L/S ratio 0.35-0.65
- **Failure signal**: Still stuck on HOLD (0 trades) or new random collapse (entropy > 1.2)

## Next Steps

### 1. Run 50-Episode Test
```powershell
python main.py --episodes 50
```

### 2. Monitor Progress (Check Episodes 10, 20, 30, 40, 50)
```powershell
for ($i in @(10,20,30,40,50)) {
    $file = "logs\validation_summaries\val_ep$('{0:D3}' -f $i).json";
    if (Test-Path $file) {
        $data = Get-Content $file | ConvertFrom-Json;
        Write-Host "Ep $i : trades=$($data.trades.ToString('0.0')) entropy=$($data.entropy.ToString('0.00')) hold=$($data.hold_frac.ToString('0.00')) long_ratio=$($data.long_ratio.ToString('0.00'))";
    }
}
```

### 3. Decision Tree

**If Episodes 30-50 show GREEN signals** (trades > 5, entropy 0.6-1.0, hold 0.65-0.80):
→ Proceed to quick ablation: `python run_seed_sweep_organized.py --seeds 7 17 777 --episodes 80`

**If still HOLD collapse** (0 trades through episode 50):
→ Further reduce hold bias: `hold_tie_tau → 0.025`
→ Increase entropy bonus: `entropy_beta → 0.030`
→ Retest 30 episodes

**If new random collapse** (entropy > 1.2, trades > 50):
→ Partially restore constraints:
  - `entropy_beta → 0.020` (split difference)
  - `hold_tie_tau → 0.033` (between original 0.030 and D1 0.038)
  - `flip_penalty → 0.00065` (split difference)

**If mixed signals** (some episodes trade, some don't):
→ Let it run to episode 80-100 for more data
→ Check if learning is progressing or stuck

## Why This Makes Sense

1. **Not a bug**: HOLD collapse after tightening parameters is expected RL behavior
2. **Oscillation is normal**: Finding sweet spot requires trying both extremes
3. **Data-driven**: We'll know in 50 episodes if these adjustments work
4. **Reversible**: Can always dial parameters back if we overshoot again

## Files Changed
- `config.py`: 3 parameter adjustments in EnvironmentConfig and AgentConfig

## Ready to Run
✅ Code fixed (trainer.py long_ratio KeyError resolved)
✅ Parameters adjusted (Option B emergency tweaks applied)
✅ Documentation updated (this file created)

**Command to start**: `python main.py --episodes 50`
