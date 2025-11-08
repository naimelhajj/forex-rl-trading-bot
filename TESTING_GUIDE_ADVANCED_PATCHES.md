# Quick Testing Guide for Advanced Learning Patches

## Immediate Smoke Test (5 minutes)

```powershell
# Run 5 episodes to verify all patches working
python main.py --episodes 5
```

**What to check in output:**
1. ✅ "Using NoisyNet" or epsilon stays at 0.0 (PATCH 1)
2. ✅ Learning starts around 360-600 steps, not 5000 (adaptive)
3. ✅ No "flat at $1000" validations (n-step helping)
4. ✅ State size includes 5 new regime features (check dimension)
5. ✅ Validation uses K=7 passes
6. ✅ See "median fitness" in logs (not just mean)
7. ✅ Action 3 (MOVE_SL_CLOSER) used at least once
8. ✅ Slippage varies (check episode logs)
9. ✅ No cost budget warnings (unless intentional)
10. ✅ `best_model_scaler.json` created in checkpoints/

## Detailed Checks

### PATCH 1: NoisyNet Verification
```python
# Check agent epsilon
import torch
from config import Config
from agent import DQNAgent

config = Config()
agent = DQNAgent(state_size=74, action_size=4, use_noisy=True, noisy_sigma_init=0.4)
print(f"Epsilon: {agent.epsilon}")  # Should be 0.0
print(f"Using noisy: {agent.use_noisy}")  # Should be True
```

### PATCH 2: N-step Returns Check
```python
# Verify replay buffer stores n-step transitions
from agent import ReplayBuffer
buffer = ReplayBuffer(capacity=1000, n_step=3, gamma=0.99)

# Push some transitions
for i in range(5):
    buffer.push(i, 0, 1.0, i+1, False)

# Check buffer contents (should have n-step rewards)
print(f"Buffer size: {len(buffer.buffer)}")
if len(buffer.buffer) > 0:
    sample = buffer.buffer[0]
    print(f"Transition format: {len(sample)} elements (should be 6)")
    print(f"Has n_step field: {sample[5] if len(sample) > 5 else 'NO'}")
```

### PATCH 3: Vol-aware Slippage Log
Look for in episode logs:
```
# Should see varying slippage values
Entry at 1.0850, slippage_eff: 0.68 pips (ATR: 12.3)
Entry at 1.0860, slippage_eff: 1.45 pips (ATR: 48.1)
```

### PATCH 4: MOVE_SL_CLOSER Usage
```powershell
# Count action 3 usage in logs
python main.py --episodes 5 > output.txt
# Then check:
grep -i "action.*3" output.txt | wc -l
# Should see multiple uses (>5)
```

### PATCH 5: Holding Cost
Check reward logs - should see tiny negative rewards while holding:
```
Step 100: reward=-0.000100 (holding, no PnL)
Step 101: reward=-0.000100 (holding, no PnL)
```

### PATCH 6: Median Fitness
Look for validation logs:
```
Validation fitness (median of 7): 0.4523 (raw values: [0.42, 0.45, 0.48, 0.44, 0.51, 0.43, 0.46])
EMA fitness: 0.4312
```

### PATCH 7: Scaler Files
```powershell
# Check scaler saved
ls checkpoints/best_model_scaler.json
# Should exist

# Check contents
cat checkpoints/best_model_scaler.json
# Should have mu, sig, feature_columns
```

### PATCH 8: Regime Features
```python
# Check state size increased by 5
from features import FeatureEngineer
fe = FeatureEngineer()
features = fe.get_feature_names()
print(f"Total features: {len(features)}")  # Should be 28 (was 23)
print("Regime features:", [f for f in features if 'vol' in f or 'trend' in f or 'trending' in f])
# Should show: realized_vol_24h_z, realized_vol_96h_z, trend_24h, trend_96h, is_trending
```

### PATCH 9: Grad Norm
```python
# After training starts
print(f"Last grad norm: {agent._last_grad_norm}")
# Should be 0.1-10.0 typically
# If >20, may need to reduce grad_clip
```

### PATCH 10: Cost Budget
```python
# Force high costs to test warning
env.spread = 0.01  # 100 pips spread
env.commission = 100.0  # $100/lot
# Run episode, should see warning:
# "Cost budget exceeded: $150.00 > $50.00 (150% threshold)"
```

## Full Run Test (30 minutes)

```powershell
python main.py --episodes 20
```

**Expected results:**
- Validation fitness more stable (EMA + median)
- Early stop at patience=10 (not 8)
- Action 3 usage: 5-15% of all actions
- Sharpe range: -0.3 to +0.8 (clearer wins)
- Fewer whipsaw fitness changes

## Performance Comparison

### Baseline (before patches)
- Learning starts: 5000 steps
- Flat validations: ~30% in first 10 episodes
- Action 3 usage: <1%
- Fitness whipsaw: ±0.3 per validation

### With patches (expected)
- Learning starts: 360-600 steps (6-8x faster)
- Flat validations: <10% in first 10 episodes
- Action 3 usage: 5-15%
- Fitness whipsaw: ±0.1 per validation (smoother)

## Troubleshooting

### Issue: "n_step not found in buffer sample"
**Fix:** Ensure both ReplayBuffer and PrioritizedReplayBuffer updated with n_step logic

### Issue: High grad norms (>50)
**Fix:** Reduce learning rate or increase grad_clip to 2.0

### Issue: Cost budget warnings every episode
**Fix:** Check spread/commission config - may be too high for $1000 balance

### Issue: Regime features all zeros
**Fix:** Check data has enough history (need 192+ bars for 96h features)

### Issue: Scaler not loading
**Fix:** Verify checkpoint path matches scaler file naming (`best_model_scaler.json`)

## TensorBoard Monitoring

```powershell
tensorboard --logdir=logs
```

**Key metrics to watch:**
- `train/grad_norm` - should be 0.5-5.0
- `val/median_fitness` - smoother than old val/mean_fitness
- `train/epsilon` - should stay at 0.0 with NoisyNet
- `q/max`, `q/mean` - should be stable, not exploding

## Success Criteria

✅ **All 10 patches working if:**
1. NoisyNet active (epsilon=0.0)
2. Learning starts <1000 steps
3. Buffer stores n-step transitions
4. Slippage varies with ATR
5. MOVE_SL_CLOSER used >0 times
6. Holding cost appears in rewards
7. Validation uses median
8. Scaler JSON files created
9. 5 regime features in state
10. Grad norms logged
11. Cost budget warnings work
12. No syntax/runtime errors

## Next Actions

1. ✅ Run smoke test (5 episodes)
2. ✅ Check all 10 patches active
3. ✅ Run full test (20 episodes)
4. ✅ Compare to baseline metrics
5. ✅ Tune hyperparameters if needed
6. 🚀 Deploy to production testing

---

**Last updated:** [Current Date]  
**Patches applied:** 10/10 ✅  
**Status:** Ready for testing
