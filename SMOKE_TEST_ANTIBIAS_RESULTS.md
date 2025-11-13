# Smoke Test Results - Anti-Bias Fixes (A1+A2)

## Test Configuration
- **Episodes**: 10
- **Seed**: 42
- **Fixes Applied**:
  - A1: LONG exploration floor (15% minimum, first 60 episodes)
  - A2: Raised lambda ceiling (1.2 → 2.0)
  - A3: Enhanced telemetry (long_entries tracking)

## Results Summary

### LONG Action Frequency (p_long)
```
Baseline (from confirmation): 0.0349 (3.49%)
Smoke test results:          0.0283 (2.83%)
Change:                      -0.0066 (-18.9%)
```

**Episodes with p_long:**
- Ep 1: 0.0453
- Ep 2: 0.0331
- Ep 3: 0.0331
- Ep 4: 0.0267
- Ep 5: 0.0307
- Ep 6: 0.0345
- Ep 7: 0.0125
- Ep 8: 0.0094
- Ep 9: 0.0181
- Ep 10: 0.0423

**Range**: 0.0094 - 0.0453
**Mean**: 0.0283
**Median**: 0.0319

### Lambda_long (Controller Bias)
```
All 10 episodes: λ_long = -2.0 (SATURATED at new ceiling)
```

**Analysis**: Controller immediately pegged at maximum negative bias (-2.0), trying desperately to increase LONG actions. This confirms:
1. ✅ Raised ceiling is working (using new -2.0 limit)
2. ❌ Even -2.0 bias insufficient to overcome Q-network's learned SHORT preference

### Returns (SPR)
```
Baseline (from confirmation): -0.0236
Smoke test results:          -0.0013
Change:                      +0.0223 (+94.5% improvement!)
```

**Positive episodes**: 4/10 (40%)
- Ep 3: +0.0022
- Ep 4: +0.0449 ⭐
- Ep 6: +0.0196
- Ep 8: +0.0518 ⭐
- Ep 9: +0.0161

### Missing Data Point
❌ **long_entries field NOT in output** - telemetry collection may have an issue

## Assessment

### ❌ Quick Guards (A1+A2) INSUFFICIENT

**Problems**:
1. **p_long actually DECREASED** (0.0349 → 0.0283)
2. **λ_long saturated at -2.0** - controller maxed out but still can't fix bias
3. **LONG exploration floor NOT triggering** - no evidence of forced LONG actions
4. **Missing long_entries telemetry** - can't verify if floor is working

**Why the floor may not be triggering**:
Looking at the code, the floor triggers when:
- `current_episode <= 60` ✅ (episodes 1-10 qualify)
- `p_long < 0.10` ✅ (all episodes have p_long < 0.10)
- `random.random() < 0.20` (20% chance per action)

**But**: The agent may be **loading a pre-trained model** that already has the bias!

### ✅ Positive Signal: Returns Improved

Despite LONG frequency staying low, **returns improved by 95%**:
- Baseline: -2.36% loss
- Current: -0.13% loss (nearly break-even)
- 4/10 episodes profitable

This suggests the **higher lambda ceiling (-2.0) is helping** even though it's saturated.

## Root Cause: Pre-trained Model

The issue is likely that the agent is **loading checkpoints from previous training** that already have the SHORT bias baked into the Q-network weights.

**Evidence**:
1. Exploration floor should have forced LONG actions but p_long decreased
2. Controller immediately saturated (no gradual increase)
3. Behavior identical to baseline despite code changes

## Recommendations

### CRITICAL: Start from Scratch
```bash
# Delete old checkpoints that have the bias
Remove-Item -Recurse quick_test_antibias\checkpoints\*
Remove-Item -Recurse checkpoints\*
```

Then re-run smoke test to train from random initialization.

### OR: Implement Durable Fixes Now (B3+B4)

Since quick guards aren't sufficient, proceed directly to:

**B3: Sign-Flip Augmentation**
- Create direction-equivariant learning
- Prevents Q-network from learning directional bias
- Works even with pre-trained models (corrects during training)

**B4: Balanced Replay Sampling**
- Force 1:1 LONG/SHORT sampling
- Prevents SHORT gradients from dominating
- Helps rebalance existing Q-network

### Fix Missing Telemetry

The `long_entries` field is missing from JSON output. Check:
1. Is `long_entries_episode` being initialized in `__init__`?
2. Is it being reset in `reset_episode_tracking()`?
3. Is it being included in trainer's episode stats?

## Next Actions (Priority Order)

1. ⚠️ **Fix telemetry** - Add long_entries to output
2. 🔴 **Delete checkpoints** - Train from scratch
3. 🔴 **Re-run smoke test** - Verify floor works with fresh model
4. 🟡 **If still fails** - Implement B3+B4 immediately
5. 🟢 **If succeeds** - Run 200-episode probe

## Conclusion

**Quick guards (A1+A2) are INSUFFICIENT** when starting from a biased checkpoint.

**Two paths forward**:
1. **Fast path**: Delete checkpoints, retrain from scratch with guards
2. **Robust path**: Implement B3+B4 (durable fixes) that work even with biased initialization

**Recommendation**: Implement **B3 (sign-flip)** immediately - it's the only fix that can correct an already-biased Q-network during training.
