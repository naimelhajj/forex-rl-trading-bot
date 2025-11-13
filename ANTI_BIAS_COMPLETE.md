# Anti-Bias Fix - Complete Implementation

## ✅ COMPLETED: Option 3 Implementation

### Phase 1: Nuclear Reset
```powershell
✅ Deleted all checkpoints with SHORT bias
   - checkpoints/*
   - quick_test_antibias/checkpoints/*
   - confirmation_results/*/checkpoints/*
```

### Phase 2: Sign-Flip Augmentation (B3)

**New File**: `augmentation.py`
- Direction-sensitive feature detection
- State flipping utilities
- Action swapping (LONG ↔ SHORT)
- Symmetry loss computation

**Modified**: `agent.py` train_step()
- Integrated symmetry loss: `L = L_TD + 0.2 * L_sym`
- Graceful fallback if augmentation fails
- Works with existing PER and n-step returns

**How It Works**:
1. Sample normal batch from replay buffer
2. Create flipped batch: flip direction features, swap LONG/SHORT
3. Compute TD loss on original batch
4. Compute symmetry loss: `MSE(Q(s',LONG), Q(s,SHORT)) + MSE(Q(s',SHORT), Q(s,LONG))`
5. Backprop combined loss

**Direction-Sensitive Features** (negated):
- OHLC prices (indices 0-3)
- Linear regression slope (index 18)
- Currency strength momentum (indices 23-34)

**Direction-Neutral Features** (unchanged):
- Time (hour, day, year - cyclical)
- Volatility (ATR, realized vol)
- RSI
- Percentiles
- Fractals

### Phase 3: Guards Still Active (A1+A2)

**A1: LONG Exploration Floor**
- Forces 15% LONG actions when p_long < 10%
- Active for first 60 episodes
- Now will work with fresh Q-network

**A2: Raised Lambda Ceiling**
- LAMBDA_MAX = 2.0 (was 1.2)
- Allows stronger controller bias
- Will revert to 1.2 after confirmation passes

## Testing Plan

### Step 1: Fresh Smoke Test (NOW)
```bash
python main.py --episodes 10 --seed 999 --telemetry extended --output-dir fresh_antibias_test
```

**Expected Results** (with fresh init + B3):
- p_long should increase from ~0% to 10-20% over 10 episodes
- Lambda_long should decrease from 0 towards -2.0 as it detects imbalance
- Some LONG entries should occur (long_entries > 0)
- Symmetry loss should decrease (network learning equivariance)

**Success Criteria**:
- p_long > 0.05 (better than 3.5% baseline)
- long_entries > 0 in multiple episodes
- No crashes or NaN values
- Symmetry loss converges

### Step 2: 200-Episode Probe (if Step 1 passes)
```bash
python main.py --episodes 200 --seed 42 --telemetry extended --output-dir probe_200ep_b3
```

**Success Criteria**:
- Episodes 1-60: LONG floor active, p_long gradually increases
- Episodes 61-200: p_long ∈ [0.30, 0.70] in ≥50% of episodes
- Lambda_long saturation < 80% (not always at -2.0)
- Mean SPR > 0 (positive returns)

### Step 3: Configuration Adjustment
After 200-episode probe passes:
```python
# In agent.py:
self.LONG_FLOOR_EPISODES = 0        # Disable exploration floor
self.LAMBDA_MAX = 1.2               # Revert to normal ceiling
```

### Step 4: Full Confirmation (Final Test)
```bash
python run_confirmation_suite.py --seeds 42,123,456,789,1011 --episodes 200
python analyze_confirmation_results.py --results-dir confirmation_results
```

**Target**: All 9 gates pass
- Gate 5 (Long Ratio): ≥70% episodes in [0.40, 0.60]
- Gate 1 (Mean SPR): ≥ +0.04
- Gate 8 (Switch Rate): ≥70% in [0.15, 0.19]

## Git Status

**Commit**: d36e635
**Message**: "Phase 2.8f: Complete anti-bias fix (B3) - Sign-flip augmentation + fresh start"

**Files Changed**:
- `agent.py`: +28 lines (symmetry loss integration)
- `augmentation.py`: +210 lines (NEW - sign-flip module)
- `ANTI_BIAS_FIX_PLAN.md`: Documentation
- `ANTI_BIAS_IMPLEMENTATION_SUMMARY.md`: Implementation details
- `SMOKE_TEST_ANTIBIAS_RESULTS.md`: Initial test analysis
- `analyze_smoke_test.py`: Diagnostic tool

## Key Technical Details

### Symmetry Loss Formula
```
L_sym = MSE(Q(flip(s), LONG), Q(s, SHORT)) + 
        MSE(Q(flip(s), SHORT), Q(s, LONG))

Total loss = L_TD + 0.2 * L_sym
```

### Why This Works

**Problem**: Q-network learned from biased data:
- Training data may have bearish trend
- Early exploration found SHORT more profitable
- Replay buffer has 7× more SHORT experiences
- Gradient updates dominated by SHORT examples

**Solution**: Sign-flip augmentation teaches invariance:
- For every bearish pattern → LONG transition, network sees mirror bullish pattern → SHORT
- Even if real data is 90% bearish, augmented batch is 50/50
- Q-values become direction-equivariant: Q(s, LONG) ≈ Q(flip(s), SHORT)
- Network can't overfit to directional bias

**Advantages**:
- ✅ Works even with biased replay buffer
- ✅ Works even with biased training data
- ✅ Prevents future bias from emerging
- ✅ No reward corruption
- ✅ No hard constraints on actions
- ✅ Clean mathematical guarantee

## Monitoring Commands

**Check progress during training**:
```bash
# Real-time monitoring
python monitor_antibias.py

# Post-hoc analysis
python analyze_smoke_test.py

# Check action distribution
python diagnose_action_bias.py
```

## Rollback Procedure

If sign-flip augmentation causes issues:

1. **Disable augmentation** (comment out in agent.py):
```python
# In train_step(), change:
use_augmentation = False  # Force disable
```

2. **Revert lambda ceiling**:
```python
self.LAMBDA_MAX = 1.2  # Back to normal
```

3. **Disable LONG floor**:
```python
self.LONG_FLOOR_EPISODES = 0  # Turn off guard
```

4. **Re-run baseline**:
```bash
python main.py --episodes 200 --seed 42
```

## Next Immediate Action

**RUN THIS NOW**:
```bash
python main.py --episodes 10 --seed 999 --telemetry extended --output-dir fresh_antibias_test
```

Then analyze with:
```bash
python analyze_smoke_test.py  # Update path in script to fresh_antibias_test
```

**Look for**:
- [ ] p_long increasing (target: >5%)
- [ ] long_entries > 0 
- [ ] Lambda_long responding to imbalance
- [ ] No crashes
- [ ] Symmetry loss in training output

## Expected Timeline

- **Now**: Fresh 10-episode smoke test (~20 min)
- **+30 min**: Analyze results, verify B3 working
- **+1 hour**: Run 200-episode probe if smoke test passes
- **+5 hours**: Analyze probe, adjust guards if needed
- **+6 hours**: Start full 5-seed confirmation (~10 hours)
- **+16 hours**: Analyze confirmation, tag v2.8f-confirmed if passes

## Success Metrics

**Immediate** (10 episodes):
- p_long > 0.05 (vs 0.0283 biased baseline)
- long_entries detected
- Training stable

**Short-term** (200 episodes):
- p_long ∈ [0.30, 0.70] in ≥50% of episodes
- Lambda saturation < 80%
- Mean SPR > 0

**Long-term** (Confirmation):
- Gate 5: ≥70% in-band
- All 9 gates pass
- Ready for production

## Documentation

All implementation details in:
- `ANTI_BIAS_FIX_PLAN.md` - Original action plan
- `ANTI_BIAS_IMPLEMENTATION_SUMMARY.md` - Complete implementation guide
- `SMOKE_TEST_ANTIBIAS_RESULTS.md` - Analysis of initial test
- `augmentation.py` - Code comments and docstrings

---

**STATUS**: ✅ Ready to test
**NEXT**: Run fresh smoke test and verify sign-flip augmentation works
