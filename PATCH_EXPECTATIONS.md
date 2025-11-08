# What to Expect After Patches

## Summary of Changes

All 6 focused patches have been successfully implemented:

1. ✅ **Fitness Hygiene**: Metrics computed on same equity series after ruin-clamp with business-day resample
2. ✅ **Frame Stacking**: Agent sees last 3 observations for temporal memory (state size increased by 3x for market features)
3. ✅ **Heuristic Prefill**: 1000-3000 baseline transitions loaded before training
4. ✅ **Meaningful SL Action**: MOVE_SL_CLOSER requires ≥1 pip meaningful tightening
5. ✅ **Deterministic Eval**: Validation uses eval_mode=True (epsilon=0, frozen noise)
6. ✅ **Training Stability**: gamma=0.97, batch=256, grad_clip=1.0, soft updates

---

## Expected Behavior

### When Running `python main.py --episodes 5`

**1. Startup (First ~10 seconds)**
```
Loading data and computing features...
[PREFILL] Collecting 1000 baseline transitions...
  Collected 1000/1000 transitions
[PREFILL] Complete. Buffer size: 1000
```

**Key Points:**
- Prefill takes ~5-10 seconds for 1000 transitions
- Buffer starts with sensible baseline actions (not random noise)
- No scipy import hang (that was fixed earlier)

**2. Training Episodes (1-5)**
```
Episode 1/5
  Steps: 450, Reward: -2.34, Trades: 28, Win Rate: 42.9%
  Validation: Fitness=0.12, Sharpe=0.31, CAGR=0.08
  
Episode 2/5
  Steps: 520, Reward: -1.87, Trades: 35, Win Rate: 45.7%
  Validation: Fitness=-0.05, Sharpe=0.18, CAGR=-0.02
```

**Key Points:**
- 20-40 trades per episode (healthy trading activity)
- Rewards can be negative initially (learning phase)
- Fitness around 0 is expected (slightly positive/negative)
- Validation is deterministic (eval_mode=True)

**3. State Size Message (First Episode)**
```
[DEBUG] State size: 143 (feature_dim=40, stack_n=3, context=23)
```

**Key Points:**
- State size = 40*3 + 23 = 143 (with frame stacking)
- Agent network auto-adjusts to new size

---

### When Running `python main.py --episodes 20`

**1. Prefill Phase**
```
[PREFILL] Collecting 3000 baseline transitions...
  Collected 1000/3000 transitions
  Collected 2000/3000 transitions
  Collected 3000/3000 transitions
[PREFILL] Complete. Buffer size: 3000
```

**Key Points:**
- Takes ~15-30 seconds to collect 3000 transitions
- More baseline data = better initial policy

**2. Training Progress**
```
Episode 1/20: Reward=-1.45, Trades=32, Fitness=0.08
Episode 5/20: Reward=-0.82, Trades=28, Fitness=0.15
Episode 10/20: Reward=0.34, Trades=24, Fitness=0.28
  ✓ New best fitness (EMA): 0.28 (raw: 0.31)
Episode 15/20: Reward=0.67, Trades=26, Fitness=0.42
  ✓ New best fitness (EMA): 0.35 (raw: 0.42)
```

**Key Points:**
- Rewards gradually improve (negative → positive)
- Fitness curve shows upward trend with oscillations
- Median validation (K=7) + EMA smoothing reduces noise
- Early stop only after min_validations=6

**3. Validation Behavior**
- Runs K=7 validation passes with different spread/commission jitters
- Uses **median** of K fitnesses (resistant to outliers)
- Applies EMA smoothing (alpha=0.3) for early stopping metric
- Deterministic policy (eval_mode=True)

---

## Key Metrics to Watch

### 1. Trades Per Episode
**Good:** 20-40 trades
**Too Low:** <10 trades (agent being too conservative)
**Too High:** >60 trades (overtrading, likely unprofitable)

### 2. Validation Fitness
**Good:** Oscillates near 0, occasionally positive, shows upward trend
**Bad:** Stuck at large negative values (<-2.0)
**Warning:** Too stable (little variation) might indicate overfitting

### 3. Sharpe Ratio
**Good:** >0.3 after 10+ episodes
**Acceptable:** 0.1-0.3 during learning
**Bad:** <0 consistently

### 4. Win Rate
**Good:** 40-55%
**Acceptable:** 35-40% if profit factor >1.2
**Bad:** <35% (poor trade quality)

---

## Debugging Tips

### If Training Hangs
- Check for infinite loops in feature computation
- Verify frame stacking not accumulating memory
- Monitor CPU/memory usage

### If Fitness Stays Very Negative
- Check prefill is working (buffer should have 1000-3000 transitions)
- Verify baseline policy makes sense (look at actions chosen)
- Check for excessive costs (spread, commission, slippage)

### If Agent Not Trading
- Check legal_action_mask() - might be blocking all actions
- Verify cooldown_bars and min_hold_bars not too restrictive
- Look at exploration (NoisyNet sigma_init might be too low)

### If Validation Unstable
- Verify eval_mode=True is working (should be deterministic)
- Check K=7 validation passes are running
- Median + EMA smoothing should reduce noise

---

## Files to Check After Running

### 1. Logs Directory
```
logs/
  events.out.tfevents.* (TensorBoard logs)
  episode_events.jsonl (detailed episode data)
  error_events.jsonl (any errors)
```

### 2. Checkpoints Directory
```
checkpoints/
  best_model.pt (best fitness checkpoint)
  best_model_scaler.json (feature normalization params)
  final_model.pt (last episode checkpoint)
  final_model_scaler.json
```

### 3. Console Output
- Look for "[PREFILL]" messages
- Check "✓ New best fitness" updates
- Monitor "⚠ Early stop" if triggered

---

## Common Issues & Fixes

### Issue: NameError about 'time' module
**Fix:** Already fixed by removing all timing instrumentation from main.py

### Issue: State size mismatch
**Fix:** Agent auto-adjusts on first forward pass. If error persists, check:
- `env.state_size` matches actual state.shape[0]
- Frame stacking properly initialized in reset()

### Issue: Replay buffer not filling
**Fix:** Check prefill_steps > 0 in config and trainer

### Issue: MOVE_SL_CLOSER never used
**Fix:** Verify min_trail_buffer_pips not too large (default 1.0 is good)

---

## Performance Expectations

### Smoke Run (5 episodes)
- **Time:** ~2-5 minutes total
- **Trades:** 20-40 per episode
- **Fitness:** -0.5 to +0.5 range
- **Win Rate:** 35-50%

### Full Run (20 episodes)
- **Time:** ~10-30 minutes total
- **Trades:** 20-40 per episode
- **Fitness:** 0 to +1.0 range by end
- **Win Rate:** 40-55% by end

### Long Run (100 episodes)
- **Time:** 1-2 hours
- **Trades:** 15-35 per episode (should decrease as policy improves)
- **Fitness:** 0.5 to +2.0 range
- **Win Rate:** 45-60%

---

## Next Actions

1. **Verify Compilation:**
   ```bash
   python -m py_compile fitness.py agent.py environment.py trainer.py config.py
   ```

2. **Run Smoke Test:**
   ```bash
   python main.py --episodes 5
   ```

3. **Monitor Output:**
   - Prefill should complete in ~10 seconds
   - Training should start immediately
   - No hangs or errors
   - Validation runs every episode

4. **Check Results:**
   - Fitness around 0 (±0.5) is good for 5 episodes
   - Trades in 20-40 range shows healthy activity
   - Win rate >35% shows some learning

5. **If All Good, Run Full Test:**
   ```bash
   python main.py --episodes 20
   ```

---

## Success Criteria

✅ **Patch Implementation Successful If:**
1. Prefill runs and loads 1000-3000 transitions
2. Training starts without hangs
3. State size correctly accounts for frame stacking
4. Validation is deterministic (same seed → same results)
5. Fitness computed consistently (no NaN or odd values)
6. MOVE_SL_CLOSER action used when appropriate
7. No syntax errors or import failures

---

## Contact/Debugging

If issues persist:
1. Check `FOCUSED_PATCHES_SUMMARY.md` for implementation details
2. Review error logs in `logs/error_events.jsonl`
3. Verify all 6 patches applied correctly using file diffs
4. Check agent network architecture matches new state size

All patches maintain backward compatibility - can be individually disabled by reverting specific config values.
