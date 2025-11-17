# Probe Analysis & Next Steps

## Probe Results Summary

**Configuration**: 
- Nuclear reset (fresh start)
- Sign-flip augmentation (B3)
- LONG exploration floor (60 episodes, 50% @ p<0.25, mask override)
- Symmetry loss weight: 0.2

**Outcome**:
- ❌ **FAILED**: p_long collapsed from 25% → 4% when floor turned off (ep 61)
- ✅ Floor worked: Maintained p_long at 25% during episodes 1-60
- ⚠️  Sign-flip NOT learned: Q-network still has SHORT bias

## Detailed Metrics

| Metric | Episodes 1-60 | Episodes 61-63 | Target | Status |
|--------|---------------|----------------|--------|--------|
| p_long mean | 24.9% | 4.0% | 40-60% | ❌ Collapsed |
| p_long in-band | 0/60 (0%) | 0/3 (0%) | ≥70% | ❌ Failed |
| Lambda saturation | 60/60 (100%) | 3/3 (100%) | <80% | ❌ Maxed out |
| Mean SPR | -0.0174 | - | ≥0.04 | ⚠️  Negative |
| Val fitness | 1.466 (best) | - | Positive | ✅ Good |

## Root Cause Analysis

### Why Did p_long Collapse?

The floor was **forcing** LONG actions but the Q-network wasn't **learning** to prefer them:

1. **Insufficient training time**: 60 episodes = ~12,000 steps
   - Neural network needs more examples to learn symmetry
   - Replay buffer size: 100,000 (only 12% filled during warmup)

2. **Weak symmetry loss**: Weight 0.2 means TD loss dominates
   - L_total = L_TD + 0.2 * L_sym
   - Small symmetry penalty → Q-network prioritizes immediate rewards (SHORT bias remains)

3. **Environment rejection**: Forced LONG actions might be invalid
   - Cooldowns, position restrictions block actual position entries
   - If LONG actions don't execute, no learning from LONG experiences

### Evidence

```
Episodes 1-60:  p_long = 25%, lambda = -2.0 (controller fighting Q-network)
Episodes 61-63: p_long = 4%,  lambda = -2.0 (bias immediately returns)
```

Controller at maximum bias (-2.0) **entire time** → Q-network never learned balance.

## Fixes Applied

### Fix 1: Extended LONG Floor

```python
# Before: 60 episodes
self.LONG_FLOOR_EPISODES = 60

# After: 120 episodes  
self.LONG_FLOOR_EPISODES = 120
```

**Rationale**: 
- Double the training time with balanced data
- 120 episodes = ~24,000 steps with LONG experiences
- Replay buffer will be ~24% filled with balanced samples

### Fix 2: Stronger Symmetry Loss

```python
# Before: 0.2 (weak)
SYMMETRY_LOSS_WEIGHT = 0.2

# After: 0.5 (medium-strong)
SYMMETRY_LOSS_WEIGHT = 0.5
```

**Rationale**:
- L_total = L_TD + 0.5 * L_sym
- Symmetry violations now have 2.5× stronger penalty
- Q-network forced to balance LONG/SHORT Q-values

**Risk**: If too high, might interfere with learning actual market patterns
**Mitigation**: Can reduce back to 0.3-0.4 if SPR degrades

## Recommended Next Steps

### Option A: Quick Retest (30 min)

Run another 10-episode test to verify fixes work:

```bash
python main.py --episodes 10 --seed 42 --telemetry extended --output-dir quick_retest
```

**Look for**:
- p_long still ~25% (floor still working)
- No crashes with new symmetry weight

### Option B: Full 200-Episode Probe (3-5 hours)

```bash
# Clean old probe
Remove-Item -Recurse -Force probe_200ep_b3

# Run new probe with extended floor + stronger symmetry
python main.py --episodes 200 --seed 42 --telemetry extended --output-dir probe_200ep_b3_v2
```

**Expected trajectory**:
- Episodes 1-120: p_long at ~25% (floor active)
- Episodes 121-200: p_long drifts toward 40-60% (learned symmetry)
- Lambda saturation decreases from 100% to <50%
- Mean SPR becomes positive

**Success criteria**:
- [ ] p_long ∈ [0.35, 0.65] in episodes 150-200
- [ ] Lambda saturation <80% in episodes 150-200
- [ ] Mean SPR (episodes 150-200) > 0

### Option C: Direct to Confirmation (10-12 hours) **[RECOMMENDED]**

Since 200 episodes should be sufficient for learning, proceed directly to confirmation:

```bash
python run_confirmation_suite.py --seeds 42,123,456,789,1011 --episodes 200
```

**Advantages**:
- 5 seeds × 200 episodes = 1000 total episodes of data
- Will reveal if bias is seed-specific or systematic
- Extended floor (120 ep) gives more learning time
- Stronger symmetry loss (0.5) should help

**Monitoring**:
After completion, check if any seeds show:
- p_long maintaining balance after episode 120
- Lambda saturation decreasing over time
- Positive SPR in later episodes

## Alternative Approaches (If This Fails)

### Plan B: Even Stronger Floor

If p_long still collapses at episode 120:

```python
# Nuclear option: Keep floor active entire training
self.LONG_FLOOR_EPISODES = 200  # Never turn off

# Or: Gradual ramp-down
# Episodes 1-100: 50% force rate
# Episodes 101-150: 30% force rate  
# Episodes 151-200: 10% force rate
```

### Plan C: Increase Symmetry Loss Further

```python
SYMMETRY_LOSS_WEIGHT = 1.0  # Equal to TD loss
```

This makes symmetry violations as important as prediction errors.

### Plan D: Architectural Change

Add explicit symmetry constraint to Q-network:

```python
# Force Q(flip(s), LONG) = Q(s, SHORT) at architecture level
# This guarantees symmetry rather than learning it
```

## Timeline Estimate

| Approach | Duration | Confidence | Recommendation |
|----------|----------|------------|----------------|
| Option A: Quick retest | 30 min | Medium | Do this first |
| Option B: Full probe | 3-5 hours | High | If retest passes |
| Option C: Confirmation | 10-12 hours | Medium | **RECOMMENDED** |
| Plan B/C/D: Escalation | Varies | Low | Only if C fails |

## My Recommendation

**Proceed directly to Option C (Full Confirmation)** because:

1. ✅ We know the floor works (25% achieved)
2. ✅ Extended floor (120 ep) gives 2× learning time
3. ✅ Stronger symmetry loss (0.5) should help learning
4. ✅ 200 episodes sufficient for neural network convergence
5. ✅ 5 seeds will reveal if solution is robust

**If confirmation fails**, we have clear escalation path (Plans B/C/D).

**Commands**:

```bash
# Start confirmation suite
python run_confirmation_suite.py --seeds 42,123,456,789,1011 --episodes 200

# While running, monitor first seed:
# (After ~2 hours, seed 42 will finish)
python analyze_probe.py  # Adapt for confirmation_results/seed_42

# If seed 42 shows p_long collapse at ep 120, cancel and escalate
# If seed 42 shows p_long balance, let full suite complete
```

Would you like me to start the confirmation suite now?
