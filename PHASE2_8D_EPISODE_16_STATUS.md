# Phase 2.8d Episode 16 Status Report

## Current Situation
Training reached episode 16 but agent is **completely stuck on HOLD-only policy**.

### Episode 16 Metrics
- **Actions**: 12,030 HOLD, 0 LONG, 0 SHORT, 0 MOVE_SL
- **Trades**: 0.0
- **Hold Rate**: 100% (1.0)
- **Action Entropy**: 0.0 bits (completely deterministic HOLD)
- **SPR**: -0.080 (full undertrade penalty)
- **Long/Short**: 0 long, 0 short (0.0 ratio)

## Root Cause Analysis

### Issue 1: Training Started Before Fix
The current run was started **before** we fixed the `long_ratio` KeyError in `trainer.py`. Evidence:
- Validation JSONs missing `spr`, `entropy`, `hold_frac`, `long_ratio` fields
- These fields were added in our recent fix but aren't appearing in output

### Issue 2: HOLD-Only Policy (Critical)
Agent learned to HOLD 100% of the time through 16 episodes:
- **Prefill**: Collected 3000 random transitions (baseline diversity)
- **Learning starts**: 1000 (agent can learn after 1000 transitions)
- **Buffer size**: 2994 at first validation
- **Result**: Despite having enough data to learn (2994 > 1000), agent collapsed to pure HOLD

### Likely Causes of HOLD Collapse
1. **Hold bias too strong**: `hold_tie_tau=0.038` may be suppressing exploration
2. **Entropy bonus too weak**: `entropy_beta=0.014` not enough to maintain action diversity
3. **Flip penalty too harsh**: `flip_penalty=0.00077` discouraging position changes
4. **Learning too slow**: Agent seeing HOLD as "safe" (no losses) vs. trading (potential losses during learning)

## Recommended Actions

### Option A: Restart with Current Config (Test Patience)
Stop current run and restart. Let it run to episode 50 to see if agent eventually breaks out of HOLD:
```powershell
# Stop current training (Ctrl+C in training terminal)
python main.py --episodes 50
```

**Pros**: Tests if more training time helps
**Cons**: May waste 50 episodes if config is fundamentally broken

### Option B: Emergency Parameter Adjustment (Recommended)
Modify config to **force exploration**, then restart:

**Changes to `config.py`**:
```python
# Reduce hold bias
hold_tie_tau: 0.038 → 0.030  # Make HOLD less attractive in ties

# Increase entropy bonus
entropy_beta: 0.014 → 0.025  # Stronger exploration reward

# Reduce flip penalty  
flip_penalty: 0.00077 → 0.0005  # Less punishment for position changes

# Reduce learning_starts
learning_starts: 5000 → 1000  # Already at 1000 in AgentConfig, verify this
```

**Rationale**:
- Lower `hold_tie_tau` makes HOLD lose in more Q-value ties → more LONG/SHORT
- Higher `entropy_beta` directly rewards action diversity → fights collapse
- Lower `flip_penalty` reduces "friction" that locks agent into positions
- Lower `learning_starts` ensures backprop starts early (though this may already be 1000)

### Option C: Diagnostic Before Rerun
Check if learning is actually happening:
```powershell
# Look for gradient/loss info in terminal output
# Check if Q-values are updating
# Verify buffer is being sampled
```

## Immediate Next Step
**STOP THE CURRENT TRAINING RUN** - it's using old broken code and stuck on HOLD-only.

Choose:
1. **Quick test**: Restart with current config → 50 episodes → see if patience helps
2. **Aggressive fix**: Apply Option B parameter changes → restart → 50 episodes
3. **Deep diagnostic**: Check training logs for learning signals before deciding

## Expected Outcomes

### If Option A (Current Config):
- Episodes 1-20: Likely still HOLD-dominant (learning lag)
- Episodes 20-40: *Might* see trading emerge if config is viable
- Episodes 40-50: Should have clear signal (trading or still stuck)

### If Option B (Adjusted Config):
- Episodes 1-10: Should see immediate trading activity (forced exploration)
- Episodes 10-30: Trading should stabilize with diversity
- Episodes 30-50: Clear assessment of new balance

## Decision Time
**What do you want to do?**
1. Stop and restart with current config (patience test)
2. Stop, adjust parameters per Option B, restart (aggressive fix)
3. Check diagnostics first (careful approach)
