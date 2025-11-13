# LONG Bias Fix - Implementation Plan

## Problem Summary
Agent learned Q(s, LONG) << Q(s, SHORT) from training data, causing:
- LONG actions: 3.5% (target: 40-60%)
- SHORT actions: 23.1%
- lambda_long saturated at -1.2 (100% of episodes)
- Negative returns across all seeds

## Implemented Fixes (Phase 1 - Quick Guards)

### A1: LONG Exploration Floor ✅
**File**: `agent.py`
**Changes**:
- Added `LONG_FLOOR = 0.15`, `LONG_FLOOR_EPISODES = 60`
- During first 60 episodes, if `p_long < 0.10`, force LONG action with 20% probability
- Tracks `long_entries_episode` in telemetry

### A2: Raised Lambda Ceiling ✅
**File**: `agent.py`
**Change**: `LAMBDA_MAX: 1.2 → 2.0`
- Allows controller more aggressive bias to overcome Q-network preferences
- Will revert to 1.2 after replay buffer balances

### A3: Enhanced Telemetry ✅
**File**: `agent.py`
**Addition**: Track `long_entries` per episode (not just action frequency)

## Pending Implementations (Phase 2 - Durable Fixes)

### B3: Sign-Flip Augmentation (CRITICAL)
**Files to modify**: `agent.py` (train_step method), potentially new `augmentation.py`

**What to implement**:
1. **Feature identification function**:
   ```python
   def get_direction_sensitive_indices(state_size):
       # Returns indices of features that flip with direction
       # Example: returns[0:10], momentum[20:30], roc[40:50]
       # Excludes: time_of_day, volatility, spreads, one-hots
       return direction_sensitive_indices
   ```

2. **Sign-flip function**:
   ```python
   def flip_state(state, sensitive_indices):
       flipped = state.copy()
       flipped[sensitive_indices] *= -1
       return flipped
   ```

3. **Action swap function**:
   ```python
   def flip_action(action):
       # 0=HOLD → 0, 1=LONG → 2, 2=SHORT → 1, 3=MOVE_SL → 3
       if action == 1: return 2
       elif action == 2: return 1
       else: return action
   ```

4. **Augmented training in train_step()**:
   - For each batch, create flipped versions
   - Compute Q(s, a) and Q(s', a')
   - Add symmetry loss: `L_sym = MSE(Q(s', LONG), Q(s, SHORT)) + MSE(Q(s', SHORT), Q(s, LONG))`
   - Total loss: `L = L_TD + 0.2 * L_sym`

### B4: Balanced Replay Sampling
**Files to modify**: `agent.py` (sampling in train_step), potentially `replay_buffer.py`

**What to implement**:
1. **Action-stratified sampling**:
   ```python
   def sample_balanced(self, batch_size):
       # Sample LONG/SHORT in 1:1 ratio
       n_long = batch_size // 4
       n_short = batch_size // 4
       n_hold = batch_size // 2
       
       # Sample from per-action buckets
       long_samples = sample_from_bucket(action=1, count=n_long)
       short_samples = sample_from_bucket(action=2, count=n_short)
       hold_samples = sample_from_bucket(action=0, count=n_hold)
       
       return concatenate([long_samples, short_samples, hold_samples])
   ```

2. **Fallback if LONG bucket empty**:
   - Use regular sampling but weight LONG samples higher in loss
   - `loss_weights[action==1] *= 10.0  # Oversample LONG gradients`

## Testing Plan

### Quick Smoke Test (Now)
```bash
python main.py --episodes 10 --seed 42 --telemetry extended --output-dir quick_test
python diagnose_action_bias.py  # Check if p_long improving
```

### 1-Seed Probe (After B3+B4)
```bash
python main.py --episodes 200 --seed 42 --telemetry extended --output-dir probe_200ep
```
**Success criteria**:
- Episodes 1-40: LONG entries increasing (warmup phase working)
- Episodes 41-200: p_long ∈ [0.30, 0.70] in ≥50% of episodes
- lambda_long saturation <80% (not pegged at -1.2)

### Full Confirmation (After Probe Passes)
```bash
# Remove LONG floor (set LONG_FLOOR_EPISODES = 0)
# Reduce LAMBDA_MAX back to 1.2
python run_confirmation_suite.py --seeds 42,123,456,789,1011 --episodes 200
```

## Audit Checks (Phase 3 - To Do)

### C5: Reward Symmetry Audit
**File**: `environment.py`
**Checks**:
- [ ] Bid/ask spread applied symmetrically
- [ ] Stop loss/take profit triggers symmetric
- [ ] PnL calculation symmetric for LONG/SHORT
- [ ] Friction jitter identical for both directions

### C6: Data Regime Audit
**Script to create**: `audit_data_bias.py`
```python
# Calculate baseline performance:
# - Long-only strategy returns
# - Short-only strategy returns
# Over training windows
```

## File Changes Summary

### Modified Files
1. **agent.py**:
   - ✅ LAMBDA_MAX: 2.0
   - ✅ LONG_FLOOR guards
   - ✅ long_entries telemetry
   - ⏳ Sign-flip augmentation in train_step()
   - ⏳ Balanced sampling

2. **analyze_confirmation_results.py**:
   - ✅ UTF-8 encoding fix

### New Files
3. **diagnose_action_bias.py**: ✅ Created
4. **diagnose_long_ratio.py**: ✅ Created

### Files to Create
5. **augmentation.py** (optional): Sign-flip utilities
6. **audit_data_bias.py**: Baseline performance check
7. **ANTI_BIAS_FIX_SUMMARY.md**: Document the full fix

## Next Immediate Steps

1. ✅ Test current guards (A1+A2) with 10-episode smoke test
2. Implement B3 (sign-flip augmentation)
3. Implement B4 (balanced sampling)
4. Run 200-episode probe
5. If probe passes, remove guards and run full confirmation

## Rollback Plan

If fixes cause instability:
1. Set `LONG_FLOOR_EPISODES = 0` (disable floor)
2. Set `LAMBDA_MAX = 1.2` (revert ceiling)
3. Comment out symmetry loss in train_step()
4. Revert to regular sampling
5. Return to Phase 2.8f baseline
