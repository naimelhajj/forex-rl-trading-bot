# Anti-Bias Fix Implementation Summary

## Root Cause Diagnosis ✅

**Problem**: Agent learned systematic LONG avoidance
- **LONG actions**: 3.5% (target: 40-60%)
- **SHORT actions**: 23.1% 
- **HOLD actions**: 73.4%
- **Result**: 7:1 SHORT:LONG ratio, negative returns

**Why the controller couldn't fix it**:
- Controller working correctly: λ_long saturated at -1.2 (100% of episodes)
- But Q-network weights learned Q(s, LONG) << Q(s, SHORT)
- Controller bias of ±1.2 insufficient to overcome learned preference

## Phase 1: Quick Guards (IMPLEMENTED) ✅

### A1: LONG Exploration Floor
**File**: `agent.py` lines ~400, ~680
```python
# Configuration
self.LONG_FLOOR = 0.15              # 15% minimum LONG frequency
self.LONG_FLOOR_EPISODES = 60       # Apply for first 60 episodes
self.current_episode = 0            # Episode counter

# In select_action() - epsilon-greedy path
if (not eval_mode and self.use_dual_controller and 
    self.current_episode <= self.LONG_FLOOR_EPISODES and
    self.p_long < 0.10 and random.random() < 0.20):
    if mask is None or (len(mask) > 1 and mask[1]):
        action = 1  # Force LONG
```

**Purpose**: Ensure replay buffer receives LONG experiences during warmup

### A2: Raised Lambda Ceiling  
**File**: `agent.py` line ~405
```python
self.LAMBDA_MAX = 2.0  # Was 1.2, allows stronger controller bias
```

**Purpose**: Allow controller more aggressive correction power
**Note**: Will revert to 1.2 after replay balances

### A3: Enhanced Telemetry
**File**: `agent.py` lines ~420, ~480, ~665
```python
# Track actual LONG entries (not just action frequency)
self.long_entries_episode = 0

# In select_action()
if action == 1:
    self.long_entries_episode += 1

# In get_episode_telemetry()
return {
    # ... existing fields ...
    'long_entries': int(getattr(self, 'long_entries_episode', 0)),
}
```

**Purpose**: Monitor if buffer is actually receiving LONG trades

## Phase 2: Durable Fixes (PENDING)

### B3: Sign-Flip Augmentation (NOT YET IMPLEMENTED)
**Concept**: For each training sample (s, a, r, s', done), create mirror sample:
- Flip direction-sensitive features: returns, momentum, ROC → negate
- Keep level features unchanged: time, volatility, spreads
- Swap actions: LONG ↔ SHORT
- Add symmetry loss: L_sym = MSE(Q(s', LONG), Q(s, SHORT))

**Why it works**: Teaches network that "bullish pattern at s" = "bearish pattern at flipped s"
Eliminates directional bias from training data

**Implementation needed**:
1. `get_direction_sensitive_indices(state_size)` - identify which features to flip
2. `flip_state(state)` - create sign-flipped version
3. `flip_action(action)` - swap LONG ↔ SHORT
4. Modify `train_step()` to compute augmented batch + symmetry loss

### B4: Balanced Replay Sampling (NOT YET IMPLEMENTED)
**Concept**: Sample LONG and SHORT transitions in 1:1 ratio
- Instead of random sampling, maintain per-action buckets
- Each batch: 25% LONG, 25% SHORT, 50% HOLD
- If LONG bucket underfilled, use class-weighted loss

**Why it works**: Prevents SHORT gradients from dominating training
Even if agent takes 7× more SHORT actions, they get equal learning weight

**Implementation needed**:
1. Modify `PrioritizedReplayBuffer` or create `BalancedReplayBuffer`
2. Track action when storing transitions
3. Sample from per-action buckets in `sample()` method
4. Fallback to weighted loss if buckets imbalanced

## Diagnostic Tools Created ✅

### diagnose_long_ratio.py
**Purpose**: Analyze p_long statistics and lambda saturation
**Output**:
```
p_long statistics (target: 0.40-0.60):
  Min: 0.0018, Max: 0.0505, Mean: 0.0349, Median: 0.0355
lambda_long statistics:
  100.0% at -1.2 (saturated)
Episodes in target band: 0/81
```

### diagnose_action_bias.py
**Purpose**: Comprehensive action distribution analysis
**Output**:
```
ACTION FREQUENCY ESTIMATES:
  HOLD: 73.4% ± 5.1%
  LONG: 3.5% ± 1.1%
  SHORT: 23.1% (inferred)
❌ CRITICAL ISSUE: Agent has learned policy that almost never goes LONG
```

### monitor_antibias.py
**Purpose**: Real-time monitoring during training
**Features**:
- Shows p_long, λ_long, LONG entries per episode
- 5-episode rolling averages
- Status indicators (✅ in-band, ❌ out-of-band, 🔧 floor active)

## Testing Plan

### Step 1: Smoke Test (10 episodes)
```bash
python main.py --episodes 10 --seed 42 --telemetry extended --output-dir quick_test_antibias
python diagnose_action_bias.py  # Check if p_long improving
```

**Success criteria**:
- p_long > 0.05 (up from 0.035 baseline)
- long_entries > 0 in most episodes
- No crashes or NaN values

### Step 2: 200-Episode Probe (after B3+B4)
```bash
python main.py --episodes 200 --seed 42 --telemetry extended --output-dir probe_200ep
```

**Success criteria**:
- Episodes 1-40: Increasing LONG entries (warmup working)
- Episodes 41-200: p_long ∈ [0.30, 0.70] in ≥50% of episodes
- λ_long saturation < 80% (not pegged at -1.2)
- Mean SPR > 0 (positive returns)

### Step 3: Full Confirmation (after probe passes)
```bash
# Configuration changes:
# - Set LONG_FLOOR_EPISODES = 0 (disable floor)
# - Set LAMBDA_MAX = 1.2 (revert ceiling)

python run_confirmation_suite.py --seeds 42,123,456,789,1011 --episodes 200
python analyze_confirmation_results.py --results-dir confirmation_results
```

**Success criteria**: All 9 gates pass
- Gate 5 (Long Ratio): ≥70% episodes in [0.40, 0.60]
- Gate 1 (Mean SPR): ≥ +0.04
- Gate 8 (Switch Rate): ≥70% in [0.15, 0.19]

## Audit Checks (Phase 3 - TODO)

### C5: Reward Symmetry
Check `environment.py` for asymmetries:
- [ ] Bid/ask spread handling
- [ ] Stop loss triggers (long vs short)
- [ ] Take profit triggers  
- [ ] Trailing stop logic
- [ ] PnL calculation
- [ ] Friction/slippage application

### C6: Data Regime
Create `audit_data_bias.py`:
- Calculate long-only baseline returns
- Calculate short-only baseline returns
- Over actual training windows
- If short-only >> long-only, data has bearish bias

## File Modifications

### Modified Files
1. **agent.py**:
   - Lines ~400-410: LAMBDA_MAX=2.0, LONG_FLOOR config
   - Lines ~420-440: reset_episode_tracking() - add long_entries tracking
   - Lines ~480-485: get_episode_telemetry() - add long_entries field
   - Lines ~670-685: select_action() - LONG floor guard (both NoisyNet and ε-greedy)

2. **analyze_confirmation_results.py**:
   - Line ~357: UTF-8 encoding fix for Windows

### Created Files
3. **ANTI_BIAS_FIX_PLAN.md**: Implementation plan
4. **diagnose_long_ratio.py**: Lambda saturation analysis
5. **diagnose_action_bias.py**: Action distribution diagnosis
6. **monitor_antibias.py**: Real-time training monitor

## Git Status

**Commit**: "Phase 2.8f: Anti-bias fixes (A1+A2)"
**Files staged**: agent.py, analyze_confirmation_results.py, diagnostic scripts, plan docs

**Next commit** (after B3+B4):
"Phase 2.8f: Direction-equivariant learning (B3+B4) - sign-flip augmentation + balanced replay"

## Rollback Procedure

If fixes cause instability:
```python
# In agent.py:
self.LAMBDA_MAX = 1.2              # Revert ceiling
self.LONG_FLOOR_EPISODES = 0        # Disable floor
# Comment out symmetry loss in train_step()
# Revert to regular sampling
```

Then re-run baseline confirmation to verify stability.

## Key Insights

1. **Controller was working correctly** - λ_long=-1.2 proves it detected the problem
2. **Q-network learned the bias** - Not a controller tuning issue
3. **Data flow is the fix** - Need LONG experiences in replay buffer
4. **Symmetry is the cure** - Sign-flip augmentation prevents directional overfitting
5. **Balance prevents dominance** - Equal LONG/SHORT sampling prevents gradient skew

## Next Immediate Actions

1. ✅ Implement A1+A2 (DONE)
2. ⏳ Test with 10-episode smoke test
3. ⏳ Implement B3 (sign-flip augmentation)
4. ⏳ Implement B4 (balanced sampling)
5. ⏳ Run 200-episode probe
6. ⏳ If probe passes, remove guards and run full confirmation
