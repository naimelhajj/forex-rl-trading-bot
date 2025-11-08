# Phase 2.8e: Soft Bias Steering Implementation Guide

## Executive Summary

**Recommendation**: Implement soft bias steering at action selection instead of reward penalties.

**Rationale**: You're absolutely right about the flaws in Fix Pack D3:
- Hard action masking = brittle, distribution mismatch, thrashing at boundaries
- Episodic penalties = still reward shaping with poor credit assignment
- Rolling window penalties = proven broken (momentum trap)

**Solution**: Soft, symmetric steering at action-selection time with clean rewards.

## Core Principles

1. **Keep rewards clean** - Optimize pure SPR, no gaming incentives
2. **Soft logit/Q rebalancing** - Gentle nudges, not hard masks
3. **Circuit-breaker with hysteresis** - Fail-safe only for extremes
4. **No episodic penalties** - Use KPIs for gates, not agent payment

## Implementation Changes

### 1. Config Parameters (config.py)

```python
# PHASE 2.8e: Soft bias steering (action-selection nudges, not reward penalties)
entropy_beta: float = 0.014  # Keep stable
directional_bias_beta: float = 0.08  # Soft nudge for L/S balance
hold_bias_gamma: float = 0.05  # Soft nudge for HOLD discouragement
bias_check_interval: int = 10  # Check every N steps
bias_margin_low: float = 0.35  # Trigger SHORT bias if long_ratio < 35%
bias_margin_high: float = 0.65  # Trigger LONG bias if long_ratio > 65%
hold_ceiling: float = 0.80  # Discourage HOLD if > 80%

# Circuit-breaker (fail-safe with hysteresis)
circuit_breaker_enabled: bool = True
circuit_breaker_threshold_low: float = 0.10  # Trigger if <10% LONG
circuit_breaker_threshold_high: float = 0.90  # Trigger if >90% LONG
circuit_breaker_lookback: int = 500  # Must persist 500 steps
circuit_breaker_mask_duration: int = 30  # Mask for 30 steps
```

**REMOVE** these old parameters:
- `ls_balance_lambda: float = 0.050`
- `hold_balance_lambda: float = 0.020`

### 2. Environment Changes (environment.py)

#### A. Constructor Parameters

```python
def __init__(self, ...,
             entropy_beta: float = 0.014,
             directional_bias_beta: float = 0.08,
             hold_bias_gamma: float = 0.05,
             bias_check_interval: int = 10,
             bias_margin_low: float = 0.35,
             bias_margin_high: float = 0.65,
             hold_ceiling: float = 0.80,
             circuit_breaker_enabled: bool = True,
             circuit_breaker_threshold_low: float = 0.10,
             circuit_breaker_threshold_high: float = 0.90,
             circuit_breaker_lookback: int = 500,
             circuit_breaker_mask_duration: int = 30):
```

#### B. Instance Variables (in __init__)

```python
# PHASE-2.8e: Soft bias steering
self.entropy_beta = entropy_beta
self.directional_bias_beta = directional_bias_beta
self.hold_bias_gamma = hold_bias_gamma
self.bias_check_interval = bias_check_interval
self.bias_margin_low = bias_margin_low
self.bias_margin_high = bias_margin_high
self.hold_ceiling = hold_ceiling

# Circuit-breaker state
self.circuit_breaker_enabled = circuit_breaker_enabled
self.circuit_breaker_threshold_low = circuit_breaker_threshold_low
self.circuit_breaker_threshold_high = circuit_breaker_threshold_high
self.circuit_breaker_lookback = circuit_breaker_lookback
self.circuit_breaker_mask_duration = circuit_breaker_mask_duration
self.circuit_breaker_active = False
self.circuit_breaker_steps_remaining = 0
self.circuit_breaker_mask_side = None  # 'LONG' or 'SHORT'
```

#### C. Reset Method

```python
def reset(self) -> np.ndarray:
    # ... existing reset code ...
    
    # PHASE-2.8e: Track balance for soft bias
    self.long_trades = 0
    self.short_trades = 0
    self.action_counts = [0, 0, 0, 0]  # [HOLD, LONG, SHORT, MOVE_SL]
    
    # Circuit-breaker state
    from collections import deque
    self.action_history = deque(maxlen=self.circuit_breaker_lookback)
    self.circuit_breaker_active = False
    self.circuit_breaker_steps_remaining = 0
    self.circuit_breaker_mask_side = None
    
    return self._get_state()
```

#### D. Step Method - REMOVE Rolling Window Penalties

**REMOVE** these lines (~510-540):

```python
# PHASE-2.8d Fix Pack D2.B: Rolling window L/S balance regularizer
# All the dir_window tracking and penalty code
# PHASE-2.8d Fix Pack D2.C: Hold-rate guardrail
# All the hold_share penalty code
```

**REPLACE** with simple action tracking:

```python
def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
    # ... existing step code ...
    
    # Track action history for circuit-breaker
    if hasattr(self, 'action_history'):
        self.action_history.append(action)
    
    # ... rest of step logic ...
```

#### E. New Method: get_action_bias()

```python
def get_action_bias(self) -> np.ndarray:
    """
    PHASE-2.8e: Compute soft biases for action selection (steering, not penalties).
    Returns biases [HOLD, LONG, SHORT, MOVE_SL] to add to Q-values.
    
    Returns:
        bias: np.ndarray of shape (4,) with biases to add to action scores
    """
    bias = np.zeros(4, dtype=np.float32)
    
    # 1. Directional bias (L/S balance)
    total_trades = self.long_trades + self.short_trades
    if total_trades >= 10:
        long_ratio = self.long_trades / total_trades
        
        if long_ratio > self.bias_margin_high:  # Too many LONG (>65%)
            bias[1] -= self.directional_bias_beta  # Discourage LONG
            bias[2] += self.directional_bias_beta  # Encourage SHORT
        elif long_ratio < self.bias_margin_low:  # Too many SHORT (<35%)
            bias[2] -= self.directional_bias_beta  # Discourage SHORT
            bias[1] += self.directional_bias_beta  # Encourage LONG
    
    # 2. Hold bias (prevent excessive passivity)
    total_actions = sum(self.action_counts)
    if total_actions > 50:
        hold_rate = self.action_counts[0] / total_actions
        if hold_rate > self.hold_ceiling:  # Too much HOLD (>80%)
            bias[0] -= self.hold_bias_gamma  # Discourage HOLD
    
    # 3. Circuit-breaker (fail-safe with hysteresis)
    if self.circuit_breaker_enabled:
        if self.circuit_breaker_active:
            # Apply strong bias
            if self.circuit_breaker_mask_side == 'LONG':
                bias[1] -= 10.0  # Strongly discourage LONG
            elif self.circuit_breaker_mask_side == 'SHORT':
                bias[2] -= 10.0  # Strongly discourage SHORT
            
            # Countdown
            self.circuit_breaker_steps_remaining -= 1
            if self.circuit_breaker_steps_remaining <= 0:
                self.circuit_breaker_active = False
                self.circuit_breaker_mask_side = None
        else:
            # Check if should trigger
            if len(self.action_history) >= self.circuit_breaker_lookback and total_trades >= 10:
                recent_long = sum(1 for a in self.action_history if a == 1)
                recent_short = sum(1 for a in self.action_history if a == 2)
                recent_directional = recent_long + recent_short
                
                if recent_directional > 0:
                    recent_long_ratio = recent_long / recent_directional
                    
                    if recent_long_ratio > self.circuit_breaker_threshold_high:  # >90%
                        self.circuit_breaker_active = True
                        self.circuit_breaker_steps_remaining = self.circuit_breaker_mask_duration
                        self.circuit_breaker_mask_side = 'LONG'
                    elif recent_long_ratio < self.circuit_breaker_threshold_low:  # <10%
                        self.circuit_breaker_active = True
                        self.circuit_breaker_steps_remaining = self.circuit_breaker_mask_duration
                        self.circuit_breaker_mask_side = 'SHORT'
    
    return bias
```

### 3. Agent Changes (agent.py)

Update `select_action()` to use bias from environment:

```python
def select_action(self, state: np.ndarray, explore: bool = True, 
                 env=None, eval_mode: bool = False) -> int:
    """Select action with optional soft bias from environment."""
    
    # Get Q-values
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q_net(s).squeeze(0).cpu().numpy()
    
    # Apply soft bias if environment provides it
    if env is not None and hasattr(env, 'get_action_bias'):
        bias = env.get_action_bias()
        q = q + bias  # Add bias to Q-values
    
    # Epsilon-greedy
    if explore and not eval_mode and random.random() < self.epsilon:
        return random.randrange(self.action_size)
    
    return int(np.argmax(q))
```

### 4. Main/Trainer Changes (main.py)

Pass environment to agent during action selection:

```python
# In training loop
action = agent.select_action(state, explore=True, env=env)

# In validation
action = agent.select_action(state, explore=False, env=env, eval_mode=True)
```

## Why This Works Better

### 1. No Reward Hacking
- Rewards optimize pure SPR
- No perverse incentives to game balance metrics
- Agent learns profitable trading, steering handles balance

### 2. Symmetric, Local Control
- Nudges adjust dynamically based on current state
- No runaway to opposite extremes
- Self-correcting system (bias reduces as balance improves)

### 3. Hysteresis Prevents Thrashing
- Circuit-breaker requires 500 steps of extreme behavior
- Short 30-step mask duration
- Only triggers for lock-in (>90% or <10%), not normal drift

### 4. Clean Evaluation
- Same bias mechanism in training and eval
- No distribution mismatch
- Evaluates agent's ability to trade profitably within guardrails

## Expected Behavior

### Episodes 1-10
- Random exploration may hit 70% → soft bias activates
- Gentle nudges keep agent exploring 40-60% range
- No penalty on rewards, just action steering

### Episodes 11-50
- Agent learns bias boundaries naturally
- Stays 40-60% without strong intervention
- Circuit-breaker rarely triggers (only if truly stuck)

### Episodes 51-200
- Stable 40-60% maintained
- SPR optimization proceeds cleanly
- No oscillations or momentum trap effects

## Testing Protocol

### Phase 1: Smoke Test (3 seeds × 20 episodes)
```powershell
python main.py --episodes 20 --seed 42
python main.py --episodes 20 --seed 123  
python main.py --episodes 20 --seed 777
```

**Check**: Episodes 10, 20 should show long_ratio 0.40-0.60

### Phase 2: Validation (3 seeds × 80 episodes)
```powershell
python main.py --episodes 80 --seed 42
python main.py --episodes 80 --seed 123
python main.py --episodes 80 --seed 777
```

**Check**: Episodes 20, 40, 60, 80 maintain 0.40-0.60

### Phase 3: Confirmation (200 episodes)
```powershell
python main.py --episodes 200 --seed 42
```

**Success Criteria** (unchanged):
- Entropy: 0.95-1.10
- Hold rate: 0.65-0.78
- Long ratio: 0.35-0.65
- Switch rate: 0.14-0.20
- Trades/ep: 24-32

## Tuning Knobs (If Needed)

### If Still Drifts to 70%+ (Too Weak)
```python
directional_bias_beta = 0.12  # Increase from 0.08
bias_margin_high = 0.60  # Tighten from 0.65
bias_margin_low = 0.40  # Tighten from 0.35
```

### If Oscillates (Too Strong)
```python
directional_bias_beta = 0.05  # Decrease from 0.08
bias_check_interval = 20  # Check less frequently
```

### If Circuit-Breaker Triggers Too Often
```python
circuit_breaker_threshold_high = 0.95  # Relax from 0.90
circuit_breaker_threshold_low = 0.05  # Relax from 0.10
circuit_breaker_lookback = 750  # Longer hysteresis from 500
```

## File Summary

**Modified Files**:
1. `config.py` - Update Phase 2.8e parameters
2. `environment.py` - Add get_action_bias(), remove rolling window penalties
3. `agent.py` - Update select_action() to accept env parameter
4. `main.py` - Pass env to agent.select_action()

**Time Estimate**: 2 hours implementation + 3 hours testing

## Next Steps

1. Backup current environment.py (corrupted during editing)
2. Implement changes carefully section by section
3. Run smoke test (3 seeds × 20 episodes)
4. Monitor Episode 10, 20 for long_ratio 0.40-0.60
5. If green → proceed to 80-episode validation
6. If red → tune bias_beta parameters

## Critical Note

The environment.py file got corrupted during my editing attempt. You'll need to:
1. Restore from backup if available
2. OR manually apply the changes section by section
3. Test imports after each section: `python -c "import environment"`

I apologize for the file corruption - the replace operations cascaded into syntax errors. The design is sound, but implementation needs careful manual application.
