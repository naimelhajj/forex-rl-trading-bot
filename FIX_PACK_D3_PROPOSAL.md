# Fix Pack D3: Episodic Balance with Hard Constraints
# Replaces broken rolling window approach

## Changes

### 1. Remove Rolling Window Penalties (environment.py)
- Remove `dir_window` tracking
- Remove step-by-step L/S balance penalties
- Remove hold-rate guardrail

### 2. Add Hard Action Masking (environment.py)
```python
def _get_action_mask(self):
    """Block actions that would worsen directional imbalance."""
    mask = [True, True, True, True]  # [HOLD, LONG, SHORT, MOVE_SL]
    
    # Only apply masking if we have enough trades to assess balance
    if self.trades_long + self.trades_short >= 10:
        long_ratio = self.trades_long / (self.trades_long + self.trades_short)
        
        # Block LONG if too many long trades (>70%)
        if long_ratio > 0.70:
            mask[1] = False
        
        # Block SHORT if too many short trades (<30%)
        if long_ratio < 0.30:
            mask[2] = False
    
    return mask
```

### 3. Add Episodic Balance Penalty (environment.py - in done)
```python
if done and (self.trades_long + self.trades_short) > 0:
    episode_long_ratio = self.trades_long / (self.trades_long + self.trades_short)
    imbalance = abs(episode_long_ratio - 0.5)
    
    # Strong episodic penalty if imbalanced
    if imbalance > 0.20:  # Outside 30-70% range
        episodic_penalty = 5.0 * imbalance  # Linear penalty
        self.cumulative_reward -= episodic_penalty
```

### 4. Update config.py
```python
# Remove rolling window parameters
# ls_balance_lambda: DELETE
# hold_balance_lambda: DELETE

# Add episodic parameters
episodic_balance_penalty: float = 5.0  # Penalty multiplier for imbalance
balance_threshold: float = 0.20  # Allow 30-70% range
action_mask_enabled: bool = True  # Enable hard constraints
```

## Why This Works

### Hard Action Masking
- **Prevents** extreme imbalances before they happen
- **No learning required** - hard constraint
- **No momentum trap** - checks current episode only
- **Allows flexibility** - 30-70% range is permissive

### Episodic Penalty
- **Simple signal**: Balance within each episode
- **No rolling window**: No old trades affecting current decisions
- **Linear penalty**: No cliff effects
- **Applied once**: At episode end only

### Combined Effect
1. Hard mask prevents >70% or <30% within episode
2. Episodic penalty discourages 60-70% extremes
3. Agent naturally settles to 40-60% range
4. No momentum traps or over-correction

## Implementation Priority

**HIGH PRIORITY - Implement immediately**:
1. Remove rolling window code from environment.py
2. Add action masking
3. Add episodic penalty
4. Update config.py
5. Test with 20-episode run

This should solve the directional collapse WITHOUT the momentum trap issues.
