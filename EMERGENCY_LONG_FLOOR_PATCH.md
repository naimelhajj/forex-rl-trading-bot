# Emergency Patch: Aggressive LONG Floor

## Problem
The LONG exploration floor isn't triggering because:
1. `p_long` starts at 0.5 (initialized in reset_episode_tracking)
2. It only drops when non-LONG actions are taken
3. By the time p_long < 0.10, many steps have passed
4. The 20% random chance isn't enough

## Solution: Make Floor More Aggressive

### Option A: Higher Floor Probability (Quick Fix)
Change from 20% to 40% chance when p_long < 0.10

### Option B: Longer Trigger Window (Better)
Trigger when p_long < 0.20 instead of < 0.10

### Option C: Step-Based Floor (Most Aggressive)
Force LONG every N steps during warmup episodes

## Recommended: Option B + Higher Probability

```python
# In select_action(), epsilon-greedy path:
if (not eval_mode and self.use_dual_controller and 
    self.current_episode <= self.LONG_FLOOR_EPISODES and
    self.p_long < 0.20 and  # Changed from 0.10
    random.random() < 0.40):  # Changed from 0.20
    if mask is None or (len(mask) > 1 and mask[1]):
        action = 1  # Force LONG
```

This doubles both the trigger threshold and the probability, giving:
- 40% chance per action when p_long < 20%
- Over 200 steps/episode, expect ~80 forced LONGs
- Should push p_long up to 10-15% range

## Apply Now

The sign-flip augmentation (B3) is working (p_long improved 12.8%) but needs
more LONG experiences in the buffer to have effect. The floor needs to be 
more aggressive to bootstrap the process.
