# Training Parameter Tuning Guide

## Issue: Low Trade Count Leading to Zero Fitness

### Symptom
Episodes show low validation trade counts (≤2 trades), causing trade-count gating to reduce fitness to near-zero, even when price action decisions might be reasonable.

### Root Cause
Conservative policy after negative feedback + strict trade gating thresholds = prolonged inactivity

### Current Thresholds (in config.py)

```python
# Validation settings
min_hold: int = 8          # Production: 8 bars minimum hold
cooldown: int = 16         # Production: 16 bars cooldown after exit
min_trades_half: int = 16  # 0.5x mult if trades < this
min_trades_full: int = 23  # 1.0x mult if trades >= this
```

### Solution Options (Pick ONE)

#### Option 1: Lower Trade Gating Thresholds (SAFEST)
**Good for:** Keeping realism while avoiding zero-trade collapse

```python
min_trades_half: int = 14  # Was 16
min_trades_full: int = 20  # Was 23
```

**Impact:** Agent gets partial credit (0.5x-0.75x) with fewer trades, reducing "all or nothing" pressure

---

#### Option 2: Reduce Hold/Cooldown Constraints (MODERATE)
**Good for:** Allowing more trade opportunities without turning it into a scalper

```python
min_hold: int = 6   # Was 8 (still realistic for H1 timeframe)
cooldown: int = 12  # Was 16
```

**Impact:** More chances to trade per episode, but still enforces meaningful position duration

---

#### Option 3: Inactivity Penalty (ADVANCED)
**Good for:** Nudging agent to take *some* action over perpetual flat

Add to `ForexTradingEnv._calculate_step_reward()` in `environment.py`:

```python
# After existing reward calculation
if self.position is None:
    # Tiny penalty for staying flat too long
    flat_steps = self.current_step - self._last_trade_step
    if flat_steps > 50:  # e.g., 50 hours flat
        reward -= 1e-5  # Tiny nudge, doesn't distort PnL learning
```

**Impact:** Agent slightly prefers action over inaction, but penalty is small enough not to override PnL signals

---

### Recommended Combo (Conservative)

Start with Option 1 (lower thresholds) for 1-2 runs:

```python
min_trades_half: int = 14
min_trades_full: int = 20
```

**Why:** No code changes needed, just config adjustment. Preserves realistic holding periods while being less punitive.

If still seeing zero-trade collapse after 20-30 episodes, add Option 2:

```python
min_hold: int = 6
cooldown: int = 12
```

**Avoid Option 3** unless absolutely necessary - it adds complexity and can interfere with PnL-based learning.

### How to Apply

1. **Edit config.py**:
   ```python
   # Around line 100-110, find ValidationConfig
   min_trades_half: int = 14  # Changed from 16
   min_trades_full: int = 20  # Changed from 23
   ```

2. **For active seed sweep**: Changes will apply to next seed (77, 777)

3. **For new training**: Start fresh with `python main.py --episodes 50`

### Monitoring

After adjusting, watch for:
- **Validation trade counts** increasing (should see >10 trades/window)
- **Fitness values** no longer stuck at ~0
- **Episode rewards** showing more variation (not flat)

### Current Status (from logs)

| Episode | Val Trades | Fitness | Multiplier |
|---------|------------|---------|------------|
| 1       | 26         | 1.087   | 1.0x ✅     |
| 2       | 16         | -1.758  | 1.0x ✅     |
| 3       | 22         | -0.861  | 1.0x ✅     |
| 19-23   | ~varies    | ~0.53   | Partial    |
| Later   | ≤2         | ~0.0    | 0.0-0.5x ❌ |

**Observation:** Early episodes had healthy trade counts. Later episodes collapsed to near-zero trades, suggesting policy became overly conservative.

**Action:** Lower thresholds (Option 1) would give partial credit to episodes with 14-19 trades instead of gating them to near-zero.

---

## Alternative: Accept the Behavior

If you believe the agent *should* learn to stay flat in unfavorable conditions:

**Do nothing.** The current gating is working as designed - it heavily penalizes low-activity episodes, forcing the agent to either:
1. Learn to trade actively when conditions permit, OR
2. Accept near-zero fitness during conservative phases

This is a valid training philosophy if you want a "trade or die" approach.

---

## Summary

**Quick fix (5 min):** Change `min_trades_half: 14` and `min_trades_full: 20` in config.py

**Test:** Run 10-20 episodes and check if validation trades increase

**Fallback:** If no improvement, add `min_hold: 6` and `cooldown: 12`

**Nuclear option:** Inactivity penalty (not recommended unless above fails)
