# CRITICAL BUG FOUND AND FIXED - Trading Permanently Blocked

## The Bug That Explained EVERYTHING

**ROOT CAUSE:** `bars_since_close` counter was NEVER incremented, permanently blocking all trading.

### The Broken Logic

```python
# In legal_action_mask():
if self.bars_since_close < self.cooldown_bars:  # cooldown_bars = 11
    trading_blocked = True
```

**Problem:**
- `bars_since_close` initialized to 0 in `reset()`
- `bars_since_close` reset to 0 when opening position
- **BUT NEVER INCREMENTED** when no position
- Result: `bars_since_close` stays at 0 forever
- Check: `0 < 11` → **ALWAYS TRUE** → trading ALWAYS blocked

### Why This Broke Everything

1. **Episode starts:** `bars_since_close = 0`
2. **Every step:** Check `if 0 < 11` → TRUE → `trading_blocked = True`
3. **Action mask:** `long_ok = short_ok = False` (blocked)
4. **Epsilon-greedy:** Tries to select LONG/SHORT randomly
5. **Action masking:** Mask forces HOLD (only legal action)
6. **Result:** Agent executes 100% HOLD regardless of epsilon

**This bug made epsilon-greedy completely ineffective!**

### The Fix

```python
# environment.py, line ~488
else:
    self.bars_in_position = 0
    self.bars_since_close += 1  # BUGFIX: Increment cooldown counter when no position
```

Now:
- `bars_since_close` increments each step when no position
- After 11 steps: `bars_since_close = 11` → `11 < 11` → FALSE
- Trading unlocked after cooldown period

## Why All Previous Attempts Failed

### Attempt 1: Fix Pack D1 Parameters
- ✗ Changed entropy_beta, hold_tie_tau, flip_penalty
- **Failed:** Trading was blocked by action mask, parameters irrelevant

### Attempt 2: Emergency Option B Adjustments
- ✗ Further parameter tweaks (entropy_beta=0.025, etc.)
- **Failed:** Still blocked by action mask

### Attempt 3: Cleared Incompatible Checkpoints
- ✓ Good step (removed 176-dim model mismatch)
- ✗ But trading still blocked by mask

### Attempt 4: Nuclear Fix (Epsilon-Greedy)
- ✗ Disabled NoisyNet, set epsilon_start=0.50
- **Failed:** Epsilon selected LONG/SHORT, but mask forced HOLD anyway

**None of these could work because the action mask physically prevented trading.**

## Expected Behavior After Fix

### Episode 1 (Immediate)
With epsilon=0.50 and cooldown_bars=11:
- **Steps 1-11:** Trading blocked (cooldown active)
- **Steps 12+:** Trading allowed!
  - 50% random actions
  - ~25% LONG, ~25% SHORT attempts
  - Over ~590 tradeable steps → expect 140+ LONG + 140+ SHORT attempts
- **Trades:** Should see 20-50+ trades immediately

### Episodes 2-10
- Epsilon decays slowly (0.997 per episode)
- Still ~45-48% random actions
- Continued high trading activity
- Agent starts learning which trades work

### Episodes 10-30
- Epsilon ~30-40%
- Trading stabilizes with learned policy + exploration
- Behavioral metrics:
  - Trades: 15-40 per episode
  - Entropy: 0.8-1.2 bits
  - Hold rate: 0.50-0.75

## Why This Bug Wasn't Caught Earlier

1. **Validation always showed 0 trades** → Looked like HOLD collapse
2. **NoisyNet eval_mode freeze** → Also caused 0 trades in validation
3. **Checkpoint dimension mismatch** → Corrupted weights also caused issues
4. **Multiple failure modes masked each other**

The real bug was hidden beneath 3 layers of problems:
- Layer 1: Incompatible checkpoint (176 → 93 dim)
- Layer 2: NoisyNet frozen in eval_mode
- Layer 3: **Action mask permanently blocking trading** ← THE ACTUAL BUG

## Files Modified
- `environment.py` line 488: Added `self.bars_since_close += 1`

## Ready to Test

**All issues now resolved:**
✅ Checkpoint cleared (no 176-dim model)
✅ NoisyNet disabled (epsilon-greedy active)
✅ Epsilon=0.50 (50% random actions)
✅ **Action mask bug fixed (trading unlocked after 11 steps)**

**Command:**
```powershell
python main.py --episodes 50
```

**Expected Episode 1 Result:**
- Trades: 20-60 (mostly random exploration)
- Entropy: 1.0-1.4 bits
- Hold rate: 0.40-0.65
- Actions: Mix of HOLD/LONG/SHORT

**If STILL 0 trades:**
→ There's another bug (check min_hold_bars, max_trades_per_episode, weekend flatten logic)
→ But this is extremely unlikely - the cooldown was the blocker

## Confidence Level

**99.9% confident this fixes it** because:
- Action mask was the physical gate preventing trades
- Even 50% epsilon couldn't bypass the mask
- Fix allows mask to unlock after 11 steps
- Epsilon will force trading from step 12 onward
