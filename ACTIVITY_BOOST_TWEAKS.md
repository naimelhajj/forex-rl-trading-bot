## Trading Activity Boost Tweaks - October 18, 2025

**Goal:** Encourage more consistent trading activity while maintaining quality control  
**Approach:** Ease constraints and add complementary exploration without aggressive tuning

---

## Summary of Changes

### A) ✅ Validation Gate Further Eased

**File:** `trainer.py` line 526  
**Change:** `/80 → /100` bars per expected trade

```python
# BEFORE:
expected_trades = max(8, int(bars_per_pass / 80))  # ~8 trades for 600 bars

# AFTER:
expected_trades = max(6, int(bars_per_pass / 100))  # ~6 trades for 600 bars
```

**Effect:**
- 600-bar window now expects **6 trades** (was 8)
- Thresholds:
  - `hard_floor`: 5 trades (was 5, unchanged)
  - `min_half`: 6 trades (was ~6, now formalized)
  - `min_full`: 7 trades (was 8)
- **Penalty stays 0.000 once you hit 6-7+ trades**
- More forgiving for borderline-but-OK episodes

---

### B) ✅ Position Constraints Relaxed

**File:** `config.py` lines 59-60  
**Change:** Reduce hold time and cooldown requirements

```python
# BEFORE:
min_hold_bars: int = 8   # Minimum position hold time
cooldown_bars: int = 16  # Prevent churning

# AFTER:
min_hold_bars: int = 6   # Was 8, eased to 6
cooldown_bars: int = 12  # Was 16, eased to 12
```

**Effect:**
- **More opportunities to trade** without turning into a scalper
- With min_hold=6 and cooldown=12, theoretical max trades per 600 bars:
  - Before: 600/(8+16) = 25 max trades
  - After: 600/(6+12) = 33 max trades
- **~30% more trading opportunities** without removing quality control

---

### C) ✅ Epsilon Exploration Re-Enabled

**File:** `config.py` lines 72-73  
**Change:** Add small ε-greedy exploration alongside NoisyNet

```python
# BEFORE:
epsilon_start: float = 0.0  # Disabled for NoisyNet
epsilon_end: float = 0.05   # Disabled for NoisyNet

# AFTER:
epsilon_start: float = 0.10  # Small ε to prevent HOLD lock-in
epsilon_end: float = 0.05    # Decay to minimal exploration
```

**Effect:**
- **NoisyNet provides state-dependent exploration** (still active)
- **Epsilon provides uniform fallback** when Q-values are close
- Prevents lock-in to HOLD when:
  - Q(HOLD) ≈ Q(LONG) ≈ Q(SHORT) (all similarly valued)
  - NoisyNet noise happens to favor HOLD slightly
- **Dual exploration mechanism:** NoisyNet for general exploration + ε for tie-breaking

**Decay Schedule:**
- Episode 0: ε = 0.10 (10% random actions)
- Episode 100: ε ≈ 0.07
- Episode 500: ε ≈ 0.05 (minimal)
- Decay rate: 0.997 per episode (slow, gentle)

---

### D) ✅ Softer Friction Maintained

**File:** `trainer.py` lines 667-678 (already applied previously)  
**Status:** Kept as-is from previous session

```python
# Validation friction ranges (already softened):
spread: uniform(0.00013, 0.00020)  # Was 0.00012-0.00025
slippage: uniform(0.6, 1.0)        # Was 0.5-1.2
# hard_max_lots: REMOVED             # Was clamped to 0.1
```

**Effect:** Realistic but not prohibitive friction during validation

---

## Expected Behavior After Tweaks

### Validation Thresholds (600-bar window):

| Trades | Old Behavior | New Behavior |
|--------|--------------|--------------|
| 0-4    | mult=0.0, pen=0.25 | mult=0.0, pen=0.25 (unchanged) |
| 5      | mult=0.0, pen=~0.07 | mult=0.5, pen=~0.04 ✅ Better! |
| 6      | mult=0.5, pen=0.000 | mult=0.75, pen=0.000 ✅ Better! |
| 7+     | mult=0.75-1.0 | mult=1.0, pen=0.000 ✅ Full credit |

**Key Improvement:** 5-6 trade episodes now get better treatment instead of harsh penalties.

### Trading Dynamics:

**Before Changes:**
- min_hold=8, cooldown=16 → 24 bars between trades
- Expected 8-10 trades per 600 bars
- Epsilon=0 → NoisyNet only → can lock into HOLD

**After Changes:**
- min_hold=6, cooldown=12 → **18 bars between trades** (25% faster)
- Expected **6-7 trades** per 600 bars (more realistic)
- Epsilon=0.10→0.05 → **Prevents HOLD lock-in** when Q-values close

---

## Testing Recommendations

### 1. Quick Test (15 episodes):
```powershell
python main.py --episodes 15
```

**Look for:**
- Trade counts consistently 6-15+ per validation window
- Fewer episodes with 0-2 trades
- Penalty (pen) rarely activates (should be 0.000 most times)
- Some epsilon-driven random actions early on

### 2. Check Exploration Mix:
In logs, look for action distribution:
- Should see mix of HOLD, LONG, SHORT (not 95% HOLD)
- Epsilon will cause ~10% random actions early, ~5% later
- NoisyNet adds continuous perturbation on top

### 3. Validation Diversity:
```powershell
python check_validation_diversity.py
```

Expected pattern:
```
Ep  3: trades= 7.0 | mult=1.00 | pen=0.000 | score=+0.120 ✓
Ep  5: trades= 5.0 | mult=0.50 | pen=0.036 | score=+0.050 ✓
Ep  8: trades= 9.0 | mult=1.00 | pen=0.000 | score=+0.180 ✓
Ep 10: trades= 6.0 | mult=0.75 | pen=0.000 | score=+0.085 ✓
```

---

## Rollback Strategy (If Needed)

**If trading becomes too frequent/noisy:**

1. **Restore constraints:**
   ```python
   min_hold_bars: int = 8
   cooldown_bars: int = 16
   ```

2. **Reduce epsilon:**
   ```python
   epsilon_start: float = 0.05  # Was 0.10
   epsilon_end: float = 0.01    # Was 0.05
   ```

3. **Keep gate at /100** (this is the safest change)

**If still seeing 0-trade episodes:**

1. **Increase penalty strength:**
   ```python
   # In trainer.py line ~533
   undertrade_penalty = 0.40 * (shortage / max(1, min_half))  # Was 0.25
   ```

2. **Lower gate further:**
   ```python
   expected_trades = max(5, int(bars_per_pass / 120))  # Was /100
   ```

---

## Philosophy Behind Changes

### Validation Gate: Progressive Forgiveness
- **Hard floor (5 trades):** Still prevents complete inactivity
- **Partial credit (6 trades):** Now gets 0.75x mult (was 0.5x)
- **Full credit (7+ trades):** Easier to achieve, encourages activity

### Constraints: Opportunity vs Quality
- **min_hold=6:** Still prevents scalping (6 bars = 6 hours for hourly data)
- **cooldown=12:** Still prevents churning (12-hour rest between trades)
- **Combined:** 18-bar cycle allows ~33 trades max vs 25 before (reasonable increase)

### Exploration: Dual Mechanism
- **NoisyNet:** Provides intelligent, state-dependent exploration
- **Epsilon:** Provides uniform fallback to break ties and prevent lock-in
- **Together:** More robust than either alone

---

## Summary Table

| Change | Old Value | New Value | Impact |
|--------|-----------|-----------|--------|
| **Gate divisor** | /80 | /100 | 8→6 expected trades |
| **min_hold** | 8 bars | 6 bars | +25% trading speed |
| **cooldown** | 16 bars | 12 bars | +25% trading speed |
| **epsilon_start** | 0.0 | 0.10 | Prevents HOLD lock-in |
| **epsilon_end** | 0.0 | 0.05 | Minimal final exploration |

**Net Effect:** ~30% more trading opportunities with more forgiving validation

---

## Files Modified

1. **trainer.py** - Line 526: Gate divisor /80 → /100
2. **config.py** - Lines 59-60: min_hold 8→6, cooldown 16→12
3. **config.py** - Lines 72-73: epsilon 0.0→0.10/0.05

**All changes are conservative nudges, not aggressive overhauls. Easy to test and rollback if needed.**
