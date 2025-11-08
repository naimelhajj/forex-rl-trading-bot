# Disjoint Validation Patches - Implementation Summary

## Overview
Implemented 4 surgical patches to make validation more robust, stable, and adaptive to different run lengths. These patches address the core issue of **validation noise** from overlapping windows and spiky outliers.

---

## Patch 1: Disjoint Validation Windows

### What Changed
- **Before**: K=7 passes with full reset each time → overlapping data, correlated results
- **After**: K=7 **disjoint** windows with no overlap → independent samples

### Implementation
```python
def _make_disjoint_windows(self, idx, k: int, min_bars: int = 600):
    """Split validation index into k disjoint windows (no overlap)."""
    n = len(idx)
    step = n // k
    windows = []
    for i in range(k):
        start = i * step
        end = n if i == k - 1 else (i + 1) * step
        if end - start >= min_bars:
            windows.append((start, end))
    return windows
```

### Why This Matters
- **Eliminates correlation** between validation passes
- Each window sees **different market conditions**
- More reliable estimate of true performance
- **Min 600 bars** per window ensures sufficient data

### Example Output
```
[VAL] window 1: 2024-01-01 00:00:00 → 2024-01-25 23:00:00  (600 bars)
[VAL] window 2: 2024-01-26 00:00:00 → 2024-02-19 23:00:00  (600 bars)
[VAL] window 3: 2024-02-20 00:00:00 → 2024-03-15 23:00:00  (600 bars)
```

---

## Patch 2: IQR Dispersion Penalty

### What Changed
- **Before**: Median fitness across K passes (vulnerable to one lucky pass)
- **After**: Median - 0.25×IQR (penalizes inconsistent/spiky performance)

### Implementation
```python
# Compute stability-adjusted median
median = float(np.median(fits)) if fits else 0.0
q75, q25 = np.percentile(fits, [75, 25]) if len(fits) >= 2 else (median, median)
iqr = float(q75 - q25)
stability_adj = median - 0.25 * iqr  # Knock down spiky runs
```

### Why This Matters
- **Rewards consistency** over lucky spikes
- IQR = Inter-Quartile Range (Q75 - Q25) measures spread
- 0.25× penalty tuned for balance (not too harsh)
- **Prevents overfitting** to volatile micro-windows

### Example
| Scenario | Passes | Median | IQR | Adjusted | Effect |
|----------|--------|--------|-----|----------|--------|
| Consistent | [0.22, 0.24, 0.23, 0.25, 0.24] | 0.24 | 0.03 | 0.233 | Small penalty |
| Spiky | [0.10, 0.45, 0.12, 0.50, 0.13] | 0.13 | 0.38 | 0.035 | Large penalty |

---

## Patch 3: Adaptive Trade Gating (Window-Length Aware)

### What Changed
- **Before**: Fixed thresholds (8/10/15 trades) regardless of window length
- **After**: Thresholds scale with `bars_per_pass / 60` (adaptive to window size)

### Implementation
```python
# Adaptive trade gate scaled by window length
bars_per_pass = max(1, windows[0][1] - windows[0][0]) if windows else 1
expected_trades = max(8, int(bars_per_pass / 60))  # ~1 trade per 30-60 bars
hard_floor = max(5, int(0.4 * expected_trades))
min_half = max(hard_floor + 1, int(0.7 * expected_trades))
min_full = max(min_half + 1, expected_trades)

# Apply gating multipliers
if median_trades < hard_floor:
    mult = 0.0
elif median_trades < min_half:
    mult = 0.5
elif median_trades < min_full:
    mult = 0.75
else:
    mult = 1.0

val_score = stability_adj * mult
```

### Why This Matters
- **Scales with window length** (600 bars → expect 10 trades, 3000 bars → expect 50 trades)
- **No magic numbers** → works for smoke tests AND full training
- **Fair gating** for different min_hold/cooldown settings
- Four-tier multipliers: 0.0× (inactive), 0.5× (partial), 0.75× (active), 1.0× (full credit)

### Example Thresholds
| Window Size | Expected Trades | Hard Floor | 50% Credit | 75% Credit | Full Credit |
|-------------|----------------|------------|------------|------------|-------------|
| 600 bars | 10 | 5 | 7 | 10 | 11+ |
| 1500 bars | 25 | 10 | 18 | 25 | 26+ |
| 3000 bars | 50 | 20 | 35 | 50 | 51+ |

---

## Patch 4: EMA-Based Early Stop

### What Changed
- **Before**: Early stop on raw `best_fitness` (sensitive to single spikes)
- **After**: Early stop on EMA of stability-adjusted score (smooth trend)

### Implementation
```python
# EMA smoothing with alpha=0.3 (responsive in smoke mode)
alpha = 0.3
self.best_fitness_ema = alpha * current_fitness + (1 - alpha) * self.best_fitness_ema

# Save best model based on EMA
if self.best_fitness_ema > self.best_fitness_ema_saved:
    self.save_checkpoint("best_model.pt")
    self.best_fitness_ema_saved = self.best_fitness_ema
    print(f"  ✓ New best fitness (EMA): {self.best_fitness_ema:.4f} (raw: {current_fitness:.4f})")
```

### Why This Matters
- **Alpha = 0.3** balances responsiveness vs stability
- EMA reacts to **trends**, not single spikes
- **Best model checkpoint** reflects sustained improvement
- Early stop doesn't trigger on noise

### Example EMA Tracking
```
Episode 10  | Raw: 0.180 | EMA: 0.180  ✓ New best
Episode 20  | Raw: 0.220 | EMA: 0.192  ✓ New best
Episode 30  | Raw: 0.150 | EMA: 0.179  (no save)
Episode 40  | Raw: 0.250 | EMA: 0.200  ✓ New best (trend improved)
```

---

## Complete Validation Output

### Before (Overlapping Windows)
```
[VAL] K=7 passes | median fitness=0.223 | trades=19.0
```

### After (Disjoint + IQR + Adaptive)
```
[VAL] window 1: 2024-01-01 → 2024-01-25  (600 bars)
[VAL] window 2: 2024-01-26 → 2024-02-19  (600 bars)
[VAL] window 3: 2024-02-20 → 2024-03-15  (600 bars)
[VAL] K=7 disjoint | median fitness=0.223 | IQR=0.087 | adj=0.201 | trades=19.0 | mult=1.00 | score=0.201
```

**Key Differences**:
- Shows **window ranges** (verify coverage)
- Shows **IQR** (measure of dispersion)
- Shows **adjusted score** (median - 0.25×IQR)
- Shows **multiplier** (trade gate effect)
- Shows **final score** (what drives early-stop)

---

## Expected Behavior After Patches

### Smoke Test (5 episodes, 600-bar windows)
```
[VAL] K=5 disjoint | median fitness=0.201 | IQR=0.045 | adj=0.190 | trades=12.0 | mult=1.00 | score=0.190
  ✓ New best fitness (EMA): 0.190 (raw: 0.190)
```
- **Adaptive thresholds**: 600 bars → expect ~10 trades → hard_floor=5, full_credit=11
- **Consistent**: IQR ~0.04 (low dispersion)
- **Active trading**: 12 trades → full credit (mult=1.0)

### Full Training (30+ episodes, 1500-bar windows)
```
[VAL] K=7 disjoint | median fitness=0.287 | IQR=0.112 | adj=0.259 | trades=38.0 | mult=0.75 | score=0.194
  ✓ New best fitness (EMA): 0.194 (raw: 0.194)
```
- **Adaptive thresholds**: 1500 bars → expect ~25 trades → hard_floor=10, full_credit=26
- **Higher IQR**: Longer windows → more variance (penalty applied)
- **Active but not full**: 38 trades on 1500 bars → 75% credit (needs 50+ for full)

---

## Testing Checklist

### 1. Verify Disjoint Windows
- [ ] Run smoke test: `python main.py --episodes 5`
- [ ] Check output shows `[VAL] window 1: ... → ...` (time ranges)
- [ ] Verify `(600 bars)` or higher (no tiny windows)
- [ ] Confirm K=5-7 windows created (depends on val data size)

### 2. Verify IQR Penalty
- [ ] Look for `IQR=X.XXX` in validation output
- [ ] Verify `adj < median` (penalty applied)
- [ ] Check `adj ≈ median - 0.25*IQR` (math correct)

### 3. Verify Adaptive Gating
- [ ] Check `mult=` in output (0.0, 0.5, 0.75, or 1.0)
- [ ] For 600 bars: expect full credit at ~11 trades
- [ ] For 1500 bars: expect full credit at ~26 trades
- [ ] For 3000 bars: expect full credit at ~51 trades

### 4. Verify EMA Early-Stop
- [ ] Check for `✓ New best fitness (EMA): X.XXX (raw: Y.YYY)`
- [ ] Verify EMA is **smoother** than raw score (not jumping around)
- [ ] Confirm `best_model.pt` saved only when EMA improves

### 5. End-to-End Stability
- [ ] Run 10-20 episodes: validation score should **trend** (not spike randomly)
- [ ] Early stop should trigger on **plateau**, not noise
- [ ] Best model should reflect **sustained performance**, not lucky pass

---

## High-Leverage Knobs (No Code Changes)

### 1. Smoke → Real Training
Once patches verified:
```bash
python main.py --episodes 30  # Bump from 5
```
- Longer episodes → adaptive gates become stricter
- More validations → EMA smoothing shows its value

### 2. Frame Stacking Tuning
If `state_size=176` is large (memory/speed issues):
```python
# In config.py
stack_n = 2  # Down from 3
```
- Reduces state size from 176 to ~120
- Faster training, still has temporal context

### 3. PER Beta Annealing (if not already done)
```python
# In trainer.py (already implemented, verify settings)
self.per_beta_start = 0.4
self.per_beta_end = 1.0
self.per_beta_anneal_steps = int(num_episodes * 0.7)
```
- Β anneals from 0.4 → 1.0 over 70% of training
- Helps stabilize later episodes

### 4. N-Step Targets (if DQN supports it)
```python
# In agent.py config (check if already implemented)
n_step = 3
```
- Helps propagate reward through `min_hold` delays
- Faster credit assignment for delayed rewards

---

## Regression Prevention

### What Could Go Wrong
1. **Window too short**: If val data <4200 bars (600×7), some windows dropped
2. **IQR too high**: If market extremely volatile, penalty might be harsh
3. **Trade gate mismatch**: If min_hold changes, expected_trades formula might need tuning
4. **EMA plateau**: If alpha too low, early-stop becomes sluggish

### Monitoring
- Watch for `K < 7` (means some windows dropped due to min_bars)
- If `IQR > 0.5` consistently, market might be too choppy (consider different val period)
- If `mult=0.0` frequently, tune `expected_trades` formula (currently `bars/60`)
- If best model never updates, check EMA isn't stuck (alpha might be too low)

---

## Summary

**What We Fixed**:
1. Validation passes now **disjoint** (no overlap) → independent samples
2. **IQR penalty** tames spiky outliers → rewards consistency
3. **Adaptive trade gating** scales with window length → fair across run types
4. **EMA early-stop** reacts to trends, not noise → stable checkpoints

**Expected Impact**:
- Validation fitness will be **much steadier** (less variance)
- Trade counts will **scale appropriately** (no magic numbers)
- Best model checkpoints reflect **sustained improvement** (not flukes)
- Early stop triggers on **plateaus**, not random dips

**Next Steps**:
1. Run smoke test to verify all patches working
2. Check output matches expected format
3. Bump to 30+ episodes for full training
4. Monitor EMA trend and best model saves
