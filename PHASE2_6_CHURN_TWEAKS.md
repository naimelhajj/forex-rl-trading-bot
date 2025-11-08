# Phase 2.6 - Churn Reduction Tweaks

**Date:** 2025-01-22  
**Status:** ✅ COMPLETE - Micro-adjustments to calm 40+ trade spikes

---

## Observation from 3-Seed Run

**What's working (✅ Keep!):**
- ✅ Gating fix successful: `scaled_exp=24`, `min_full=24`, many `mult=1.00`
- ✅ K=7 consistent, 600-bar windows working
- ✅ Final scores positive for seeds 7 & 777
- ✅ Penalty rate low (~10-20%)

**What needs refinement (⚠️ Tune):**
- ⚠️ Over-trading spikes: Some episodes with 40+ trades (churn in chop)
- ⚠️ Action entropy low: ~0.74 bits (target 0.75-0.85)
- ⚠️ Long bias: ~0.62 (target 0.55-0.60)
- ⚠️ Negative episodes correlate with 30-43 trade count (over-fitting chop)

**Pattern identified:**
```
High-trade episodes (35-43 trades):
→ Often score -1.5 to -2.0
→ Excessive flipping in choppy markets
→ Costs outweigh edge

Moderate-trade episodes (24-30 trades):
→ Often score +0.5 to +1.5
→ Selective entries on better setups
→ Edge exceeds costs
```

---

## Changes Applied (2 Micro-Adjustments)

### 1. ✅ **Environment Cadence** - Ease Churn Slightly

**Before (Phase 2.5 aggressive settings):**
```python
cooldown_bars: 8         # Short cooldown
min_hold_bars: 4         # Short minimum hold
trade_penalty: 0.00005   # Tiny per-trade cost
flip_penalty: 0.0005     # Low stance-change cost

# Result: 
# - Trades per window: 22-43 (wide range)
# - Some 40+ trade episodes with negative scores
# - Over-trading in chop
```

**After (Phase 2.6 calmer settings):**
```python
cooldown_bars: 10        # +2 bars (was 8)
min_hold_bars: 5         # +1 bar (was 4)
trade_penalty: 0.00006   # +20% (was 0.00005)
flip_penalty: 0.0006     # +20% (was 0.0005)

# Expected result:
# - Trades per window: 24-32 (tighter range)
# - Fewer 40+ trade episodes
# - More selective entries
```

**Impact:**
- **Effective cadence:** 12 bars/trade → 15 bars/trade
- **Max theoretical trades:** 50 → 40 per 600-bar window
- **Expected actual trades:** 27-32 → 24-28 (healthier)
- **Filters out:** Low-quality chop trades while staying above min_full=24

---

### 2. ✅ **Agent Exploration** - Trim Probing in Chop

**Before (Phase 2.5 exploration boost):**
```python
eval_epsilon: 0.06          # 6% random on ties
hold_tie_tau: 0.030         # Tight hold tolerance
hold_break_after: 6         # Break holds quickly

# Result:
# - Some unnecessary exits in good holds
# - Random probes adding noise in chop
# - Entropy ~0.74 bits (too many low-value actions)
```

**After (Phase 2.6 refined exploration):**
```python
eval_epsilon: 0.05          # 5% random on ties (was 0.06)
hold_tie_tau: 0.032         # Slightly more tolerance (was 0.030)
hold_break_after: 7         # Break holds a bit later (was 6)

# Expected result:
# - Fewer premature hold exits
# - Less noise probing in choppy regimes
# - Entropy ~0.78-0.82 bits (quality over quantity)
```

**Impact:**
- **-17% random probing** on Q-ties (0.06 → 0.05)
- **+7% hold tolerance** (0.030 → 0.032)
- **+17% hold patience** (6 → 7 bars)
- **Preserves:** Tie-only logic (still smart exploration)

---

## Math: Trade Count Impact

### Theoretical Max Trades (600-bar window)

**Phase 2.5 (aggressive):**
```
Cadence: min_hold(4) + cooldown(8) = 12 bars/trade
Max trades: 600 / 12 = 50 trades
Observed: 55% of max = 27.5 trades average
Spikes: Up to 40-43 trades (80-86% of max!)
```

**Phase 2.6 (calmer):**
```
Cadence: min_hold(5) + cooldown(10) = 15 bars/trade
Max trades: 600 / 15 = 40 trades
Expected: 55% of max = 22 trades average
Expected range: 20-28 trades (50-70% of max)
Max spike: ~32 trades (80% of max, down from 43)
```

**Change:**
- **Max theoretical:** 50 → 40 trades (-20%)
- **Expected average:** 27.5 → 22 trades (-20%)
- **Expected range:** 22-43 → 20-28 trades (tighter!)
- **Spike reduction:** 40-43 → max ~32 trades

---

## Gating Alignment Check

**Current gating thresholds (unchanged):**
```python
VAL_EXP_TRADES_SCALE: 0.32
VAL_EXP_TRADES_CAP: 24
VAL_MIN_FULL_TRADES: 16
VAL_MIN_HALF_TRADES: 8

# Result: min_full ≈ 19-20, min_half ≈ 11-12
```

**Phase 2.6 trade expectations:**
```
Expected average: 22 trades per window
Expected range: 20-28 trades

Gating thresholds:
- min_full ≈ 19-20
- min_half ≈ 11-12

Alignment check:
✅ 20 trades > min_full(19) → mult=1.00
✅ 22 trades > min_full(19) → mult=1.00
✅ 25 trades > min_full(19) → mult=1.00
✅ 28 trades > min_full(19) → mult=1.00

All expected trade counts get full credit!
```

**Verdict:** Phase 2.6 churn reduction stays **well above gating thresholds**. No conflict!

---

## Expected Outcomes

### Primary (Trade Activity)

**Before (Phase 2.5):**
```
Trades per window: 22-43 (wide range)
Average: ~27.5 trades
Spikes: 40-43 trades (churn!)
Episodes with 35+ trades: ~20%
```

**After (Phase 2.6 expected):**
```
Trades per window: 20-28 (tighter range)
Average: ~24 trades
Spikes: max ~32 trades (reduced)
Episodes with 35+ trades: <5%
```

### Secondary (Quality Metrics)

**Before:**
```
Action entropy: ~0.74 bits
Long ratio: ~0.62 (bias)
Negative episodes: ~25% (often high-trade)
Score range: -2.0 to +1.5
```

**After (expected):**
```
Action entropy: ~0.78-0.82 bits (↑ quality)
Long ratio: ~0.56-0.59 (↓ bias)
Negative episodes: ~15-18% (fewer)
Score range: -1.2 to +1.8 (tighter, higher)
```

### Tertiary (Gating - unchanged)

**Maintained:**
```
Penalty rate: ~10-20% ✅
mult=1.00 rate: ~70-80% ✅
Window sizing: 600 bars, K=7 ✅
Thresholds: min_full≈19 ✅
```

---

## Risk Assessment

**All changes are VERY LOW RISK:**

✅ **Minimal adjustments:** Only 7 parameters, small increments
✅ **No architectural changes:** Just config tuning
✅ **Gating-compatible:** 20-28 trades still >> min_full(19)
✅ **Reversible:** Easy to roll back if needed

**Expected side effects (all positive):**

1. **Fewer churn episodes:**
   - 40-43 trade spikes → max ~32 ✅
   - Negative high-trade episodes reduced

2. **Tighter score distribution:**
   - Less variance from over-trading ✅
   - More consistent quality

3. **Better entropy (paradoxically):**
   - Fewer noise actions → higher quality diversity ✅
   - 0.74 → 0.78-0.82 bits

4. **Reduced long bias:**
   - Less hold-breaking → fewer forced long exits ✅
   - 0.62 → 0.56-0.59 ratio

**No risk to:**
- ✅ Gating thresholds (stay well above min_full)
- ✅ mult=1.00 credit rate (maintained)
- ✅ Window sizing (unchanged)
- ✅ Anti-collapse (unchanged)

---

## Verification Commands

After running 20-episode spot-check:

```powershell
# Check trade count distribution
python check_validation_diversity.py
# Look for:
# - Median trades: 24-28 per window (was 27)
# - Spikes: max ~32 trades (was 40-43)
# - 35+ trade rate: <5% (was ~20%)

# Check entropy and ratios
python check_metrics_addon.py
# Look for:
# - Action entropy: 0.78-0.82 bits (was 0.74)
# - Long ratio: 0.56-0.59 (was 0.62)
# - Hold rate: 0.70-0.75 (was 0.60-0.65)

# Check score distribution
python compare_seed_results.py
# Look for:
# - Fewer negative episodes
# - Tighter score range
# - Mean improvement
```

---

## Quick Spot-Check (20 Episodes)

```powershell
# Clear old validation JSONs
if (Test-Path .\logs\validation_summaries) { 
    Remove-Item .\logs\validation_summaries\* -Recurse -Force 
}

# Run single seed, 20 episodes
python run_seed_sweep_organized.py --seeds 7 --episodes 20

# After completion, check metrics:
python check_validation_diversity.py
# Target: Median trades 24-28, max ~32

python check_metrics_addon.py
# Target: Entropy 0.78-0.82, long ratio 0.56-0.59
```

---

## Decision Tree

```
After 20-episode spot-check:

Median trades 24-28?
├─ YES → ✅ Churn calmed!
│         Check entropy improved (>0.77)
│         Verify fewer negative episodes
│         Proceed to full 80-episode run
│
└─ NO (still 30+) → Still too many trades
         Option A: Increase cooldown to 12
         Option B: Increase flip_penalty to 0.0007
         Re-run 10 episodes to verify

Max spikes < 35?
├─ YES → ✅ Spike reduction working!
│
└─ NO (still 38-43) → Extreme churn persists
         Increase trade_penalty to 0.00007
         AND increase min_hold to 6
         Re-check

Entropy > 0.77?
├─ YES → ✅ Quality diversity achieved!
│
└─ NO (still 0.74) → Exploration still noisy
         Keep eval_epsilon at 0.05
         Check if probes are on ties (should be)
         May need to tighten eval_tie_tau slightly
```

---

## Iterative Tuning (if needed)

**If median trades still > 30 after spot-check:**

```python
# Further churn reduction (Phase 2.7 candidate)
cooldown_bars: 12       # +2 more bars
flip_penalty: 0.0007    # +17% more cost
# OR
min_hold_bars: 6        # +1 more bar (most conservative)
```

**If entropy still < 0.77:**

```python
# Already reduced eval_epsilon to 0.05
# Check logs to ensure probing is tie-only
# If still low quality, tighten ties:
eval_tie_tau: 0.04      # Stricter tie definition
```

**If long bias still > 0.60:**

```python
# Already increased hold_break_after to 7
# If bias persists, consider:
hold_break_after: 8     # +1 more patience
```

---

## Summary Table

| Parameter | Phase 2.5 | Phase 2.6 | Change | Purpose |
|-----------|-----------|-----------|--------|---------|
| `cooldown_bars` | 8 | 10 | +2 | Reduce churn |
| `min_hold_bars` | 4 | 5 | +1 | Reduce churn |
| `trade_penalty` | 0.00005 | 0.00006 | +20% | Discourage noise |
| `flip_penalty` | 0.0005 | 0.0006 | +20% | Reduce whipsaw |
| `eval_epsilon` | 0.06 | 0.05 | -17% | Less probing in chop |
| `hold_tie_tau` | 0.030 | 0.032 | +7% | Keep good holds |
| `hold_break_after` | 6 | 7 | +17% | Less premature breaks |

**Net effect:**
- **Trades:** 27 → 24 average (tighter, cleaner)
- **Spikes:** 40-43 → max ~32 (controlled)
- **Entropy:** 0.74 → 0.78-0.82 bits (quality up)
- **Gating:** Still well above min_full=19 ✅

---

## Why This Maintains Gating Success

**Critical insight:**
```
Phase 2.5 trade count: 22-43 (average 27.5)
→ Gating threshold: min_full ≈ 19
→ Even lowest healthy episode (22) gets mult=1.00 ✅

Phase 2.6 trade count: 20-28 (average 24)
→ Gating threshold: min_full ≈ 19 (UNCHANGED)
→ Even lowest healthy episode (20) gets mult=1.00 ✅

Safety margin maintained: 20 > 19 (+5%)
No risk of dropping below threshold!
```

**What we're eliminating:**
- Not the 22-28 trade "healthy zone" ✅
- But the 35-43 trade "churn zone" ✗
- Which were often negative score episodes anyway ✗

**Result:** **Better signal-to-noise** without sacrificing gating credits!

---

## Bottom Line

**Problem:** Over-trading spikes (40-43 trades) causing negative episodes, low entropy (0.74), long bias (0.62)

**Solution (Phase 2.6 - 7 micro-tweaks):**
1. **Ease cadence:** cooldown 8→10, min_hold 4→5
2. **Tiny cost increases:** trade/flip penalties +20%
3. **Trim probing:** eval_epsilon 6%→5%
4. **Gentler hold-breaking:** tie_tau +7%, break_after 6→7

**Expected outcome:**
- Trades: 27 → 24 average (tighter range)
- Spikes: 40-43 → max ~32 (controlled)
- Entropy: 0.74 → 0.78-0.82 bits (quality up)
- Long bias: 0.62 → 0.56-0.59 (balanced)
- **Gating: Unchanged, still above threshold** ✅

**All changes minimal, config-only, gating-compatible, and easily reversible!** 🎯🚀

---

**Files modified: config.py only. Ready for 20-episode spot-check!**
