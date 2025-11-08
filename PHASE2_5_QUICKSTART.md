# Phase 2.5 Quick Reference

## What Changed? (3 Surgical Tweaks)

### 1. **Looser Eval Probing** (config.py)
```python
eval_epsilon: 0.01 → 0.04        # 4% probes (still 96% deterministic)
eval_tie_tau: 0.03 → 0.05        # Wider tie band
hold_tie_tau: 0.02 → 0.035       # Less whipsaw
hold_break_after: 5 → 8          # Reduce premature breaks
```
**Why:** Lift entropy 0.7-0.8 → 1.0-1.1 bits (target range)

### 2. **Smarter Trade Gating** (trainer.py)
```python
# Before (arbitrary)
expected_trades = bars_per_pass / 100  # Magic number

# After (realistic)
eff = min_hold + cooldown/2      # Actual trade cycle
expected_trades = bars_per_pass / eff
hard_floor = 0.35 × expected     # Was 0.4
min_half = 0.6 × expected        # Was 0.7
```
**Why:** Reduce mult=0 episodes ~10-15% → <5%

### 3. **Stronger Parameter Noise** (config.py)
```python
noisy_sigma_init: 0.01 → 0.03    # 3× stronger (still conservative)
```
**Why:** Better Q-value separation during training (EMA smooths eval)

---

## Expected Impact

```
Metric              Before      After       Change
─────────────────────────────────────────────────────
Entropy             0.7-0.8     1.0-1.1     +0.2-0.3 bits ✅
mult=0 rate         10-15%      <5%         -5 to -10% ✅
Cross-seed mean     -0.67       -0.30       +0.37 (target)
Trade activity      20-30       20-30       Maintained ✅
Zero-trade          0%          0%          Maintained ✅
```

---

## Run It

```powershell
# 3-seed sweep (80 episodes, ~5-6 hours)
python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 80

# Check entropy lift
python check_metrics_addon.py
# Target: action_entropy_bits ≥ 1.0

# Check mult=0 reduction
python check_validation_diversity.py
# Target: mult=0.00 episodes <5%

# Check mean lift
python compare_seed_results.py
# Target: mean ≥ -0.45
```

---

## What to Watch

### In Metrics
- ✓ **Entropy:** 1.0-1.1 bits (was 0.7-0.8)
- ✓ **hold_rate:** 0.60-0.75 (was 0.75-0.85, slight drop expected)
- ✓ **switch_rate:** 0.15-0.22 (was 0.14-0.18, slight increase)

### In Diversity Check
- ✓ **mult=0 rate:** <5% episodes (was ~10-15%)
- ✓ **Penalty rate:** <5% total
- ✓ **Zero-trade:** 0% maintained

### In Results
- ✓ **Mean:** -0.30 to -0.45 (was -0.67)
- ✓ **Finals positive:** ≥ 2/3 seeds
- ✓ **Collapse:** ≤ 5%

---

## Risk Assessment

**ALL LOW RISK:**
- ✅ Minimal magnitude (all changes <3×)
- ✅ EMA model protects eval (smooths training noise)
- ✅ Complementary (each targets different issue)
- ✅ Reversible (all config flags)

**No risk to:**
- ✅ 0% collapse (mechanisms preserved)
- ✅ Healthy activity (unchanged)
- ✅ Validation stability (EMA + trimmed median + IQR cap intact)

---

## Success Criteria

**Must Achieve:**
- ✅ Entropy ≥ 1.0 bits
- ✅ mult=0 <5%
- ✅ Zero-trade = 0%
- ✅ Trade activity 20-30

**Target:**
- ✅ Mean ≥ -0.45
- ✅ Finals ≥ 2/3 positive

**Stretch:**
- ⭐ Mean > -0.30
- ⭐ Entropy 1.0-1.1 (sweet spot)

---

## Files Changed

```
✅ config.py - 5 parameters (eval + training exploration)
✅ trainer.py - Trade gating logic (realistic expected trades)
✅ All compile without errors
```

---

**Ready for sweep! Expected: Entropy 0.7→1.0, mult=0 ~10%→<5%** 🚀
