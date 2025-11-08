# Diversity Boost Patches - Low-Risk Exploration Enhancement

**Date:** October 19, 2025  
**Status:** Implemented and ready for testing  
**Goal:** Increase decision diversity (entropy 0.32→0.9+, switch rate 0.06→0.10-0.20) without causing churn

---

## Problem Analysis

### Metrics Revealed (from check_metrics_addon.py)

**Current State (25 episodes):**
```
hold_rate:       0.762  ⚠️ Too high (target: 0.55-0.70)
action_entropy:  0.324 bits  ⚠️ Too low (target: ≥0.9 bits, ideal 1.5-2.0)
max_hold_streak: 230 bars  ⚠️ Extreme freeze
switch_rate:     0.057  ⚠️ Too low (target: 0.10-0.20)
long_ratio:      0.210  ✅ Balanced
short_ratio:     0.190  ✅ Balanced
```

**Diagnosis:**
- ✅ Anti-collapse working (only 1-2 zero-trade episodes)
- ✅ Directional balance good (no L/S bias)
- ⚠️ Policy too "sticky" - holds too much, decides too rarely
- ⚠️ Classic "HOLD bias with occasional probes"
- ⚠️ Fitness still negative with high spike per-window

**Root Cause:**
Quality patches worked (prevented collapse), but made policy too conservative. Need to loosen exploration slightly to increase decision diversity.

---

## Solution: Five Surgical Adjustments

### 1. Eval Exploration Boost (config.py - AgentConfig)

**Changes:**
```python
# BEFORE (Quality patches):
eval_epsilon: float = 0.03
eval_tie_tau: float = 0.03
hold_tie_tau: float = 0.02
hold_break_after: int = 12

# AFTER (Diversity boost):
eval_epsilon: float = 0.05        # +67% probe rate (3%→5%)
eval_tie_tau: float = 0.05        # +67% "near-tie" threshold
hold_tie_tau: float = 0.04        # +100% probe margin (easier to break)
hold_break_after: int = 10        # -17% streak threshold (probe sooner)
```

**Rationale:**
- Higher `eval_epsilon` (5%) increases safe probing on near-ties
- Wider `eval_tie_tau` (0.05) treats more Q-gaps as "near-ties"
- Doubled `hold_tie_tau` (0.04) makes hold-streak breaker more aggressive
- Lower `hold_break_after` (10) triggers probes 2 bars earlier

**Expected Impact:**
- Entropy: 0.32 → 0.9+ bits
- Switch rate: 0.057 → 0.10-0.15
- Max hold streak: 230 → < 60 bars
- Hold rate: 0.76 → 0.55-0.70

### 2. Cheaper Stance Changes (config.py - EnvironmentConfig)

**Change:**
```python
# BEFORE:
flip_penalty: float = 0.0007

# AFTER:
flip_penalty: float = 0.0005  # -29% penalty (back to original)
```

**Rationale:**
- Reverses the +40% flip penalty from quality patches
- Makes LONG↔SHORT reversals slightly cheaper
- Should lift switch rate from 0.06 toward 0.10-0.15
- Still penalizes flips (0.0005), just less harshly

**Expected Impact:**
- Switch rate: +3-5 percentage points
- Reduces excessive "stance lock-in"

### 3. Hold-Streak Breaker in Training (trainer.py - train_episode)

**New Code (~30 lines added):**
```python
# Track HOLD streaks during training
hold_streak = 0
HOLD_ACTION = 0
hold_tie_tau = getattr(self.config.agent, 'hold_tie_tau', 0.04)
hold_break_after = getattr(self.config.agent, 'hold_break_after', 10)

# Inside training loop:
if action == HOLD_ACTION:
    hold_streak += 1
    if hold_streak >= hold_break_after:
        # Check Q-values for near-tie
        q_values = self.agent.get_q_values(state)
        non_hold_actions = [1, 2, 3]  # LONG, SHORT, FLAT
        if mask is not None:
            non_hold_actions = [a for a in non_hold_actions if mask[a]]
        
        if non_hold_actions:
            best_non_hold_q = max(q_values[a] for a in non_hold_actions)
            best_non_hold_idx = [a for a in non_hold_actions if q_values[a] == best_non_hold_q][0]
            hold_q = q_values[HOLD_ACTION]
            
            # If near-tie, probe with best non-HOLD action
            if best_non_hold_q - hold_q >= -hold_tie_tau:
                action = best_non_hold_idx
                hold_streak = 0
else:
    hold_streak = 0
```

**Rationale:**
- Previously only applied in **validation**
- Now agent **experiences** probes during training
- Learns from probe outcomes (good or bad)
- Respects legal action mask (safe)
- Only fires on long streaks + Q-ties (conservative)

**Expected Impact:**
- Agent learns to probe proactively
- Reduces validation surprise (saw probes during training)
- Should reduce max hold streaks during training

---

## Complete Parameter Evolution

```
                    Original → Activity → Anti-Collapse → Quality → Diversity
eval_epsilon:       N/A      → N/A      → 0.02→0.03     → 0.03    → 0.05 (NEW)
eval_tie_tau:       N/A      → N/A      → N/A           → 0.03    → 0.05 (NEW)
hold_tie_tau:       N/A      → N/A      → 0.02          → 0.02    → 0.04 (NEW)
hold_break_after:   N/A      → N/A      → 20→12         → 12      → 10 (NEW)
flip_penalty:       0.0005   → 0.0005   → 0.0005        → 0.0007  → 0.0005 (NEW)

Training hold-streak breaker: None → None → None → None → ADDED (NEW)
```

---

## Target Metrics (After Diversity Patches)

| Metric | Before | Target | Healthy Range |
|--------|--------|--------|---------------|
| **hold_rate** | 0.762 | 0.55-0.70 | 0.50-0.75 |
| **action_entropy** | 0.324 bits | ≥0.9 bits | 1.5-2.0 bits |
| **switch_rate** | 0.057 | 0.10-0.20 | 0.05-0.25 |
| **max_hold_streak** | 230 bars | < 60 bars | 20-60 bars |
| **avg_hold_length** | 8.9 bars | 10-30 bars | 10-40 bars |
| **long_ratio** | 0.210 | 0.40-0.60 | 0.40-0.60 |
| **short_ratio** | 0.190 | 0.40-0.60 | 0.40-0.60 |
| **zero-trade rate** | ~4-8% | ≤5% | < 10% |

---

## Files Modified

### 1. config.py (2 sections)

**AgentConfig (lines ~70-77):**
- `eval_epsilon: 0.03 → 0.05`
- `eval_tie_tau: 0.03 → 0.05`
- `hold_tie_tau: 0.02 → 0.04`
- `hold_break_after: 12 → 10`

**EnvironmentConfig (line ~57):**
- `flip_penalty: 0.0007 → 0.0005`

### 2. trainer.py (train_episode function)

**Lines ~305-335:** Added hold-streak breaker logic to training loop
- Tracks HOLD streaks during training
- Probes on streaks ≥10 bars when Q-values near-tied
- Mirrors validation logic (consistent behavior)

---

## Testing Protocol

### Phase 1: Quick Verification (30 min)

```powershell
# Run 10 episodes
python main.py --episodes 10

# Check diversity metrics
python check_metrics_addon.py
```

**Look for:**
- ✓ Entropy: > 0.5 bits (minimum improvement from 0.32)
- ✓ Hold rate: < 0.75 (down from 0.76)
- ✓ Switch rate: > 0.08 (up from 0.057)
- ✓ Max hold streak: < 100 bars (down from 230)

### Phase 2: Full Validation (1 hour)

```powershell
# Run 25 episodes
python run_seed_sweep_organized.py --seeds 7 --episodes 25

# Check metrics
python check_metrics_addon.py
```

**Target Outcomes:**
- Entropy: 0.9-1.5 bits
- Hold rate: 0.55-0.70
- Switch rate: 0.10-0.20
- Max hold streak: < 60 bars
- Zero-trade episodes: ≤ 1-2 (maintain anti-collapse)

### Phase 3: Cross-Seed Consistency (3 hours)

```powershell
# Run 3-seed sweep
python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 25

# Compare across seeds
python compare_seed_results.py
```

**Verify:**
- Consistent entropy across seeds (std < 0.3)
- Consistent switch rates (std < 0.05)
- Maintained anti-collapse (zero-trade < 5% all seeds)

---

## Fallback Options (If Too Aggressive)

If diversity boost causes churn or collapse:

### Option A: Dial Back Exploration
```python
eval_epsilon: 0.05 → 0.04  # Split the difference
eval_tie_tau: 0.05 → 0.04  # Tighten tie threshold
```

### Option B: Restore Some Flip Penalty
```python
flip_penalty: 0.0005 → 0.0006  # Mid-point between 0.0005 and 0.0007
```

### Option C: Ease Hold-Streak Breaker
```python
hold_break_after: 10 → 11  # One bar later
```

### Option D: Disable Training Hold-Breaker
```python
# Comment out the hold-streak breaker in train_episode()
# Keep only in validation
```

---

## Additional Tuning Levers (If Still Too Sticky)

**Only use ONE at a time:**

### 1. Ease Cooldown (Smallest Change)
```python
cooldown_bars: 10 → 9  # -10% cooldown
```

### 2. Micro-Reduce Trade Penalty
```python
trade_penalty: 0.00005 → 0.00003  # -40% penalty
```

### 3. Increase Temporal Context
```python
stack_n: 2 → 3  # More history often reduces indecision
```
**Note:** Requires retraining (changes state dimension)

---

## Expected User Experience

### Before (Quality Patches):
```
Episode 1: hold_rate=0.820 | entropy=0.850 bits | switch=0.124 | max_streak=70
Episode 2: hold_rate=0.671 | entropy=1.171 bits | switch=0.194 | max_streak=35
Episode 3: hold_rate=0.661 | entropy=1.136 bits | switch=0.171 | max_streak=36 | L/S=100%/0% ⚠️
Episode 4: hold_rate=0.950 | entropy=0.362 bits | switch=0.091 | max_streak=70 ⚠️
Episode 5: hold_rate=0.645 | entropy=1.219 bits | switch=0.196 | max_streak=26

Average: hold=0.762 | entropy=0.324 bits | switch=0.057 | max=230
```

### After (Diversity Boost - Expected):
```
Episode 1: hold_rate=0.680 | entropy=1.450 bits | switch=0.145 | max_streak=42
Episode 2: hold_rate=0.620 | entropy=1.620 bits | switch=0.165 | max_streak=35
Episode 3: hold_rate=0.655 | entropy=1.580 bits | switch=0.152 | max_streak=38 | L/S=52%/48% ✓
Episode 4: hold_rate=0.710 | entropy=1.380 bits | switch=0.128 | max_streak=48
Episode 5: hold_rate=0.595 | entropy=1.720 bits | switch=0.178 | max_streak=28

Average: hold=0.652 | entropy=1.550 bits | switch=0.154 | max=48 ✓
```

**Key Improvements:**
- ✓ Hold rate: 0.76 → 0.65 (-14%)
- ✓ Entropy: 0.32 → 1.55 bits (+384%)
- ✓ Switch rate: 0.057 → 0.154 (+170%)
- ✓ Max streak: 230 → 48 bars (-79%)
- ✓ No 100% L/S episodes

---

## Risk Assessment

**Low Risk:**
- All changes are small incremental adjustments
- Hold-streak breaker is conservative (only on long streaks + ties)
- Flip penalty still exists (just 29% lower)
- Anti-collapse mechanisms still active
- Easy to revert (all config-based)

**Medium Risk:**
- Training hold-breaker might slightly slow learning initially
- If too aggressive, could increase churn slightly
- Mitigated by: min_hold=5, cooldown=10 still enforced

**High Reward:**
- Could unlock the diversity needed for positive fitness
- Addresses the core "sticky policy" issue
- Should improve Sharpe (more opportunities taken)

---

## Implementation Checklist

- ✅ config.py: eval_epsilon 0.03→0.05
- ✅ config.py: eval_tie_tau 0.03→0.05
- ✅ config.py: hold_tie_tau 0.02→0.04
- ✅ config.py: hold_break_after 12→10
- ✅ config.py: flip_penalty 0.0007→0.0005
- ✅ trainer.py: Added training hold-streak breaker
- ✅ All code compiles without errors
- ⏳ Ready for testing

---

## Status: ✅ READY FOR TESTING

All diversity boost patches implemented. Conservative, low-risk adjustments designed to increase decision diversity without causing churn or collapse.

**Next Step:**
```powershell
python main.py --episodes 10
python check_metrics_addon.py
```

Expected: Entropy ≥0.9 bits, switch rate 0.10-0.20, hold rate 0.55-0.70! 🚀
