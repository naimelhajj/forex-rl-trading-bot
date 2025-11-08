# Phase 2.8d Fix Pack D2: Rolling Window Anti-Collapse - COMPLETE ✅

**Status**: Implementation complete, validation pending  
**Date**: 2025-01-15  
**Trigger**: Episode 52 directional collapse (98% long trades) detected in seed 42 run

---

## Problem Summary

### Episode 52 Collapse Discovery
- **Seed 42** (current run): Episode 52/80 → SPR=+0.954, trades=24, **long_ratio=0.98** (98% long)
- **Seed 777** (old run): Episode 80 → SPR=-0.204, trades=31, **long_ratio=0.98** (98% long)

### Root Cause Analysis
1. **Episode-level L/S balance too weak**: Only checked total trade counts at episode end, not continuous behavior
2. **No hold-rate guardrail**: Agent could park in >90% HOLD state without penalty
3. **Hold bias mechanics**: Small hold_tie_tau + eval_tie_tau allowed "direction parking" strategy
4. **Weak regularization**: ls_balance_lambda=0.003 too small to prevent directional drift over 500+ bars

### User Insight
> "Same directional collapse to long under two different entropy regimes"

- Seed 777: High entropy (1.14) but still 98% long
- Seed 42: Low entropy (0.58) but still 98% long
- **Conclusion**: Collapse independent of exploration level → systemic regularization failure

---

## Fix Pack D2 Strategy

### Four-Pillar Approach

**D2.A: Parameter Tuning (5 changes)**
- `ls_balance_lambda`: 0.003 → **0.006** (2x stronger L/S regularization)
- `hold_balance_lambda`: **0.002** (NEW - penalize >82% HOLD parking)
- `cooldown_bars`: 11 → **12** (slower re-entries to prevent rapid directional stacking)
- `hold_tie_tau`: 0.038 → **0.030** (-21%, reduce HOLD bias in training)
- `eval_tie_tau`: 0.05 → **0.035** (-30%, tighten tie band in validation)

**D2.B: Rolling Window L/S Balance**
- **500-bar deque** tracking position state: +1 (long), 0 (flat), -1 (short)
- **Margin tolerance**: 35-65% (±15% from 50/50 balance)
- **Quadratic penalty**: `ls_lambda * max(0, |rolling_long_ratio - 0.5| - 0.15)^2`
- Applied after 50 bars minimum (allows initial exploration)

**D2.C: Hold-Rate Guardrail**
- **Upper limit**: 82% (0.82)
- **Quadratic penalty**: `hold_lambda * max(0, hold_share - 0.82)^2`
- Computed from same 500-bar rolling window
- Applied after 50 bars minimum

**D2.D: Adaptive Epsilon "Unstick"** (DEFERRED)
- Not implemented in this iteration
- Can add later if D2.A-C insufficient

---

## Implementation Details

### config.py Changes (Lines 56-82)

```python
# PHASE-2.8d Fix Pack D2 parameters
ls_balance_lambda: float = 0.006  # 2x stronger L/S regularizer
hold_balance_lambda: float = 0.002  # NEW: Hold-rate guardrail
cooldown_bars: int = 12  # Slower re-entries
hold_tie_tau: float = 0.030  # Reduce HOLD bias
eval_tie_tau: float = 0.035  # Tighten tie band
```

### environment.py Changes

**1. Constructor Signature (Line 135-137)**
```python
entropy_beta: float = 0.014,
ls_balance_lambda: float = 0.006,  # Updated default
hold_balance_lambda: float = 0.002  # NEW parameter
```

**2. Instance Variables (Line 186-189)**
```python
self.entropy_beta = entropy_beta
self.ls_balance_lambda = ls_balance_lambda
self.hold_balance_lambda = hold_balance_lambda  # NEW
```

**3. Rolling Window Initialization (Line 250 in reset())**
```python
self.dir_window = deque(maxlen=500)
```

**4. Rolling Window Logic (Lines 510-535 in step())**

**Old implementation** (Episode-level, weak):
```python
# Count trades by direction at episode end
if self.trades_long > 0 or self.trades_short > 0:
    ratio = self.trades_long / (self.trades_long + self.trades_short)
    imbalance = abs(ratio - 0.5)
    ls_penalty = self.ls_balance_lambda * (imbalance ** 2)
    reward -= ls_penalty
```

**New implementation** (Rolling window, strong):
```python
# Update 500-bar rolling window
state_val = 0  # flat/hold
if self.position == 1:
    state_val = 1  # long
elif self.position == -1:
    state_val = -1  # short
self.dir_window.append(state_val)

# Rolling L/S balance (D2.B)
if len(self.dir_window) >= 50:
    long_bars = sum(1 for s in self.dir_window if s == 1)
    short_bars = sum(1 for s in self.dir_window if s == -1)
    if long_bars + short_bars > 0:
        rolling_long_ratio = long_bars / (long_bars + short_bars)
        margin = 0.15  # Allow 35-65% range
        imbalance = max(0, abs(rolling_long_ratio - 0.5) - margin)
        ls_penalty = self.ls_balance_lambda * (imbalance ** 2)
        reward -= ls_penalty

# Hold-rate guardrail (D2.C)
if len(self.dir_window) >= 50:
    hold_share = sum(1 for s in self.dir_window if s == 0) / len(self.dir_window)
    upper_limit = 0.82
    if hold_share > upper_limit:
        excess = hold_share - upper_limit
        hold_penalty = self.hold_balance_lambda * (excess ** 2)
        reward -= hold_penalty
```

### main.py Changes (Lines 243-247)

Added reward shaping parameters to `env_kwargs`:
```python
fx_lookup=fx_lookup,  # Dynamic pip value conversion
# PHASE-2.8d Fix Pack D2: Reward shaping parameters
entropy_beta=config.environment.entropy_beta,
ls_balance_lambda=config.environment.ls_balance_lambda,
hold_balance_lambda=config.environment.hold_balance_lambda
```

---

## Behavioral Targets (Fix Pack D2 Success Criteria)

| Metric | Target Range | Rationale |
|--------|--------------|-----------|
| **Entropy** | 0.95–1.05 bits | Balanced exploration (not too random, not too deterministic) |
| **Hold rate** | 0.65–0.78 | Healthy passivity (not >0.90 parking) |
| **Long ratio** | 0.35–0.65 | **Critical**: No directional collapse (not 0.98!) |
| **Switch rate** | 0.14–0.20 | Reasonable flip frequency |
| **Trades/episode** | 24–32 | Moderate activity |
| **Penalty rate** | ≤10% | Not overtrade |
| **SPR** | ≥+0.03 | Positive but NOT from directional bias |

---

## Validation Plan

### Current State
- **Seed 42 run** (Episode 52/80): Still running with **OLD Fix Pack D1** parameters
- Will continue showing directional collapse (98% long)

### Decision Point: Stop or Continue?

**Option A: STOP current run (Recommended)**
```powershell
# Stop current terminal (Ctrl+C)
# Backup old checkpoints
Move-Item checkpoints\*.pt checkpoints_backup_fixpack_d1\

# Start fresh 80-episode run with Fix Pack D2
python main.py --episodes 80 --seed 42
```

**Pros**: Don't waste ~2 hours on known-bad parameters  
**Cons**: Lose Episode 52-80 baseline data

**Option B: LET FINISH to Episode 80**
```powershell
# Wait for completion (~2 hours)
# Then backup and restart
Move-Item logs\validation_summaries logs_backup_seed42_fixpack_d1\
Move-Item checkpoints\*.pt checkpoints_backup_fixpack_d1\

python main.py --episodes 80 --seed 42
```

**Pros**: Complete baseline for comparison  
**Cons**: Wastes time on parameters we know will fail

### Checkpoint Monitoring (Every 10 Episodes)

```powershell
# Episode 10 check
$val = Get-Content logs\validation_summaries\val_ep010.json | ConvertFrom-Json
Write-Host "Ep 10: SPR=$([math]::Round($val.score,3)) trades=$($val.trades) entropy=$([math]::Round($val.action_entropy_bits,2)) hold=$([math]::Round($val.hold_rate,2)) long_ratio=$([math]::Round($val.long_short.long_ratio,2))"

# Expected: long_ratio 0.35-0.65 (NOT 0.98!)
```

**Early success signal (Episode 10-20)**:
- ✅ long_ratio 0.40-0.60 → D2 working
- ⚠️ long_ratio >0.75 or <0.25 → Need stronger penalties
- ❌ long_ratio >0.90 or <0.10 → Immediate stop, increase lambdas

### Cross-Validation (If Episode 80 successful)
1. Run **seed 123** (80 episodes)
2. Run **seed 777** (80 episodes)
3. Check all 3 seeds maintain long_ratio 0.35-0.65
4. If ≥2/3 pass → Proceed to full confirmation (5 seeds × 150 episodes)

---

## Escalation Plan

### If Episodes 10-20 Show Continued Collapse

**Level 1: Stronger Penalties**
```python
# config.py
ls_balance_lambda: 0.010  # +67% increase
hold_balance_lambda: 0.004  # 2x increase
```

**Level 2: Tighter Margin**
```python
# environment.py (line ~520)
margin = 0.10  # Tighten to 45-55% range (from 35-65%)
```

**Level 3: Episodic Penalty**
```python
# environment.py (in episode_done())
if episode_long_ratio > 0.70 or episode_long_ratio < 0.30:
    final_reward -= 5.0  # Large episodic penalty
```

### If Fix Pack D2 Completely Fails

**Fallback Plan**:
1. Revert to Fix Pack D1 (accept some directional bias)
2. Analyze training data regime (is LONG actually optimal?)
3. Consider multi-objective optimization (SPR + balance)
4. Investigate data artifacts (systematic bias in features?)

---

## Technical Notes

### Why Rolling Window Over Episode-Level?

**Episode-level problem** (old approach):
- Agent can park in 98% long for 450 bars, then take 2 short trades at end
- Episode metrics: long_ratio=0.94 (still collapsed) but episode-level penalty minimal
- No continuous feedback signal to prevent drift

**Rolling window solution** (new approach):
- Every step checks last 500 bars
- Penalty grows quadratically as directional drift increases
- Immediate feedback prevents "direction parking" strategy
- Margin tolerance (35-65%) allows natural variation without over-constraining

### Why Quadratic Penalties?

**Linear penalty**: `reward -= lambda * imbalance`
- Problem: Small imbalances (<0.10) barely penalized
- Agent can "tolerate" 60-40 splits indefinitely

**Quadratic penalty**: `reward -= lambda * (imbalance^2)`
- Small deviations (0.05) get light penalty: 0.006 * 0.0025 = 0.000015
- Large deviations (0.30) get heavy penalty: 0.006 * 0.09 = 0.00054
- Exponential growth prevents extreme imbalances

### Why 500-Bar Window?

- **Training episodes**: ~500-600 bars typical
- **500 bars** = ~1 full episode of context
- Too small (100 bars) → Noisy signal, agent exploits local patterns
- Too large (1000 bars) → Stale signal, slow response to regime shifts
- Sweet spot for intra-episode behavioral control

### Why 82% Hold Limit?

- **Historical data** (10-episode smoke test): hold_rate=0.76 (76%)
- **Target range**: 0.65-0.78 (65-78%)
- **Upper limit**: 0.82 = Target + 0.04 buffer
- Above 82% = "parking" behavior, not healthy passivity

---

## Code Verification Checklist

- [x] `config.py`: 5 parameter changes applied
- [x] `environment.py`: Constructor signature updated (line 135-137)
- [x] `environment.py`: Instance variables stored (line 186-189)
- [x] `environment.py`: `dir_window` deque initialized in reset() (line 250)
- [x] `environment.py`: Rolling L/S balance implemented (line 510-535)
- [x] `environment.py`: Hold-rate guardrail implemented (line 510-535)
- [x] `main.py`: `env_kwargs` includes reward shaping parameters (line 243-247)
- [x] No syntax errors (all files compile)

---

## Expected Outcomes

### If Fix Pack D2 Works (Episodes 10-80)

**Behavioral improvements**:
- ✅ long_ratio 0.35-0.65 throughout (no collapse)
- ✅ hold_rate 0.65-0.78 (healthy passivity)
- ✅ entropy 0.95-1.05 (balanced exploration)
- ✅ SPR > 0 but NOT from directional parking

**Cross-validation (seeds 42, 123, 777)**:
- ≥2/3 seeds pass behavioral targets
- Consistent metrics across seeds
- No outlier collapse events

**Next step**: Full confirmation run (5 seeds × 150 episodes)

### If Fix Pack D2 Partially Works

**Scenario**: long_ratio 0.60-0.80 (better but not ideal)
- Increase `ls_balance_lambda` from 0.006 → 0.008
- Tighten margin from 0.15 → 0.12 (38-62% range)
- Continue iterating

### If Fix Pack D2 Fails

**Scenario**: long_ratio >0.90 persists
- Investigate data regime (is LONG optimal in training data?)
- Check for feature engineering bias (e.g., trend features favor long?)
- Consider episodic penalties
- May need multi-objective optimization

---

## Comparison: Fix Pack D1 vs D2

| Component | Fix Pack D1 | Fix Pack D2 | Change |
|-----------|-------------|-------------|--------|
| **L/S regularizer** | 0.003 (episode-level) | 0.006 (rolling window) | 2x stronger + continuous |
| **Hold guardrail** | None | 0.002 (>82% limit) | NEW penalty |
| **Cooldown bars** | 11 | 12 | +9% slower re-entries |
| **hold_tie_tau** | 0.038 | 0.030 | -21% reduce HOLD bias |
| **eval_tie_tau** | 0.05 | 0.035 | -30% tighten tie band |
| **Rolling window** | ❌ None | ✅ 500-bar deque | Continuous feedback |
| **Penalty type** | Linear (weak) | Quadratic (strong) | Exponential growth |

**Key insight**: D1 checked episode-level trade counts (lagging indicator), D2 checks rolling bar-level position state (leading indicator).

---

## Next Actions

1. **User Decision**: Stop current seed 42 run OR let finish to Episode 80
2. **Start validation run**: 80 episodes with Fix Pack D2
3. **Monitor Episode 10/20/30**: Check long_ratio staying 0.35-0.65
4. **If successful**: Run seeds 123, 777 for cross-validation
5. **If collapsed**: Escalate to stronger penalties (see Escalation Plan)

---

## Success Declaration

**Fix Pack D2 GREEN** if:
- ✅ Seed 42: long_ratio 0.35-0.65 for Episodes 10-80
- ✅ Seeds 123, 777: Same behavioral targets met
- ✅ ≥2/3 seeds show no directional collapse
- ✅ SPR > 0 across all seeds
- ✅ Entropy/hold/switch rates within target ranges

**Then proceed to**: Phase 2.8d Full Confirmation (5 seeds × 150 episodes)

---

**Implementation Status**: ✅ COMPLETE  
**Validation Status**: ⏳ PENDING (awaiting fresh 80-episode run)  
**Estimated Validation Time**: ~4 hours (80 episodes × 3 minutes/episode)
