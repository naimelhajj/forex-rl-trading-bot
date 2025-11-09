# Phase 2.8f: Per-Step Dual-Variable Controller

## Overview

Phase 2.8f replaces the **delayed, coarse-grained soft bias** (Phase 2.8e) with a **per-step, dead-zone, dual-variable controller** that prevents directional collapse without creating oscillations or momentum traps.

## Root Cause of Phase 2.8e Failure

Phase 2.8e exhibited:
- **Long ratio oscillations**: 0.23 → 0.75 → varied (std=0.251, range=0.768)
- **Entropy collapses**: Episodes 10, 15 had H=0.42–0.45 (target: 0.95–1.10)
- **Hold rate spikes**: 93%+ holding in some episodes (target: 65–80%)
- **Delayed, under-damped control**: Bias applied every 10 steps → error accumulates → overcorrection

**Diagnosis**: Fixed β=0.08 applied every 10 steps = **proportional controller without dead-zone** → waits, drifts, then overcorrects → oscillation.

## Phase 2.8f Solution

### 1. **Per-Step Control** (Not Every N Steps)
- Controller updates **every single step**
- Eliminates delay that causes error accumulation
- Smooth, continuous steering instead of discrete jolts

### 2. **Dead-Zone Hysteresis**
- **Long ratio**: Center=0.50, Band=±0.10 → Acceptable: [0.40, 0.60]
- **Hold rate**: Center=0.72, Band=±0.07 → Acceptable: [0.65, 0.79]
- **Inside band**: Controller does nothing (prevents chatter)
- **Outside band**: Compute signed error and update dual variable

### 3. **Dual Variables (Lagrange Multipliers)**

Two controller states with proportional-integral behavior:

#### λ_long (Long/Short Balance)
```
e_long = deadzone_error(p_long, center=0.50, band=0.10)
λ_long ← clip(γ·λ_long + K_long·e_long, -λ_max, λ_max)

Apply to Q-values:
  Q[LONG] -= λ_long
  Q[SHORT] += λ_long
```

#### λ_hold (Hold Rate Balance)
```
e_hold = deadzone_error(p_hold, center=0.72, band=0.07)
λ_hold ← clip(γ·λ_hold + K_hold·e_hold, -λ_max, λ_max)

Apply to Q-values:
  Q[HOLD] -= λ_hold
```

**Parameters**:
- K_long = 0.8 (proportional gain for directional balance)
- K_hold = 0.6 (proportional gain for hold balance)
- γ = 0.995 (leak factor to prevent wind-up)
- λ_max = 1.2 (saturation limit)

### 4. **Entropy Governor (Temperature Scaling)**

Prevents policy collapse by adjusting temperature τ:

```
H = -Σ p_i log₂(p_i)  # entropy in bits

If H < H_min (0.95):
    τ ← min(1.5, τ × 1.05)  # increase temperature → more exploration
    
If H > H_max (1.10):
    τ ← max(0.8, τ × 0.95)  # decrease temperature → sharper policy

Q-values ← Q-values / τ
```

### 5. **Anti-Stickiness Nudge**

Prevents run-length collapse:
```
If run_length > 80:
    Q[last_action] -= 0.05
    Q[opposite_action] += 0.05
```

### 6. **EWMA Proportions (W=64 steps)**

Smooth tracking of action frequencies:
```
α = 1/64
p_long ← (1-α)·p_long + α·1{action=LONG}
p_hold ← (1-α)·p_hold + α·1{action=HOLD}
```

## Implementation Location

**Agent-side (not environment-side)**:
- `agent.py`: Controller state, methods, and integration in `select_action()`
- Applied to Q-values **before argmax** (logit-level nudges)
- **No reward shaping** - clean learning signal preserved

## Key Differences from Phase 2.8e

| Aspect | Phase 2.8e (Failed) | Phase 2.8f (New) |
|--------|---------------------|------------------|
| **Update frequency** | Every 10 steps | Every step |
| **Controller type** | Fixed bias (β=0.08) | Dual-variable PI with leak |
| **Dead-zone** | None (always applies bias) | ±0.10 (long), ±0.07 (hold) |
| **Entropy control** | None | Temperature governor (τ ∈ [0.8, 1.5]) |
| **Anti-stickiness** | Circuit-breaker (hard mask) | Soft nudge at run_len>80 |
| **Location** | Environment (`get_action_bias()`) | Agent (`_apply_controller()`) |
| **Oscillation risk** | High (no hysteresis) | Low (dead-zone prevents chatter) |

## Tuning Knobs (Start Values)

```python
# EWMA window
ALPHA = 1/64  # 64-step window

# Dead-zone bands
LONG_CENTER = 0.50, LONG_BAND = 0.10  # [0.40, 0.60]
HOLD_CENTER = 0.72, HOLD_BAND = 0.07  # [0.65, 0.79]

# Controller gains
K_LONG = 0.8   # directional balance gain
K_HOLD = 0.6   # hold balance gain
LAMBDA_MAX = 1.2
LAMBDA_LEAK = 0.995

# Entropy governor
H_MIN = 0.95, H_MAX = 1.10
TAU_MIN = 0.8, TAU_MAX = 1.5

# Anti-stickiness
RUNLEN_CAP = 80
```

## Expected Results (20-Episode Smoke Test)

**Gate Targets**:
1. Long ratio: Mean 0.45–0.55, ≥70% episodes in [0.40, 0.60]
2. Hold rate: Mean 0.68–0.76, ≥70% episodes in [0.65, 0.79]
3. Entropy: Mean 0.95–1.10, no catastrophic collapses (<0.50)
4. Switch rate: 0.15–0.19
5. **Volatility**: std(long_ratio) < 0.15, range < 0.40 (no oscillation)

## Troubleshooting

### If long_ratio still swings:
1. Widen dead-zone: `LONG_BAND = 0.12`
2. Reduce gain: `K_LONG = 0.6`
3. Increase leak: `LAMBDA_LEAK = 0.99`

### If entropy still dips:
1. Raise τ_max: `TAU_MAX = 1.7`
2. Increase response: Change `τ × 1.05` → `τ × 1.07` when H < H_min

### If hold rate drifts:
1. Tighten band: `HOLD_BAND = 0.05`
2. Increase gain: `K_HOLD = 0.8`

## Validation Protocol

1. **20-episode smoke test** (seed 42)
   - Check long_ratio, hold_rate, entropy at Episodes 10, 15, 20
   - Verify no oscillations (std < 0.15)
   
2. **If passed**: Run 3-seed × 80-episode validation
   - Seeds: 42, 123, 777
   - All gates must pass ≥70% of episodes

3. **Compare to Phase 2.8e**:
   - Long ratio std: 0.251 → <0.12 (target)
   - Entropy collapses: Yes (0.42) → None (target)
   - Oscillation: Yes (0.23→0.75) → No (target)

## Why This Works

1. **Per-step updates** → No delay, no error accumulation
2. **Dead-zone** → No chatter, controller silent when in-range
3. **Dual variables with leak** → Integral behavior without wind-up
4. **Entropy governor** → Prevents collapse via temperature
5. **EWMA smoothing** → Ignores transient spikes
6. **Logit-level nudges** → Preserves clean learning signal

**Bottom line**: This is a **stable, continuous controller** that addresses the exact failure modes (oscillation, entropy collapse, delayed response) observed in Phase 2.8e.
