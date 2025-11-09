# Phase 2.8f Confirmation Protocol

## Overview
Production-grade validation protocol for Phase 2.8f dual-variable PI controller before promotion to `v2.8f-confirmed`.

---

## 1. Confirmation Run Specification

### Design
- **Seeds**: 5 independent runs (42, 123, 456, 789, 1011)
- **Episodes per seed**: 200
- **Friction jitter**: ON (uniform distribution as used in smoke test)
- **Total training time**: ~10-12 hours (5 seeds × 2-2.5 hrs/seed)

### Acceptance Gates (Cross-Seed)

#### Performance Gates
1. **Mean Sharpe ≥ +0.04** with **σ(mean) ≤ 0.035**
2. **Trail-5 median ≥ +0.25** (median of last 5 episode Sharpes)
3. **≥3/5 seeds with positive mean Sharpe**

#### Behavioral In-Band Episode Share
4. **Long ratio [0.40, 0.60]**: ≥70% of episodes
5. **Hold rate [0.65, 0.79]**: ≥70% of episodes  
6. **Entropy [0.95, 1.10]**: ≥80% of episodes
7. **Switch rate [0.15, 0.19]**: ≥70% of episodes

#### Stability Gate
8. **Penalty/failsafe rate**: ≤10% of episodes

---

## 2. Telemetry Per Episode

**Required logging** (append to episode metrics):
```python
{
    'p_long_smoothed': float,      # EWMA long ratio
    'p_hold_smoothed': float,      # EWMA hold rate
    'lambda_long': float,          # PI controller variable for L/S
    'lambda_hold': float,          # PI controller variable for hold
    'tau': float,                  # Entropy temperature
    'H_bits': float,               # Policy entropy (bits)
    'run_len_max': int,            # Max consecutive same actions
    'trades': int,                 # Total trades
    'SPR': float,                  # Sharpe ratio
    'switch_rate': float,          # Action switches / total steps
}
```

---

## 3. Guardrails (Mandatory)

### Controller Inference-Only
- **No gradients through controller**: Use `detach()` or `stop_gradient()` on adjusted Q-values
- Controller operates purely at action selection, not in loss computation

### Episode Reset Protocol
```python
# At start of each episode:
lambda_long = 0.0
lambda_hold = 0.0
tau = 1.0
run_len = 0
ewma_long = 0.5  # neutral
ewma_hold = 0.72  # center of [0.65, 0.79]
```

### Hard Limits
- **λ bounds**: `|lambda_long| ≤ 1.2`, `|lambda_hold| ≤ 1.2`
- **τ bounds**: `[0.8, 1.5]` (widen to [0.8, 1.7] only if entropy still dips)

---

## 4. Tuning Ladder (Single Knob Changes)

### A. Long Ratio Oscillates/Drifts
**Symptoms**: std(long_ratio) > 0.15 or episodes outside [0.40, 0.60] > 30%

**Fixes** (try in order):
1. Widen dead-zone: `LONG_BAND = 0.12` (was 0.10)
2. Reduce gain: `K_LONG = 0.6` (was 0.8)
3. Increase leak: `LEAK = 0.997` (was 0.95)

### B. Hold Rate Too High
**Symptoms**: hold_rate > 0.79 in >30% of episodes

**Fixes** (try in order):
1. Increase gain: `K_HOLD = 0.7` (was 0.6)
2. Keep band: `HOLD_BAND = 0.07`
3. Anti-stickiness: After 40 consecutive HOLDs, `logits[HOLD] -= 0.03`

### C. Entropy Dips
**Symptoms**: H < 0.95 in >20% of episodes

**Fixes** (try in order):
1. Faster correction: `tau *= 1.07` (was 1.05) when below H_MIN
2. Raise ceiling: `TAU_MAX = 1.7` (was 1.5)

### D. High Cross-Seed Variance
**Symptoms**: σ(mean_sharpe) > 0.035

**Fixes** (try in order):
1. Smoother EWMA: `W = 96` (was 64)
2. Reduce both gains: `K_LONG = 0.6, K_HOLD = 0.5`

---

## 5. Fix Pack D3 Status

**Assessment**: Hard masks + episodic penalties will re-introduce:
- Boundary surfing (agent gaming 70/30 limits)
- Delayed corrections (episodic penalties lag real-time issues)

**Recommendation**: 
- **Primary**: Keep Phase 2.8f PI controller as main stabilizer
- **Failsafe only**: Emergency clamp if `p_long_smoothed` leaves [0.05, 0.95] for >W steps
  - Duration: 1-3 steps with exponential decay
  - Strength: Temporary mask, not reward penalty

---

## 6. Stress Tests (Post-Confirmation)

### Friction Robustness
- **Spiky mixture**: 90% baseline spread, 10% spikes at 1.5-2× spread
- **Expected**: Behavior stays in-band, SPR drops <20%

### Latency Shock
- **Random delays**: 1-3 bar delays on 10% of actions
- **Expected**: Controller remains stable, no runaway

### Regime Shift OOS
- **Holdout windows**: +2 weeks, +6 weeks from training
- **Gates**: SPR ≥ 0, trail-5 ≥ +0.15

### Instrument Generalization
- **Quick test**: 80 episodes on GBPUSD or USDJPY
- **Gates**: Behavior in-band (even if SPR modest)

### Ablation Studies
Test removal of each component:
1. No entropy governor (remove temperature scaling)
2. No hold controller (remove λ_hold)
3. No long/short controller (remove λ_long)

**Expected**: Full model outperforms all ablations by ≥15% on mean SPR

---

## 7. Code Hardening (Unit Tests)

### Dead-Zone Function Test
```python
def test_deadzone():
    # Inside band: error = 0
    assert deadzone_err(0.50, 0.50, 0.10) == 0.0
    assert deadzone_err(0.45, 0.50, 0.10) == 0.0
    
    # Outside band: linear, sign-correct
    assert deadzone_err(0.65, 0.50, 0.10) > 0  # Above band
    assert deadzone_err(0.35, 0.50, 0.10) < 0  # Below band
```

### LSE Softmax Test
```python
def test_lse_softmax():
    # Extreme logits
    q_extreme = np.array([-100.0, 100.0, 50.0, -50.0])
    for tau in [0.8, 1.0, 1.5]:
        probs = lse_softmax(q_extreme, tau)
        assert np.allclose(probs.sum(), 1.0)
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))
```

### Controller Monotonicity Test
```python
def test_controller_monotonic():
    # Fixed action stream drives EWMA above band
    # Expect lambda to increase until clipped
    controller = DualController()
    lambdas = []
    for _ in range(100):
        controller.update(long_ratio=0.8)  # Above band
        lambdas.append(controller.lambda_long)
    
    # Should increase monotonically (or plateau at clip)
    assert all(lambdas[i] <= lambdas[i+1] for i in range(len(lambdas)-1))
```

### Provenance Logging
```python
# In validation summary:
{
    'config_hash': hash(json.dumps(config, sort_keys=True)),
    'controller_params': {
        'W': 64, 'K_LONG': 0.8, 'K_HOLD': 0.6,
        'LONG_BAND': 0.10, 'HOLD_BAND': 0.07,
        'TAU_MIN': 0.8, 'TAU_MAX': 1.5,
        'H_MIN': 0.95, 'H_MAX': 1.10,
        'LEAK': 0.95
    },
    'version': 'v2.8f',
    'timestamp': datetime.now().isoformat()
}
```

---

## 8. Deployment Checklist (Paper Trading)

### Kill-Switches
- **Daily MDD limit**: 3× backtest MDD (auto-flatten all positions)
- **Per-hour trade limit**: ≤20 trades/hour
- **Consecutive loss cooldown**: After 5 losses, pause 30 minutes

### Live Monitoring
**Dashboard metrics** (update every 5 minutes):
- `lambda_long`, `lambda_hold` (color-coded if near clip)
- `tau`, `H_bits` (alarm if outside [0.95, 1.10])
- `p_long_smoothed`, `p_hold_smoothed` (alarm if outside bands >10 mins)
- Realized vs. simulated PnL divergence

**Alarms**:
- λ clipped for >5 consecutive minutes
- Entropy outside [0.95, 1.10] for >3 minutes
- Behavior outside bands for >10 minutes
- Realized slippage >1.5× backtest assumption

### Shadow Mode (1 Week)
- Run Phase 2.8f alongside baseline (Phase 2.8e or buy-and-hold)
- Compare:
  - Realized fills vs. simulated fills
  - Actual spread/commission vs. backtest assumptions
  - Trade frequency stability
- **Gate**: <10% degradation from backtest metrics before going live

---

## 9. Promotion Workflow

### Step 1: Run Confirmation Suite
```bash
python run_confirmation_suite.py --seeds 42,123,456,789,1011 --episodes 200
```

### Step 2: Analyze Results
```bash
python analyze_confirmation_results.py --output confirmation_report.md
```

### Step 3: Gate Check
- Review all 8 acceptance gates
- If **all pass**: Proceed to Step 4
- If **any fail**: Apply tuning ladder (§4), retest 1 seed × 200 eps

### Step 4: Tag and Document
```bash
git add agent.py PHASE_2_8F*.md test_dual_controller.py
git commit -m "Phase 2.8f: Passed 5-seed × 200-ep confirmation

Results:
- Mean SPR: {value} ± {std}
- Trail-5 median: {value}
- Behavioral in-band: L={pct}%, H={pct}%, E={pct}%
- Seeds positive: {count}/5

All 8 acceptance gates passed.
Ready for stress tests."

git tag -a v2.8f-confirmed -m "Phase 2.8f: Production-ready after confirmation suite"
git push origin main --tags
```

### Step 5: Stress Tests
Run tests from §6, document in `PHASE_2_8F_STRESS_TEST_RESULTS.md`

### Step 6: Paper Trading
Follow §8 deployment checklist for 1 week

---

## 10. Success Criteria Summary

| Gate | Metric | Threshold | Purpose |
|------|--------|-----------|---------|
| 1 | Mean SPR | ≥ +0.04 | Profitable on average |
| 2 | σ(mean SPR) | ≤ 0.035 | Consistent across seeds |
| 3 | Trail-5 median | ≥ +0.25 | Strong recent performance |
| 4 | Positive seeds | ≥ 3/5 | Majority profitable |
| 5 | Long ratio in-band | ≥ 70% | Balanced L/S |
| 6 | Hold rate in-band | ≥ 70% | Healthy activity |
| 7 | Entropy in-band | ≥ 80% | Policy diversity |
| 8 | Switch rate in-band | ≥ 70% | Reasonable churn |
| 9 | Penalty rate | ≤ 10% | Rare failsafes |

**All 9 gates must pass for promotion to `v2.8f-confirmed`**

---

*Protocol version: 1.0*  
*Created: November 9, 2025*  
*Next review: After confirmation run completion*
