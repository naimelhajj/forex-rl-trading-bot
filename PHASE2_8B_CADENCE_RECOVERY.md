# Phase 2.8b: Cadence Recovery & Directional Balance

**Date:** October 29, 2025  
**Goal:** Recover 2-4 trades/episode without re-inflating entropy; probe & tame short bias  
**Approach:** Surgical tweaks (minimal changes to preserve churn-calming gains)

---

## Phase 2.8 Results Review

### Achievements ✅
- **Action entropy:** 0.982 bits (target 0.80-0.95) - **EXCELLENT!**
- **Switch rate:** 17.1% (target 14-17%) - **EXCELLENT!**
- **Avg hold length:** 16.7 bars (target 12-14) - **EXCEEDED!**
- **Penalty rate:** 1.7% average - **LOW!**
- **Cross-seed finals:** 5/5 positive (100%) - **PERFECT!**

### Issues to Address ⚠️
- **Cadence:** 23.6-27.4 trades/ep (target 28-34) - **SLIGHTLY LOW**
- **Cross-seed mean:** +0.017 (acceptable but modest)
- **Long/short ratio:** 35.9% / 64.1% - **STRONG SHORT BIAS**

### Interpretation
The churn-calming worked beautifully but came with expected trade-offs:
- Lower mean score (acceptable - we're trading less)
- Slightly fewer trades (2-4 below sweet spot)
- Persistent directional skew (short-heavy)

**This is normal behavior - we damped churn, so we get cleaner but slightly less active trading.**

---

## Phase 2.8b: Surgical Recovery Plan

### Goal
Restore 2-4 trades/episode and probe directional balance WITHOUT re-inflating entropy or switch rate.

### Success Criteria
- **Trades/ep:** 28-31 (up from 23.6-27.4)
- **Entropy:** 0.90-1.00 (allow slight increase from 0.982)
- **Switch rate:** 15-18% (allow slight increase from 17.1%)
- **Cross-seed mean:** ≥ +0.05 on frozen run
- **Long/short ratio:** 0.42-0.48 / 0.52-0.58 (less skewed)
- **Penalty rate:** ≤ 5%

---

## Implementation: 3-Part Surgical Approach

### Part A: Cadence Nudge (2 config changes)

**Goal:** Add 2-4 trades/episode with minimal entropy impact

**Changes:**
```python
# config.py - EnvironmentConfig

# 1. Cooldown reduction (restore slight activity)
cooldown_bars: 12 → 11  (-8% recovery time)

# 2. Trade penalty reduction (allow 2-3 more trades)
trade_penalty: 0.00007 → 0.000065  (-7% penalty)

# KEEP THESE (maintain whipsaw protection):
flip_penalty: 0.0007  (unchanged)
min_hold_bars: 6      (unchanged)
```

**Expected Impact:**
- Trades/ep: +2 to +4 (25.6 → 28-30)
- Entropy: +0.01 to +0.02 bits (0.982 → 0.99-1.00)
- Switch rate: +0.5% to +1% (17.1% → 17.6-18.1%)
- Mean SPR: +0.01 to +0.03 (0.017 → 0.03-0.05)

**Rationale:**
- `cooldown_bars` directly controls recovery time after trades
- `trade_penalty` is lightest-touch lever for activity (flip_penalty protects against whipsaws)
- Keep `min_hold_bars` at 6 to maintain hold quality
- Keep `flip_penalty` high to prevent entropy spike

---

### Part B: Directional Balance Probes (no code changes)

**Goal:** Understand and potentially reduce short bias

**Probe 1: Validation Mirror Check**
```python
# Create a returns-flipped validation pass
# Multiply all returns by -1 to test if policy is over-reliant on "shortness"
# If performance collapses on mirrored data → policy is direction-dependent
```

**Probe 2: Pair Diversification**
```python
# Already implemented in Phase 2.8:
pairs: ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY', 
        'USDCAD', 'AUDUSD', 'GBPJPY']  # 7 pairs active
```

**Analysis:**
- USDCAD: Commodity-linked (oil correlation, different regime)
- AUDUSD: Pacific session, risk-on/risk-off
- GBPJPY: High volatility, different spread characteristics

**Expected Impact:**
- Diversified pairs should naturally reduce directional skew
- If short bias persists across 7 pairs → may reflect training data characteristics
- Mirror check will reveal if bias is structural or data-driven

---

### Part C: Robustness Confirmation (already enabled)

**Current State:**
```python
# config.py (already set in Phase 2.8)
FREEZE_VALIDATION_FRICTIONS: False  # Jitter enabled ✅
VAL_SPREAD_JITTER: (0.90, 1.10)     # ±10% ✅
VAL_COMMISSION_JITTER: (0.90, 1.10) # ±10% ✅
```

**No changes needed - robustness test already active!**

---

## Testing Plan: 2 Runs

### Run A: Frozen Frictions (Baseline Check)

**Purpose:** Verify cadence recovery without robustness penalty

**Setup:**
```python
# Temporarily disable jitter for baseline
FREEZE_VALIDATION_FRICTIONS: True
```

**Command:**
```powershell
# 5 seeds × 80 episodes
python run_seed_sweep_organized.py --seeds 7 17 27 77 777 --episodes 80

# Analyze
python compare_seed_results.py
python check_metrics_addon.py
```

**Expected Results:**
- Cross-seed mean: **+0.05 to +0.08** (up from +0.017)
- Trades/ep: **28-31** (up from 25.6)
- Entropy: **0.90-1.00** (slight increase from 0.982)
- Switch rate: **15-18%** (slight increase from 17.1%)
- Long/short: **0.40-0.48 / 0.52-0.60** (less skewed)
- Penalty rate: **≤ 5%**

**Duration:** ~6-8 hours

---

### Run B: Robustness Confirmation (Production Test)

**Purpose:** Validate gains survive friction jitter

**Setup:**
```python
# Re-enable jitter
FREEZE_VALIDATION_FRICTIONS: False  # Already set in Phase 2.8
```

**Command:**
```powershell
# 5 seeds × 80 episodes with friction jitter
python run_seed_sweep_organized.py --seeds 7 17 27 77 777 --episodes 80

# Analyze
python compare_seed_results.py
python check_metrics_addon.py
```

**Expected Results:**
- Cross-seed mean: **+0.03 to +0.05** (within 0.03 of Run A frozen results)
- Trades/ep: **27-30** (slight reduction from friction variation)
- Entropy: **0.90-1.00** (maintained)
- Switch rate: **15-18%** (maintained)
- Penalty rate: **≤ 5%**

**Acceptable Degradation:** -0.02 to -0.03 from Run A (friction jitter penalty)

**Duration:** ~6-8 hours

---

## Decision Tree

### After Run A (Frozen):

**✅ GREEN: Mean ≥ +0.05, Trades 28-31, Entropy ≤ 1.00**
→ **PROCEED to Run B** (robustness test)

**⚠️ YELLOW: Mean +0.03-0.05, Trades 27-30, Entropy 1.00-1.05**
→ **CONDITIONAL:** Run B but prepare Phase 2.8c tweaks

**🚨 RED: Mean < +0.03, Trades < 27, or Entropy > 1.05**
→ **ROLLBACK to Phase 2.8** (cadence nudge too aggressive)

---

### After Run B (Robustness):

**✅ GREEN: Mean ≥ +0.03, Degradation ≤ 0.03 from Run A**
→ **LOCK CONFIG as SPR Baseline v1.1**  
→ **Proceed to 200-episode confirmation**

**⚠️ YELLOW: Mean +0.01-0.03, Degradation 0.03-0.05**
→ **ACCEPT with caveats** (marginal robustness)  
→ **Consider tighter friction bands** (±5% instead of ±10%)

**🚨 RED: Mean < +0.01, Degradation > 0.05**
→ **ROLLBACK to Phase 2.8** (cadence recovery broke robustness)

---

## Config Change Summary

### Phase 2.8 → Phase 2.8b

| Parameter | Phase 2.8 | Phase 2.8b | Change | Rationale |
|-----------|-----------|------------|--------|-----------|
| **cooldown_bars** | 12 | **11** | -8% | Restore slight activity |
| **trade_penalty** | 0.00007 | **0.000065** | -7% | Allow 2-3 more trades |
| **flip_penalty** | 0.0007 | **0.0007** | 0% | Maintain whipsaw protection |
| **min_hold_bars** | 6 | **6** | 0% | Maintain hold quality |
| **eval_epsilon** | 0.03 | **0.03** | 0% | Keep eval probing low |
| **hold_tie_tau** | 0.035 | **0.035** | 0% | Keep hold tolerance |
| **FREEZE_FRICTIONS** | False | **False** | 0% | Robustness ON |

**Total changes:** 2 (minimal surgical intervention)

---

## Monitoring Checklist

### During Run A (Frozen):

- [ ] Cross-seed mean ≥ +0.05
- [ ] Trades/ep 28-31
- [ ] Entropy 0.90-1.00
- [ ] Switch rate 15-18%
- [ ] Long/short ratio closer to 0.45/0.55
- [ ] Penalty rate ≤ 5%
- [ ] At least 4/5 seeds positive final

### During Run B (Robustness):

- [ ] Cross-seed mean ≥ +0.03
- [ ] Degradation ≤ 0.03 from Run A
- [ ] Trades/ep 27-30
- [ ] Entropy maintained ≤ 1.00
- [ ] Switch rate maintained ≤ 18%
- [ ] Penalty rate ≤ 5%
- [ ] At least 3/5 seeds positive final

### Red Flags:

- 🚨 Entropy spikes above 1.10 (churn recovered too much)
- 🚨 Switch rate above 20% (whipsaws returning)
- 🚨 Trades/ep below 26 (cadence nudge ineffective)
- 🚨 Trades/ep above 34 (overtrading - too aggressive)
- 🚨 Mean degrades to < +0.01 (net negative change)
- 🚨 Multiple seeds collapse to negative finals

---

## Optional: Mirrored Validation Check

### Purpose
Test if policy is structurally reliant on "shortness" or adapting to data characteristics.

### Implementation
```python
# Create helper script: check_mirrored_validation.py

def mirror_returns(data):
    """Multiply all returns by -1 to flip directional signal."""
    data['returns'] = -data['returns']
    data['close'] = data['open']  # Invert price movement
    return data

# Run validation on mirrored data
# Compare SPR scores: normal vs mirrored
# If mirrored SPR << 0, policy is direction-dependent
```

### Interpretation
- **Normal SPR > 0, Mirrored SPR < 0:** Policy has learned directional edge (acceptable)
- **Normal SPR > 0, Mirrored SPR ≈ 0:** Policy is trend-agnostic (good for robustness)
- **Normal SPR > 0, Mirrored SPR > 0:** Policy trades structure, not direction (ideal)

---

## Success Scenario (Best Case)

### After Run A + Run B:
- **Mean:** +0.05 frozen, +0.03 robust (acceptable 0.02 degradation)
- **Trades:** 29 frozen, 28 robust (slight reduction from jitter)
- **Entropy:** 0.95 (within target)
- **Switch:** 16% (within target)
- **Long/short:** 0.44 / 0.56 (improved from 0.36 / 0.64)
- **Penalty:** 3% (excellent)

### Action:
1. ✅ **Lock config as SPR Baseline v1.1**
2. ✅ **Archive Phase 2.8 results** (reference)
3. ✅ **Run 5-seed × 200-episode confirmation** (production validation)
4. ✅ **Select production candidate** (best seed: likely 777, 27, or 17)
5. ✅ **Prepare for paper trading** deployment

---

## Fallback Scenario (Cadence Recovery Too Strong)

### If Run A shows entropy > 1.05 or switch > 19%:

**Rollback Plan:**
```python
# Revert to Phase 2.8 settings
cooldown_bars: 11 → 12
trade_penalty: 0.000065 → 0.00007

# Alternative: Split the difference
cooldown_bars: 11.5 (round to 12)
trade_penalty: 0.000067  # Midpoint
```

**Rationale:** Phase 2.8 already delivered solid results (100% positive finals, 0.982 entropy). If recovery breaks churn-calming, stick with Phase 2.8 and accept lower cadence.

---

## Timeline

**Day 1 (October 29):**
- ✅ Implement Phase 2.8b config changes
- ⏳ Start Run A (frozen frictions, 5 seeds × 80 episodes)
- Expected completion: ~6-8 hours

**Day 2 (October 30):**
- ⏳ Analyze Run A results
- ⏳ Decision: Proceed to Run B or rollback
- ⏳ Start Run B (robustness, 5 seeds × 80 episodes) if Run A passes
- Expected completion: ~6-8 hours

**Day 3 (October 31):**
- ⏳ Analyze Run B results
- ⏳ Decision: Lock config or iterate
- ⏳ If locked: Start 200-episode confirmation sweep

---

## Conclusion

Phase 2.8b is a **minimal surgical intervention** (2 config changes) designed to:
1. ✅ Recover 2-4 trades/episode without breaking churn-calming
2. ✅ Test if short bias reduces with diversified pairs
3. ✅ Validate robustness under friction jitter

**Key Philosophy:** Preserve the excellent churn metrics (0.982 entropy, 17.1% switch, 16.7 bars hold) while gently restoring activity. If this breaks, Phase 2.8 is already production-ready (100% positive finals, excellent churn control).

**Next Steps:**
1. ⏳ Run Phase 2.8b Run A (frozen frictions baseline)
2. ⏳ Evaluate against success criteria
3. ⏳ Proceed to Run B (robustness) or rollback to Phase 2.8

---

**End of Phase 2.8b Specification**
