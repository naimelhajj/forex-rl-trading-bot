# Diversity Boost - Quick Reference

**Goal:** Increase decision diversity without churn  
**Status:** ✅ Ready for testing

---

## Changes Made (5 total)

### Config Adjustments
```python
# AgentConfig
eval_epsilon:       0.03 → 0.05  (+67% probe rate)
eval_tie_tau:       0.03 → 0.05  (+67% tie threshold)
hold_tie_tau:       0.02 → 0.04  (+100% probe margin)
hold_break_after:   12 → 10      (-17% streak threshold)

# EnvironmentConfig  
flip_penalty:       0.0007 → 0.0005  (-29% flip cost)
```

### New Feature
- ✅ Hold-streak breaker added to **training** (was only in validation)

---

## Target Improvements

| Metric | Before | After (Target) |
|--------|--------|----------------|
| Hold rate | 76.2% | 55-70% |
| Entropy | 0.324 bits | ≥0.9 bits |
| Switch rate | 5.7% | 10-20% |
| Max hold streak | 230 bars | < 60 bars |
| Zero-trade rate | ~5% | ≤5% (maintain) |

---

## Quick Test

```powershell
# Run 10 episodes
python main.py --episodes 10

# Check metrics
python check_metrics_addon.py
```

**Look for:**
- Entropy > 0.5 bits (up from 0.32)
- Switch rate > 8% (up from 5.7%)
- Hold rate < 75% (down from 76%)
- Max streak < 100 bars (down from 230)

---

## If Too Aggressive

**Dial back exploration:**
```python
eval_epsilon: 0.05 → 0.04
eval_tie_tau: 0.05 → 0.04
```

**Or restore flip penalty:**
```python
flip_penalty: 0.0005 → 0.0006
```

---

## Status

✅ All patches applied  
✅ Code compiles cleanly  
✅ Low-risk incremental changes  
✅ Easy to revert if needed  
🎯 Ready to test!
