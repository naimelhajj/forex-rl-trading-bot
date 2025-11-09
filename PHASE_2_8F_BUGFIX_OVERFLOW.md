# Phase 2.8f Overflow Bugfix

## Issue

During initial Phase 2.8f smoke test run, encountered many `RuntimeWarning: overflow encountered in exp` errors in the entropy governor:

```
C:\Development\forex_rl_bot\agent.py:470: RuntimeWarning: overflow encountered in exp
  probs = np.exp(q / self.tau)
C:\Development\forex_rl_bot\agent.py:471: RuntimeWarning: invalid value encountered in divide
  probs = probs / (probs.sum() + 1e-12)
```

## Root Cause

The entropy governor computes softmax probabilities from Q-values to estimate policy entropy. When Q-values are very large (common with untrained networks during prefill), `np.exp(q / self.tau)` overflows to infinity, causing `NaN` values in the probability distribution.

##Fix Applied

**Before (agent.py lines 468-472):**
```python
# Compute entropy from Q-values (approximate policy entropy)
probs = np.exp(q / self.tau)
probs = probs / (probs.sum() + 1e-12)
H = -np.sum(probs * np.log2(probs.clip(1e-12, None)))  # entropy in bits
```

**After (agent.py lines 468-475):**
```python
# Compute entropy from Q-values (approximate policy entropy)
# Use log-sum-exp trick to prevent overflow
q_scaled = q / self.tau
q_max = np.max(q_scaled)
exp_q = np.exp(q_scaled - q_max)
probs = exp_q / (exp_q.sum() + 1e-12)
H = -np.sum(probs * np.log2(probs.clip(1e-12, None)))  # entropy in bits
```

## Explanation

The **log-sum-exp trick** prevents overflow by:
1. Finding the maximum scaled Q-value: `q_max = np.max(q / τ)`
2. Subtracting it before exponentiation: `exp(q/τ - q_max)`
3. This is mathematically equivalent to the original softmax, but numerically stable

**Why it works:**
```
softmax(x) = exp(x_i) / Σ exp(x_j)
           = exp(x_i - c) / Σ exp(x_j - c)  [for any constant c]
           = exp(x_i - max(x)) / Σ exp(x_j - max(x))  [choose c = max(x)]
```

Now the largest exponent is `exp(0) = 1`, preventing overflow while preserving relative probabilities.

## Impact

- **No functional change**: Controller logic remains identical
- **Numerical stability**: Prevents overflow warnings and `NaN` probabilities
- **Early training robustness**: Works correctly even when untrained network outputs extreme Q-values

## Status

- ✅ Fix applied to `agent.py`
- ⏳ Waiting for current 20-episode smoke test to complete (started before fix)
- Next: Restart smoke test with fixed code

## Testing

The unit test (`test_dual_controller.py`) passed before this fix because it used moderate Q-values (`[1, 15, 1, 1]`). The overflow only appears with very large Q-values from untrained networks (e.g., Q-values in the range of 100-1000).

To verify the fix, the entropy governor test in `test_dual_controller.py` should be updated to use extreme Q-values:

```python
# Test with extreme Q-values that would cause overflow without log-sum-exp
extreme_q = np.array([100.0, 1000.0, 100.0, 100.0])
# Should not overflow or produce NaN
```

---

**Commit message for this fix:**
```
Phase 2.8f: Fix overflow in entropy governor (log-sum-exp trick)

- Prevents RuntimeWarning during training with large Q-values
- Uses numerically stable softmax computation
- No functional change to controller logic
```
