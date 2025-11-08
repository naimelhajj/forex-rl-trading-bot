# Phase 2.8d Fix Pack D1 - Recovery Plan

## Status: REVERTING from Nuclear Fix → Fix Pack D1

**Context:** After 200-ep Phase 2.8c failed gates (entropy 1.18-1.22, hold 0.58-0.62, L/S collapse), we applied Nuclear Fix (epsilon=0.50) which fixed the trading blockage bug but gives us too-random policy. Now reverting to the **proper surgical fixes** from Fix Pack D1.

## Fix Pack D1 Parameter Changes

### Current (Phase 2.8d Nuclear Fix)
These are emergency parameters that fixed the trading bug but are too aggressive:
- `entropy_beta: 0.025` (too high, forces randomness)
- `hold_tie_tau: 0.030` (too low, allows overtrading)
- `flip_penalty: 0.0005` (too low, allows churn)
- `epsilon_start: 0.50` (NUCLEAR - way too high for production)
- `epsilon_end: 0.10` (too high for final policy)
- `use_noisy: False` (disabled due to eval_mode freeze bug)

### Target (Fix Pack D1 - Surgical Recovery)
Returning to stricter, production-quality parameters:

1. **Reduce exploration** (Fix D1.1)
   - `entropy_beta: 0.025 → 0.014` (-44% reduction)
   - Rationale: Cut entropy bonus to reduce too-random behavior

2. **Strengthen hold-on-ties** (Fix D1.2)
   - `hold_tie_tau: 0.030 → 0.038` (+27% increase)
   - Rationale: Bias toward HOLD when Q-values are close, reduce overtrading

3. **Discourage churn** (Fix D1.3)
   - `flip_penalty: 0.0005 → 0.00077` (+54% increase)
   - Rationale: Penalize rapid position flips more heavily

4. **Tighten expected-trades gate** (Fix D1.4)
   - `VAL_EXP_TRADES_SCALE: 0.42` (already applied, keep)
   - Rationale: Down-weight low-quality high-turnover regimes

5. **Trim luck in aggregation** (Fix D1.5)
   - `VAL_TRIM_FRACTION: 0.25` (already applied, keep)
   - Rationale: Remove top/bottom 25% to reduce outlier influence

6. **L/S balance regularizer** (Fix D1.6)
   - `ls_balance_lambda: 0.003` (already applied, keep)
   - Rationale: Prevent seed-driven directional collapse (0.065 or 0.934 long_ratio)

7. **Restore epsilon-greedy to production values**
   - `epsilon_start: 0.50 → 0.12` (back to Phase 2.8b baseline)
   - `epsilon_end: 0.10 → 0.06` (back to Phase 2.8b baseline)
   - `use_noisy: False → False` (keep disabled, epsilon-greedy works better)

## Expected Behavioral Changes

### Before (Nuclear Fix - Too Random)
- Training trades: 45-50 per episode (too high)
- Validation trades: 18-22 per episode
- Entropy: ~1.4 bits (too random)
- Hold rate: ~40-50% (too low)
- Policy: Dominated by random exploration

### After (Fix Pack D1 - Balanced)
- Training trades: 22-30 per episode (target range)
- Validation trades: 18-25 per episode
- Entropy: 0.90-1.10 bits (balanced)
- Hold rate: 0.65-0.80 (more conservative)
- Policy: Learned + measured exploration

## Ablation Study Plan (Fast Loop)

After applying Fix Pack D1, run **3 seeds × 80 episodes** to validate:

### Target Gates (Ablation)
- Mean SPR ≥ **+0.03**
- Trail-5 median ≥ **+0.20**
- Entropy: **0.90–1.10 bits**
- Hold rate: **0.65–0.80**
- Long ratio: **0.40–0.60**
- Success: ≥2/3 seeds hit targets

### If Ablation GREEN → Full Confirmation
Run **5 seeds × 150 episodes** with same parameters:
- Mean SPR ≥ **+0.04**
- Trail-5 ≥ **+0.25**
- σ(means) ≤ **0.035**
- Penalty ≤ **10%**
- ≥3/5 seeds with trail-5 > 0
- Behavioral metrics within bands

## Rollback Strategy

If Fix Pack D1 causes HOLD collapse again:
1. Check `bars_since_close` increment is still present (the critical bug)
2. Nudge `hold_tie_tau` down by -0.002 (0.038 → 0.036)
3. Increase `entropy_beta` by +10% (0.014 → 0.015)
4. Check action masking isn't blocking trades

If still collapsing after above:
- **Option A:** Increase `epsilon_start` to 0.20 (middle ground)
- **Option B:** Increase L/S regularizer `ls_balance_lambda` to 0.006
- **Option C:** Reduce `flip_penalty` by -15% (0.00077 → 0.00065)

## Files Modified
- `config.py`: Parameter changes (7 parameters)
- This document: Recovery plan and ablation strategy

## Next Commands

```powershell
# After applying Fix Pack D1 parameters
python main.py --episodes 80 --seed 42

# Monitor first 10 episodes closely
# Check logs/validation_summaries/val_ep010.json for metrics
```

## Success Criteria (Episode 10 checkpoint)
- Trades: 20-35 per episode
- Entropy: 0.85-1.15 bits
- Hold rate: 0.60-0.80
- Long/short mix: not collapsed (both > 0.15)
- No trading blockage (trades > 0)

If Episode 10 looks good → continue to Episode 80 for ablation validation.
