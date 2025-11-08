# Phase 2.8b Run B - Robustness Test Specification

**Date:** 2025-10-30  
**Goal:** Validate Phase 2.8b config under realistic friction variation (±10% jitter)  
**Configuration:** Phase 2.8b cadence recovery (cooldown=11, trade_penalty=0.000065)  
**Frictions:** **UNFROZEN** (FREEZE_VALIDATION_FRICTIONS=False)  
**Test Scope:** 3 seeds × 80 episodes (reduced for quick validation)  

---

## Test Parameters

### Selected Seeds (Diverse Profile)
- **Seed 7:** Neutral mean (-0.014), zero final, 16% penalty rate → Baseline/stress test
- **Seed 77:** Neutral mean (-0.000), good final (+0.408), 14% penalty → Mid performer
- **Seed 777:** **Best mean** (+0.033), excellent final (+0.873), 6% penalty → Elite performer

**Rationale:** Testing worst/mid/best from Run A to assess robustness across performance spectrum

### Configuration Changes
```python
# From Run A (frozen):
FREEZE_VALIDATION_FRICTIONS: True

# To Run B (jitter):
FREEZE_VALIDATION_FRICTIONS: False  # Enable ±10% spread/commission/slippage variation
```

**No other config changes** - keeping cooldown=11, trade_penalty=0.000065 from Phase 2.8b

---

## Success Criteria

### ✅ GREEN (Pass - Proceed to 200-ep confirmation)
- **Cross-seed mean:** ≥ +0.03
- **Degradation from Run A:** ≤ 0.03 (mean drop ≤ 0.03)
- **Entropy:** 0.90-1.00 bits
- **Switch rate:** 15-18%
- **Trades/ep:** 26-31
- **Penalty rate:** ≤ 8%
- **No catastrophic collapses:** All seeds ≥ -0.10 mean

### 🟡 YELLOW (Borderline - Fine-tune and re-test)
- **Cross-seed mean:** +0.02 to +0.03
- **Degradation from Run A:** 0.03-0.05
- **Action:** Increase penalties slightly (trade_penalty +8%, cooldown +1)

### 🔴 RED (Fail - Rollback to Phase 2.8)
- **Cross-seed mean:** < +0.02
- **Degradation from Run A:** > 0.05
- **Action:** Revert to Phase 2.8 config, investigate root cause

---

## Expected Results (Based on Phase 2.8 vs Run A)

### Baseline Comparison
| Metric | Phase 2.8 (Jitter) | Run A (Frozen) | Expected Run B |
|--------|-------------------|----------------|----------------|
| **Mean** | +0.017 ± 0.040 | -0.004 ± 0.021 | **+0.01 to +0.04** |
| **Final** | +0.333 ± 0.437 | +0.451 ± 0.384 | +0.35 to +0.45 |
| **Variance** | ±0.040 | ±0.021 | **±0.025 to ±0.035** |
| **Penalty** | 1.7% | 0-16% | **2-8%** |

### Hypothesis
- **Friction jitter will increase variance** (±0.021 → ±0.03) but remain tighter than Phase 2.8
- **Mean should improve vs Run A** (frozen baseline revealed policy behavior, jitter adds realism)
- **Elite seeds (777) should maintain positive performance**
- **Stress seed (7) may show larger degradation** (already had 16% penalties)

---

## Run Command

```bash
python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 80
```

**Estimated Duration:** 4-5 hours (3 seeds × 80 episodes)

---

## Post-Run Analysis

### Immediate Checks
1. **Compare to Run A baseline:**
   ```bash
   python compare_seed_results.py
   ```
   - Check mean degradation (should be ≤ 0.03)
   - Verify variance increase reasonable (±0.025-0.035)

2. **Measure action metrics:**
   ```bash
   python check_metrics_addon.py
   ```
   - Entropy: 0.90-1.00 bits
   - Switch: 15-18%
   - Hold length: 14-17 bars
   - Long/short ratio: monitor for bias

3. **Verify stability:**
   - Check for zero-trade episodes (should be 0%)
   - Penalty rate ≤ 8%
   - No catastrophic collapses

### Decision Tree

```
Run B Complete
    ↓
Mean ≥ +0.03 AND Degradation ≤ 0.03?
    ↓
  YES → ✅ GREEN
    ↓
    Lock config as SPR Baseline v1.1
    Run 5 seeds × 200 episodes confirmation
    Select production seed (777 or 27)
    
  NO → Mean ≥ +0.02?
    ↓
  YES → 🟡 YELLOW
    ↓
    Fine-tune penalties:
      trade_penalty: 0.000065 → 0.000070
      cooldown_bars: 11 → 12
    Re-run 3 seeds × 80 eps
    
  NO → 🔴 RED
    ↓
    Revert to Phase 2.8 config
    Investigate root cause
    Consider alternative approaches
```

---

## Monitoring During Run

### Key Indicators (check periodically)
- **Episode 20:** Should see learning stabilizing (non-zero trades, positive spikes)
- **Episode 40:** Midpoint check - mean should be trending neutral/positive
- **Episode 60:** Late-run convergence - finals should be improving
- **Episode 80:** Final scores should show 2/3+ positive

### Warning Signs
- ⚠️ Persistent zero-trade episodes (>10%)
- ⚠️ Mean dropping below -0.10 after episode 40
- ⚠️ Penalty rate exceeding 15% consistently
- ⚠️ Variance explosion (>0.05)

---

## Rationale for 3-Seed Test

**Why reduced scope vs 5 seeds?**
1. **Quick validation:** 4-5 hours vs 6-8 hours (saves time for iteration)
2. **Diverse coverage:** Worst/mid/best seeds from Run A captures performance spectrum
3. **Statistical validity:** 3 seeds sufficient for robustness check (Phase 2.8 used 5 but had higher variance)
4. **Resource efficiency:** If fails, we save 2-3 hours before fine-tuning
5. **Precedent:** Phase 2.8 showed tight clustering (seeds 7/17/77 very similar), so 3 likely representative

**If results ambiguous:** Can always extend to 5 seeds × 80 episodes for confirmation

---

## Next Steps After Run B

### If GREEN ✅
1. Archive Run B results
2. Update config comment to "SPR Baseline v1.1 candidate"
3. Run full 5 seeds × 200 episodes confirmation
4. Document final config
5. Select production seed

### If YELLOW 🟡
1. Implement penalty fine-tuning
2. Document changes in PHASE2_8C doc
3. Re-run 3 seeds × 80 eps
4. If passes → proceed to 200-ep confirmation
5. If fails again → revert to Phase 2.8

### If RED 🔴
1. Revert FREEZE_VALIDATION_FRICTIONS to False
2. Revert cooldown/trade_penalty to Phase 2.8 values
3. Run diagnostic: compare Phase 2.8 vs 2.8b under jitter
4. Investigate: Why did frozen baseline not predict jitter performance?
5. Consider: More conservative cadence recovery (smaller tweaks)

---

## Status

**Current:** ✅ Config updated (FREEZE_VALIDATION_FRICTIONS=False)  
**Ready to run:** `python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 80`  
**Expected completion:** ~4-5 hours from start  
**Next checkpoint:** Run B analysis & decision (GREEN/YELLOW/RED)  

---

## Notes

- Run B uses **same config as Run A** except friction jitter enabled
- This isolates the impact of friction variation on performance
- Results will guide whether cadence recovery (Phase 2.8b) is robust enough for production
- If successful, this becomes the foundation for SPR Baseline v1.1
