# Phase 2.8c - 200-Episode Confirmation Plan

## ✅ Run C v1 Results (120 episodes, 3 seeds)
- **Cross-seed mean SPR:** +0.048 ± 0.057 ✅
- **Cross-seed trail-5:** +0.232 ± 0.328
- **Penalty rate:** 0.0% ✅
- **Trades/ep:** 27.8 ± 0.3 ✅
- **Friction jitter:** WORKING (120 unique spreads per seed) ✅
- **Best seed:** 777 (mean +0.128, trail-5 +0.695) ⭐

**Verdict:** 🟢 GREEN - Proceed to 200-episode confirmation

---

## 🚀 200-Episode Confirmation Protocol

### Test Configuration
- **Seeds:** 7, 17, 27, 77, 777 (5 seeds for diversity)
- **Episodes:** 200 per seed
- **Duration:** ~25-30 hours total
- **Friction jitter:** ±10% (K=3 jitter-averaged validation)
- **Gating:** VAL_EXP_TRADES_SCALE = 0.38

### Command
```powershell
python run_seed_sweep_organized.py --seeds 7 17 27 77 777 --episodes 200
```

---

## 🎯 GO/NO-GO Acceptance Gates (MUST MEET ALL)

### Primary Gates
1. ✅ **Cross-seed Mean SPR ≥ +0.04**
2. ✅ **Cross-seed Trail-5 median ≥ +0.25**
3. ✅ **Std of means ≤ 0.035** (tighter than Run C v1's ±0.057)
4. ✅ **Penalty rate ≤ 10%**
5. ✅ **≥3/5 seeds with Trail-5 median > 0**

### Behavioral Metrics (Per Seed)
- **Action entropy:** 0.90–1.10 bits
- **Switch rate:** 0.14–0.20
- **Hold rate:** 0.65–0.80
- **Long/Short mix:** 40–60% / 60–40% (balanced)

**Flag for review:** Any seed drifting outside these ranges

---

## 🔬 Secondary Evaluations

### 1. Fixed-Friction Sanity Check (Every 25 Episodes)
**Purpose:** Verify agent isn't over-tuned to jitter sampling

**Method:**
- Save checkpoint every 25 episodes (ep25, ep50, ep75, ..., ep200)
- Re-evaluate each checkpoint with FIXED frictions (no jitter)
- Compare fixed-friction mean to jitter-avg mean

**Acceptance:**
- Fixed-friction mean within ±0.01-0.02 of jitter-avg mean
- If collapses, agent is fragile to friction variations

### 2. Harsher Jitter Stress Test (Post-Run)
**Purpose:** Test robustness to micro-cost variations

**Method:**
- After 200-ep run completes
- Re-run evaluation on LAST 10 episodes with K=5 jitter draws (vs K=3 in training)
- Compare K=5 results to K=3 results

**Acceptance:**
- Mean degrades <0.01 (K=5 vs K=3)
- Trail-5 median remains > +0.20
- If fails, policy is fragile to micro-costs

---

## 📊 Monitoring Checklist

**During run, track:**
- [ ] Mean SPR per seed (every 25 episodes)
- [ ] Trail-5 median per seed (rolling window)
- [ ] Penalty episodes per seed (running %)
- [ ] Trades/episode (should stay 25-30)
- [ ] Action entropy (should stay 0.90-1.10)
- [ ] Switch rate (should stay 0.14-0.20)
- [ ] Hold rate (should stay 0.65-0.80)
- [ ] Long/Short balance (should stay balanced)

**Post-run, verify:**
- [ ] All 5 seeds completed 200 episodes
- [ ] Friction jitter working (200+ unique spreads per seed)
- [ ] All primary gates passed
- [ ] Behavioral metrics in range
- [ ] Fixed-friction sanity check passed
- [ ] K=5 jitter stress test passed

---

## 🔧 Optional Micro-Tunes (IF Confirmation Misses One Gate)

### If Finals Too Luck-Heavy (big spikes, weak trail-5)
**Action:** Raise `VAL_TRIM_FRACTION` from 0.20 → 0.25
- Trims more tail outliers in aggregation
- Stabilizes median computation

### If Trade Counts Creep High or Quality Dips
**Action:** Nudge `VAL_EXP_TRADES_SCALE` +0.02 (0.38 → 0.40)
- Firms up expected-trades threshold
- Filters under-trade spikes more aggressively

### If Whipsaw Rises (entropy >1.1, switch >0.20)
**Action:** Raise `hold_tie_tau` slightly (0.035 → 0.038-0.040)
- Biases "hold" on near-ties
- Reduces excessive switching

---

## 📈 Expected Results

### GREEN Scenario (All Gates Pass)
- Cross-seed mean: +0.04 to +0.06
- Cross-seed trail-5: +0.30 to +0.50
- Std of means: ≤0.035
- Penalty rate: <10%
- Positive seeds: 4-5/5 (80-100%)
- Best seed: +0.10+ final score

**Action:** Lock Phase 2.8c as SPR Baseline v1.1 → Production seed selection

### YELLOW Scenario (1-2 Gates Marginal)
- Cross-seed mean: +0.03 to +0.04
- Some gates slightly missed (e.g., trail-5 +0.23, std 0.037)
- Behavioral metrics mostly in range

**Action:** Apply micro-tunes above, re-run 120-episode test with 5 seeds

### RED Scenario (Multiple Gates Failed)
- Cross-seed mean: <+0.03
- High variance (>0.05)
- Penalty rate >15%
- Behavioral drift (entropy <0.80 or >1.2)

**Action:** Revert to Phase 2.8b, analyze what degraded

---

## 🎯 Success Criteria Summary

**MUST ACHIEVE:**
1. ✅ Mean SPR ≥ +0.04
2. ✅ Trail-5 ≥ +0.25
3. ✅ Std ≤ 0.035
4. ✅ Penalty ≤ 10%
5. ✅ ≥3/5 positive trail-5 seeds
6. ✅ Fixed-friction sanity (within ±0.02)
7. ✅ K=5 stress test (mean degrade <0.01)

**IF ALL PASS:** 🟢 **Production-ready! Lock as SPR Baseline v1.1**

---

## 📝 Run Notes Template

```
PHASE 2.8c - 200-EPISODE CONFIRMATION RUN
==========================================
Date: [DATE]
Seeds: 7, 17, 27, 77, 777
Episodes: 200 per seed
Duration: [START] - [END]

PRIMARY GATES:
[ ] Cross-seed Mean SPR ≥ +0.04: ______
[ ] Cross-seed Trail-5 ≥ +0.25: ______
[ ] Std of means ≤ 0.035: ______
[ ] Penalty rate ≤ 10%: ______
[ ] ≥3/5 positive trail-5 seeds: ______

BEHAVIORAL METRICS:
[ ] Entropy 0.90-1.10: ______
[ ] Switch 0.14-0.20: ______
[ ] Hold 0.65-0.80: ______
[ ] L/S balanced: ______

SECONDARY CHECKS:
[ ] Fixed-friction sanity (every 25 eps): ______
[ ] K=5 jitter stress (last 10 eps): ______

VERDICT: [ ] GREEN  [ ] YELLOW  [ ] RED

NOTES:
_____________________________________
_____________________________________
```

---

## 🚀 Next Steps After GREEN

1. Lock Phase 2.8c as SPR Baseline v1.1
2. Archive config as `config_phase2.8c_baseline_v1.1.py`
3. Select production seed (likely 777 or 27)
4. Create comprehensive documentation
5. Paper trading integration (1-week test)
6. Live trading preparation

**Timeline to production:** ~48 hours after confirmation completes
