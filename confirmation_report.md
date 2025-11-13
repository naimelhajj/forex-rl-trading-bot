================================================================================
PHASE 2.8f CONFIRMATION REPORT
================================================================================
Generated: 2025-11-12T22:59:44.383666
Results directory: confirmation_results

## SUMMARY

❌ **5 GATE(S) FAILED**
   Review tuning ladder in PHASE_2_8F_CONFIRMATION_PROTOCOL.md

Gates passed: 4/9

## ACCEPTANCE GATES

### GATE_1: Mean SPR
**Status**: ❌ FAIL
**Threshold**: ≥ 0.04
**Measured**: -0.0236

Per-seed breakdown:
  - Seed 42: -0.0021
  - Seed 123: -0.0125
  - Seed 456: -0.0434
  - Seed 789: -0.0311
  - Seed 1011: -0.0287

### GATE_2: Cross-Seed σ(SPR)
**Status**: ✅ PASS
**Threshold**: ≤ 0.035
**Measured**: 0.0163

### GATE_3: Trail-5 Median SPR
**Status**: ❌ FAIL
**Threshold**: ≥ 0.25
**Measured**: -0.0283

Per-seed breakdown:
  - Seed 42: -0.0046
  - Seed 123: 0.0065
  - Seed 456: -0.0558
  - Seed 789: -0.0283
  - Seed 1011: -0.0321

### GATE_4: Positive Seeds
**Status**: ❌ FAIL
**Threshold**: ≥ 3/5
**Measured**: 0/5

### GATE_5: Long Ratio In-Band
**Status**: ❌ FAIL
**Threshold**: ≥ 70%
**Measured**: 0.0%

Per-seed breakdown:
  - Seed 42: 0.0%
  - Seed 123: 0.0%
  - Seed 456: 0.0%
  - Seed 789: 0.0%
  - Seed 1011: 0.0%

### GATE_6: Hold Rate In-Band
**Status**: ✅ PASS
**Threshold**: ≥ 70%
**Measured**: 72.9%

Per-seed breakdown:
  - Seed 42: 79.0%
  - Seed 123: 62.3%
  - Seed 456: 76.9%
  - Seed 789: 62.1%
  - Seed 1011: 84.4%

### GATE_7: Entropy In-Band
**Status**: ✅ PASS
**Threshold**: ≥ 80%
**Measured**: 98.3%

Per-seed breakdown:
  - Seed 42: 98.8%
  - Seed 123: 96.1%
  - Seed 456: 100.0%
  - Seed 789: 100.0%
  - Seed 1011: 96.9%

### GATE_8: Switch Rate In-Band
**Status**: ❌ FAIL
**Threshold**: ≥ 70%
**Measured**: 32.5%

Per-seed breakdown:
  - Seed 42: 54.3%
  - Seed 123: 59.7%
  - Seed 456: 28.8%
  - Seed 789: 6.9%
  - Seed 1011: 12.5%

### GATE_9: Penalty Rate
**Status**: ✅ PASS
**Threshold**: ≤ 10%
**Measured**: 0.0%

Per-seed breakdown:

## RECOMMENDATIONS

### Tuning Required:

**Mean SPR**:

**Trail-5 Median SPR**:

**Positive Seeds**:

**Long Ratio In-Band**:
  - Widen dead-zone: LONG_BAND = 0.12
  - Reduce gain: K_LONG = 0.6
  - Increase leak: LEAK = 0.997

**Switch Rate In-Band**:

After tuning, retest with 1 seed × 200 episodes before full suite.

================================================================================