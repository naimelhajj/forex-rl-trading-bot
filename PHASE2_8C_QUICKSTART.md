# Phase 2.8c Quick-Start Guide

## 🎯 Quick Summary
Phase 2.8c implements stabilization tweaks to turn Run B v2's 🟡 YELLOW into 🟢 GREEN:
- **Jitter-averaged validation** (K=3 draws per window)
- **Tighter trade-count gating** (VAL_EXP_TRADES_SCALE = 0.38)
- **Enhanced reporting** (trail-5 median for finals)

## 🚀 Running Phase 2.8c

### Step 1: Verify Configuration
```powershell
# Check config settings
python -c "import config; print(f'VAL_JITTER_DRAWS = {config.Config.VAL_JITTER_DRAWS}'); print(f'VAL_EXP_TRADES_SCALE = {config.Config.VAL_EXP_TRADES_SCALE}'); print(f'FREEZE_VALIDATION_FRICTIONS = {config.Config.FREEZE_VALIDATION_FRICTIONS}')"
```

**Expected output:**
```
VAL_JITTER_DRAWS = 3
VAL_EXP_TRADES_SCALE = 0.38
FREEZE_VALIDATION_FRICTIONS = False
```

### Step 2: Archive Old Results (Optional)
```powershell
# Archive Run B v2 results for reference
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Move-Item -Path "logs\seed_sweep_results\seed_7" -Destination "logs\seed_sweep_results\seed_7_RUN_B_V2_$timestamp" -ErrorAction SilentlyContinue
Move-Item -Path "logs\seed_sweep_results\seed_77" -Destination "logs\seed_sweep_results\seed_77_RUN_B_V2_$timestamp" -ErrorAction SilentlyContinue
Move-Item -Path "logs\seed_sweep_results\seed_777" -Destination "logs\seed_sweep_results\seed_777_RUN_B_V2_$timestamp" -ErrorAction SilentlyContinue
```

### Step 3: Start Run C v1 (Quick Test)
```powershell
# Start 3-seed × 120-episode test
python run_seed_sweep_organized.py --seeds 7 77 777 --episodes 120
```

**Expected duration:** ~6-8 hours (3 seeds × 120 episodes)

### Step 4: Monitor Progress
```powershell
# Check seed directories (should update every ~3-4 minutes per episode)
Get-ChildItem logs\seed_sweep_results\seed_* -Directory | Select-Object Name, LastWriteTime | Sort-Object LastWriteTime

# Check if process running
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, CPU, StartTime

# Watch validation logs (should show "jitter-avg K=3" in VAL output)
Get-Content logs\training.log -Tail 50 -Wait
```

**Expected VAL output:**
```
[VAL] 7 passes | window=600 | stride~90 | coverage~1.15x | jitter-avg K=3
```

### Step 5: Analyze Results
```powershell
# After run completes, analyze results
python compare_run_c_v1.py
```

**Expected GREEN output:**
```
RUN C v1 CROSS-SEED SUMMARY:
Cross-seed mean SPR:  +0.040 ± 0.030
Cross-seed trail-5:   +0.400 ± 0.100
Cross-seed trades/ep: 25.5 ± 0.8
Cross-seed penalty:   8.5%
Positive trail-5: 3/3 seeds

VERDICT:
🟢 GREEN - Agent is robust! Proceed to 200-episode confirmation.
```

## 📊 What to Look For

### 🟢 GREEN Signals (Proceed to 200-episode)
- Cross-seed mean ≥ +0.03
- Cross-seed trail-5 ≥ +0.30
- Mean variance ≤ ±0.035
- Penalty rate <10%
- Positive trail-5: 3/3 seeds
- Friction jitter working (verify unique spreads)

### 🟡 YELLOW Signals (Fine-tune)
- Cross-seed mean +0.02-0.03
- Cross-seed trail-5 +0.20-0.30
- Mean variance ±0.035-0.045
- Penalty rate 10-15%
- Action: Run 120-episode test with 5 seeds, adjust gating

### 🔴 RED Signals (Revert or Debug)
- Cross-seed mean <+0.02
- Degradation vs Run B v2
- High variance (>±0.05)
- Penalty rate >15%
- Action: Revert to Phase 2.8b, analyze what broke

## 🔍 Verification Checks

### Check 1: Jitter-Averaging Working
```powershell
# Verify jitter-avg appears in logs
python -c "with open('logs/training.log', 'r') as f: lines = [l for l in f if 'jitter-avg K=3' in l]; print(f'Found {len(lines)} validation passes with jitter-averaging')"
```

**Expected:** Multiple validation passes with "jitter-avg K=3" message

### Check 2: Friction Randomization Working
```powershell
# Verify friction variation in JSONs
python check_friction_jitter.py
```

**Expected:** 120+ unique spread values per seed (K=3 draws × 40 windows × 7 passes)

### Check 3: Trade-Count Gating Impact
```powershell
# Check penalty episodes
python -c "
import json
from pathlib import Path
for seed in [7, 77, 777]:
    seed_dir = Path(f'logs/seed_sweep_results/seed_{seed}')
    if seed_dir.exists():
        penalty_eps = sum(1 for f in seed_dir.glob('val_ep*.json') if json.load(f.open()).get('penalty_applied', False))
        total = len(list(seed_dir.glob('val_ep*.json')))
        print(f'Seed {seed}: {penalty_eps}/{total} penalty episodes ({penalty_eps/total:.1%})')
"
```

**Expected:** <10% penalty episodes per seed

## 🎯 Next Steps After GREEN

### 1. Lock Phase 2.8c as Baseline v1.1
```powershell
# Archive configuration
Copy-Item config.py -Destination config_phase2.8c_baseline_v1.1.py

# Create release tag (if using git)
git add config.py trainer.py PHASE2_8C_STABILIZATION_TWEAKS.md
git commit -m "Phase 2.8c: Stabilization tweaks (jitter-avg K=3, gating 0.38)"
git tag -a phase2.8c_baseline_v1.1 -m "SPR Baseline v1.1 - Robustness confirmed"
```

### 2. Run 200-Episode Confirmation
```powershell
# Full validation with 5 seeds
python run_seed_sweep_organized.py --seeds 7 17 27 77 777 --episodes 200
```

**Expected duration:** ~20-25 hours (5 seeds × 200 episodes)

### 3. Select Production Seed
- Analyze 200-episode results
- Select seed with:
  - 200-ep mean ≥ +0.05
  - Trail-5 median ≥ +0.30
  - Consistent positive performance
  - Low penalty rate (<5%)
  - Good long/short balance

**Likely candidates:** Seed 777 (best in Run B v2) or Seed 27 (excellent finals in Run A)

## 🛠️ Troubleshooting

### Issue: Jitter-averaging not appearing in logs
**Check:** Verify `VAL_JITTER_DRAWS = 3` in config.py  
**Fix:** Update config.py and restart run

### Issue: Penalty rate too high (>15%)
**Check:** Verify `VAL_EXP_TRADES_SCALE = 0.38`  
**Fix:** Lower to 0.36 for gentler gating

### Issue: No unique spread values
**Check:** Verify `FREEZE_VALIDATION_FRICTIONS = False`  
**Fix:** Enable friction jitter and restart

### Issue: Results worse than Run B v2
**Check:** Verify jitter-averaging logic in trainer.py  
**Debug:** Add print statements to verify K=3 draws per window  
**Fallback:** Revert to Phase 2.8b config

## 📝 File Checklist

**Modified files:**
- ✅ `config.py` - VAL_JITTER_DRAWS, VAL_EXP_TRADES_SCALE
- ✅ `trainer.py` - Jitter-averaged validation logic

**New files:**
- ✅ `compare_run_c_v1.py` - Enhanced comparison script
- ✅ `PHASE2_8C_STABILIZATION_TWEAKS.md` - Full documentation
- ✅ `PHASE2_8C_QUICKSTART.md` - This guide

**Scripts to use:**
- `run_seed_sweep_organized.py` - Start training
- `compare_run_c_v1.py` - Analyze results
- `check_friction_jitter.py` - Verify friction randomization
- `monitor_sweep.py` - Monitor progress

## 🎉 Success Criteria

✅ **Phase 2.8c is successful if:**
1. Cross-seed mean ≥ +0.03 (improved vs Run B v2 +0.018)
2. Cross-seed trail-5 ≥ +0.30 (stable finals)
3. Mean variance ≤ ±0.035 (acceptable for ±10% jitter)
4. Penalty rate <10% (improved gating)
5. Positive trail-5: 3/3 seeds (100% success rate)
6. Friction jitter working (verified unique spreads)

**Result:** Agent is robust to market friction variations → Ready for 200-episode production validation!
