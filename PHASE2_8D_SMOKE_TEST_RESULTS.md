# Phase 2.8d Smoke Test Results

**Date**: November 5, 2025  
**Status**: ⚠️ **COMPLETED WITH ISSUES**

---

## 🔴 Critical Issue: Agent Not Trading

### Validation Output
```
[VAL] K=7 overlapping | SPR=-0.080 | PF=0.00 | MDD=1.50% | MMR=0.00% | TPY=0.0 | SIG=0.00 | trades=0.0 | mult=0.00 | pen=0.080
```

**Problem**: Agent made **0 trades** during validation after 20 episodes of training.

**Penalty**: −0.080 (full undertrade penalty for 0 trades)

---

## 🔍 Root Cause Analysis

### Possible Causes:

1. **Training Duration Too Short**
   - 20 episodes may be insufficient for initial learning
   - Agent might still be in exploration/buffer-filling phase
   - Default `learning_starts=5000` might gate early learning

2. **Overly Conservative Policy**
   - New reward shaping (entropy bonus + L/S regularizer) might suppress trading
   - Higher `hold_tie_tau=0.038` increases hold bias excessively
   - Agent learning to avoid trading to minimize penalties

3. **Action Masking Issues**
   - Cooldown/gating preventing trade execution
   - Min hold bars blocking position changes

4. **Missing Behavioral Metrics Export** ✅ **FIXED**
   - Validation JSONs weren't exporting `entropy`, `hold_frac`, `long_ratio`
   - Added field aliases in trainer.py summary dict

---

## ✅ What Was Fixed

Added behavioral metrics export to `trainer.py`:
```python
"entropy": policy_metrics["action_entropy_bits"],  # Alias
"hold_frac": policy_metrics["hold_rate"],  # Alias
"long_ratio": policy_metrics["long_short"][0] / max(1.0, sum(policy_metrics["long_short"])),
```

Now validation JSONs will include these fields for monitoring.

---

## 🎯 Next Steps

### Option A: Run Longer Smoke Test
```powershell
python main.py --episodes 50
```
- Let agent train longer to see if it starts trading
- Check episode 40-50 for trading activity

### Option B: Diagnostic Check
```powershell
# Check training logs for learning progress
Get-Content logs\training.log | Select-String "Episode" | Select-Object -Last 20

# Check replay buffer size
Get-Content logs\training.log | Select-String "buffer" | Select-Object -Last 5

# Check if agent is executing actions
Get-Content logs\training.log | Select-String "LONG|SHORT" | Select-Object -Last 10
```

### Option C: Reduce Learning Barriers
Temporarily adjust config.py to speed up learning:
```python
# Reduce learning starts gate
learning_starts: int = 1000  # Down from 5000

# Reduce hold bias slightly
hold_tie_tau: float = 0.035  # Down from 0.038

# Increase exploration temporarily
entropy_beta: float = 0.020  # Up from 0.014
```

---

## 📊 What to Monitor

### Training Logs
```powershell
# Real-time monitoring
Get-Content logs\training.log -Wait -Tail 20
```

Look for:
- ✅ Buffer filling: "Replay buffer: 1000/100000"
- ✅ Training starting: "Episode X | Steps: Y | Reward: Z"
- ✅ Actions taken: "LONG", "SHORT", "FLATTEN"
- ❌ All HOLD actions (problem sign)

### Validation Files
```powershell
# Check if trades are happening in validation
foreach ($i in 1..20) {
    $file = "logs\validation_summaries\val_ep$('{0:D3}' -f $i).json";
    if (Test-Path $file) {
        $data = Get-Content $file | ConvertFrom-Json;
        Write-Host "Ep $i : trades=$($data.trades) score=$($data.score.ToString('0.000'))";
    }
}
```

---

## 🔧 Recommended Action

**Run 50-episode test** to see if agent starts trading:

```powershell
python main.py --episodes 50
```

Then check last 10 episodes for trading activity:

```powershell
for ($i=41; $i -le 50; $i++) {
    $file = "logs\validation_summaries\val_ep$('{0:D3}' -f $i).json";
    if (Test-Path $file) {
        $data = Get-Content $file | ConvertFrom-Json;
        Write-Host "Ep $i : trades=$($data.trades.ToString('0.0')) entropy=$($data.entropy.ToString('0.00')) hold=$($data.hold_frac.ToString('0.00'))";
    }
}
```

**Success Indicators**:
- ✅ trades > 5 by episode 50
- ✅ entropy > 0.5
- ✅ hold_frac < 0.95

If still 0 trades → we need to debug further (likely learning_starts gate or overly conservative rewards).

---

**Status**: Metrics export fixed ✅, but agent not trading after 20 eps ⚠️  
**Next**: Run 50-episode test to diagnose learning progress
