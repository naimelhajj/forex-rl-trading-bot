# Phase 2.8d - Ready to Restart Training

## Problem Identified
**Root cause**: Incompatible model checkpoint (176-dim → 93-dim mismatch)

## Fix Applied
✅ Old 176-dimensional checkpoints backed up to `checkpoints_backup_176dim/`
✅ Checkpoints directory cleared (only scaler JSON files remain)
✅ Emergency parameters still active (entropy_beta=0.025, hold_tie_tau=0.030, flip_penalty=0.0005)

## Current Configuration
- `entropy_beta`: 0.025 (strong exploration bonus)
- `hold_tie_tau`: 0.030 (reduced HOLD bias)
- `flip_penalty`: 0.0005 (low churn penalty)
- `learning_starts`: 1000 (SMOKE mode)
- `use_noisy`: True (NoisyNet exploration)

## Next Step: Fresh Training Run

```powershell
python main.py --episodes 50
```

## Expected Behavior (If Fix Worked)

### Episodes 1-5: Initial Exploration
- Random baseline collection (prefill)
- Some LONG/SHORT actions should appear
- Entropy > 0.0 (not completely deterministic)

### Episodes 5-15: Early Learning
- Trading activity should emerge
- Hold rate should drop below 1.0
- Trades > 0 in validation

### Episodes 15-30: Stabilization  
- Consistent trading behavior
- Entropy 0.3-0.8 bits
- Hold rate 0.6-0.8
- Long/short balance improving

### Episodes 30-50: Assessment
**SUCCESS signals:**
- Trades ≥ 5 per episode
- Entropy 0.5-1.0 bits
- Hold rate 0.65-0.80
- Long ratio 0.30-0.70

**FAILURE signals (still broken):**
- 0 trades (still HOLD collapse)
- Entropy = 0.0
- Hold rate = 1.0

## If Still Fails After Fresh Start

Then it's NOT the checkpoint - it's a true code/parameter issue.

### Nuclear Options if HOLD Collapse Persists
1. **Disable NoisyNet, use epsilon-greedy**
   ```python
   # config.py - AgentConfig
   use_noisy: False
   epsilon_start: 0.5
   epsilon_end: 0.1
   ```

2. **Remove ALL HOLD bias**
   ```python
   # config.py - AgentConfig  
   hold_tie_tau: 0.0  # No HOLD preference
   ```

3. **Massively boost entropy**
   ```python
   # config.py - EnvironmentConfig
   entropy_beta: 0.050  # 2x current value
   ```

4. **Check if entropy bonus is actually being applied**
   - Add print statements in environment.py step() function
   - Verify reward includes entropy term
   - Check if long_trades/short_trades counters update

## Monitoring During Run

Watch for first signs of trading:

```powershell
# Check every 5 episodes
for ($i=5; $i -le 50; $i+=5) {
    Start-Sleep -Seconds 600  # Wait ~10 minutes per episode
    $file = "logs\validation_summaries\val_ep$('{0:D3}' -f $i).json"
    if (Test-Path $file) {
        $data = Get-Content $file | ConvertFrom-Json
        Write-Host "Ep $i : trades=$($data.trades) entropy=$($data.entropy) hold=$($data.hold_frac)"
    }
}
```

## Decision Points

**After Episode 10:**
- If trades > 0: ✓ Fix worked, continue to 50
- If trades = 0: ✗ Still broken, stop and apply Nuclear Option 1 or 2

**After Episode 30:**
- If trading is stable: ✓ Proceed to ablation study
- If still random/collapsed: Apply more aggressive fixes

## Ready State
✅ Checkpoints cleared
✅ Config verified (emergency params loaded)
✅ State size documented (93-dim expected, will be confirmed on run)
✅ Monitoring plan ready

**Command to run:** `python main.py --episodes 50`
