# ROOT CAUSE FOUND: Model Dimension Mismatch

## Critical Discovery

The 29-episode HOLD collapse was caused by:

**INCOMPATIBLE MODEL CHECKPOINT**
- Current state size: **93 dimensions**
- Checkpoint state size: **176 dimensions** (from old feature set)
- Training tried to load old 176-dim model into 93-dim architecture
- Load failed silently or used corrupted weights → HOLD collapse

## Evidence

```
Error loading best_model.pt:
size mismatch for feature.0.weight_mu: 
  checkpoint shape: torch.Size([256, 176])
  current model:    torch.Size([256, 93])
```

All checkpoints in `checkpoints/` are from the old 176-dimensional state space:
- `best_model.pt` (11/06 10:10) - 176 dims
- `final_model.pt` (11/06 11:15) - 176 dims  
- `checkpoint_ep200.pt` (11/05 19:47) - 176 dims

## Why This Caused HOLD Collapse

1. **Training started**: Tried to load incompatible checkpoint
2. **Load failed**: Weights didn't match dimensions
3. **Fallback**: Either:
   - Used random initialization (untrained network)
   - Partially loaded some layers (corrupted state)
4. **Random/corrupted Q-values**: Made HOLD appear "safest"
5. **Locked in**: Once HOLD Q-value higher, hold_tie_tau bias kept it there
6. **Never recovered**: 29 episodes not enough to overcome initial bias

## Fix: Delete Old Checkpoints and Retrain

### Step 1: Clear Old Checkpoints
```powershell
# Backup old checkpoints
New-Item -ItemType Directory -Path checkpoints_backup_176dim -Force
Move-Item checkpoints\*.pt checkpoints_backup_176dim\

# Verify clean slate
Get-ChildItem checkpoints\*.pt  # Should return nothing
```

### Step 2: Verify Current State Size
```python
python -c "from features import build_features; from data_loader import load_data; df = load_data('data/EURUSD_M30_prepared.csv'); feats, names = build_features(df, stack_n=2); print(f'State size: {feats.shape[1]}')"
```

Expected: **93 dimensions** (current feature set with stack_n=2)

### Step 3: Start Fresh Training
```powershell
# Train from scratch with correct dimensions
python main.py --episodes 50
```

Now the agent will:
- Initialize fresh Q-network with 93-dim input
- No checkpoint loading (no best_model.pt exists)
- Learn from scratch with correct architecture
- Emergency parameter adjustments (entropy_beta=0.025, etc.) will actually work

## Why Parameter Tweaks Had No Effect

We tried:
1. Fix Pack D1 (entropy_beta, hold_tie_tau, flip_penalty)
2. Emergency Option B (further adjustments)

**Both failed** because the agent was using a **corrupted/incompatible model**. Parameter changes to reward shaping can't fix fundamentally broken Q-network weights.

## Expected Outcome After Fix

With clean slate + emergency parameters:
- Episodes 1-10: Should see some LONG/SHORT actions (exploration)
- Episodes 10-20: Trading activity should stabilize
- Episodes 20-30: Entropy should rise above 0.0, hold_rate drop below 1.0

If STILL 100% HOLD after fresh start → then it's a true parameter/code issue.

## Action Plan

1. **Backup old checkpoints** (save 176-dim models for reference)
2. **Delete checkpoints/*.pt** (force fresh start)
3. **Verify state size = 93** (confirm feature config)
4. **Run 50 episodes** from scratch
5. **Monitor first 10 episodes** for ANY trading activity

## Lesson Learned

**Always verify model architecture matches when loading checkpoints!**

The dimension mismatch should have:
- Thrown a clear error (it did, but may have been caught/ignored)
- Prevented training from starting
- Been caught by a validation check

Add to future code:
```python
# In main.py or trainer.py before loading checkpoint
expected_state_size = env.observation_space.shape[0]
checkpoint_state_size = checkpoint['policy_net_state_dict']['feature.0.weight_mu'].shape[1]
assert expected_state_size == checkpoint_state_size, f"State size mismatch: {expected_state_size} != {checkpoint_state_size}"
```
