# Environment.py Corruption Summary and Fix Guide

## Problem
The environment.py file became corrupted during automated editing, with over 400 syntax errors caused by overlapping text fragments and malformed statements.

## Root Cause
Multiple `replace_string_in_file` operations cascaded incorrectly, causing text fragments to merge improperly.

## Quick Fix Options

### Option 1: Restore from Parent Directory Backup (RECOMMENDED)
```powershell
# Check if backup exists
Get-ChildItem -Path ..\ -Filter "*forex_rl_bot*.zip" | Sort-Object LastWriteTime -Descending | Select-Object -First 3

# If found, extract environment.py from latest backup
# Example:
Expand-Archive -Path "..\forex_rl_bot-backup-YYYYMMDD.zip" -DestinationPath ".\temp_backup"
Copy-Item ".\temp_backup\environment.py" ".\environment.py" -Force
```

### Option 2: Run Automated Fix Script
```powershell
python fix_environment.py
# Review environment_fixed.py
# If good, backup current and replace:
Copy-Item environment.py environment_corrupted_backup.py
Copy-Item environment_fixed.py environment.py
```

### Option 3: Manual Fix Key Corruption Points

The corruption is concentrated in these sections:

#### 1. Reset Method (lines 260-305)
**Corrupted lines**:
- Line 264: `self.balance = self.initial_balance by trainer)`
- Line 272: `self.trades_this_ep = 0 Dict with position info`
- Line 280: `self.bars_since_close = 00.0`
- Line 281: `self.last_action = [0, 0, 0, 0]dget tracking`

**Fix**: Remove all text after the actual statement (e.g., "by trainer)", "Dict with position info", etc.)

#### 2. Step Method (lines 310-350)
**Corrupted lines**:
- Line 320: `current_price = current_data['close']# Track position state`
- Line 329: `reward = 0.0dget_pct = 0.05`
- Line 330: `did_trade = Falsetrol state`
- Line 333: `if self.position is not None:0]`

**Fix**: Split merged statements onto separate lines

#### 3. Portfolio Features Method (lines 580-600)
**Corrupted lines**:
- Line 581: `"""urn next_state, reward, done, info`
- Line 593: `if self.position is None:osition and self.position['type']`
- Line 594: `lots_norm = 0.0 short_on # [-1, 0, 1]`

**Fix**: Separate merged statements

## Complete Fix (If Manual)

Since there are 400+ errors, here's the systematic approach:

1. **Backup current file**:
   ```powershell
   Copy-Item environment.py environment_corrupted.py
   ```

2. **Check if Git or ZIP backup exists** (most reliable):
   ```powershell
   # Check parent directory for zip backups
   ls ..\ | Where-Object {$_.Name -like "*forex*"}
   ```

3. **If no backup, apply PHASE 2.8e changes to last known good version**:
   - The file worked before my edits
   - You need to restore to Fix Pack D2 Level 2 state
   - Then manually apply soft bias changes from `PHASE_2_8E_SOFT_BIAS_IMPLEMENTATION.md`

## Key Sections That Need Soft Bias Changes (Once File is Clean)

### 1. Constructor Parameters (line ~130-145)
Remove:
```python
ls_balance_lambda: float = 0.050
hold_balance_lambda: float = 0.020
```

Add:
```python
directional_bias_beta: float = 0.08,
hold_bias_gamma: float = 0.05,
bias_check_interval: int = 10,
bias_margin_low: float = 0.35,
bias_margin_high: float = 0.65,
hold_ceiling: float = 0.80,
circuit_breaker_enabled: bool = True,
circuit_breaker_threshold_low: float = 0.10,
circuit_breaker_threshold_high: float = 0.90,
circuit_breaker_lookback: int = 500,
circuit_breaker_mask_duration: int = 30
```

### 2. Instance Variables (line ~185-210)
Remove:
```python
self.ls_balance_lambda = ls_balance_lambda
self.hold_balance_lambda = hold_balance_lambda
```

Add:
```python
self.directional_bias_beta = directional_bias_beta
self.hold_bias_gamma = hold_bias_gamma
self.bias_check_interval = bias_check_interval
self.bias_margin_low = bias_margin_low
self.bias_margin_high = bias_margin_high
self.hold_ceiling = hold_ceiling
self.circuit_breaker_enabled = circuit_breaker_enabled
self.circuit_breaker_threshold_low = circuit_breaker_threshold_low
self.circuit_breaker_threshold_high = circuit_breaker_threshold_high
self.circuit_breaker_lookback = circuit_breaker_lookback
self.circuit_breaker_mask_duration = circuit_breaker_mask_duration
self.circuit_breaker_active = False
self.circuit_breaker_steps_remaining = 0
self.circuit_breaker_mask_side = None
```

### 3. Reset Method (line ~260)
Replace:
```python
self.dir_window = deque(maxlen=500)
```

With:
```python
self.action_history = deque(maxlen=self.circuit_breaker_lookback)
self.circuit_breaker_active = False
self.circuit_breaker_steps_remaining = 0
self.circuit_breaker_mask_side = None
```

### 4. Step Method (line ~510-540)
Remove entire rolling window penalty section:
```python
# PHASE-2.8d Fix Pack D2.B: Rolling window L/S balance regularizer
# ... all dir_window tracking code ...
# PHASE-2.8d Fix Pack D2.C: Hold-rate guardrail
# ... all hold_share penalty code ...
```

Replace with:
```python
# PHASE-2.8e: Track action history for circuit-breaker
if hasattr(self, 'action_history'):
    self.action_history.append(action)
```

### 5. Add New Method (before _get_state)
Add the complete `get_action_bias()` method from `PHASE_2_8E_SOFT_BIAS_IMPLEMENTATION.md` (lines 155-225)

## Testing After Fix

```powershell
# Test import
python -c "import environment; print('SUCCESS')"

# If successful, run smoke test
python main.py --episodes 3 --seed 42
```

## Recovery Contacts

If you can't fix it:
1. Check parent folder for `.zip` backups dated before today
2. Extract environment.py from most recent backup
3. Apply soft bias changes manually using the implementation guide

## Apologies

I apologize for corrupting the file. The cascading replace operations caused overlapping edits that merged text incorrectly. In the future, I should:
- Read larger sections before editing
- Use more targeted string replacements
- Test imports after each change
- Create backups before major refactoring

The soft bias design is sound and correct - the implementation just needs to be applied to a clean file.
