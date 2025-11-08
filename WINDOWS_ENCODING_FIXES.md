# Windows Encoding Fixes - Implementation Summary

## Date: 2025-10-18

## Problem
Windows PowerShell uses CP1252 encoding by default, causing:
1. `UnicodeEncodeError` when printing Unicode characters (→, ✓, ⚠, etc.)
2. `OSError: [Errno 22] Invalid argument` when stdout pipe closes during tee operations

## Solutions Implemented

### 1. ✅ Robust Tee Class (`tee.py`)
- **Purpose**: Native Python tee for simultaneous console and file output
- **Features**:
  - Graceful encoding error handling
  - No PowerShell dependencies (avoids pipe issues)
  - Context manager for clean setup/teardown
  - Fallback encoding strategies for console output

**Usage:**
```python
from tee import Tee

with Tee('output.log'):
    # All prints go to both console and file
    main()
```

### 2. ✅ UTF-8 Reconfiguration in main.py
- Added `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` at module top
- Prevents crashes when Unicode sneaks through
- Falls back gracefully on older Python versions

### 3. ✅ Fixed run_seed_sweep_auto.py
- Replaced PowerShell pipe (`| Tee-Object`) with native Python tee
- Uses `subprocess.Popen` with direct line-by-line streaming
- Sets `PYTHONUNBUFFERED=1` and `PYTHONIOENCODING=utf-8`
- No more `OSError [Errno 22]` from broken pipes

### 4. ✅ Fixed run_seed_sweep_simple.py
- Fixed SyntaxError: `f r'\g<1>{seed}'` → `fr'\g<1>{seed}'`
- Now compiles and runs correctly

### 5. ✅ ASCII-Only in Production Files
**Already clean (verified with check_ascii.py):**
- `main.py` - All arrows removed (→ became "to")
- `trainer.py` - All checkmarks/arrows removed
- `environment.py` - Comments only (no print statements with Unicode)
- `agent.py` - Clean

## Files Still Containing Unicode (Non-Critical)

These are **test files only** - not used in production training:

```
test_system.py                  (19 ✓ symbols)
test_balance_invariance.py      (8 ✓ symbols)
test_enhancements.py            (13 ✓ symbols)  
test_hardening.py               (11 ✓ symbols)
test_hang_diagnosis.py          (11 ✓/✗ symbols)
verify_strengths.py             (10 ✓/✗ symbols)
test_unicode_and_hangs.py       (2 → and ✓ in test data)
smoke_test_improvements.py      (7 ✓ symbols)
test_*.py                       (various diagnostic scripts)
```

**Recommendation:** Leave these as-is since they're not part of the training pipeline. If you need to run them on Windows, either:
1. Run `chcp 65001` in PowerShell first, OR
2. Batch replace `✓` with `[OK]` and `✗` with `[X]` if needed

## Seed Sweep Scripts Status

### ✅ run_seed_sweep_auto.py
- **Status**: FIXED and WORKING
- **Changes**:
  - Uses native Python tee (no PowerShell)
  - Streams output line-by-line
  - Properly handles encoding
- **Current**: Running seed sweep successfully

### ✅ run_seed_sweep_simple.py  
- **Status**: FIXED
- **Changes**: Fixed regex syntax error
- **Usage**: Manual config updates, good for debugging

### ⚠️ run_seed_sweep.py (older version)
- **Status**: Contains ✓ symbols
- **Recommendation**: Use `run_seed_sweep_auto.py` instead

## Optional: Force UTF-8 Console (If You Want Unicode Back)

If you want to keep Unicode symbols for better readability, run this **once per PowerShell session**:

```powershell
chcp 65001 > $null
$env:PYTHONUTF8 = 1
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
```

Then all Unicode will work correctly.

## Testing

### Verify Tee Class
```powershell
python tee.py
```
Should print test output and show "[OK] Tee test passed!"

### Verify Main.py Encoding
```powershell
python -c "import main; print('Import successful')"
```

### Verify Seed Sweep
```powershell
python run_seed_sweep_auto.py
```
Should run without encoding errors and populate log files.

## Current Status

**Production Pipeline**: ✅ **FULLY ASCII-SAFE**
- main.py ✅
- trainer.py ✅  
- environment.py ✅
- agent.py ✅
- run_seed_sweep_auto.py ✅
- run_seed_sweep_simple.py ✅

**Seed Sweep**: 🔄 **RUNNING**
- Seed 7: In progress
- Estimated completion: ~90 minutes for all 3 seeds

## Training Behavior Notes

From seed sweep logs, observed:
- Validation windowing (K=6) working correctly
- Trade count gating functioning (0.5x, 0.75x, 1.0x multipliers)
- Early stopping logic operational
- Episodes 19-23 showed good fitness (~0.53)
- Later episodes had low trade counts → fitness gated to near 0

**If low trade count is problematic**, consider:
1. Reduce `min_trades_full` from 23 → 20
2. Reduce `min_trades_half` from 16 → 14  
3. Lower `min_hold` from 8 → 6 bars
4. Lower `cooldown` from 16 → 12 bars

These are in `config.py` under validation settings.

## Summary

All critical Unicode issues resolved. The system is now robust against Windows CP1252 encoding limitations. The seed sweep is running successfully with proper output capture.
