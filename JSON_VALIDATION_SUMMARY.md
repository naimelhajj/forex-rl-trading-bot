## JSON Validation Summaries - Implementation Complete

**Date:** October 18, 2025  
**Feature:** Automated JSON export of validation metrics for easy analysis

---

## What Was Added

### 1. JSON Export in Trainer ✅

**File:** `trainer.py` lines 572-596  
**Location:** End of `validate()` method, after all metrics computed

```python
# --- Write compact JSON summary for analysis tools ---
import os
import json
import datetime as dt

summary = {
    "episode": int(self.episode),
    "k": int(len(windows)),
    "median_fitness": float(median),
    "iqr": float(iqr),
    "adj": float(stability_adj),
    "trades": float(median_trades),
    "mult": float(mult),
    "penalty": float(undertrade_penalty),
    "score": float(val_score),
    "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    "seed": int(getattr(self.config, "random_seed", -1))
}

out_dir = os.path.join("logs", "validation_summaries")
os.makedirs(out_dir, exist_ok=True)

fname = os.path.join(out_dir, f"val_ep{summary['episode']:03d}.json")
with open(fname, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=True, indent=2)
```

**Features:**
- ✅ One JSON file per validation (episode)
- ✅ `ensure_ascii=True` prevents Windows encoding issues
- ✅ Formatted with `indent=2` for readability
- ✅ Auto-creates `logs/validation_summaries/` directory
- ✅ Filename pattern: `val_ep001.json`, `val_ep002.json`, etc.

---

### 2. Updated Diversity Checker ✅

**File:** `check_validation_diversity.py` (complete rewrite)  
**Change:** Read from JSON files instead of parsing log text

```python
import json
from pathlib import Path

summary_dir = Path("logs/validation_summaries")
json_files = sorted(summary_dir.glob("val_ep*.json"))

for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Direct access to clean data (no regex parsing!)
    ep = data['episode']
    trades = data['trades']
    mult = data['mult']
    pen = data['penalty']
    score = data['score']
```

**Benefits:**
- ✅ No regex parsing (more reliable)
- ✅ Handles UTF-8 correctly
- ✅ Easy to extend with more metrics
- ✅ Summary statistics at the end

---

## JSON Format

Each validation creates a file like `logs/validation_summaries/val_ep003.json`:

```json
{
  "episode": 3,
  "k": 6,
  "median_fitness": 0.145,
  "iqr": 0.082,
  "adj": 0.116,
  "trades": 7.5,
  "mult": 1.0,
  "penalty": 0.0,
  "score": 0.116,
  "timestamp": "2025-10-18T14:32:15",
  "seed": 777
}
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `episode` | int | Training episode number (1-indexed) |
| `k` | int | Number of overlapping validation windows (typically 6-7) |
| `median_fitness` | float | Median raw fitness across K windows |
| `iqr` | float | Interquartile range (fitness stability measure) |
| `adj` | float | IQR-adjusted fitness (median - 0.35*IQR) |
| `trades` | float | Median trade count across K windows |
| `mult` | float | Quality multiplier (0.0, 0.5, 0.75, 1.0) based on trade count |
| `penalty` | float | Under-trade penalty (0.0 to 0.25) |
| `score` | float | Final validation score (adj * mult - penalty) |
| `timestamp` | string | ISO format timestamp of validation |
| `seed` | int | Random seed used for this run |

---

## Usage

### Run Training (Generates JSONs):
```powershell
python main.py --episodes 5
```

**Output:**
- Training runs normally
- After each validation, writes JSON to `logs/validation_summaries/`
- Console still shows `[VAL]` line as before

### Check Results:
```powershell
python check_validation_diversity.py
```

**Example Output:**
```
======================================================================
VALIDATION SLICE DIVERSITY CHECK
======================================================================

Found 5 validation summaries

Ep   1: K=6 | trades= 3.0 | mult=0.00 | pen=0.214 | score=-0.214 ⚠
Ep   2: K=6 | trades= 7.5 | mult=1.00 | pen=0.000 | score=+0.145 ✓
Ep   3: K=6 | trades= 5.0 | mult=0.50 | pen=0.036 | score=+0.082 ✓
Ep   4: K=6 | trades= 9.0 | mult=1.00 | pen=0.000 | score=+0.198 ✓
Ep   5: K=6 | trades= 6.5 | mult=0.75 | pen=0.000 | score=+0.131 ✓

======================================================================
SUMMARY STATISTICS:
  Trade count range: 3.0 - 9.0
  Score range: -0.214 - +0.198
  Episodes with 0 trades: 0
  Episodes with 5+ trades: 4
======================================================================
```

### List JSON Files:
```powershell
Get-ChildItem logs\validation_summaries -Recurse -Include *.json | Select-Object FullName, Length, LastWriteTime
```

### Read Specific JSON:
```powershell
Get-Content logs\validation_summaries\val_ep003.json | ConvertFrom-Json | Format-List
```

---

## Benefits Over Log Parsing

### Before (Log Parsing):
- ❌ Regex fragile to format changes
- ❌ Unicode encoding issues
- ❌ Hard to extract multiple metrics
- ❌ Requires full log file scanning
- ❌ No structured data for tools

### After (JSON Export):
- ✅ Direct structured access to all metrics
- ✅ No encoding issues (`ensure_ascii=True`)
- ✅ Easy to extend with new fields
- ✅ Fast file-by-file processing
- ✅ Ready for analysis tools (pandas, plotting, etc.)

---

## Advanced Usage

### Load All Validations in Python:
```python
import json
from pathlib import Path

summary_dir = Path("logs/validation_summaries")
validations = []

for json_file in sorted(summary_dir.glob("val_ep*.json")):
    with open(json_file, 'r') as f:
        validations.append(json.load(f))

# Now you have a list of dicts for analysis
```

### Convert to Pandas DataFrame:
```python
import pandas as pd

df = pd.DataFrame(validations)
print(df[['episode', 'trades', 'score']].describe())

# Plot learning curve
import matplotlib.pyplot as plt
plt.plot(df['episode'], df['score'])
plt.xlabel('Episode')
plt.ylabel('Validation Score')
plt.title('Validation Learning Curve')
plt.savefig('validation_curve.png')
```

### Compare Across Seeds:
```python
# After running seed sweeps
df = pd.DataFrame(validations)
df.groupby('seed')['score'].agg(['mean', 'std', 'max'])
```

---

## Directory Structure

After running training:
```
logs/
├── validation_summaries/
│   ├── val_ep001.json
│   ├── val_ep002.json
│   ├── val_ep003.json
│   ├── val_ep004.json
│   └── val_ep005.json
├── training_curves.png
└── training_log_*.txt
```

**Note:** Each training run creates new JSON files. Files from previous runs are overwritten if episode numbers match.

---

## Optional: Config Flag

To make JSON export optional, add to `config.py`:

```python
@dataclass
class TrainingConfig:
    # ...existing fields...
    save_validation_json: bool = True  # Enable JSON export
```

Then in `trainer.py`, guard the write:

```python
if getattr(self.config.training, 'save_validation_json', True):
    # ... JSON export code ...
```

**Current Implementation:** Always enabled (recommended for debugging)

---

## Troubleshooting

### No JSON files created?
1. Check if `logs/validation_summaries/` directory exists
2. Verify training reached validation step (episode 1)
3. Check permissions on `logs/` directory

### JSON encoding errors?
- Using `ensure_ascii=True` prevents Unicode issues
- All numeric fields explicitly cast to `float()` or `int()`

### Wrong episode numbers?
- Ensure `self.episode` is set in trainer before validation
- Check if episode counter starts at 0 or 1

---

## Future Enhancements

**Potential Additions to JSON:**
- Per-window fitness values (not just median)
- Per-window trade counts (not just median)
- Computational time per validation
- Agent epsilon value at validation time
- Buffer size at validation time

**Example Extended Format:**
```json
{
  "episode": 3,
  "k": 6,
  "windows": [
    {"fitness": 0.12, "trades": 8},
    {"fitness": 0.15, "trades": 7},
    // ... more windows
  ],
  "median_fitness": 0.145,
  // ... rest of fields
}
```

---

## Testing Checklist

- [x] JSON export code added to `trainer.py`
- [x] Checker script updated to read JSON
- [x] Code compiles without errors
- [ ] Test with 5-episode run
- [ ] Verify JSON files created
- [ ] Verify checker script works
- [ ] Check file contents are valid JSON
- [ ] Verify summary statistics accurate

---

## Summary

**Files Modified:**
1. `trainer.py` - Added JSON export (27 lines)
2. `check_validation_diversity.py` - Rewritten to read JSON (50 lines)

**Benefits:**
- ✅ Structured data export for analysis
- ✅ No regex parsing fragility
- ✅ No encoding issues
- ✅ Easy to extend
- ✅ Ready for plotting/analysis tools

**Testing:**
```powershell
# Generate data
python main.py --episodes 5

# Check results
python check_validation_diversity.py

# List files
gci logs\validation_summaries -Include *.json | select Name, Length
```

**All changes are backward compatible - console output unchanged, just adds JSON export.**
