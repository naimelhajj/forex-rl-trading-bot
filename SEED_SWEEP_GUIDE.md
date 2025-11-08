# Seed Sweep Guide - 3 Seeds × 25 Episodes

**Purpose:** Test if your RL agent learns consistently across different random seeds  
**Configuration:** 3 seeds × 25 episodes = 75 total episodes  
**Estimated Time:** ~30-45 minutes

---

## 🚀 Quick Start

### Method 1: Simple (Fastest)
```powershell
python run_seed_sweep_simple.py
```
- Runs 3 seeds × 25 episodes sequentially
- Output goes to console
- **Warning:** Each seed overwrites previous JSON files

### Method 2: Organized (Recommended)
```powershell
python run_seed_sweep_organized.py
```
- Runs 3 seeds × 25 episodes sequentially
- **Automatically saves each seed's results separately**
- Creates: `logs/seed_sweep_results/seed_7/`, `seed_77/`, `seed_777/`
- Then compare:
```powershell
python compare_seed_results.py
```

---

## What Happens

### Seeds Tested:
- **Seed 7**: First run
- **Seed 77**: Second run
- **Seed 777**: Third run

### Per Seed:
- 25 training episodes
- 25 validation passes (one per episode)
- 25 JSON files with validation metrics

### Total:
- 75 episodes
- 75 validation passes
- 75 JSON files (if using organized script)

---

## Expected Output

### During Run:
```
============================================================
FOREX RL BOT - SEED SWEEP (Auto-Organized)
============================================================
Seeds: [7, 77, 777]
Episodes per seed: 25
Results will be saved to: logs/seed_sweep_results

============================================================
SEED 1/3: 7
============================================================

[OK] Updated config.py: random_seed = 7
[RUN] Starting training with seed 7...

[Episode 1/25] ...
[VAL] K=6 | trades=14.0 | score=+0.120
...
[Episode 25/25] ...
[VAL] K=6 | trades=16.5 | score=+0.185

[OK] Seed 7 completed successfully
[OK] Saved 25 JSON files to logs\seed_sweep_results\seed_7

============================================================
SEED 2/3: 77
============================================================
...
```

### After Comparison:
```
======================================================================
SEED SWEEP COMPARISON
======================================================================

SUMMARY BY SEED
----------------------------------------------------------------------
Seed     Episodes   Score Mean   Score Final  Score Best   Trades    
----------------------------------------------------------------------
7        25         +0.142       +0.185       +0.230       13.5
77       25         +0.138       +0.172       +0.245       14.2
777      25         +0.151       +0.198       +0.268       13.8

======================================================================
CROSS-SEED COMPARISON
======================================================================

Score Mean across seeds:    +0.144 ± 0.007
Score Final across seeds:   +0.185 ± 0.013
Best seed (by mean):        777
Best seed (by final):       777

Consistency:
  ✓ CONSISTENT - Seed variation (0.007) < within-seed variation (0.042)
    → Learning is stable across seeds
```

---

## Analyzing Results

### 1. Check File Organization:
```powershell
# List all seed directories
Get-ChildItem logs\seed_sweep_results

# Count files per seed
Get-ChildItem logs\seed_sweep_results\seed_7\*.json | Measure-Object
Get-ChildItem logs\seed_sweep_results\seed_77\*.json | Measure-Object
Get-ChildItem logs\seed_sweep_results\seed_777\*.json | Measure-Object
```

### 2. View Individual Seed:
```powershell
# Seed 7
Get-ChildItem logs\seed_sweep_results\seed_7\*.json | Select-Object Name | Format-Table

# View specific episode
Get-Content logs\seed_sweep_results\seed_7\val_ep010.json | ConvertFrom-Json | Format-List
```

### 3. Compare All Seeds:
```powershell
python compare_seed_results.py
```

---

## What to Look For

### ✅ Good Signs:
- **Consistent learning**: All seeds show improving scores
- **Similar final scores**: Within ~20% across seeds
- **Healthy trades**: 10-20 trades per validation
- **Low penalties**: < 20% of validations penalized
- **Score progression**: Negative → Positive over episodes

### ⚠️ Warning Signs:
- **Divergent seeds**: One succeeds, others fail
- **Zero trades**: Stuck in HOLD mode
- **High penalties**: > 50% validations penalized
- **No improvement**: Flat or declining scores
- **Excessive variance**: One seed much better/worse

---

## Customization

### Change Seeds:
Edit `run_seed_sweep_organized.py` line 10:
```python
SEEDS = [42, 123, 999]  # Your choice
```

### Change Episodes:
Edit line 11:
```python
EPISODES = 50  # More episodes = better curves
```

### Different Combinations:
```python
# Quick test (5 min)
SEEDS = [7, 77]
EPISODES = 10

# Standard test (30 min)
SEEDS = [7, 77, 777]
EPISODES = 25

# Thorough test (90 min)
SEEDS = [7, 77, 777, 7777]
EPISODES = 50
```

---

## Script Differences

| Feature | `run_seed_sweep_simple.py` | `run_seed_sweep_organized.py` |
|---------|----------------------------|-------------------------------|
| **Run command** | `python run_seed_sweep_simple.py` | `python run_seed_sweep_organized.py` |
| **Saves results** | ❌ Overwrites | ✓ Separate dirs per seed |
| **Compare script** | ❌ Manual | ✓ `compare_seed_results.py` |
| **Use case** | Quick check | Production analysis |

---

## Troubleshooting

### Script Stops Unexpectedly:
```powershell
# Check if Python still running
Get-Process python -ErrorAction SilentlyContinue

# Check last log file
Get-ChildItem logs -Filter "training_log*.txt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content -Tail 50
```

### Missing JSON Files:
```powershell
# Verify directory exists
Test-Path logs\seed_sweep_results\seed_7

# List all files
Get-ChildItem logs\seed_sweep_results -Recurse -Include *.json
```

### Memory Issues:
Reduce buffer size in `config.py`:
```python
buffer_capacity: int = 50000  # was 100000
batch_size: int = 128         # was 256
```

---

## Advanced: Plot Learning Curves

Create `plot_seed_curves.py`:
```python
import json
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_BASE = Path("logs/seed_sweep_results")
SEEDS = [7, 77, 777]

plt.figure(figsize=(12, 6))

for seed in SEEDS:
    seed_dir = RESULTS_BASE / f"seed_{seed}"
    json_files = sorted(seed_dir.glob("val_ep*.json"))
    
    episodes = []
    scores = []
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            episodes.append(data['episode'])
            scores.append(data['score'])
    
    plt.plot(episodes, scores, marker='o', label=f"Seed {seed}", alpha=0.7)

plt.xlabel('Episode')
plt.ylabel('Validation Score')
plt.title('Learning Curves - Seed Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.savefig('logs/seed_comparison.png', dpi=150, bbox_inches='tight')
print('✓ Saved: logs/seed_comparison.png')
```

Then run:
```powershell
python plot_seed_curves.py
```

---

## Summary

### Recommended Workflow:
```powershell
# 1. Run sweep (30-45 min)
python run_seed_sweep_organized.py

# 2. Compare results
python compare_seed_results.py

# 3. Optional: Plot curves
python plot_seed_curves.py
```

### Files Created:
```
logs/
└── seed_sweep_results/
    ├── seed_7/
    │   ├── val_ep001.json
    │   ├── val_ep002.json
    │   └── ... (25 files)
    ├── seed_77/
    │   └── ... (25 files)
    └── seed_777/
        └── ... (25 files)
```

### Key Metrics:
- **Score progression**: Should improve over episodes
- **Cross-seed consistency**: Std dev < mean
- **Trade activity**: 10-20 trades per validation
- **Penalty rate**: < 20% of validations

**Ready to run!** 🚀
