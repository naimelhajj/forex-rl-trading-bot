# Quick Start: Seed Sweep in 20 Seconds

## Option 1: Simple (Fastest)
```powershell
python run_seed_sweep_simple.py
```
- **Time**: ~30-45 minutes
- **Seeds**: 7, 77, 777
- **Episodes**: 25 per seed
- **Note**: Overwrites JSON files between seeds

---

## Option 2: Organized (Recommended)
```powershell
python run_seed_sweep_organized.py
```
- **Time**: ~30-45 minutes  
- **Seeds**: 7, 77, 777
- **Episodes**: 25 per seed
- **Saves**: Separate directory per seed
- **Then run**: `python compare_seed_results.py`

---

## What You Get

### During Run:
- Console output showing progress
- Each episode displays: `[VAL] K=6 | trades=14.0 | score=+0.120`

### After Run (Organized Method):
```
logs/seed_sweep_results/
├── seed_7/    (25 JSON files)
├── seed_77/   (25 JSON files)
└── seed_777/  (25 JSON files)
```

### After Comparison:
```
Seed     Episodes   Score Mean   Score Final  Trades
7        25         +0.142       +0.185       13.5
77       25         +0.138       +0.172       14.2
777      25         +0.151       +0.198       13.8

Consistency: ✓ CONSISTENT (learning is stable)
```

---

## Files You Have

| File | Purpose |
|------|---------|
| `run_seed_sweep_simple.py` | Basic sweep, overwrites files |
| `run_seed_sweep_organized.py` | Saves results per seed ✓ |
| `compare_seed_results.py` | Analyzes across seeds |
| `SEED_SWEEP_GUIDE.md` | Full documentation |

---

## Customize

Edit `run_seed_sweep_organized.py` lines 10-11:

```python
SEEDS = [7, 77, 777]    # Change seeds
EPISODES = 25           # Change episode count
```

---

## That's It!

**Just run and wait:**
```powershell
python run_seed_sweep_organized.py
```

**Then analyze:**
```powershell
python compare_seed_results.py
```

**Total time:** ~30-45 minutes for 3 seeds × 25 episodes

See `SEED_SWEEP_GUIDE.md` for full details.
