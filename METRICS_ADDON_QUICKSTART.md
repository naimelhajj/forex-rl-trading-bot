# Metrics Add-On Quick Reference Card

## 🎯 What It Does

Adds **6 new diagnostic metrics** to every validation JSON:
- `action_entropy_bits` - Policy diversity (Shannon entropy)
- `hold_streak_max` - Longest HOLD sequence
- `hold_streak_mean` / `avg_hold_length` - Average HOLD duration
- `switch_rate` - Action transition frequency
- `long_short` - Directional bias (long/short counts & ratios)

## 📊 Healthy Ranges (Hourly Data)

| Metric | Healthy | Warning Low | Warning High |
|--------|---------|-------------|--------------|
| `action_entropy_bits` | 1.5-2.0 | < 1.0 (collapse) | > 2.5 (random) |
| `hold_rate` | 0.70-0.80 | < 0.50 (churn) | > 0.90 (freeze) |
| `hold_streak_max` | 20-60 | < 10 (churn) | > 100 (freeze) |
| `avg_hold_length` | 10-40 | < 5 (churn) | > 60 (inactive) |
| `switch_rate` | 0.05-0.20 | < 0.02 (stuck) | > 0.30 (thrash) |
| `long_ratio` | 0.40-0.60 | - | > 0.70 (bias) |
| `short_ratio` | 0.40-0.60 | - | > 0.70 (bias) |

## 🚀 Quick Start

```powershell
# Run training
python main.py --episodes 10

# View new metrics
python check_metrics_addon.py
```

## 📈 Example Output

```
Found 10 validation summaries

First 5 episodes:
  Ep   1: score=-0.125 | trades=  18 | hold_rate=0.750 | H(avg,max)=(12.50,45) | Hbits=1.850 | switch=0.085 | L/S=(0.520/0.480)
  Ep   2: score=-0.082 | trades=  21 | hold_rate=0.720 | H(avg,max)=(15.30,52) | Hbits=1.920 | switch=0.095 | L/S=(0.485/0.515)
  ...

Averages:
  hold_rate:       0.715
  avg_hold_length: 13.66
  max_hold_streak: 52
  action_entropy:  1.980 bits  ✅ HEALTHY
  switch_rate:     0.104       ✅ HEALTHY
  long_ratio:      0.496       ✅ BALANCED
```

## 🔧 Quick Fixes

**Low Entropy (< 1.0)?**
→ Increase `eval_epsilon` from 0.03 to 0.05

**Long Streaks (> 100)?**
→ Lower `hold_break_after` from 12 to 8

**High Switch Rate (> 0.30)?**
→ Increase `min_hold_bars` from 5 to 8

**Long/Short Bias (> 70%)?**
→ Check if data is directional (expected)
→ If not: verify training diversity

## 📝 Files Created

1. `check_metrics_addon.py` - View metrics from existing runs
2. `augment_existing_json_metrics.py` - Backfill old JSONs (partial)
3. `METRICS_ADDON_SUMMARY.md` - Full documentation

## ✅ Status

**All integrated in `trainer.py`**
- ✅ Compiles without errors
- ✅ Backward compatible
- ✅ Zero config changes needed
- ✅ Works with existing anti-collapse patches
- ✅ Ready for immediate use

---

**That's it! Just run training and check metrics.** 🎉
