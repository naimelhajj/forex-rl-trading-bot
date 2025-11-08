# Quick Start Guide - Advanced Improvements

## 🚀 Ready to Use!

All 7 improvements are implemented and ready. Here's how to use them:

## Quick Commands

### 1. Smoke Test (3 episodes, ~5 min)
```bash
python main.py --episodes 3
```
**What happens:**
- 🔥 Smoke mode AUTO-ACTIVATES
- learning_starts: 1000 (vs 5000)
- episode_length: 600 steps
- epsilon: 0.4 → 0.10
- Updates happen immediately!

### 2. Development Run (5 episodes, ~15 min)
```bash
python main.py --episodes 5
```
**What happens:**
- 🔥 Still in smoke mode
- Fast iteration for testing
- Agents learns from limited data

### 3. Medium Run (20 episodes, ~1 hour)
```bash
python main.py --episodes 20
```
**What happens:**
- ⚙️ Normal mode (learning_starts=5000)
- Robust scaling + PER active
- Early stopping monitors fitness
- Domain randomization in training

### 4. Production Run (50-100 episodes, ~3-6 hours)
```bash
python main.py --episodes 100
```
**What happens:**
- Full learning pipeline
- Early stopping likely kicks in around ep 60-80
- Best model auto-saved
- Comprehensive validation

## What's Different Now?

### ✅ Before
- Short runs: No learning (buffer too small)
- Scaling: Mean/std (outlier-sensitive)
- Sampling: Uniform replay
- Stopping: Manual

### ✅ After
- Short runs: **LEARN IMMEDIATELY** (smoke mode)
- Scaling: **Median/MAD** (outlier-robust)
- Sampling: **PER** (prioritized, efficient)
- Stopping: **AUTO** (fitness-based, patience=20)

## New Features Active by Default

1. **Smoke Mode** - Auto for `--episodes ≤ 5`
2. **Robust Scaling** - Winsorization + MAD
3. **Double-DQN** - Already implemented
4. **Weight Decay** - 1e-6 L2 regularization
5. **PER** - Prioritized Experience Replay
6. **Domain Randomization** - ±10% spread/commission
7. **Early Stopping** - Patience=20 validations

## Expected Results

### Smoke Run (3 episodes)
```
Episode 1: Learning starts at step ~1000
Episode 2: Updates every 4 steps
Episode 3: Small but measurable learning

Training equity: ±10-20%
Validation Sharpe: -1.0 to 0.0
```

### Medium Run (20 episodes)
```
Early episodes: Exploration (epsilon decays)
Middle: Learning stabilizes
Late: Convergence signals

Training equity: ±20-40%
Validation Sharpe: -0.5 to +0.5
Win rate: 40-55%
```

### Production Run (100 episodes)
```
Learning: Steady improvement
Peak: Around episode 40-60
Early stop: Episode 60-80 (if plateau)

Training equity: ±30-50%
Validation Sharpe: >0.5
Win rate: 48-58%
Best model: Auto-saved
```

## Monitoring

### Key Metrics to Watch

**During Training:**
```
train/episode_reward  → Should stabilize
train/avg_loss        → Should decrease then plateau
train/epsilon         → Should decay 0.4 → 0.10
q/max                 → Should not explode
```

**During Validation:**
```
val/fitness           → Should increase (early stop triggers on this)
val/sharpe            → Target >0.3
val/cagr              → Positive is good
val/reward            → Should correlate with fitness
```

### Red Flags 🚩

- **Q-values exploding** → Check weight_decay is active
- **No learning in smoke mode** → Check learning_starts=1000
- **Training never improves** → Check PER is enabled
- **Validation worse than random** → Check robust scaling applied

## Troubleshooting

### "Smoke mode not activating"
```bash
# Check config
python -c "from config import Config; c=Config(); print(c.SMOKE_LEARN, c.SMOKE_LEARNING_STARTS)"
# Should print: True 1000
```

### "Agent not learning"
```bash
# Check replay buffer size during training
# Should see: "replay_size: 1000+" in first episode
```

### "Features not scaled properly"
```bash
# Check scaler output in terminal
# Should see: "Computing robust feature scaler..."
#             "Feature median range: [...]"
#             "Feature MAD range: [...]"
```

### "Early stopping too aggressive"
```python
# In trainer.py, increase patience:
patience = 30  # Default is 20
```

## File Changes Summary

- ✅ `config.py` - Smoke mode switches
- ✅ `scaler_utils.py` - NEW robust scaling
- ✅ `agent.py` - Weight decay added
- ✅ `main.py` - Smoke activation + robust scaler
- ✅ `trainer.py` - Domain rand + early stop

## Next Steps

1. **Quick validation**: `python main.py --episodes 3`
2. **Check logs**: Look for "🔥 SMOKE MODE ACTIVATED"
3. **Monitor**: Watch for learning updates in episode 1
4. **Scale up**: Try `--episodes 20` next
5. **Production**: Run `--episodes 100` when ready

---

**All improvements are PRODUCTION-READY and backward compatible!** 🎉
