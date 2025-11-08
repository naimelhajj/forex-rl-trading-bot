# Quick Start Guide

## Installation

```bash
# Navigate to project directory
cd forex_rl_bot

# Install dependencies
pip install -r requirements.txt
```

## Verify Installation

```bash
# Run system test (takes ~10 seconds)
python test_system.py
```

You should see all tests pass with a ✓ symbol.

## Your First Training Run

```bash
# Train for 50 episodes (takes ~5-10 minutes)
python main.py --mode train --episodes 50
```

This will:
- Generate synthetic training data
- Train a DQN agent
- Save checkpoints to `./checkpoints/`
- Save training logs to `./logs/`
- Create training curve plots

## Evaluate the Model

```bash
# Evaluate the best model on test data
python main.py --mode evaluate --checkpoint checkpoints/best_model.pt
```

Results will be saved to `./results/test_results.json`

## Understanding the Output

### During Training

```
Episode 10/50
  Train - Reward: 45.23, Equity: $1045.23, Trades: 12, Win Rate: 58.33%
  Val   - Reward: 38.15, Equity: $1038.15, Fitness: 0.8234
```

- **Reward**: Total reward accumulated in the episode
- **Equity**: Final account balance
- **Trades**: Number of trades executed
- **Win Rate**: Percentage of profitable trades
- **Fitness**: Stability-adjusted performance score (higher is better)

### After Evaluation

```
Test Results:
  Final Equity: $1125.50
  Return: 12.55%
  Total Trades: 45
  Win Rate: 62.22%
  Profit Factor: 1.85
  Fitness Score: 1.2345
  Sharpe Ratio: 1.4567
  CAGR: 15.23%
  Max Drawdown: 8.45%
```

## Next Steps

1. **Read the documentation**:
   - `README.md` - Full documentation
   - `ARCHITECTURE.md` - System design
   - `EXAMPLES.md` - Usage examples

2. **Customize configuration**:
   - Edit `config.py` to adjust hyperparameters
   - Modify risk settings, learning rates, etc.

3. **Use your own data**:
   - See `EXAMPLES.md` for loading CSV files
   - Format: time, open, high, low, close, volume

4. **Experiment**:
   - Try different risk levels
   - Adjust fitness weights
   - Modify network architecture

## Common Issues

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### Training is slow
- Reduce number of episodes: `--episodes 20`
- Use GPU if available (automatic)
- Reduce data size in `config.py`

### Low win rate
- Increase training episodes
- Adjust exploration (epsilon_decay in config)
- Check if spread/commission is too high

### Agent not trading
- Agent may be too conservative
- Try increasing risk_per_trade
- Check if features are normalized properly

## Getting Help

- Check module docstrings for API details
- Run individual module tests: `python features.py`
- Review `EXAMPLES.md` for code samples
- Inspect training logs in `./logs/`

## Important Notes

⚠️ **This is for educational purposes only**
- Not tested with real money
- Past performance ≠ future results
- Trading involves substantial risk
- Always use proper risk management

## File Structure After Training

```
forex_rl_bot/
├── checkpoints/
│   ├── best_model.pt          # Best model by fitness
│   ├── final_model.pt         # Final model
│   └── checkpoint_ep50.pt     # Periodic checkpoint
├── logs/
│   ├── training_history_*.json
│   ├── validation_history_*.json
│   └── training_curves.png
└── results/
    └── test_results.json
```

## Support

For detailed information, see:
- Full documentation: `README.md`
- Usage examples: `EXAMPLES.md`
- System architecture: `ARCHITECTURE.md`

