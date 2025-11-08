# Forex RL Trading Bot

A comprehensive Deep Q-Network (DQN) based reinforcement learning system for Forex trading with advanced risk management and stability-adjusted fitness metrics.

## Features

### Core Capabilities

**Advanced Feature Engineering**
- OHLC price data processing
- Technical indicators: ATR, RSI
- Short/medium/long-term percentile calculations
- Multi-currency strength analysis with lagged features
- Temporal features: hour of day, day of week, day of year
- Fractal detection (top and bottom fractals)
- Linear regression slope analysis

**DQN Agent**
- Deep Q-Network with experience replay
- Target network for stable learning
- Epsilon-greedy exploration strategy
- Configurable network architecture
- GPU/CPU support

**Trading Environment**
- Gym-compatible interface
- Single position constraint (one trade at a time)
- Automatic position switching (long ↔ short)
- Weekend/holiday position flattening
- Realistic spread and commission modeling

**Risk Management**
- Risk-based position sizing (default 2% per trade)
- Margin constraint enforcement
- Drawdown survivability checks
- Automatic SL/TP on every trade
- Adjustable stop loss functionality
- Broker constraint compliance (volume steps, min/max lots)

**Fitness Metric**
- Stability-adjusted fitness score combining:
  - Sharpe ratio (risk-adjusted returns)
  - CAGR (compound annual growth rate)
  - Stagnation penalty (time below equity peak)
  - Loss year penalty (negative yearly returns)
  - Ruin clamp (account blow-up protection)
- Configurable component weights

### Action Space

The agent can take four actions:
- **HOLD**: Maintain current position or stay flat
- **LONG**: Open long position (closes short if exists)
- **SHORT**: Open short position (closes long if exists)
- **MOVE_SL_CLOSER**: Tighten stop loss to lock in profits

## Installation

### Requirements

- Python 3.11+
- PyTorch 2.0+
- NumPy, Pandas, SciPy
- Matplotlib (for visualization)

### Setup

```bash
# Clone or download the project
cd forex_rl_bot

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

Train the agent on historical data:

```bash
# Train with default settings (500 episodes)
python main.py --mode train

# Train with custom number of episodes
python main.py --mode train --episodes 100

# Train and evaluate
python main.py --mode both
```

### Evaluation

Evaluate a trained model:

```bash
# Evaluate using best checkpoint
python main.py --mode evaluate --checkpoint checkpoints/best_model.pt

# Evaluate final model
python main.py --mode evaluate --checkpoint checkpoints/final_model.pt
```

### Configuration

All hyperparameters can be customized in `config.py`.

## Project Structure

```
forex_rl_bot/
├── agent.py              # DQN agent implementation
├── config.py             # Configuration management
├── data_loader.py        # Data loading and preprocessing
├── environment.py        # Trading environment (Gym-compatible)
├── features.py           # Feature engineering
├── fitness.py            # Fitness metric calculation
├── risk_manager.py       # Position sizing and risk management
├── trainer.py            # Training loop management
├── main.py               # Main entry point
├── requirements.txt      # Python dependencies
├── ARCHITECTURE.md       # System architecture documentation
└── README.md             # This file
```

## Position Sizing

The system uses a sophisticated position sizing algorithm that considers:

1. **Risk-based sizing**: Maximum loss per trade (default 2% of balance)
2. **Margin constraints**: Available free margin and leverage
3. **Drawdown survivability**: Ensures sufficient margin after maximum drawdown
4. **Broker constraints**: Volume steps, minimum/maximum lots

## Fitness Metric

The stability-adjusted fitness score is calculated as:

```
Fitness = w₁×Sharpe + w₂×CAGR - w₃×Stagnation - w₄×LossYears - [5 if ruined]
```

**Default weights:** Sharpe=1.0, CAGR=2.0, Stagnation=1.0, LossYears=1.0, Ruin=5.0

## Disclaimer

**This is an educational project for learning reinforcement learning and algorithmic trading concepts.**

- Past performance does not guarantee future results
- Trading Forex involves substantial risk of loss
- This system has not been tested with real money
- Always test thoroughly before live deployment
- Use proper risk management
- Consult with financial professionals before trading

## License

This project is provided as-is for educational purposes.

