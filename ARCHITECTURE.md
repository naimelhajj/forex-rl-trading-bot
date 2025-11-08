# Forex RL Trading Bot Architecture

## Overview
A DQN-based reinforcement learning system for Forex trading with comprehensive risk management and stability-adjusted fitness metrics.

## System Components

### 1. Feature Engineering (`features.py`)
- **OHLC data processing**
- **Technical indicators**: ATR, RSI
- **Percentile calculations**: Short/medium/long percentile values
- **Currency strength**: Multi-pair strength calculation with lagged features
- **Temporal features**: Time of day, day of week, day of year
- **Fractal detection**: Top and bottom fractal values
- **Linear regression**: Slope calculation for HLC prices

### 2. DQN Agent (`agent.py`)
- **Neural network architecture**: Deep Q-Network
- **Experience replay buffer**
- **Target network with periodic updates**
- **Epsilon-greedy exploration**
- **Action space**: HOLD, LONG, SHORT, MOVE_SL_CLOSER

### 3. Trading Environment (`environment.py`)
- **Gym-compatible environment**
- **State management**: Current position, equity, features
- **Action execution**: Position opening/closing, SL/TP management
- **Reward calculation**: Based on PnL and risk metrics
- **Weekend/holiday handling**: Flatten positions 3+ hours before closure
- **Single position constraint**: Only one trade at a time

### 4. Risk Management (`risk_manager.py`)
- **Position sizing**: Risk-based lot calculation
- **Margin management**: Leverage and free margin constraints
- **Drawdown survivability**: Post-DD margin requirements
- **Stop loss and take profit**: Automatic SL/TP on each trade
- **Broker constraint compliance**: Volume steps, min/max lots

### 5. Fitness Metric (`fitness.py`)
- **Sharpe ratio**: Risk-adjusted returns
- **CAGR**: Compound annual growth rate
- **Stagnation penalty**: Time below equity peak
- **Loss year penalty**: Negative yearly returns count
- **Ruin clamp**: Terminal penalty for account blow-up
- **Configurable weights**: Customizable metric components

### 6. Data Pipeline (`data_loader.py`)
- **Historical data loading**
- **Multi-pair data synchronization**
- **Currency strength calculation across pairs**
- **Data preprocessing and normalization**

### 7. Training System (`trainer.py`)
- **Episode management**
- **Model checkpointing**
- **Performance tracking**
- **Hyperparameter configuration**

### 8. Live Trading Interface (`live_trader.py`)
- **MT5 integration** (optional)
- **Real-time feature calculation**
- **Order execution**
- **Position monitoring**

## Data Flow

```
Historical Data → Feature Engineering → State Representation
                                              ↓
                                         DQN Agent
                                              ↓
                                         Action Selection
                                              ↓
                                    Trading Environment
                                              ↓
                                    Risk Manager → Order Execution
                                              ↓
                                    Reward Calculation
                                              ↓
                                    Experience Replay Buffer
                                              ↓
                                    Agent Training (backprop)
```

## Key Design Decisions

1. **Single position constraint**: Simplifies risk management and prevents over-exposure
2. **Mandatory SL/TP**: Every trade starts with defined risk parameters
3. **Weekend/holiday protection**: Automatic position flattening before market closure
4. **Extensible architecture**: Modular design for easy feature/strategy additions
5. **Stability-focused fitness**: Prioritizes consistent returns over raw profit

## Configuration

All hyperparameters, risk parameters, and fitness weights are configurable via a central config file.

