# Usage Examples

## Basic Training

### Train with default settings

```bash
python main.py --mode train
```

This will:
- Generate 10,000 bars of synthetic data for 4 currency pairs
- Train for 500 episodes
- Validate every 10 episodes
- Save checkpoints every 50 episodes
- Save the best model based on validation fitness

### Quick test with fewer episodes

```bash
python main.py --mode train --episodes 50
```

## Evaluation

### Evaluate the best model

```bash
python main.py --mode evaluate --checkpoint checkpoints/best_model.pt
```

### Train and evaluate in one run

```bash
python main.py --mode both --episodes 100
```

## Custom Configuration

### Example 1: Conservative Trading

```python
# Create a file: train_conservative.py
from config import Config
from main import *

# Create custom config
config = Config()

# Conservative risk settings
config.risk.risk_per_trade = 0.01  # 1% risk per trade
config.risk.leverage = 50
config.risk.tp_multiplier = 2.0  # 2:1 reward:risk

# Lower exploration
config.agent.epsilon_decay = 0.99

# Train
set_random_seeds(config.random_seed)
train_data, val_data, test_data, feature_columns = prepare_data(config)
train_env, val_env, test_env = create_environments(
    train_data, val_data, test_data, feature_columns, config
)
agent = create_agent(train_env.state_size, config)
history = train_agent(agent, train_env, val_env, config)
```

### Example 2: Aggressive Trading

```python
# Create a file: train_aggressive.py
from config import Config
from main import *

config = Config()

# Aggressive risk settings
config.risk.risk_per_trade = 0.05  # 5% risk per trade
config.risk.leverage = 200
config.risk.tp_multiplier = 5.0  # 5:1 reward:risk

# More exploration
config.agent.epsilon_start = 1.0
config.agent.epsilon_end = 0.05
config.agent.epsilon_decay = 0.995

# Train
# ... (same as above)
```

### Example 3: Custom Fitness Weights

```python
config = Config()

# Prioritize Sharpe ratio and minimize drawdown
config.fitness.sharpe_weight = 3.0
config.fitness.cagr_weight = 1.0
config.fitness.stagnation_weight = 2.0
config.fitness.loss_years_weight = 1.5
```

## Using Real Data

### Load data from CSV files

```python
from data_loader import DataLoader
from features import FeatureEngineer

# Initialize loader
loader = DataLoader(data_dir="./my_data")

# Load multiple pairs
pair_files = {
    'EURUSD': './my_data/EURUSD_H1.csv',
    'GBPUSD': './my_data/GBPUSD_H1.csv',
    'USDJPY': './my_data/USDJPY_H1.csv',
}

multi_pair_data = loader.load_multiple_pairs(pair_files)

# Compute features
fe = FeatureEngineer()
primary_data = multi_pair_data['EURUSD']
data_with_features = fe.compute_all_features(primary_data, multi_pair_data)

# Continue with training...
```

Expected CSV format:
```
time,open,high,low,close,volume
2023-01-01 00:00:00,1.1000,1.1010,1.0990,1.1005,1000
2023-01-01 01:00:00,1.1005,1.1015,1.0995,1.1010,1200
...
```

## Testing Individual Components

### Test Feature Engineering

```python
from features import FeatureEngineer
import pandas as pd
import numpy as np

# Create sample data
dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
data = pd.DataFrame({
    'time': dates,
    'open': np.random.randn(1000).cumsum() + 1.1000,
    'high': np.random.randn(1000).cumsum() + 1.1010,
    'low': np.random.randn(1000).cumsum() + 1.0990,
    'close': np.random.randn(1000).cumsum() + 1.1000,
})

# Compute features
fe = FeatureEngineer()
features_df = fe.compute_all_features(data)

print(features_df.columns)
print(features_df.tail())
```

### Test Risk Manager

```python
from risk_manager import RiskManager

rm = RiskManager(
    contract_size=100000,
    point=0.00001,
    leverage=100,
    risk_per_trade=0.02
)

# Calculate position size
sizing = rm.calculate_position_size(
    balance=1000.0,
    free_margin=900.0,
    price=1.1000,
    atr=0.0015
)

print(f"Position size: {sizing['lots']} lots")
print(f"Stop loss: {sizing['sl_pips']} pips")
print(f"Take profit: {sizing['tp_pips']} pips")
```

### Test Agent

```python
from agent import DQNAgent
import numpy as np

# Create agent
agent = DQNAgent(state_size=30, action_size=4)

# Test action selection
state = np.random.randn(30)
action = agent.select_action(state)
print(f"Selected action: {action}")

# Get Q-values
q_values = agent.get_q_values(state)
print(f"Q-values: {q_values}")
```

## Analyzing Results

### Load and analyze training history

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load training history
with open('logs/training_history_YYYYMMDD_HHMMSS.json', 'r') as f:
    history = json.load(f)

df = pd.DataFrame(history)

# Plot equity progression
plt.figure(figsize=(12, 6))
plt.plot(df['episode'], df['final_equity'])
plt.xlabel('Episode')
plt.ylabel('Final Equity ($)')
plt.title('Equity Progression During Training')
plt.grid(True)
plt.show()

# Plot win rate
plt.figure(figsize=(12, 6))
plt.plot(df['episode'], df['win_rate'])
plt.xlabel('Episode')
plt.ylabel('Win Rate')
plt.title('Win Rate Progression')
plt.grid(True)
plt.show()
```

### Compare multiple training runs

```python
from fitness import compare_strategies
import pandas as pd

# Load equity curves from different runs
equity1 = pd.Series(...)  # Load from run 1
equity2 = pd.Series(...)  # Load from run 2
equity3 = pd.Series(...)  # Load from run 3

# Compare
comparison = compare_strategies({
    'Conservative': equity1,
    'Balanced': equity2,
    'Aggressive': equity3,
})

print(comparison)
```

## Advanced Usage

### Custom Reward Function

Edit `environment.py` to modify the reward calculation:

```python
def step(self, action: int):
    # ... existing code ...
    
    # Custom reward calculation
    if pnl > 0:
        reward = pnl * 2  # Amplify positive rewards
    else:
        reward = pnl * 0.5  # Reduce negative penalties
    
    # Add bonus for high win rate
    if len(self.trade_history) >= 10:
        recent_trades = self.trade_history[-10:]
        recent_wins = sum(1 for t in recent_trades if t['pnl'] > 0)
        if recent_wins >= 7:
            reward += 10  # Bonus for consistency
    
    # ... rest of code ...
```

### Multi-Pair Trading

To trade multiple pairs simultaneously, you would need to:

1. Modify the environment to handle multiple positions
2. Expand the state space to include all pairs
3. Expand the action space for pair selection
4. Update risk management for portfolio-level constraints

### Hyperparameter Optimization

```python
from config import Config
import optuna

def objective(trial):
    config = Config()
    
    # Suggest hyperparameters
    config.agent.learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    config.agent.gamma = trial.suggest_uniform('gamma', 0.95, 0.99)
    config.risk.risk_per_trade = trial.suggest_uniform('risk', 0.01, 0.05)
    
    # Train and evaluate
    # ... (training code) ...
    
    # Return fitness score
    return final_fitness

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best parameters: {study.best_params}")
```

## Tips and Best Practices

1. **Start small**: Test with 50-100 episodes before full training
2. **Monitor validation**: Watch validation fitness to detect overfitting
3. **Save frequently**: Use `--save_every` to avoid losing progress
4. **Adjust exploration**: If agent doesn't explore enough, increase epsilon_end
5. **Check trade statistics**: Low trade count may indicate overly conservative agent
6. **Review equity curves**: Look for smooth, consistent growth
7. **Test on unseen data**: Always evaluate on test set before deployment

