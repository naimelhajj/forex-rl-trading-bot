"""Minimal test to find the hang"""
import sys
import numpy as np
from config import Config
from agent import DQNAgent
from environment import ForexTradingEnv
import pandas as pd

print("Creating config...")
config = Config()
config.agent.use_noisy = True
config.agent.noisy_sigma_init = 0.4

print("Creating minimal data...")
dates = pd.date_range('2023-01-01', periods=100, freq='h')
data = pd.DataFrame({
    'time': dates,
    'open': 1.08 + np.random.randn(100) * 0.001,
    'high': 1.08 + np.random.randn(100) * 0.001 + 0.0005,
    'low': 1.08 + np.random.randn(100) * 0.001 - 0.0005,
    'close': 1.08 + np.random.randn(100) * 0.001,
    'atr': 0.0015,
    'rsi': 50.0,
})
data = data.set_index('time')

# Minimal features
feature_cols = ['open', 'high', 'low', 'close', 'atr', 'rsi']

print("Creating environment...")
env = ForexTradingEnv(
    data=data,
    feature_columns=feature_cols,
    initial_balance=1000.0,
    max_steps=50
)

print(f"State size: {env.state_size}")

print("Creating agent...")
agent = DQNAgent(
    state_size=env.state_size,
    action_size=4,
    use_noisy=True,
    noisy_sigma_init=0.4,
    buffer_type='simple',
    buffer_capacity=1000,
    n_step=3,
    learning_starts=10,
    update_every=4,
    grad_steps=1
)

print("Running episode...")
state = env.reset()
done = False
step = 0

while not done and step < 20:
    print(f"Step {step}: buffer_size={len(agent.replay_buffer)}", end='')
    
    action = agent.select_action(state, explore=True)
    next_state, reward, done, info = env.step(action)
    
    agent.store_transition(state, action, reward, next_state, done)
    print(f" -> {len(agent.replay_buffer)} (after push)")
    
    # Try training
    if len(agent.replay_buffer) >= agent.replay_batch_size:
        print(f"  Training...")
        loss = agent.train_step()
        print(f"  Loss: {loss}")
    
    state = next_state
    step += 1

print(f"\nCompleted {step} steps!")
print(f"Final buffer size: {len(agent.replay_buffer)}")
print("✅ Test passed - no hang!")
