"""
Minimal test to verify state_size attribute exists.
"""
import pandas as pd
import numpy as np
import sys

print("Testing state_size attribute fix...")

from environment import ForexTradingEnv

# Create minimal test data
data = pd.DataFrame({
    'open': [1.0] * 50,
    'high': [1.01] * 50,
    'low': [0.99] * 50,
    'close': [1.0] * 50,
    'atr': [0.001] * 50,
    'rsi': [50.0] * 50,
})

feature_columns = ['close', 'atr', 'rsi']

print(f"Creating environment with {len(feature_columns)} features...")
env = ForexTradingEnv(
    data=data,
    feature_columns=feature_columns,
    initial_balance=1000
)

print(f"✓ Environment created")
print(f"  Feature dim: {env.feature_dim}")
print(f"  Stack N: {env.stack_n}")
print(f"  Context dim: {env.context_dim}")
print(f"  State size: {env.state_size}")

expected_size = len(feature_columns) * 3 + 23
print(f"  Expected: {expected_size}")

assert hasattr(env, 'state_size'), "state_size attribute missing!"
assert env.state_size == expected_size, f"State size mismatch: {env.state_size} != {expected_size}"

state = env.reset()
print(f"✓ Reset successful, state shape: {state.shape}")
assert state.shape[0] == env.state_size, f"State shape mismatch: {state.shape[0]} != {env.state_size}"

print("\n✅ ALL CHECKS PASSED!")
print(f"State size attribute is working correctly: {env.state_size}")
