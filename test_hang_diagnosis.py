"""
Minimal diagnostic to find where training hangs.
"""
import sys
import time
import numpy as np
import pandas as pd
import torch

print("=" * 60)
print("HANG DIAGNOSIS TEST")
print("=" * 60)

# Test 1: Import all modules
print("\n[1/6] Testing imports...")
try:
    from config import Config
    from data_loader import DataLoader
    from features import FeatureEngineer
    from environment import TradingEnv
    from agent import DQNAgent
    from trainer import Trainer
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load config
print("\n[2/6] Loading config...")
try:
    config = Config()
    print(f"✓ Config loaded (episodes=5 override)")
except Exception as e:
    print(f"✗ Config failed: {e}")
    sys.exit(1)

# Test 3: Load data
print("\n[3/6] Loading data...")
try:
    start = time.time()
    loader = DataLoader(config.data_dir, config.pair)
    train_df, val_df, test_df = loader.load_split_data(
        train_size=config.train_size,
        val_size=config.val_size
    )
    elapsed = time.time() - start
    print(f"✓ Data loaded in {elapsed:.2f}s")
    print(f"  Train: {len(train_df)} bars, Val: {len(val_df)} bars")
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    sys.exit(1)

# Test 4: Engineer features
print("\n[4/6] Engineering features...")
try:
    start = time.time()
    engineer = FeatureEngineer()
    train_df_feat = engineer.compute_all_features(train_df)
    elapsed = time.time() - start
    print(f"✓ Features computed in {elapsed:.2f}s")
    print(f"  Feature count: {len(train_df_feat.columns)}")
    if elapsed > 2.0:
        print(f"  ⚠ WARNING: Feature computation took {elapsed:.2f}s (expected <1s)")
except Exception as e:
    print(f"✗ Feature engineering failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Create environment
print("\n[5/6] Creating environment...")
try:
    start = time.time()
    env = TradingEnv(
        df=train_df_feat,
        initial_balance=config.initial_balance,
        max_position_size=config.max_position_size,
        leverage=config.leverage,
        spread=config.spread,
        commission=config.commission
    )
    elapsed = time.time() - start
    print(f"✓ Environment created in {elapsed:.2f}s")
except Exception as e:
    print(f"✗ Environment creation failed: {e}")
    sys.exit(1)

# Test 6: Create agent and test reset_noise
print("\n[6/6] Creating agent and testing NoisyNet...")
try:
    start = time.time()
    agent = DQNAgent(
        state_dim=env.state_size,
        action_dim=env.action_space,
        config=config
    )
    elapsed = time.time() - start
    print(f"✓ Agent created in {elapsed:.2f}s")
    
    # Test reset_noise (should be instant, not hang)
    print("  Testing reset_noise() 100 times...")
    start = time.time()
    for _ in range(100):
        agent.reset_noise()
    elapsed = time.time() - start
    print(f"✓ 100 reset_noise() calls in {elapsed:.4f}s")
    if elapsed > 1.0:
        print(f"  ⚠ WARNING: reset_noise is slow ({elapsed:.4f}s for 100 calls)")
except Exception as e:
    print(f"✗ Agent creation/testing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Single episode step
print("\n[7/7] Testing single episode step...")
try:
    state = env.reset()
    print(f"✓ Environment reset, state shape: {state.shape}")
    
    # Take 10 steps
    for step in range(10):
        mask = env.legal_action_mask()
        action = agent.select_action(state, explore=True, mask=mask)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
        print(f"  Step {step+1}: action={action}, reward={reward:.4f}, done={done}")
    
    print(f"✓ 10 steps completed without hanging")
except Exception as e:
    print(f"✗ Episode step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED - No hang detected in isolated components")
print("=" * 60)
print("\nIf main.py still hangs, the issue is likely in:")
print("  - Trainer initialization")
print("  - Validation loop")
print("  - TensorBoard logging")
print("  - File I/O operations")
