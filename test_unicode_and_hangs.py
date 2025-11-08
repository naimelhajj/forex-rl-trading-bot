"""
Comprehensive test for Unicode issues and potential hangs.
Tests all critical components in isolation before full training.
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

print("="*60)
print("UNICODE AND HANG DIAGNOSTIC TEST")
print("="*60)

# Test 1: Check Python encoding
print("\n[TEST 1] Python encoding check...")
print(f"  Default encoding: {sys.getdefaultencoding()}")
print(f"  stdout encoding: {sys.stdout.encoding}")
print(f"  filesystem encoding: {sys.getfilesystemencoding()}")

# Test 2: Test problematic Unicode characters
print("\n[TEST 2] Unicode character test...")
test_chars = {
    "arrow": "→",
    "checkmark": "✓",
    "warning": "⚠",
    "epsilon": "ε",
    "plus-minus": "±",
    "less-equal": "≤",
    "en-dash": "–"
}

for name, char in test_chars.items():
    try:
        print(f"  Testing {name}: {char}")
    except UnicodeEncodeError as e:
        print(f"  FAILED {name}: {e}")

# Test 3: Import all modules
print("\n[TEST 3] Module import test...")
try:
    from config import Config
    print("  [OK] config")
except Exception as e:
    print(f"  [FAIL] config: {e}")
    sys.exit(1)

try:
    from data_loader import DataLoader
    print("  [OK] data_loader")
except Exception as e:
    print(f"  [FAIL] data_loader: {e}")
    sys.exit(1)

try:
    from features import FeatureEngineer
    print("  [OK] features")
except Exception as e:
    print(f"  [FAIL] features: {e}")
    sys.exit(1)

try:
    from environment import ForexTradingEnv
    print("  [OK] environment")
except Exception as e:
    print(f"  [FAIL] environment: {e}")
    sys.exit(1)

try:
    from agent import DDQNAgent
    print("  [OK] agent")
except Exception as e:
    print(f"  [FAIL] agent: {e}")
    sys.exit(1)

try:
    from trainer import Trainer
    print("  [OK] trainer")
except Exception as e:
    print(f"  [FAIL] trainer: {e}")
    sys.exit(1)

# Test 4: Quick feature computation
print("\n[TEST 4] Feature computation test...")
try:
    config = Config()
    loader = DataLoader(config.data)
    
    print("  Generating sample data...")
    start = time.time()
    all_pairs = loader.load_data()
    elapsed = time.time() - start
    print(f"  Data loaded in {elapsed:.2f}s")
    
    print("  Computing features...")
    start = time.time()
    fe = FeatureEngineer()
    primary_pair = config.data.primary_pair
    df = all_pairs[primary_pair].copy()
    df = fe.compute_all_features(df, all_pairs)
    elapsed = time.time() - start
    print(f"  Features computed in {elapsed:.2f}s")
    print(f"  Feature shape: {df.shape}")
    
except Exception as e:
    print(f"  [FAIL] Feature computation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Environment creation
print("\n[TEST 5] Environment creation test...")
try:
    from scaler_utils import create_robust_scaler
    
    # Split data
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size].copy()
    
    # Create scaler
    print("  Creating scaler...")
    scaler = create_robust_scaler(train_df)
    
    # Create environment
    print("  Creating environment...")
    env = ForexTradingEnv(
        df=train_df,
        scaler=scaler,
        config=config.environment,
        initial_balance=1000.0
    )
    print(f"  State size: {env.state_size}")
    print(f"  Action space: {env.action_space}")
    
except Exception as e:
    print(f"  [FAIL] Environment creation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Agent creation
print("\n[TEST 6] Agent creation test...")
try:
    agent = DDQNAgent(
        state_size=env.state_size,
        action_size=env.action_space,
        config=config.agent
    )
    print(f"  Agent created successfully")
    print(f"  Learning starts: {agent.learning_starts}")
    print(f"  Buffer capacity: {agent.buffer_capacity}")
    
except Exception as e:
    print(f"  [FAIL] Agent creation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Single episode step
print("\n[TEST 7] Single episode test (checking for hangs)...")
try:
    state = env.reset()
    print(f"  Initial state shape: {state.shape}")
    
    steps = 0
    max_steps = 100
    start = time.time()
    
    while steps < max_steps:
        action = agent.select_action(state, is_training=False)
        next_state, reward, done, info = env.step(action)
        
        steps += 1
        if steps % 20 == 0:
            elapsed = time.time() - start
            print(f"  Step {steps}/{max_steps} - {elapsed:.2f}s - Action: {action}")
        
        if done:
            break
        
        state = next_state
    
    elapsed = time.time() - start
    print(f"  Episode completed in {elapsed:.2f}s ({steps} steps)")
    print(f"  Average: {elapsed/steps:.4f}s per step")
    
except Exception as e:
    print(f"  [FAIL] Episode test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Validation window computation
print("\n[TEST 8] Validation window test...")
try:
    val_size = int(len(df) * 0.15)
    val_df = df.iloc[-val_size:].copy()
    
    val_env = ForexTradingEnv(
        df=val_df,
        scaler=scaler,
        config=config.environment,
        initial_balance=1000.0
    )
    
    trainer = Trainer(
        agent=agent,
        train_env=env,
        val_env=val_env,
        config=config
    )
    
    print("  Computing validation windows...")
    start = time.time()
    windows = trainer._compute_val_windows()
    elapsed = time.time() - start
    
    print(f"  Windows computed in {elapsed:.2f}s")
    print(f"  Number of windows: {len(windows)}")
    if windows:
        print(f"  First window: {windows[0]}")
        print(f"  Last window: {windows[-1]}")
    
except Exception as e:
    print(f"  [FAIL] Validation window test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Single validation pass
print("\n[TEST 9] Single validation pass test...")
try:
    if windows:
        lo, hi = windows[0]
        print(f"  Testing window: [{lo}, {hi})")
        
        start = time.time()
        stats = trainer._run_validation_slice(lo, hi, base_spread=0.00015, base_commission=7.0)
        elapsed = time.time() - start
        
        print(f"  Validation pass completed in {elapsed:.2f}s")
        print(f"  Fitness: {stats['fitness']:.4f}")
        print(f"  Trades: {stats.get('trade_count', 0)}")
        print(f"  Final equity: ${stats.get('final_equity', 0):.2f}")
    
except Exception as e:
    print(f"  [FAIL] Validation pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 10: Learning step (small batch)
print("\n[TEST 10] Learning step test...")
try:
    # Prefill buffer with some transitions
    print("  Prefilling buffer...")
    state = env.reset()
    for i in range(100):
        action = agent.select_action(state, is_training=False)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        
        if done:
            state = env.reset()
        else:
            state = next_state
    
    print(f"  Buffer size: {agent.replay_size}")
    
    if agent.replay_size >= 32:
        print("  Testing learning step...")
        start = time.time()
        loss = agent.learn()
        elapsed = time.time() - start
        
        print(f"  Learning step completed in {elapsed:.4f}s")
        print(f"  Loss: {loss:.6f}")
    else:
        print("  [SKIP] Not enough samples for learning")
    
except Exception as e:
    print(f"  [FAIL] Learning step: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
print("\nSystem appears to be working correctly.")
print("You can now run full training with:")
print("  python main.py --episodes 5  (smoke test)")
print("  python main.py --episodes 50 (production)")
print("  python run_seed_sweep_auto.py (seed sweep)")
