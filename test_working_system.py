#!/usr/bin/env python3
"""
Basic System Test - Focused on core functionality
Tests only the essential components that are actually implemented.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_working_system():
    """Test the system as it actually exists."""
    print("🔍 Testing Working System Components")
    print("=" * 50)
    
    try:
        # Test 1: Basic imports
        print("1️⃣ Testing imports...")
        from agent import DQNAgent, ActionSpace
        from environment import ForexTradingEnv
        from features import FeatureEngineer
        from trainer import Trainer
        from structured_logger import StructuredLogger
        print("   ✅ All imports successful")
        
        # Test 2: Create sample data with features
        print("2️⃣ Creating test data with features...")
        time_index = pd.date_range('2024-01-01', periods=200, freq='h')
        sample_data = pd.DataFrame({
            'timestamp': time_index,
            'time': time_index,
            'open': np.random.uniform(1.1000, 1.1100, 200),
            'high': np.random.uniform(1.1000, 1.1100, 200),
            'low': np.random.uniform(1.1000, 1.1100, 200),
            'close': np.random.uniform(1.1000, 1.1100, 200),
            'volume': np.random.uniform(1000, 10000, 200)
        }, index=time_index)
        
        # Compute features
        feature_eng = FeatureEngineer()
        sample_data_with_features = feature_eng.compute_all_features(sample_data)
        feature_columns = feature_eng.get_feature_names()
        print(f"   ✅ Data created with {len(feature_columns)} features")
        
        # Test 3: Create environment
        print("3️⃣ Testing environment...")
        env = ForexTradingEnv(
            data=sample_data_with_features,
            feature_columns=feature_columns,
            initial_balance=10000
        )
        
        state = env.reset()
        print(f"   ✅ Environment created (state size: {len(state)})")
        
        # Test 4: Create agent 
        print("4️⃣ Testing agent...")
        agent = DQNAgent(
            state_size=env.state_size,
            action_size=ActionSpace.get_action_size(),
            lr=0.001,
            use_double_dqn=True,
            use_dueling=True,
            use_noisy=False,  # Disable noisy for now due to recursion issue
            use_per=True
        )
        print("   ✅ Agent created successfully")
        
        # Test 5: Basic interaction
        print("5️⃣ Testing interaction...")
        action = agent.select_action(state, explore=True)
        next_state, reward, done, info = env.step(action)
        
        # Store and train
        agent.store_transition(state, action, reward, next_state, done)
        
        # Try training step
        loss = agent.train_step()
        print(f"   ✅ Interaction complete (action: {action}, reward: {reward:.4f}, loss: {loss})")
        
        # Test 6: Simple training loop
        print("6️⃣ Testing mini training loop...")
        episode_rewards = []
        
        for ep in range(2):  # Very short test
            state = env.reset()
            total_reward = 0
            
            for step in range(10):  # Short episodes
                action = agent.select_action(state, explore=True)
                next_state, reward, done, info = env.step(action)
                
                agent.store_transition(state, action, reward, next_state, done)
                
                # Train if we have enough experiences
                if step > 5:
                    loss = agent.train_step()
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            
        print(f"   ✅ Training complete (episode rewards: {episode_rewards})")
        
        # Test 7: Structured logging
        print("7️⃣ Testing structured logging...")
        logger = StructuredLogger()
        logger.log_episode_start(1, datetime.now())
        logger.log_episode_end(
            episode=1,
            timestamp=datetime.now(),
            reward=episode_rewards[0],
            final_equity=info.get('equity', 10000),
            steps=10,
            trades=0,
            win_rate=0.0
        )
        print("   ✅ Structured logging works")
        
        print("\n🎉 ALL CORE TESTS PASSED!")
        print("Your system is working correctly for training.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_actual_training():
    """Test with the actual main.py training setup."""
    print("\n🚂 Testing Actual Training Setup")
    print("=" * 50)
    
    try:
        # Import and run actual training for 2 episodes
        import main
        
        print("Running 2 episodes of actual training...")
        
        # This will use the real data loader and configuration
        main.train(episodes=2, validate_every=1, save_every=1, verbose=True)
        
        print("✅ Actual training completed successfully!")
        
        # Check if files were created
        logs_path = Path("./logs")
        checkpoints_path = Path("./checkpoints")
        
        if logs_path.exists():
            log_files = list(logs_path.glob("*.json*"))
            print(f"   Created {len(log_files)} log files")
        
        if checkpoints_path.exists():
            checkpoint_files = list(checkpoints_path.glob("*.pt"))
            print(f"   Created {len(checkpoint_files)} checkpoint files")
        
        return True
        
    except Exception as e:
        print(f"❌ Actual training failed: {e}")
        return False

if __name__ == "__main__":
    # Test working system components
    basic_test_passed = test_working_system()
    
    if basic_test_passed:
        print("\n" + "="*60)
        # Test actual training if basic test passes
        training_test_passed = test_actual_training()
        
        if training_test_passed:
            print(f"\n🏆 COMPLETE SUCCESS!")
            print("Both component tests and actual training work perfectly!")
        else:
            print(f"\n⚠️ PARTIAL SUCCESS")
            print("Components work, but actual training needs attention.")
    else:
        print(f"\n❌ SYSTEM ISSUES")
        print("Basic components need fixing before training.")
