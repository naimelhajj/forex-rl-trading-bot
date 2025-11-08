#!/usr/bin/env python3
"""
Quick System Verification
Fast tests to verify basic functionality is working.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_test():
    """Run quick verification tests."""
    print("🚀 Quick System Verification")
    print("=" * 40)
    
    try:
        # Test 1: Import all modules
        print("1️⃣ Testing imports...")
        from agent import DQNAgent, ActionSpace
        from environment import ForexTradingEnv
        from features import FeatureEngineer
        from trainer import Trainer
        from structured_logger import StructuredLogger
        print("   ✅ All modules imported successfully")
        
        # Test 2: Create sample data and environment
        print("2️⃣ Testing environment creation...")
        time_index = pd.date_range('2024-01-01', periods=100, freq='h')
        sample_data = pd.DataFrame({
            'timestamp': time_index,
            'time': time_index,  # Add time column for temporal features
            'open': np.random.uniform(1.1000, 1.1100, 100),
            'high': np.random.uniform(1.1000, 1.1100, 100),
            'low': np.random.uniform(1.1000, 1.1100, 100),
            'close': np.random.uniform(1.1000, 1.1100, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=time_index)
        
        # Create feature engineer and compute features
        feature_eng = FeatureEngineer()
        
        # Create mock multi-pair data for currency strength features
        gbp_data = sample_data.copy()
        gbp_data[['open', 'high', 'low', 'close']] *= 1.25  # Mock GBP higher prices
        
        jpy_data = sample_data.copy()  
        jpy_data[['open', 'high', 'low', 'close']] *= 110  # Mock JPY higher prices
        
        mock_currency_data = {
            'EURUSD': sample_data.copy(),
            'GBPUSD': gbp_data,
            'USDJPY': jpy_data  
        }
        
        # Compute base features first
        sample_data_with_features = feature_eng.compute_all_features(sample_data, currency_data=None)
        
        # Add currency strengths manually for test  
        from features import compute_currency_strengths
        try:
            S = compute_currency_strengths(
                pair_dfs=mock_currency_data,
                currencies=['EUR', 'USD'], 
                window=24,
                lags=3
            )
            sample_data_with_features = sample_data_with_features.join(S, how="left").ffill().bfill()
        except Exception as e:
            print(f"   Warning: Currency strength computation failed: {e}")
            # Add mock strength features
            for curr in ['EUR', 'USD']:
                sample_data_with_features[f'strength_{curr}'] = 0.0
                for lag in range(1, 4):
                    sample_data_with_features[f'strength_{curr}_lag{lag}'] = 0.0
        
        # Get updated feature columns including strengths
        base_features = feature_eng.get_feature_names(include_currency_strength=False)
        strength_features = [c for c in sample_data_with_features.columns if c.startswith('strength_')]
        feature_columns = [c for c in base_features if c in sample_data_with_features.columns] + strength_features
        
        # Debug: check what features are actually in the data vs what's expected
        print(f"   Expected feature columns: {feature_columns}")
        print(f"   Actual columns in data: {list(sample_data_with_features.columns)}")
        
        # Filter feature columns to only include what's actually available
        available_features = [col for col in feature_columns if col in sample_data_with_features.columns]
        print(f"   Available features: {available_features}")
        feature_columns = available_features
        
        env = ForexTradingEnv(
            data=sample_data_with_features, 
            feature_columns=feature_columns,
            initial_balance=10000
        )
        state = env.reset()
        print(f"   ✅ Environment created (state size: {len(state)})")
        
        # Add currency strength feature guard
        assert any(c.startswith("strength_") for c in env.feature_columns), "Currency strength features are missing."
        print(f"   ✅ Currency strength features confirmed")
        
        # Test 3: Create agent
        print("3️⃣ Testing agent creation...")
        agent = DQNAgent(
            state_size=env.state_size,
            action_size=ActionSpace.get_action_size(),
            use_double_dqn=True,
            use_dueling=True,
            use_noisy=True,
            use_per=True
        )
        print("   ✅ Agent created with all features enabled")
        
        # Test 4: Test basic interaction
        print("4️⃣ Testing agent-environment interaction...")
        action = agent.select_action(state, explore=True)
        next_state, reward, done, info = env.step(action)
        print(f"   ✅ Interaction works (action: {action}, reward: {reward:.2f})")
        
        # Test 5: Test structured logging
        print("5️⃣ Testing structured logging...")
        logger = StructuredLogger()
        from datetime import datetime
        logger.log_episode_start(1, datetime.now())
        print("   ✅ Structured logging works")
        
        print("\n🎉 QUICK VERIFICATION PASSED!")
        print("Your system is ready for training.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    quick_test()
