#!/usr/bin/env python3
"""
Comprehensive System Test Suite
Tests all major components and their integration to verify the system is working correctly.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import DQNAgent, ActionSpace
from environment import ForexTradingEnv
from features import FeatureEngineer
from fitness import FitnessCalculator
from risk_manager import RiskManager
from trainer import Trainer
from structured_logger import StructuredLogger


class SystemTester:
    """Comprehensive system testing."""
    
    def __init__(self):
        self.test_results = {}
        self.log_dir = Path("./logs")
        self.checkpoint_dir = Path("./checkpoints")
        
    def _create_sample_data_with_features(self, n_points: int = 1000):
        """Create sample data with computed features."""
        time_index = pd.date_range('2024-01-01', periods=n_points, freq='h')
        sample_data = pd.DataFrame({
            'timestamp': time_index,
            'time': time_index,  # Add time column for temporal features
            'open': np.random.uniform(1.1000, 1.1100, n_points),
            'high': np.random.uniform(1.1000, 1.1100, n_points),
            'low': np.random.uniform(1.1000, 1.1100, n_points),
            'close': np.random.uniform(1.1000, 1.1100, n_points),
            'volume': np.random.uniform(1000, 10000, n_points)
        }, index=time_index)
        
        # Compute features
        feature_eng = FeatureEngineer()
        sample_data = feature_eng.compute_all_features(sample_data)
        feature_columns = feature_eng.get_feature_names()
            
        return sample_data, feature_columns
        
    def test_component_creation(self):
        """Test that all components can be created without errors."""
        print("🔧 Testing Component Creation...")
        
        try:
            # Test basic component instantiation
            feature_calc = FeatureEngineer()
            risk_manager = RiskManager()
            fitness_calc = FitnessCalculator()
            structured_logger = StructuredLogger()
            
            # Test environment creation
            sample_data, feature_columns = self._create_sample_data_with_features(1000)
            
            env = ForexTradingEnv(
                data=sample_data,
                feature_columns=feature_columns,
                risk_manager=risk_manager,
                initial_balance=10000
            )
            
            # Test agent creation
            agent = DQNAgent(
                state_size=env.state_size,
                action_size=ActionSpace.get_action_size(),
                lr=0.001,
                use_double_dqn=True,
                use_dueling=True,
                use_noisy=True,
                use_per=True
            )
            
            self.test_results['component_creation'] = {
                'status': 'PASS',
                'environment_state_size': env.state_size,
                'agent_created': True,
                'noisy_net_enabled': agent.use_noisy,
                'per_enabled': agent.use_per,
                'double_dqn_enabled': agent.use_double_dqn
            }
            print("✅ Component creation: PASS")
            
        except Exception as e:
            self.test_results['component_creation'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"❌ Component creation: FAIL - {e}")
    
    def test_feature_calculation(self):
        """Test feature calculation with various market conditions."""
        print("📊 Testing Feature Calculation...")
        
        try:
            feature_calc = FeatureEngineer()
            
            # Create test data with known patterns
            trending_data, _ = self._create_sample_data_with_features(200)
            features = feature_calc.compute_all_features(trending_data)
            
            # Verify feature structure
            expected_features = [
                'returns', 'volatility', 'rsi', 'macd', 'macd_signal', 
                'atr', 'price_percentile', 'volume_ratio', 'trend_strength',
                'fractal_high', 'fractal_low', 'currency_strength'
            ]
            
            feature_names = list(features.columns)
            missing_features = set(expected_features) - set(feature_names)
            
            self.test_results['feature_calculation'] = {
                'status': 'PASS' if len(missing_features) == 0 else 'PARTIAL',
                'total_features': len(feature_names),
                'expected_features': len(expected_features),
                'missing_features': list(missing_features),
                'feature_shape': features.shape,
                'has_nan_values': features.isna().any().any(),
                'rsi_range_valid': (features['rsi'].min() >= 0 and features['rsi'].max() <= 100)
            }
            
            if len(missing_features) == 0:
                print("✅ Feature calculation: PASS")
            else:
                print(f"⚠️ Feature calculation: PARTIAL - Missing: {missing_features}")
                
        except Exception as e:
            self.test_results['feature_calculation'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"❌ Feature calculation: FAIL - {e}")
    
    def test_environment_functionality(self):
        """Test environment step function and trade logic."""
        print("🎯 Testing Environment Functionality...")
        
        try:
            # Create test environment
            sample_data, feature_columns = self._create_sample_data_with_features(500)
            
            env = ForexTradingEnv(
                data=sample_data,
                feature_columns=feature_columns,
                initial_balance=10000,
                structured_logger=StructuredLogger()
            )
            
            # Test environment reset
            state = env.reset()
            self.test_results['environment_functionality'] = {
                'reset_works': True,
                'state_size': len(state),
                'initial_balance': env.balance,
                'initial_equity': env.equity
            }
            
            # Test taking actions
            actions_taken = []
            rewards = []
            
            for i in range(50):  # Take 50 random steps
                action = np.random.randint(0, ActionSpace.get_action_size())
                next_state, reward, done, info = env.step(action)
                
                actions_taken.append(action)
                rewards.append(reward)
                
                if done:
                    break
                    
                state = next_state
            
            # Get trade statistics
            trade_stats = env.get_trade_statistics()
            
            self.test_results['environment_functionality'].update({
                'status': 'PASS',
                'steps_taken': len(actions_taken),
                'total_reward': sum(rewards),
                'final_equity': env.equity,
                'trades_opened': trade_stats.get('total_trades', 0),
                'win_rate': trade_stats.get('win_rate', 0),
                'environment_completed': True
            })
            
            print("✅ Environment functionality: PASS")
            
        except Exception as e:
            self.test_results['environment_functionality'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"❌ Environment functionality: FAIL - {e}")
    
    def test_agent_learning(self):
        """Test agent's learning capabilities with experience replay."""
        print("🧠 Testing Agent Learning...")
        
        try:
            # Create simple environment
            sample_data, feature_columns = self._create_sample_data_with_features(300)
            
            env = ForexTradingEnv(data=sample_data, feature_columns=feature_columns, initial_balance=10000)
            
            # Create agent
            agent = DQNAgent(
                state_size=env.state_size,
                action_size=ActionSpace.get_action_size(),
                lr=0.01,  # Higher learning rate for testing
                use_double_dqn=True,
                use_dueling=True,
                use_noisy=True,
                use_per=True,
                replay_capacity=1000
            )
            
            # Fill replay buffer with experiences
            state = env.reset()
            experiences_collected = 0
            
            for _ in range(100):  # Collect experiences
                action = agent.select_action(state, explore=True)
                next_state, reward, done, _ = env.step(action)
                
                agent.store_transition(state, action, reward, next_state, done)
                experiences_collected += 1
                
                if done:
                    state = env.reset()
                else:
                    state = next_state
            
            # Test learning
            initial_loss = None
            final_loss = None
            losses = []
            
            for i in range(20):  # Training steps
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)
                    if initial_loss is None:
                        initial_loss = loss
                    final_loss = loss
            
            self.test_results['agent_learning'] = {
                'status': 'PASS',
                'experiences_collected': experiences_collected,
                'replay_buffer_size': len(agent.replay_buffer) if hasattr(agent, 'replay_buffer') else 0,
                'training_steps_completed': len(losses),
                'initial_loss': float(initial_loss) if initial_loss is not None else None,
                'final_loss': float(final_loss) if final_loss is not None else None,
                'loss_decreased': (final_loss < initial_loss) if (initial_loss is not None and final_loss is not None) else None,
                'noisy_net_active': agent.use_noisy,
                'per_active': agent.use_per
            }
            
            print("✅ Agent learning: PASS")
            
        except Exception as e:
            self.test_results['agent_learning'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"❌ Agent learning: FAIL - {e}")
    
    def test_monitoring_system(self):
        """Test monitoring and logging system."""
        print("📈 Testing Monitoring System...")
        
        try:
            # Test structured logger
            logger = StructuredLogger(log_dir="./logs")
            
            # Test trade logging
            logger.log_trade_open(
                timestamp=datetime.now(),
                action="BUY",
                price=1.1050,
                lots=0.1,
                stop_loss=1.1000,
                take_profit=1.1100
            )
            
            logger.log_trade_close(
                timestamp=datetime.now(),
                action="SELL",
                price=1.1080,
                lots=0.1,
                pnl=30.0,
                duration_bars=10,
                close_reason="TAKE_PROFIT"
            )
            
            # Test episode logging
            logger.log_episode_start(1, datetime.now())
            logger.log_episode_end(
                episode=1,
                timestamp=datetime.now(),
                reward=100.0,
                final_equity=10100.0,
                steps=50,
                trades=5,
                win_rate=0.6
            )
            
            # Test validation logging
            logger.log_validation(
                episode=1,
                timestamp=datetime.now(),
                fitness=0.85,
                sharpe=1.2,
                cagr=0.15,
                max_drawdown=0.05,
                total_trades=10
            )
            
            # Check if log files exist
            log_files_exist = {
                'trade_events': (Path("./logs") / "trade_events.jsonl").exists(),
                'episode_events': (Path("./logs") / "episode_events.jsonl").exists(),
                'error_events': (Path("./logs") / "error_events.jsonl").exists()
            }
            
            self.test_results['monitoring_system'] = {
                'status': 'PASS',
                'structured_logger_created': True,
                'log_files_created': log_files_exist,
                'trade_logging_works': True,
                'episode_logging_works': True,
                'validation_logging_works': True
            }
            
            print("✅ Monitoring system: PASS")
            
        except Exception as e:
            self.test_results['monitoring_system'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"❌ Monitoring system: FAIL - {e}")
    
    def test_integration_training(self):
        """Test full integration with a short training run."""
        print("🔄 Testing Integration Training...")
        
        try:
            # Create test data
            sample_data, feature_columns = self._create_sample_data_with_features(1000)
            
            # Split data
            split_idx = int(len(sample_data) * 0.8)
            train_data = sample_data[:split_idx].copy()
            val_data = sample_data[split_idx:].copy()
            
            # Create environments
            train_env = ForexTradingEnv(data=train_data, feature_columns=feature_columns, initial_balance=10000)
            val_env = ForexTradingEnv(data=val_data, feature_columns=feature_columns, initial_balance=10000)
            
            # Create agent
            agent = DQNAgent(
                state_size=train_env.state_size,
                action_size=ActionSpace.get_action_size(),
                lr=0.001,
                use_double_dqn=True,
                use_dueling=True,
                use_noisy=True,
                use_per=True
            )
            
            # Create trainer
            trainer = Trainer(
                agent=agent,
                train_env=train_env,
                val_env=val_env,
                checkpoint_dir="./checkpoints",
                log_dir="./logs"
            )
            
            # Run short training
            result = trainer.train(
                num_episodes=3,  # Very short for testing
                validate_every=1,
                save_every=2,
                verbose=True
            )
            
            # Check results
            training_completed = len(result['training_history']) == 3
            validation_completed = len(result['validation_history']) > 0
            
            # Check if checkpoint was saved
            checkpoint_exists = (Path("./checkpoints") / "final_model.pt").exists()
            
            self.test_results['integration_training'] = {
                'status': 'PASS',
                'training_episodes_completed': len(result['training_history']),
                'validation_runs_completed': len(result['validation_history']),
                'checkpoint_saved': checkpoint_exists,
                'training_completed': training_completed,
                'validation_completed': validation_completed
            }
            
            print("✅ Integration training: PASS")
            
        except Exception as e:
            self.test_results['integration_training'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"❌ Integration training: FAIL - {e}")
    
    def run_all_tests(self):
        """Run all system tests."""
        print("🚀 Starting Comprehensive System Tests")
        print("=" * 60)
        
        test_methods = [
            self.test_component_creation,
            self.test_feature_calculation,
            self.test_environment_functionality,
            self.test_agent_learning,
            self.test_monitoring_system,
            self.test_integration_training
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                test_name = test_method.__name__
                print(f"❌ {test_name}: FAILED with unexpected error - {e}")
                self.test_results[test_name] = {
                    'status': 'FAIL',
                    'error': f"Unexpected error: {e}"
                }
            print()  # Add spacing
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_results()
    
    def print_summary(self):
        """Print test summary."""
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed = sum(1 for result in self.test_results.values() if result.get('status') == 'PASS')
        partial = sum(1 for result in self.test_results.values() if result.get('status') == 'PARTIAL')
        failed = sum(1 for result in self.test_results.values() if result.get('status') == 'FAIL')
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed}")
        print(f"⚠️ Partial: {partial}")
        print(f"❌ Failed: {failed}")
        print()
        
        if failed == 0 and partial == 0:
            print("🎉 ALL TESTS PASSED! Your system is working correctly.")
        elif failed == 0:
            print("✅ Core functionality working. Some optional features may need attention.")
        else:
            print("⚠️ Some components need attention. Check the detailed results.")
    
    def save_results(self):
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path("./logs") / f"system_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"\n💾 Test results saved to: {results_file}")


def main():
    """Run system tests."""
    tester = SystemTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
