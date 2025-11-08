"""
Quick system test to verify all components work together.
"""

import numpy as np
import pandas as pd
from config import Config
from data_loader import DataLoader
from features import FeatureEngineer
from agent import DQNAgent
from risk_manager import RiskManager
from environment import ForexTradingEnv, pip_size, pip_value_usd
from fitness import FitnessCalculator

def test_system():
    print("=" * 60)
    print("FOREX RL TRADING BOT - SYSTEM TEST")
    print("=" * 60)
    
    # 1. Test Configuration
    print("\n1. Testing Configuration...")
    config = Config()
    print(f"   ✓ Config loaded: {config.data.pairs}")
    
    # 2. Test Data Loader
    print("\n2. Testing Data Loader...")
    loader = DataLoader()
    data = loader.generate_sample_data(n_bars=500)
    print(f"   ✓ Generated {len(data)} bars of sample data")
    
    # 3. Test Feature Engineering
    print("\n3. Testing Feature Engineering...")
    fe = FeatureEngineer()
    features_df = fe.compute_all_features(data)
    feature_cols = [c for c in features_df.columns if c != 'time']
    print(f"   ✓ Computed {len(feature_cols)} features")
    
    # 4. Test Risk Manager
    print("\n4. Testing Risk Manager...")
    rm = RiskManager()
    sizing = rm.calculate_position_size(1000, 900, 1.1, 0.0015)
    print(f"   ✓ Position size: {sizing['lots']:.2f} lots")
    
    # 5. Test Environment
    print("\n5. Testing Trading Environment...")
    env = ForexTradingEnv(
        data=features_df,
        feature_columns=feature_cols,
        initial_balance=1000.0,
        risk_manager=rm
    )
    state = env.reset()
    print(f"   ✓ Environment initialized, state size: {len(state)}")
    
    # 6. Test Agent
    print("\n6. Testing DQN Agent...")
    agent = DQNAgent(state_size=len(state), action_size=4)
    action = agent.select_action(state)
    print(f"   ✓ Agent created, selected action: {action}")
    
    # 7. Test Episode
    print("\n7. Testing Episode Execution...")
    state = env.reset()
    total_reward = 0
    for i in range(50):
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break
    print(f"   ✓ Executed {i+1} steps, total reward: {total_reward:.2f}")
    print(f"   ✓ Final equity: ${info['equity']:.2f}")
    
    # 8. Test Training
    print("\n8. Testing Agent Training...")
    loss = agent.train_step()
    if loss:
        print(f"   ✓ Training step completed, loss: {loss:.4f}")
    else:
        print(f"   ✓ Collecting more data before training")
    
    # 9. Test Fitness Calculation
    print("\n9. Testing Fitness Calculation...")
    equity_series = pd.Series(
        env.equity_history,
        index=pd.date_range('2024-01-01', periods=len(env.equity_history), freq='h')
    )
    fc = FitnessCalculator()
    metrics = fc.calculate_all_metrics(equity_series)
    print(f"   ✓ Fitness: {metrics['fitness']:.4f}")
    print(f"   ✓ Sharpe: {metrics['sharpe']:.4f}")
    print(f"   ✓ CAGR: {metrics['cagr']:.2%}")
    
    # 10. Test Trade Statistics
    print("\n10. Testing Trade Statistics...")
    stats = env.get_trade_statistics()
    print(f"   ✓ Total trades: {stats['total_trades']}")
    print(f"   ✓ Win rate: {stats['win_rate']:.2%}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print("\nSystem is ready for training!")
    print("Run: python main.py --mode train --episodes 50")

def test_known_pnl():
    prices = [1.10000, 1.10100, 1.10050]  # +10 pips, then -5 pips
    df = pd.DataFrame({
        'open': prices,
        'high': prices,
        'low': prices,
        'close': prices
    })
    env = ForexTradingEnv(data=df, feature_columns=['open','high','low','close'], initial_balance=1000.0, symbol='EURUSD')
    _ = env.reset()
    # open a long position at first price using internal helper
    env._open_position('long', price=prices[0], atr=0.001)
    # simulate first bar (price moves to 1.10100)
    env._calculate_equity(prices[1])
    # calculate expected unrealized PnL using the environment's entry (includes slippage)
    ps = pip_size(env.symbol)
    pips_move = (prices[1] - env.position['entry']) / ps * (1.0 if env.position['type'] == 'long' else -1.0)
    pv = pip_value_usd(env.symbol, prices[1], lots=env.position['lots'])
    expected_unreal = pips_move * pv
    assert abs(env.equity - (env.balance + expected_unreal)) < 1e-6

    # next bar moves to 1.10050 (-5 pips from previous price, net +5 pips from entry)
    env._calculate_equity(prices[2])
    pips_move = (prices[2] - env.position['entry']) / ps * (1.0 if env.position['type'] == 'long' else -1.0)
    pv = pip_value_usd(env.symbol, prices[2], lots=env.position['lots'])
    expected_unreal = pips_move * pv
    assert abs(env.equity - (env.balance + expected_unreal)) < 1e-6


if __name__ == "__main__":
    test_system()
    test_known_pnl()
    print('Sanity test passed')
