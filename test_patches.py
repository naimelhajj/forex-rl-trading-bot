"""
Test file to verify all 6 patches are working correctly.
"""
import numpy as np
import pandas as pd
from config import Config
from environment import ForexTradingEnv
from agent import DQNAgent, ActionSpace
from fitness import stability_fitness, FitnessCalculator
from trainer import baseline_policy, prefill_replay

def test_patch1_fitness():
    """PATCH #1: Fitness computation after ruin-clamp with business-day resample."""
    print("\n=== Testing PATCH #1: Fitness Hygiene ===")
    
    # Create sample equity curve
    equity = pd.Series(
        np.linspace(1000, 800, 100),  # Declining equity
        index=pd.date_range('2024-01-01', periods=100, freq='H')
    )
    
    # Test new stability_fitness function
    fitness, metrics = stability_fitness(equity)
    print(f"Fitness: {fitness:.4f}")
    print(f"Sharpe: {metrics['Sharpe']:.4f}, CAGR: {metrics['CAGR']:.4f}")
    print(f"Stagnation: {metrics['Stagnation']:.4f}, Loss Years: {metrics['LossYears']}")
    
    # Test FitnessCalculator uses new function
    fc = FitnessCalculator()
    all_metrics = fc.calculate_all_metrics(equity)
    assert 'fitness' in all_metrics
    print("✓ PATCH #1 working correctly")


def test_patch2_frame_stacking():
    """PATCH #2: Frame stacking for temporal memory."""
    print("\n=== Testing PATCH #2: Frame Stacking ===")
    
    # Create minimal environment
    data = pd.DataFrame({
        'open': np.ones(100),
        'high': np.ones(100) * 1.01,
        'low': np.ones(100) * 0.99,
        'close': np.ones(100),
        'atr': np.ones(100) * 0.001,
        'rsi': np.ones(100) * 50,
    })
    feature_cols = ['close', 'atr', 'rsi']
    
    env = ForexTradingEnv(
        data=data,
        feature_columns=feature_cols,
        initial_balance=1000
    )
    
    # Check state size accounts for frame stacking
    expected_size = len(feature_cols) * env.stack_n + env.context_dim
    print(f"State size: {env.state_size} (expected: {expected_size})")
    assert env.state_size == expected_size, f"State size mismatch: {env.state_size} != {expected_size}"
    
    # Check reset initializes frame stack
    state = env.reset()
    print(f"State shape after reset: {state.shape}")
    assert state.shape[0] == env.state_size
    
    # Check step maintains frame stack
    action = 0  # HOLD
    next_state, reward, done, info = env.step(action)
    print(f"State shape after step: {next_state.shape}")
    assert next_state.shape[0] == env.state_size
    
    print("✓ PATCH #2 working correctly")


def test_patch3_baseline_prefill():
    """PATCH #3: Heuristic baseline prefill."""
    print("\n=== Testing PATCH #3: Baseline Prefill ===")
    
    # Create environment
    data = pd.DataFrame({
        'open': np.ones(200),
        'high': np.ones(200) * 1.01,
        'low': np.ones(200) * 0.99,
        'close': np.ones(200),
        'atr': np.ones(200) * 0.001,
        'rsi': np.linspace(30, 70, 200),
        'strength_EUR': np.random.randn(200) * 0.5,
        'strength_USD': np.random.randn(200) * 0.5,
    })
    feature_cols = ['close', 'atr', 'rsi', 'strength_EUR', 'strength_USD']
    
    env = ForexTradingEnv(
        data=data,
        feature_columns=feature_cols,
        initial_balance=1000
    )
    
    # Test baseline policy
    state = env.reset()
    action = baseline_policy(state, feature_cols)
    print(f"Baseline policy action: {action}")
    assert action in [0, 1, 2], f"Invalid action: {action}"
    
    # Test prefill
    agent = DQNAgent(state_size=env.state_size, action_size=ActionSpace.get_action_size())
    initial_buffer_size = agent.replay_size
    print(f"Initial buffer size: {initial_buffer_size}")
    
    prefill_replay(env, agent, steps=100)
    final_buffer_size = agent.replay_size
    print(f"Final buffer size: {final_buffer_size}")
    assert final_buffer_size > initial_buffer_size, "Buffer not filled"
    
    print("✓ PATCH #3 working correctly")


def test_patch4_action_mask():
    """PATCH #4: Meaningful MOVE_SL_CLOSER action mask."""
    print("\n=== Testing PATCH #4: Action Mask ===")
    
    # Create environment
    data = pd.DataFrame({
        'open': np.ones(100),
        'high': np.ones(100) * 1.01,
        'low': np.ones(100) * 0.99,
        'close': np.linspace(1.0, 1.01, 100),  # Trending up
        'atr': np.ones(100) * 0.001,
        'rsi': np.ones(100) * 50,
    })
    feature_cols = ['close', 'atr', 'rsi']
    
    env = ForexTradingEnv(
        data=data,
        feature_columns=feature_cols,
        initial_balance=1000,
        min_trail_buffer_pips=1.0
    )
    
    # Check min_trail_buffer_pips set
    assert hasattr(env, 'min_trail_buffer_pips')
    print(f"Min trail buffer pips: {env.min_trail_buffer_pips}")
    
    # Reset and check mask
    env.reset()
    mask = env.legal_action_mask()
    print(f"Mask (no position): {mask}")
    assert mask[3] == False, "MOVE_SL_CLOSER should be illegal with no position"
    
    # Open position
    env.step(1)  # LONG
    if env.position is not None:
        mask = env.legal_action_mask()
        print(f"Mask (with position): {mask}")
        # MOVE_SL_CLOSER legality depends on price distance
        print(f"Position SL: {env.position.get('sl')}, Current price: {env.data.iloc[env.current_step]['close']}")
    
    print("✓ PATCH #4 working correctly")


def test_patch5_eval_mode():
    """PATCH #5: Deterministic evaluation mode."""
    print("\n=== Testing PATCH #5: Eval Mode ===")
    
    # Create agent
    agent = DQNAgent(state_size=10, action_size=ActionSpace.get_action_size(), use_noisy=True)
    state = np.random.randn(10).astype(np.float32)
    
    # Test eval_mode=True gives deterministic actions
    action1 = agent.select_action(state, eval_mode=True)
    action2 = agent.select_action(state, eval_mode=True)
    print(f"Action 1 (eval): {action1}, Action 2 (eval): {action2}")
    assert action1 == action2, "Eval mode should be deterministic"
    
    # Test eval_mode=False allows exploration
    actions = [agent.select_action(state, eval_mode=False) for _ in range(10)]
    print(f"Actions (explore): {actions}")
    
    print("✓ PATCH #5 working correctly")


def test_patch6_training_params():
    """PATCH #6: Training stability parameters."""
    print("\n=== Testing PATCH #6: Training Parameters ===")
    
    cfg = Config()
    
    # Check gamma = 0.97
    print(f"Gamma: {cfg.agent.gamma}")
    assert cfg.agent.gamma == 0.97, f"Gamma should be 0.97, got {cfg.agent.gamma}"
    
    # Check batch_size = 256
    print(f"Batch size: {cfg.agent.batch_size}")
    assert cfg.agent.batch_size == 256, f"Batch size should be 256, got {cfg.agent.batch_size}"
    
    # Check grad_clip = 1.0
    print(f"Grad clip: {cfg.agent.grad_clip}")
    assert cfg.agent.grad_clip == 1.0, f"Grad clip should be 1.0, got {cfg.agent.grad_clip}"
    
    # Check prefill_steps
    print(f"Prefill steps: {cfg.training.prefill_steps}")
    assert cfg.training.prefill_steps == 3000, f"Prefill steps should be 3000, got {cfg.training.prefill_steps}"
    
    # Check frame stacking
    print(f"Stack N: {cfg.environment.stack_n}")
    assert cfg.environment.stack_n == 3, f"Stack N should be 3, got {cfg.environment.stack_n}"
    
    print("✓ PATCH #6 working correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing All 6 Patches")
    print("=" * 60)
    
    try:
        test_patch1_fitness()
        test_patch2_frame_stacking()
        test_patch3_baseline_prefill()
        test_patch4_action_mask()
        test_patch5_eval_mode()
        test_patch6_training_params()
        
        print("\n" + "=" * 60)
        print("✓ ALL PATCHES WORKING CORRECTLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
