"""
Minimal smoke test - just run 1 episode to check for Unicode errors
"""
import sys

print("="*60)
print("MINIMAL SMOKE TEST - 1 EPISODE")
print("="*60)

try:
    # Import main components
    from config import Config
    from data_loader import DataLoader
    from features import FeatureEngineer
    from environment import ForexTradingEnv
    from agent import DDQNAgent
    from trainer import Trainer
    from scaler_utils import create_robust_scaler
    
    print("[OK] All imports successful")
    
    # Create config
    config = Config()
    config.training.num_episodes = 1  # Just 1 episode
    print("[OK] Config created (1 episode)")
    
    # Load data
    loader = DataLoader(config.data)
    all_pairs = loader.load_data()
    print("[OK] Data loaded")
    
    # Compute features
    fe = FeatureEngineer()
    primary_df = all_pairs[config.data.primary_pair].copy()
    primary_df = fe.compute_all_features(primary_df, all_pairs)
    print("[OK] Features computed")
    
    # Split data
    train_size = int(len(primary_df) * 0.7)
    val_size = int(len(primary_df) * 0.15)
    train_df = primary_df.iloc[:train_size].copy()
    val_df = primary_df.iloc[train_size:train_size+val_size].copy()
    
    # Create scaler
    scaler = create_robust_scaler(train_df)
    print("[OK] Scaler created")
    
    # Create environments
    train_env = ForexTradingEnv(train_df, scaler, config.environment, initial_balance=1000.0)
    val_env = ForexTradingEnv(val_df, scaler, config.environment, initial_balance=1000.0)
    print(f"[OK] Environments created (state_size={train_env.state_size})")
    
    # Create agent
    agent = DDQNAgent(train_env.state_size, train_env.action_space, config.agent)
    print(f"[OK] Agent created (learning_starts={agent.learning_starts})")
    
    # Create trainer
    trainer = Trainer(agent, train_env, val_env, config)
    print("[OK] Trainer created")
    
    # Run 1 episode
    print("\n" + "="*60)
    print("RUNNING 1 EPISODE...")
    print("="*60 + "\n")
    
    history = trainer.train(num_episodes=1, validate_every=1, verbose=True)
    
    print("\n" + "="*60)
    print("SUCCESS! 1 episode completed without errors")
    print("="*60)
    print("\nSystem is ready for full training:")
    print("  python main.py --episodes 5   (smoke test)")
    print("  python main.py --episodes 50  (production)")
    
    sys.exit(0)
    
except Exception as e:
    print("\n" + "="*60)
    print("ERROR DETECTED")
    print("="*60)
    print(f"\n{type(e).__name__}: {e}")
    
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
    
    sys.exit(1)
