"""
Main Application Module
Entry point for training and evaluating the Forex RL trading bot.
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import argparse
import json

# Configure stdout/stderr for UTF-8 to prevent Windows encoding errors
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, Exception):
    # Python < 3.7 or other issue - continue anyway
    pass

from config import Config
from data_loader import DataLoader
from features import FeatureEngineer
from agent import DQNAgent, ActionSpace
from risk_manager import RiskManager
from environment import ForexTradingEnv
from fitness import FitnessCalculator
from trainer import Trainer


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def prepare_data(config: Config):
    """
    Prepare data for training.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_data, val_data, test_data, feature_columns)
    """
    from itertools import combinations
    from features import compute_currency_strengths
    
    # Canonical 21-pair major FX universe (market-standard naming)
    PAIRS_7MAJ = [
        # USD majors
        "EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD",
        # EUR crosses
        "EURGBP","EURJPY","EURCHF","EURAUD","EURCAD",
        # GBP crosses
        "GBPJPY","GBPCHF","GBPAUD","GBPCAD",
        # AUD crosses
        "AUDJPY","AUDCHF","AUDCAD",
        # CAD & CHF cross
        "CADJPY","CADCHF","CHFJPY",
    ][:21]

    print("\n" + "=" * 50)
    print("PREPARING DATA")
    print("=" * 50)
    
    # Initialize data loader
    loader = DataLoader(config.data.data_dir)

    # Build pair universe
    pairs = config.PAIRS or PAIRS_7MAJ
    print(f"\nGenerating sample data for {len(pairs)} pairs: {pairs}")

    # Generate multi-pair data
    pair_dfs = loader.generate_multiple_pairs(
        pairs=pairs,
        n_bars=config.data.n_bars,
        start_date=config.data.start_date,
        freq=config.data.freq
    )

    # Use primary pair for trading (EURUSD or first pair)
    primary_pair = "EURUSD" if "EURUSD" in pair_dfs else pairs[0]
    data = pair_dfs[primary_pair]

    print(f"\nPrimary trading pair: {primary_pair}")
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    # Initialize feature engineer
    fe = FeatureEngineer(
        short_window=config.feature.short_window,
        medium_window=config.feature.medium_window,
        long_window=config.feature.long_window,
        atr_period=config.feature.atr_period,
        rsi_period=config.feature.rsi_period,
        lr_window=config.feature.lr_window,
        fractal_window=config.feature.fractal_window
    )

    # Compute base features on primary pair
    print("\nComputing features...")
    data_with_features = fe.compute_all_features(data, currency_data=None)

    # Compute currency strengths ONCE for all configured currencies, then join and split
    from features import compute_currency_strengths
    print(f"Computing currency strengths for majors: {config.CURRENCIES}")
    S = compute_currency_strengths(
        pair_dfs=pair_dfs,
        currencies=config.CURRENCIES,
        window=config.STRENGTH_WINDOW,
        lags=config.STRENGTH_LAGS
    )

    # Join strengths once, then split
    data_with_features = data_with_features.join(S, how="left").ffill().bfill()

    # Build feature list (include all strengths or pair-only)
    base, quote = primary_pair[:3], primary_pair[3:] if len(primary_pair) == 6 else "USD"

    if config.INCLUDE_ALL_STRENGTHS:
        strength_cols = []
        for c in config.CURRENCIES:
            col = f"strength_{c}"
            if col in data_with_features.columns:
                strength_cols.append(col)
                for k in range(1, config.STRENGTH_LAGS+1):
                    lag_col = f"strength_{c}_lag{k}"
                    if lag_col in data_with_features.columns:
                        strength_cols.append(lag_col)
    else:
        strength_cols = []
        for c in {base, quote}:
            col = f"strength_{c}"
            if col in data_with_features.columns:
                strength_cols.append(col)
                for k in range(1, config.STRENGTH_LAGS+1):
                    lag_col = f"strength_{c}_lag{k}"
                    if lag_col in data_with_features.columns:
                        strength_cols.append(lag_col)

    base_feature_cols = fe.get_feature_names(include_currency_strength=False)
    feature_columns = [c for c in base_feature_cols if c in data_with_features.columns] + strength_cols

    print(f"Using {len(feature_columns)} features:")
    print(f"  - Base features: {len([c for c in base_feature_cols if c in data_with_features.columns])}")
    print(f"  - Currency strengths: {len(strength_cols)}")
    print(f"  - Total: {len(feature_columns)}")

    # Split AFTER join so all splits share same feature set
    print("\nSplitting data...")
    train_data, val_data, test_data = loader.split_data(
        data_with_features,
        train_ratio=config.training.train_ratio,
        val_ratio=config.training.val_ratio,
        test_ratio=config.training.test_ratio
    )
    
    # Compute robust feature scaler from training data only (winsorized + MAD)
    print("\nComputing robust feature scaler from training data...")
    from scaler_utils import robust_fit
    
    scaler_stats = robust_fit(train_data, feature_columns)
    scaler = {
        'mu': scaler_stats['med'],   # Use median instead of mean
        'sig': scaler_stats['mad']    # Use MAD instead of std
    }
    print(f"  Feature median range: [{pd.Series(scaler['mu']).min():.4f}, {pd.Series(scaler['mu']).max():.4f}]")
    print(f"  Feature MAD range: [{pd.Series(scaler['sig']).min():.4f}, {pd.Series(scaler['sig']).max():.4f}]")
    
    return train_data, val_data, test_data, feature_columns, scaler, pair_dfs


def create_environments(train_data, val_data, test_data, feature_columns, scaler, pair_dfs, config: Config):
    """
    Create training, validation, and test environments.
    
    Args:
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        feature_columns: List of feature columns
        scaler: Feature normalization parameters (mu, sig)
        pair_dfs: Dict of {pair: DataFrame} with all currency pair data
        config: Configuration object
        
    Returns:
        Tuple of (train_env, val_env, test_env)
    """
    print("\n" + "=" * 50)
    print("CREATING ENVIRONMENTS")
    print("=" * 50)
    
    # Build fx_lookup for dynamic pip value calculation
    # Map each pair to its close price series aligned to primary pair index
    primary_pair = "EURUSD" if "EURUSD" in pair_dfs else list(pair_dfs.keys())[0]
    primary_index = pair_dfs[primary_pair].index
    
    fx_lookup = {}
    for pair, df in pair_dfs.items():
        if 'close' in df.columns:
            # Reindex to primary pair's index and forward-fill missing values
            fx_lookup[pair] = df['close'].reindex(primary_index).ffill()
    
    print(f"\nFX lookup created for {len(fx_lookup)} pairs (for dynamic pip values)")
    
    # Initialize risk manager
    risk_manager = RiskManager(
        contract_size=config.risk.contract_size,
        point=config.risk.point,
        leverage=config.risk.leverage,
        risk_per_trade=config.risk.risk_per_trade,
        atr_multiplier=config.risk.atr_multiplier,
        max_dd_threshold=config.risk.max_dd_threshold,
        margin_safety_factor=config.risk.margin_safety_factor,
        tp_multiplier=config.risk.tp_multiplier
    )
    
    # Build common environment kwargs (all balance-invariant parameters)
    env_kwargs = dict(
        feature_columns=feature_columns,
        initial_balance=config.environment.initial_balance,
        risk_manager=risk_manager,
        spread=config.environment.spread,
        commission=config.environment.commission,
        slippage_pips=config.environment.slippage_pips,
        weekend_close_hours=config.environment.weekend_close_hours,
        max_steps=config.training.max_steps_per_episode,
        scaler_mu=scaler['mu'],
        scaler_sig=scaler['sig'],
        cooldown_bars=config.environment.cooldown_bars,
        min_hold_bars=config.environment.min_hold_bars,
        trade_penalty=config.environment.trade_penalty,
        flip_penalty=config.environment.flip_penalty,
        max_trades_per_episode=config.environment.max_trades_per_episode,
        risk_per_trade=config.risk.risk_per_trade,
        atr_mult_sl=config.risk.atr_multiplier,  # Use existing atr_multiplier
        tp_mult=config.risk.tp_multiplier,  # Use existing tp_multiplier
        fx_lookup=fx_lookup  # Dynamic pip value conversion
    )
    
    print(f"\nEnvironment config:")
    print(f"  Spread: {env_kwargs['spread']:.5f}")
    print(f"  Commission: ${env_kwargs['commission']:.2f}")
    print(f"  Slippage: {env_kwargs['slippage_pips']:.1f} pips")
    print(f"  Risk/trade: {env_kwargs['risk_per_trade']*100:.2f}%")
    print(f"  ATR SL mult: {env_kwargs['atr_mult_sl']:.1f}")
    print(f"  TP mult: {env_kwargs['tp_mult']:.1f}")
    print(f"  Min hold: {env_kwargs['min_hold_bars']} bars")
    print(f"  Cooldown: {env_kwargs['cooldown_bars']} bars")
    print(f"  Max trades/ep: {env_kwargs['max_trades_per_episode']}")
    print(f"  Flip penalty: {env_kwargs['flip_penalty']}")
    print(f"  Trade penalty: {env_kwargs['trade_penalty']}")
    
    # Create environments with data-specific overrides
    train_env = ForexTradingEnv(data=train_data, **env_kwargs)
    val_env = ForexTradingEnv(data=val_data, **env_kwargs)
    test_env = ForexTradingEnv(data=test_data, **env_kwargs)
    
    print(f"\nEnvironments created:")
    print(f"  State size: {train_env.state_size}")
    print(f"  Action space: {train_env.action_space_size}")
    print(f"  Initial balance: ${config.environment.initial_balance}")
    
    return train_env, val_env, test_env


def create_agent(state_size: int, config: Config, train_env=None):
    """
    Create DQN agent.
    
    Args:
        state_size: Size of state space
        config: Configuration object
        train_env: Training environment (for adaptive learning_starts)
        
    Returns:
        DQN agent
    """
    print("\n" + "=" * 50)
    print("CREATING AGENT")
    print("=" * 50)
    
    # SURGICAL PATCH: Adaptive learning_starts based on episode length
    steps_per_ep = getattr(config.training, 'max_steps_per_episode', None)
    if steps_per_ep is None and train_env is not None:
        steps_per_ep = getattr(train_env, 'max_steps', len(train_env.data))
    
    # Force exact SMOKE value when in smoke mode (no adaptive calculation)
    if getattr(config, 'SMOKE_LEARN', False):
        learning_starts = config.SMOKE_LEARNING_STARTS
    else:
        learning_starts = min(5000, int(1.0 * steps_per_ep)) if steps_per_ep else 5000
    
    # Use NoisyNet if configured
    use_noisy = getattr(config.agent, 'use_noisy', False)
    noisy_sigma = getattr(config.agent, 'noisy_sigma_init', 0.017)
    
    # SURGICAL PATCH #2: Get SMOKE mode parameters (ensure minimums)
    update_every = max(getattr(config.agent, 'update_every', 4), 4)
    grad_steps = max(getattr(config.agent, 'grad_steps', 2), 2)
    
    # SURGICAL PATCH #7: Get gradient clipping parameter
    grad_clip = getattr(config.agent, 'grad_clip', 1.0)
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=ActionSpace.get_action_size(),
        learning_rate=config.agent.learning_rate,
        gamma=config.agent.gamma,
        epsilon_start=config.agent.epsilon_start,
        epsilon_end=config.agent.epsilon_end,
        epsilon_decay=config.agent.epsilon_decay,
        buffer_capacity=config.agent.buffer_capacity,
        batch_size=config.agent.batch_size,
        target_update_freq=config.agent.target_update_freq,
        hidden_sizes=config.agent.hidden_sizes,
        learning_starts=learning_starts,
        weight_decay=1e-6,  # Small L2 regularization
        buffer_type='prioritized',  # Use PER for better sample efficiency
        prioritized_replay_alpha=0.6,
        use_noisy=use_noisy,
        noisy_sigma_init=noisy_sigma,
        update_every=update_every,
        grad_steps=grad_steps,
        grad_clip=grad_clip
    )
    
    # Sync learning_starts with SMOKE mode (keeps header and agent consistent)
    # Force to exact SMOKE value when in smoke mode
    if getattr(config, 'SMOKE_LEARN', False):
        agent.learning_starts = config.SMOKE_LEARNING_STARTS
    
    print(f"\nAgent created:")
    print(f"  State size: {state_size}")
    print(f"  Action size: {ActionSpace.get_action_size()}")
    print(f"  Learning rate: {config.agent.learning_rate}")
    print(f"  Learning starts: {agent.learning_starts}")  # Print actual value from agent
    print(f"  Update every: {update_every} steps")
    print(f"  Grad steps: {grad_steps}")
    print(f"  Hidden layers: {config.agent.hidden_sizes}")
    print(f"  NoisyNet: {use_noisy}")
    print(f"  Buffer type: prioritized")
    print(f"  Weight decay: 1e-6")
    
    return agent


def train_agent(agent, train_env, val_env, config: Config, telemetry_mode='standard', output_dir=None):
    """
    Train the agent.
    
    Args:
        agent: DQN agent
        train_env: Training environment
        val_env: Validation environment
        config: Configuration object
        telemetry_mode: 'standard' or 'extended' telemetry logging
        output_dir: Output directory for results (optional)
        
    Returns:
        Training history
    """
    print("\n" + "=" * 50)
    print("TRAINING AGENT")
    print("=" * 50)
    
    # Initialize fitness calculator
    fitness_calculator = FitnessCalculator(
        sharpe_weight=config.fitness.sharpe_weight,
        cagr_weight=config.fitness.cagr_weight,
        stagnation_weight=config.fitness.stagnation_weight,
        loss_years_weight=config.fitness.loss_years_weight,
        ruin_penalty=config.fitness.ruin_penalty,
        ruin_threshold=config.fitness.ruin_threshold,
        clip_sharpe=getattr(config, "VAL_CLIP_SHARPE", 5.0),
        clip_cagr=getattr(config, "VAL_CLIP_CAGR", 1.0),
        min_bdays=getattr(config, "VAL_MIN_BDAYS", 60)
    )
    
    # Initialize trainer
    trainer = Trainer(
        agent=agent,
        train_env=train_env,
        val_env=val_env,
        fitness_calculator=fitness_calculator,
        checkpoint_dir=config.checkpoint_dir,
        log_dir=config.log_dir,
        val_spread_jitter=config.VAL_SPREAD_JITTER,
        val_commission_jitter=config.VAL_COMMISSION_JITTER,
        config=config  # Pass config for validation parameters
    )
    
    # Train
    print(f"\nStarting training for {config.training.num_episodes} episodes...")
    print(f"Telemetry mode: {telemetry_mode}")
    if output_dir:
        print(f"Output directory: {output_dir}")
    
    history = trainer.train(
        num_episodes=config.training.num_episodes,
        validate_every=config.training.validate_every,
        save_every=config.training.save_every,
        verbose=True,
        telemetry_mode=telemetry_mode,
        output_dir=output_dir
    )
    
    # Plot training curves
    trainer.plot_training_curves()
    
    return history


def evaluate_agent(agent, test_env, config: Config):
    """
    Evaluate agent on test set.
    
    Args:
        agent: DQN agent
        test_env: Test environment
        config: Configuration object
        
    Returns:
        Evaluation results
    """
    print("\n" + "=" * 50)
    print("EVALUATING AGENT")
    print("=" * 50)
    
    # Run episode
    state = test_env.reset()
    episode_reward = 0
    steps = 0
    
    done = False
    while not done:
        action = agent.select_action(state, explore=False, eval_mode=True, env=test_env)
        next_state, reward, done, info = test_env.step(action)
        state = next_state
        episode_reward += reward
        steps += 1
    
    # Calculate metrics
    trade_stats = test_env.get_trade_statistics()
    
    # Calculate fitness
    equity_series = pd.Series(
        test_env.equity_history,
        index=pd.date_range(start='2024-01-01', periods=len(test_env.equity_history), freq='H')
    )
    
    fitness_calculator = FitnessCalculator(
        sharpe_weight=config.fitness.sharpe_weight,
        cagr_weight=config.fitness.cagr_weight,
        stagnation_weight=config.fitness.stagnation_weight,
        loss_years_weight=config.fitness.loss_years_weight,
        ruin_penalty=config.fitness.ruin_penalty,
        ruin_threshold=config.fitness.ruin_threshold,
        clip_sharpe=getattr(config, "VAL_CLIP_SHARPE", 5.0),
        clip_cagr=getattr(config, "VAL_CLIP_CAGR", 1.0),
        min_bdays=getattr(config, "VAL_MIN_BDAYS", 60)
    )
    
    fitness_metrics = fitness_calculator.calculate_all_metrics(equity_series)
    
    # Combine results
    results = {
        'test_reward': episode_reward,
        'test_steps': steps,
        'test_final_equity': info['equity'],
        **{f'test_{k}': v for k, v in trade_stats.items()},
        **{f'test_{k}': v for k, v in fitness_metrics.items() if isinstance(v, (int, float, bool))},
    }
    
    # Print results
    print("\nTest Results:")
    print(f"  Final Equity: ${results['test_final_equity']:.2f}")
    print(f"  Return: {results['test_return']:.2f}%")
    print(f"  Total Trades: {results['test_total_trades']}")
    print(f"  Win Rate: {results['test_win_rate']:.2%}")
    print(f"  Profit Factor: {results['test_profit_factor']:.2f}")
    print(f"  Fitness Score: {results['test_fitness']:.4f}")
    print(f"  Sharpe Ratio: {results['test_sharpe']:.4f}")
    print(f"  CAGR: {results['test_cagr']:.2%}")
    print(f"  Max Drawdown: {results['test_max_drawdown_pct']:.2f}%")
    
    # Save results
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_file}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Forex RL Trading Bot')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'evaluate', 'both'],
                       help='Mode: train, evaluate, or both')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint to load for evaluation')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of training episodes (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (overrides config)')
    parser.add_argument('--telemetry', type=str, default='standard',
                       choices=['standard', 'extended'],
                       help='Telemetry level: standard or extended (for confirmation suite)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (overrides default)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Override episodes if specified
    if args.episodes is not None:
        config.training.num_episodes = args.episodes
    
    # Override seed if specified
    if args.seed is not None:
        config.random_seed = args.seed
    
    # Override output directory if specified
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config.checkpoint_dir = str(output_dir / 'checkpoints')
        config.log_dir = str(output_dir / 'logs')
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Apply smoke profile for short runs
    if config.SMOKE_LEARN and args.episodes is not None and args.episodes <= 5:
        print("\n[SMOKE] MODE ACTIVATED (short run optimizations)")
        config.training.max_steps_per_episode = config.SMOKE_MAX_STEPS_PER_EPISODE
        config.agent.target_update_freq = config.SMOKE_TARGET_UPDATE
        config.agent.buffer_capacity = config.SMOKE_BUFFER_CAPACITY
        # Keep exploration higher for short runs
        config.agent.epsilon_start = 0.4
        config.agent.epsilon_end = 0.10
        # SURGICAL PATCH #2: Force SMOKE learning parameters
        config.agent.update_every = 16  # Less frequent updates for stability
        config.agent.grad_steps = 1     # Single gradient step per update
        # Override risk settings for more trade signal in smoke runs
        config.environment.cooldown_bars = 12
        config.environment.min_hold_bars = 6
        print(f"  - Learning starts: {config.SMOKE_LEARNING_STARTS}")
        print(f"  - Max steps/episode: {config.SMOKE_MAX_STEPS_PER_EPISODE}")
        print(f"  - Target update freq: {config.SMOKE_TARGET_UPDATE}")
        print(f"  - Update every: 16 steps")
        print(f"  - Grad steps: 1")
        print(f"  - Epsilon range: 0.4 to 0.10")
        print(f"  - Min hold: 6 bars | Cooldown: 12 bars (reduced for signal)\n")
    else:
        # Production settings for longer runs (>5 episodes)
        config.agent.update_every = 4   # More frequent updates
        config.agent.grad_steps = 2      # Double gradient steps
    
    # Set random seeds
    set_random_seeds(config.random_seed)
    
    print("=" * 50)
    print("FOREX RL TRADING BOT")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    
    # Prepare data
    train_data, val_data, test_data, feature_columns, scaler, pair_dfs = prepare_data(config)
    
    # Create environments
    train_env, val_env, test_env = create_environments(
        train_data, val_data, test_data, feature_columns, scaler, pair_dfs, config
    )
    
    # Create agent
    agent = create_agent(train_env.state_size, config, train_env)
    
    # Train or evaluate
    if args.mode in ['train', 'both']:
        history = train_agent(agent, train_env, val_env, config, 
                             telemetry_mode=args.telemetry,
                             output_dir=args.output_dir)
    
    if args.mode in ['evaluate', 'both']:
        if args.checkpoint:
            agent.load(args.checkpoint)
        elif args.mode == 'evaluate':
            print("\nWarning: No checkpoint specified, using untrained agent")
        
        results = evaluate_agent(agent, test_env, config)
    
    print("\n" + "=" * 50)
    print("COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()

