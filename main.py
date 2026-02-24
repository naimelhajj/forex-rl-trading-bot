"""
Main Application Module
Entry point for training and evaluating the Forex RL trading bot.
"""

import sys
import os
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import argparse
import json
from datetime import datetime, timezone, timedelta

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
from spr_fitness import compute_spr_fitness
from trainer import Trainer


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _read_git_commit(repo_root: Path) -> str:
    head_path = repo_root / ".git" / "HEAD"
    try:
        head = head_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    if head.startswith("ref: "):
        ref = head.split("ref: ", 1)[1].strip()
        ref_path = repo_root / ".git" / ref
        try:
            return ref_path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
    return head


def write_run_artifacts(config: Config, args: argparse.Namespace):
    artifact_root = Path(args.output_dir) if args.output_dir else Path(config.log_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": args.mode,
        "seed": int(getattr(config, "random_seed", -1)),
        "episodes": int(getattr(config.training, "num_episodes", -1)),
        "output_dir": str(args.output_dir) if args.output_dir else None,
        "checkpoint_dir": str(config.checkpoint_dir),
        "log_dir": str(config.log_dir),
        "results_dir": str(config.results_dir),
        "argv": list(sys.argv),
        "git_commit": _read_git_commit(Path.cwd()),
    }

    config_path = artifact_root / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=True, default=str)

    metadata_path = artifact_root / "run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2, ensure_ascii=True, default=str)


def parse_pair_files_arg(arg: str) -> dict:
    if not arg:
        return None
    path = Path(arg)
    if path.exists():
        payload = path.read_text(encoding="utf-8")
    else:
        payload = arg
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("pair_files must be a JSON object of {PAIR: path}.")
    return {str(k): str(v) for k, v in data.items()}


def parse_allow_actions(arg: str) -> list[int]:
    if not arg:
        return None
    mapping = {
        "hold": 0,
        "long": 1,
        "short": 2,
        "close": 3,
        "flat": 3,
        "move_sl": 4,
        "movesl": 4,
        "trail": 4,
        "move": 4,
        "move_sl_aggr": 5,
        "movesl_aggr": 5,
        "move_fast": 5,
        "trail_fast": 5,
        "reverse_long": 6,
        "rev_long": 6,
        "revlong": 6,
        "reverse_short": 7,
        "rev_short": 7,
        "revshort": 7,
    }
    indices = []
    for raw in arg.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token.isdigit():
            idx = int(token)
        else:
            if token not in mapping:
                raise ValueError(
                    f"Unknown action token '{token}'. Use hold,long,short,close,move_sl,move_sl_aggr,reverse_long,reverse_short or 0-7."
                )
            idx = mapping[token]
        if idx < 0 or idx > 7:
            raise ValueError(f"Action index out of range: {idx} (allowed 0-7).")
        indices.append(idx)
    if not indices:
        return None
    return sorted(set(indices))

def resolve_pair_files(pair_files: dict, data_dir: str) -> dict:
    resolved = {}
    base = Path(data_dir) if data_dir else Path(".")
    base_name = base.name if data_dir else None
    for pair, filepath in pair_files.items():
        path = Path(filepath)
        if not path.is_absolute():
            if base_name and path.parts and path.parts[0] == base_name:
                path = path
            else:
                path = base / path
        resolved[pair] = str(path)
    return resolved


def validate_pair_dfs(pair_dfs: dict, require_datetime_index: bool = True):
    required_cols = {"open", "high", "low", "close"}
    for pair, df in pair_dfs.items():
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"Missing columns for {pair}: {sorted(missing)}")
        if require_datetime_index and not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{pair} index is not a DatetimeIndex.")
        if df.index.has_duplicates:
            raise ValueError(f"{pair} has duplicate timestamps.")
        if not df.index.is_monotonic_increasing:
            raise ValueError(f"{pair} index is not sorted ascending.")
        if df[list(required_cols)].isna().any().any():
            raise ValueError(f"{pair} has NaNs in OHLC columns.")


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

    # Load real CSV data or generate synthetic data
    use_real_data = (getattr(config.data, "data_mode", "synthetic") == "csv"
                     or getattr(config.data, "pair_files", None))
    if use_real_data:
        if not getattr(config.data, "pair_files", None):
            raise ValueError("CSV data mode requires data.pair_files.")
        pair_files = resolve_pair_files(config.data.pair_files, config.data.data_dir)
        print(f"\nLoading CSV data for {len(pair_files)} pairs: {list(pair_files.keys())}")
        pair_dfs = loader.load_multiple_pairs(
            pair_files,
            time_col=config.data.time_col,
            parse_dates=config.data.parse_dates,
            date_col=config.data.date_col,
            csv_sep=config.data.csv_sep
        )
        for pair, df in pair_dfs.items():
            pair_dfs[pair] = df.sort_index()
        validate_pair_dfs(pair_dfs, require_datetime_index=config.data.parse_dates)
        pairs = list(pair_dfs.keys())
    else:
        # Build pair universe
        pairs = config.PAIRS or config.data.pairs or PAIRS_7MAJ
        print(f"\nGenerating sample data for {len(pairs)} pairs: {pairs}")

        # Generate multi-pair data
        pair_dfs = loader.generate_multiple_pairs(
            pairs=pairs,
            n_bars=config.data.n_bars,
            start_date=config.data.start_date,
            freq=config.data.freq,
            volatility=getattr(config.data, "synthetic_volatility", 0.0005),
            drift=getattr(config.data, "synthetic_drift", 0.0)
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
    strength_cols = []
    if config.USE_CURRENCY_STRENGTHS:
        from features import compute_currency_strengths
        print(f"Computing currency strengths for majors: {config.CURRENCIES}")
        S = compute_currency_strengths(
            pair_dfs=pair_dfs,
            currencies=config.CURRENCIES,
            window=config.STRENGTH_WINDOW,
            lags=config.STRENGTH_LAGS
        )

        # Join strengths once, then split (no backfill to avoid lookahead)
        data_with_features = data_with_features.join(S, how="left")
        data_with_features = data_with_features.sort_index().ffill().dropna()
    else:
        data_with_features = data_with_features.sort_index().ffill().dropna()
    if data_with_features.empty:
        raise ValueError("No data left after feature generation. Check input data and windows.")

    # Build feature list (include all strengths or pair-only)
    base, quote = primary_pair[:3], primary_pair[3:] if len(primary_pair) == 6 else "USD"

    if config.USE_CURRENCY_STRENGTHS and config.INCLUDE_ALL_STRENGTHS:
        for c in config.CURRENCIES:
            col = f"strength_{c}"
            if col in data_with_features.columns:
                strength_cols.append(col)
                for k in range(1, config.STRENGTH_LAGS+1):
                    lag_col = f"strength_{c}_lag{k}"
                    if lag_col in data_with_features.columns:
                        strength_cols.append(lag_col)
    elif config.USE_CURRENCY_STRENGTHS:
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
    
    # Build isolated risk managers per environment to avoid cross-env state bleed.
    def _build_risk_manager() -> RiskManager:
        rm = RiskManager(
            contract_size=config.risk.contract_size,
            point=config.risk.point,
            leverage=config.risk.leverage,
            risk_per_trade=config.risk.risk_per_trade,
            atr_multiplier=config.risk.atr_multiplier,
            max_dd_threshold=config.risk.max_dd_threshold,
            margin_safety_factor=config.risk.margin_safety_factor,
            tp_multiplier=config.risk.tp_multiplier
        )
        rm.slippage_pips = float(config.environment.slippage_pips)
        return rm
    
    # Resolve swap settings (support optional per-symbol overrides).
    active_swap_type = getattr(config.environment, "swap_type", "usd")
    active_swap_long = float(getattr(config.environment, "swap_long_usd_per_lot_night", 0.0))
    active_swap_short = float(getattr(config.environment, "swap_short_usd_per_lot_night", 0.0))
    swap_map = getattr(config.environment, "swap_by_symbol", {}) or {}
    if isinstance(swap_map, dict):
        sym_cfg = swap_map.get(primary_pair)
        if isinstance(sym_cfg, dict):
            active_swap_type = str(sym_cfg.get("type", active_swap_type))
            active_swap_long = float(sym_cfg.get("long", active_swap_long))
            active_swap_short = float(sym_cfg.get("short", active_swap_short))

    # Build common environment kwargs (all balance-invariant parameters)
    env_kwargs = dict(
        feature_columns=feature_columns,
        initial_balance=config.environment.initial_balance,
        spread=config.environment.spread,
        commission=config.environment.commission,
        slippage_pips=config.environment.slippage_pips,
        swap_type=active_swap_type,
        swap_long_usd_per_lot_night=active_swap_long,
        swap_short_usd_per_lot_night=active_swap_short,
        swap_rollover_hour_utc=config.environment.swap_rollover_hour_utc,
        swap_triple_weekday=config.environment.swap_triple_weekday,
        weekend_close_hours=config.environment.weekend_close_hours,
        scaler_mu=scaler['mu'],
        scaler_sig=scaler['sig'],
        cooldown_bars=config.environment.cooldown_bars,
        min_hold_bars=config.environment.min_hold_bars,
        min_trail_buffer_pips=config.environment.min_trail_buffer_pips,
        disable_move_sl=config.environment.disable_move_sl,
        allowed_actions=config.environment.allowed_actions,
        trade_penalty=config.environment.trade_penalty,
        flip_penalty=config.environment.flip_penalty,
        min_atr_cost_ratio=config.environment.min_atr_cost_ratio,
        use_regime_filter=config.environment.use_regime_filter,
        regime_min_vol_z=config.environment.regime_min_vol_z,
        regime_align_trend=config.environment.regime_align_trend,
        regime_require_trending=config.environment.regime_require_trending,
        max_trades_per_episode=config.environment.max_trades_per_episode,
        risk_per_trade=config.risk.risk_per_trade,
        atr_mult_sl=config.risk.atr_multiplier,  # Use existing atr_multiplier
        tp_mult=config.risk.tp_multiplier,  # Use existing tp_multiplier
        reward_clip=config.environment.reward_clip,
        holding_cost=config.environment.holding_cost,
        r_multiple_reward_weight=config.environment.r_multiple_reward_weight,
        r_multiple_reward_clip=config.environment.r_multiple_reward_clip,
        fx_lookup=fx_lookup  # Dynamic pip value conversion
    )
    
    print(f"\nEnvironment config:")
    print(f"  Spread: {env_kwargs['spread']:.5f}")
    print(f"  Commission: ${env_kwargs['commission']:.2f}")
    print(f"  Slippage: {env_kwargs['slippage_pips']:.1f} pips")
    if str(env_kwargs.get('swap_type', 'usd')).lower().startswith('point'):
        print(f"  Swap type: points")
        print(f"  Swap long/short: {env_kwargs['swap_long_usd_per_lot_night']:.2f} / {env_kwargs['swap_short_usd_per_lot_night']:.2f} points")
    else:
        print(f"  Swap type: USD/lot/night")
        print(f"  Swap long/short: ${env_kwargs['swap_long_usd_per_lot_night']:.2f} / ${env_kwargs['swap_short_usd_per_lot_night']:.2f}")
    print(f"  Swap rollover UTC: {env_kwargs['swap_rollover_hour_utc']:02d}:00 (triple weekday={env_kwargs['swap_triple_weekday']})")
    print(f"  Leverage: 1:{config.risk.leverage}")
    print(f"  Risk/trade: {env_kwargs['risk_per_trade']*100:.2f}%")
    print(f"  ATR SL mult: {env_kwargs['atr_mult_sl']:.1f}")
    print(f"  TP mult: {env_kwargs['tp_mult']:.1f}")
    print(f"  Min hold: {env_kwargs['min_hold_bars']} bars")
    print(f"  Cooldown: {env_kwargs['cooldown_bars']} bars")
    print(f"  Max trades/ep: {env_kwargs['max_trades_per_episode']}")
    print(f"  Disable MOVE_SL: {env_kwargs['disable_move_sl']}")
    if env_kwargs.get('allowed_actions') is not None:
        print(f"  Allowed actions: {env_kwargs['allowed_actions']}")
    print(f"  Reward clip: {env_kwargs['reward_clip']}")
    print(f"  Holding cost: {env_kwargs['holding_cost']}")
    print(f"  Flip penalty: {env_kwargs['flip_penalty']}")
    print(f"  Trade penalty: {env_kwargs['trade_penalty']}")
    print(f"  Min ATR cost ratio: {env_kwargs['min_atr_cost_ratio']}")
    print(f"  Regime filter: {env_kwargs['use_regime_filter']}")
    print(f"  Regime min vol z: {env_kwargs['regime_min_vol_z']}")
    print(f"  Regime align trend: {env_kwargs['regime_align_trend']}")
    print(f"  Regime require trending: {env_kwargs['regime_require_trending']}")
    print(f"  R-multiple reward weight: {env_kwargs['r_multiple_reward_weight']}")
    print(f"  R-multiple reward clip: {env_kwargs['r_multiple_reward_clip']}")
    print(f"  Train random starts: {getattr(config.training, 'random_episode_start', True)}")
    
    # Create environments with data-specific overrides
    train_env = ForexTradingEnv(
        data=train_data,
        risk_manager=_build_risk_manager(),
        max_steps=config.training.max_steps_per_episode,
        random_episode_start=getattr(config.training, 'random_episode_start', True),
        symbol=primary_pair,
        **env_kwargs,
    )
    val_env = ForexTradingEnv(
        data=val_data,
        risk_manager=_build_risk_manager(),
        max_steps=config.training.max_steps_per_episode,
        random_episode_start=False,
        symbol=primary_pair,
        **env_kwargs,
    )
    test_env = ForexTradingEnv(
        data=test_data,
        risk_manager=_build_risk_manager(),
        max_steps=config.training.max_steps_per_episode,
        random_episode_start=False,
        symbol=primary_pair,
        **env_kwargs,
    )
    
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
        grad_clip=grad_clip,
        use_dual_controller=config.agent.use_dual_controller,
        use_symmetry_loss=config.agent.use_symmetry_loss,
        symmetry_loss_weight=config.agent.symmetry_loss_weight,
        trade_gate_margin=config.agent.trade_gate_margin,
        trade_gate_z=config.agent.trade_gate_z,
        disable_trade_gate_in_eval=getattr(config, "VAL_DISABLE_TRADE_GATING", False),
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
    print(f"  Eval trade gate disabled: {getattr(config, 'VAL_DISABLE_TRADE_GATING', False)}")
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
    def _resolve_step_timestamp(env, base_timestamp):
        try:
            idx = env.data.index[env.current_step]
        except Exception:
            idx = None
        if hasattr(idx, "to_pydatetime"):
            return idx.to_pydatetime()
        if isinstance(idx, np.datetime64):
            return idx
        if isinstance(idx, datetime):
            return idx
        return base_timestamp + timedelta(hours=env.current_step)

    def _compute_pf_override(equity_curve, pf_cap):
        if len(equity_curve) < 2:
            return None
        eq_array = np.asarray(equity_curve, dtype=float)
        diffs = np.diff(eq_array)
        gross_profit = diffs[diffs > 0].sum()
        gross_loss = -diffs[diffs < 0].sum()
        if gross_loss <= 0 and gross_profit > 0:
            return float(pf_cap)
        if gross_loss > 0:
            return float(min(gross_profit / gross_loss, pf_cap))
        return 0.0

    def _extract_trade_count(trade_stats):
        return int(
            trade_stats.get("trades") or
            trade_stats.get("total_trades") or
            trade_stats.get("num_trades") or
            0
        )

    def _run_walkforward_slice(env, start_idx, end_idx):
        state = env.reset()
        if start_idx > 0:
            env._frame_stack = None
            stack_n = int(getattr(env, "stack_n", 1))
            start_fill = max(0, int(start_idx) - max(stack_n - 1, 0))
            for idx in range(start_fill, int(start_idx) + 1):
                env.current_step = idx
                state = env._get_state()
        else:
            state = env._get_state()
        if hasattr(env, "equity_history"):
            env.equity_history = [getattr(env, "equity", config.environment.initial_balance)]
        if hasattr(env, "trade_history"):
            env.trade_history = []

        base_timestamp = datetime(2024, 1, 1)
        timestamps = []
        equity_curve = []

        done = False
        while not done and env.current_step < end_idx:
            timestamps.append(_resolve_step_timestamp(env, base_timestamp))
            mask = getattr(env, "legal_action_mask", lambda: None)()
            action = agent.select_action(state, explore=False, mask=mask, eval_mode=True, env=env)
            next_state, _, done, info = env.step(action)
            equity_curve.append(info.get("equity", getattr(env, "equity", 0.0)))
            state = next_state

        trade_stats = env.get_trade_statistics()
        trade_pnls = [t.get("pnl", 0.0) for t in env.trade_history if isinstance(t, dict)]
        trade_count = _extract_trade_count(trade_stats)
        pf_override = None if trade_pnls else _compute_pf_override(equity_curve, config.fitness.spr_pf_cap)

        spr_score, spr_info = compute_spr_fitness(
            timestamps=timestamps,
            equity_curve=equity_curve,
            trade_pnls=trade_pnls if trade_pnls else None,
            pf_override=pf_override,
            trade_count_override=trade_count if not trade_pnls else None,
            initial_balance=config.environment.initial_balance,
            seconds_per_bar=3600,
            pf_cap=config.fitness.spr_pf_cap,
            dd_floor_pct=config.fitness.spr_dd_floor_pct,
            target_trades_per_year=config.fitness.spr_target_trades_per_year,
            use_pandas=config.fitness.spr_use_pandas,
        )

        return {
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "spr": float(spr_score),
            "pf": float(spr_info.get("pf", 0.0)),
            "mdd_pct": float(spr_info.get("mdd_pct", 0.0)),
            "mmr_pct_mean": float(spr_info.get("mmr_pct_mean", 0.0)),
            "trades": int(trade_count),
        }

    def _walkforward_windows(total_bars):
        window_bars = config.fitness.test_walkforward_window_bars
        if window_bars is None:
            window_bars = getattr(config, "VAL_WINDOW_BARS", None)
        if window_bars is None:
            window_frac = getattr(config, "VAL_WINDOW_FRAC", 0.06)
            window_bars = max(1, int(total_bars * window_frac))
        stride_frac = config.fitness.test_walkforward_stride_frac
        if stride_frac is None:
            stride_frac = getattr(config, "VAL_STRIDE_FRAC", 0.12)
        stride_bars = max(1, int(window_bars * stride_frac))

        windows = []
        start = 0
        while start + window_bars <= total_bars:
            windows.append((start, start + window_bars))
            start += stride_bars
        if not windows and total_bars > 0:
            windows = [(0, total_bars)]
        return windows

    
    print("\n" + "=" * 50)
    print("EVALUATING AGENT")
    print("=" * 50)
    
    # Run episode
    state = test_env.reset()
    episode_reward = 0
    steps = 0
    base_timestamp = datetime(2024, 1, 1)
    step_timestamps = []
    step_equity = []
    
    done = False
    while not done:
        step_timestamps.append(_resolve_step_timestamp(test_env, base_timestamp))
        mask = getattr(test_env, "legal_action_mask", lambda: None)()
        action = agent.select_action(state, explore=False, mask=mask, eval_mode=True, env=test_env)
        next_state, reward, done, info = test_env.step(action)
        state = next_state
        episode_reward += reward
        steps += 1
        step_equity.append(info.get("equity", getattr(test_env, "equity", 0.0)))
    
    # Calculate metrics
    trade_stats = test_env.get_trade_statistics()
    
    fitness_mode = getattr(config.fitness, "mode", "legacy")

    equity_series = pd.Series(
        test_env.equity_history,
        index=pd.date_range(start="2024-01-01", periods=len(test_env.equity_history), freq="h")
    )

    fitness_metrics = {}
    spr_info = {}
    spr_score = 0.0

    if fitness_mode == "spr":
        trade_pnls = [t.get("pnl", 0.0) for t in test_env.trade_history if isinstance(t, dict)]
        trade_count = _extract_trade_count(trade_stats)
        pf_override = None if trade_pnls else _compute_pf_override(step_equity, config.fitness.spr_pf_cap)
        spr_score, spr_info = compute_spr_fitness(
            timestamps=step_timestamps,
            equity_curve=step_equity,
            trade_pnls=trade_pnls if trade_pnls else None,
            pf_override=pf_override,
            trade_count_override=trade_count if not trade_pnls else None,
            initial_balance=config.environment.initial_balance,
            seconds_per_bar=3600,
            pf_cap=config.fitness.spr_pf_cap,
            dd_floor_pct=config.fitness.spr_dd_floor_pct,
            target_trades_per_year=config.fitness.spr_target_trades_per_year,
            use_pandas=config.fitness.spr_use_pandas,
        )
        fitness_metrics = {
            "fitness": float(spr_score),
            "spr": float(spr_score),
            "spr_pf": float(spr_info.get("pf", 0.0)),
            "spr_mdd_pct": float(spr_info.get("mdd_pct", 0.0)),
            "spr_mmr_pct_mean": float(spr_info.get("mmr_pct_mean", 0.0)),
            "spr_trades_per_year": float(spr_info.get("trades_per_year", 0.0)),
            "spr_significance": float(spr_info.get("significance", 0.0)),
            "spr_stagnation_penalty": float(spr_info.get("stagnation_penalty", 0.0)),
        }
    else:
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
    
    # Compute return and max drawdown for reporting
    if len(equity_series):
        equity_start = float(equity_series.iloc[0])
        equity_end = float(equity_series.iloc[-1])
        test_return = (equity_end / max(equity_start, 1e-9) - 1.0) * 100.0
        peak = equity_series.expanding(min_periods=1).max()
        drawdown = (equity_series - peak) / peak
        max_drawdown_pct = float(drawdown.min() * 100.0)
    else:
        equity_start = 0.0
        equity_end = 0.0
        test_return = 0.0
        max_drawdown_pct = 0.0

    walkforward = {}
    if fitness_mode == "spr":
        total_bars = len(test_env.data)
        max_steps = getattr(test_env, "max_steps", None)
        if max_steps:
            total_bars = min(total_bars, int(max_steps))
        windows = _walkforward_windows(total_bars)
        wf_results = [_run_walkforward_slice(test_env, start, end) for start, end in windows]
        sprs = [w["spr"] for w in wf_results]
        pfs = [w["pf"] for w in wf_results]
        pos_frac = float(sum(1 for s in sprs if s > 0.0) / max(1, len(sprs)))
        median_spr = float(np.median(sprs)) if sprs else 0.0
        median_pf = float(np.median(pfs)) if pfs else 0.0
        wf_pass = (
            len(wf_results) >= config.fitness.test_walkforward_min_windows and
            median_spr >= config.fitness.test_walkforward_min_spr and
            median_pf >= config.fitness.test_walkforward_min_pf and
            pos_frac >= config.fitness.test_walkforward_min_pos_frac
        )
        walkforward = {
            "windows": wf_results,
            "summary": {
                "windows": len(wf_results),
                "median_spr": median_spr,
                "median_pf": median_pf,
                "positive_frac": pos_frac,
            },
            "pass": bool(wf_pass),
            "requirements": {
                "min_windows": config.fitness.test_walkforward_min_windows,
                "min_spr": config.fitness.test_walkforward_min_spr,
                "min_pf": config.fitness.test_walkforward_min_pf,
                "min_positive_frac": config.fitness.test_walkforward_min_pos_frac,
            },
        }

    # Combine results
    results = {
        'test_reward': episode_reward,
        'test_steps': steps,
        'test_final_equity': info['equity'],
        'test_return': test_return,
        'test_max_drawdown_pct': max_drawdown_pct,
        **{f'test_{k}': v for k, v in trade_stats.items()},
        **{f'test_{k}': v for k, v in fitness_metrics.items() if isinstance(v, (int, float, bool))},
    }
    if walkforward:
        results["walkforward"] = walkforward
        results["test_walkforward_pass"] = walkforward.get("pass", False)
        summary = walkforward.get("summary", {})
        results["test_walkforward_windows"] = summary.get("windows", 0)
        results["test_walkforward_median_spr"] = summary.get("median_spr", 0.0)
        results["test_walkforward_median_pf"] = summary.get("median_pf", 0.0)
        results["test_walkforward_positive_frac"] = summary.get("positive_frac", 0.0)
    
    # Print results
    print("\nTest Results:")
    print(f"  Final Equity: ${results['test_final_equity']:.2f}")
    print(f"  Return: {results['test_return']:.2f}%")
    print(f"  Total Trades: {results['test_total_trades']}")
    print(f"  Win Rate: {results['test_win_rate']:.2%}")
    print(f"  Profit Factor: {results['test_profit_factor']:.2f}")
    if fitness_mode == "spr":
        print(f"  SPR Score: {results.get('test_spr', results.get('test_fitness', 0.0)):.4f}")
        print(f"  SPR PF: {results.get('test_spr_pf', 0.0):.2f}")
        print(f"  SPR MDD: {results.get('test_spr_mdd_pct', 0.0):.2f}%")
        print(f"  SPR MMR: {results.get('test_spr_mmr_pct_mean', 0.0):.3f}%")
        if walkforward:
            wf_summary = walkforward.get("summary", {})
            print(f"  Walk-forward: {walkforward.get('pass', False)} | "
                  f"median SPR {wf_summary.get('median_spr', 0.0):.4f} | "
                  f"pos frac {wf_summary.get('positive_frac', 0.0):.2f}")
    else:
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
    parser.add_argument('--max-steps-per-episode', type=int, default=None,
                       help='Override max steps per episode')
    parser.add_argument('--train-random-start', dest='train_random_start', action='store_true',
                       help='Sample random episode start bars during training')
    parser.add_argument('--no-train-random-start', dest='train_random_start', action='store_false',
                       help='Always train from the start of the split (deterministic episodes)')
    parser.set_defaults(train_random_start=None)
    parser.add_argument('--episode-timeout-min', type=float, default=None,
                       help='Abort a training episode if wall-clock time exceeds this many minutes')
    parser.add_argument('--heartbeat-secs', type=float, default=None,
                       help='Print episode heartbeat every N seconds (training only)')
    parser.add_argument('--heartbeat-steps', type=int, default=None,
                       help='Print episode heartbeat every N env steps (training only)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (overrides config)')
    parser.add_argument('--telemetry', type=str, default='standard',
                       choices=['standard', 'extended'],
                       help='Telemetry level: standard or extended (for confirmation suite)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (overrides default)')
    parser.add_argument('--data-mode', type=str, default=None,
                       choices=['synthetic', 'csv'],
                       help='Data mode: synthetic or csv')
    parser.add_argument('--n-bars', type=int, default=None,
                       help='Override synthetic data bars (synthetic mode only)')
    parser.add_argument('--synthetic-drift', type=float, default=None,
                       help='Override synthetic drift (synthetic mode only)')
    parser.add_argument('--synthetic-volatility', type=float, default=None,
                       help='Override synthetic volatility (synthetic mode only)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Base directory for CSV data files')
    parser.add_argument('--pair-files', type=str, default=None,
                       help='JSON dict or JSON file path: {PAIR: csv_path}')
    parser.add_argument('--use-symmetry-loss', dest='use_symmetry_loss', action='store_true',
                       help='Enable symmetry loss augmentation')
    parser.add_argument('--no-symmetry-loss', dest='use_symmetry_loss', action='store_false',
                       help='Disable symmetry loss augmentation')
    parser.set_defaults(use_symmetry_loss=None)
    parser.add_argument('--symmetry-loss-weight', type=float, default=None,
                       help='Override symmetry loss weight')
    parser.add_argument('--use-dual-controller', dest='use_dual_controller', action='store_true',
                       help='Enable dual controller action balancing')
    parser.add_argument('--no-dual-controller', dest='use_dual_controller', action='store_false',
                       help='Disable dual controller action balancing')
    parser.set_defaults(use_dual_controller=None)
    parser.add_argument('--use-strengths', dest='use_strengths', action='store_true',
                       help='Include currency strength features')
    parser.add_argument('--no-strengths', dest='use_strengths', action='store_false',
                       help='Disable currency strength features')
    parser.set_defaults(use_strengths=None)
    parser.add_argument('--strengths-all', dest='strengths_all', action='store_true',
                       help='Include all currency strengths (when enabled)')
    parser.add_argument('--strengths-pair-only', dest='strengths_all', action='store_false',
                       help='Include only base/quote strengths (when enabled)')
    parser.set_defaults(strengths_all=None)
    parser.add_argument('--csv-sep', type=str, default=None,
                       help='CSV delimiter (e.g., \",\" or \"\\t\")')
    parser.add_argument('--date-col', type=str, default=None,
                        help='Date column name for split date/time CSVs')
    parser.add_argument('--time-col', type=str, default=None,
                        help='Time column name for CSV data')
    parser.add_argument('--broker-profile', type=str, default=None,
                        choices=['hfm-premium'],
                        help='Apply broker defaults (costs/leverage) for a known account profile')
    parser.add_argument('--eval-zero-costs', action='store_true',
                        help='Zero spread/commission/slippage in evaluate mode')
    parser.add_argument('--spread', type=float, default=None,
                        help='Override spread (applies to train/eval)')
    parser.add_argument('--commission', type=float, default=None,
                        help='Override commission (applies to train/eval)')
    parser.add_argument('--slippage-pips', type=float, default=None,
                        help='Override slippage in pips (applies to train/eval)')
    parser.add_argument('--leverage', type=int, default=None,
                        help='Override account leverage used for margin sizing')
    parser.add_argument('--swap-long', type=float, default=None,
                        help='Override overnight swap for LONG positions (unit depends on --swap-type)')
    parser.add_argument('--swap-short', type=float, default=None,
                        help='Override overnight swap for SHORT positions (unit depends on --swap-type)')
    parser.add_argument('--swap-type', type=str, default=None,
                        choices=['usd', 'points'],
                        help='Interpret swap-long/swap-short as USD per lot/night or broker swap points')
    parser.add_argument('--swap-rollover-hour-utc', type=int, default=None,
                        help='Override rollover hour in UTC for daily swap charging (0-23)')
    parser.add_argument('--swap-triple-weekday', type=int, default=None,
                        help='Override triple-swap weekday (0=Mon..6=Sun, FX typically 2=Wed)')
    parser.add_argument('--reward-clip', type=float, default=None,
                        help='Override reward clip for log-return (applies to train/eval)')
    parser.add_argument('--holding-cost', type=float, default=None,
                        help='Override per-step holding cost (applies to train/eval)')
    parser.add_argument('--validate-every', type=int, default=None,
                        help='Override validation frequency (episodes)')
    parser.add_argument('--val-window-bars', type=int, default=None,
                        help='Override validation window size (bars)')
    parser.add_argument('--val-jitter-draws', type=int, default=None,
                        help='Override validation jitter draws (K)')
    parser.add_argument('--val-min-k', type=int, default=None,
                        help='Override minimum validation windows')
    parser.add_argument('--val-max-k', type=int, default=None,
                        help='Override maximum validation windows')
    parser.add_argument('--trade-penalty', type=float, default=None,
                        help='Override trade penalty')
    parser.add_argument('--flip-penalty', type=float, default=None,
                        help='Override flip penalty')
    parser.add_argument('--min-atr-cost-ratio', type=float, default=None,
                        help='Gate trades unless ATR exceeds cost ratio (0 disables)')
    parser.add_argument('--disable-move-sl', dest='disable_move_sl', action='store_true',
                        help='Disable MOVE_SL_CLOSER action (simpler action space)')
    parser.add_argument('--enable-move-sl', dest='disable_move_sl', action='store_false',
                        help='Enable MOVE_SL_CLOSER action')
    parser.set_defaults(disable_move_sl=None)
    parser.add_argument('--allow-actions', type=str, default=None,
                        help='Comma-separated list: hold,long,short,close,move_sl,move_sl_aggr,reverse_long,reverse_short or 0-7')
    parser.add_argument('--use-regime-filter', dest='use_regime_filter', action='store_true',
                        help='Enable regime filter (trend/vol gating)')
    parser.add_argument('--no-regime-filter', dest='use_regime_filter', action='store_false',
                        help='Disable regime filter')
    parser.set_defaults(use_regime_filter=None)
    parser.add_argument('--regime-min-vol-z', type=float, default=None,
                        help='Minimum realized_vol_24h_z to allow trades (0 disables)')
    parser.add_argument('--regime-align-trend', dest='regime_align_trend', action='store_true',
                        help='Require trend_96h alignment for LONG/SHORT entries')
    parser.add_argument('--regime-no-align-trend', dest='regime_align_trend', action='store_false',
                        help='Disable trend alignment requirement')
    parser.set_defaults(regime_align_trend=None)
    parser.add_argument('--regime-require-trending', dest='regime_require_trending', action='store_true',
                        help='Require is_trending flag to allow trades')
    parser.add_argument('--regime-no-require-trending', dest='regime_require_trending', action='store_false',
                        help='Disable is_trending requirement')
    parser.set_defaults(regime_require_trending=None)
    parser.add_argument('--cooldown-bars', type=int, default=None,
                        help='Override cooldown bars')
    parser.add_argument('--min-hold-bars', type=int, default=None,
                        help='Override minimum hold bars')
    parser.add_argument('--max-trades-per-episode', type=int, default=None,
                        help='Override max trades per episode')
    parser.add_argument('--hold-tie-tau', type=float, default=None,
                        help='Override hold tie tolerance for hold-break logic')
    parser.add_argument('--hold-break-after', type=int, default=None,
                        help='Override hold streak length before probing non-hold actions')
    parser.add_argument('--epsilon-start', type=float, default=None,
                        help='Override epsilon start for exploration')
    parser.add_argument('--epsilon-end', type=float, default=None,
                        help='Override epsilon end for exploration')
    parser.add_argument('--epsilon-decay', type=float, default=None,
                        help='Override epsilon decay for exploration')
    parser.add_argument('--trade-gate-margin', type=float, default=None,
                        help='Override minimum Q-gap over HOLD required to allow new trades')
    parser.add_argument('--trade-gate-z', type=float, default=None,
                        help='Override minimum Q-gap over HOLD in Q std units required to allow new trades')
    parser.add_argument('--r-multiple-reward-weight', type=float, default=None,
                        help='Reward shaping weight for realized R-multiples (0 disables)')
    parser.add_argument('--r-multiple-reward-clip', type=float, default=None,
                        help='Clip for realized R-multiple shaping')
    parser.add_argument('--atr-mult-sl', type=float, default=None,
                        help='Override ATR multiplier for stop loss')
    parser.add_argument('--tp-mult', type=float, default=None,
                        help='Override take-profit multiplier relative to SL')
    parser.add_argument('--prefill-policy', type=str, default=None,
                        choices=['baseline', 'random', 'none'],
                        help='Replay prefill policy: baseline, random, or none')
    parser.add_argument('--anti-regression-checkpoint', dest='anti_regression_checkpoint_selection', action='store_true',
                        help='Enable anti-regression checkpoint tournament at end of training')
    parser.add_argument('--no-anti-regression-checkpoint', dest='anti_regression_checkpoint_selection', action='store_false',
                        help='Disable anti-regression checkpoint tournament')
    parser.set_defaults(anti_regression_checkpoint_selection=None)
    parser.add_argument('--anti-regression-top-k', type=int, default=None,
                        help='Top-K checkpoints to evaluate in anti-regression tournament')
    parser.add_argument('--anti-regression-candidate-keep', type=int, default=None,
                        help='How many candidate checkpoints to retain during training')
    parser.add_argument('--anti-regression-min-validations', type=int, default=None,
                        help='Minimum validations required before anti-regression tournament')
    parser.add_argument('--anti-regression-selector-mode', type=str, default=None,
                        choices=['tail_holdout', 'future_first', 'auto_rescue', 'base_first'],
                        help='Checkpoint selector mode: tail_holdout (default), future_first, auto_rescue, or base_first')
    parser.add_argument('--anti-regression-auto-rescue', dest='anti_regression_auto_rescue_enabled', action='store_true',
                        help='Enable auto-rescue trigger when selector mode is auto_rescue')
    parser.add_argument('--anti-regression-no-auto-rescue', dest='anti_regression_auto_rescue_enabled', action='store_false',
                        help='Disable auto-rescue trigger even when selector mode is auto_rescue')
    parser.set_defaults(anti_regression_auto_rescue_enabled=None)
    parser.add_argument('--anti-regression-rescue-winner-forward-return-max', type=float, default=None,
                        help='Auto-rescue trigger: max forward return allowed for tail winner before rescue is considered')
    parser.add_argument('--anti-regression-rescue-forward-return-edge-min', type=float, default=None,
                        help='Auto-rescue trigger: min forward return edge required for future-first challenger')
    parser.add_argument('--anti-regression-rescue-forward-pf-edge-min', type=float, default=None,
                        help='Auto-rescue trigger: min forward PF edge required for future-first challenger')
    parser.add_argument('--anti-regression-rescue-challenger-base-return-max', type=float, default=None,
                        help='Auto-rescue trigger: challenger base return must be <= this threshold')
    parser.add_argument('--anti-regression-rescue-challenger-forward-pf-min', type=float, default=None,
                        help='Auto-rescue trigger: challenger forward PF must be >= this threshold')
    parser.add_argument('--anti-regression-eval-min-k', type=int, default=None,
                        help='Override VAL_MIN_K only during anti-regression tournament')
    parser.add_argument('--anti-regression-eval-max-k', type=int, default=None,
                        help='Override VAL_MAX_K only during anti-regression tournament')
    parser.add_argument('--anti-regression-eval-jitter-draws', type=int, default=None,
                        help='Override VAL_JITTER_DRAWS only during anti-regression tournament')
    parser.add_argument('--anti-regression-alt-stride-frac', type=float, default=None,
                        help='Stride fraction for anti-regression secondary hold-out validation')
    parser.add_argument('--anti-regression-alt-window-bars', type=int, default=None,
                        help='Window bars for anti-regression secondary hold-out validation')
    parser.add_argument('--anti-regression-tail-start-frac', type=float, default=None,
                        help='Tail-segment start fraction for anti-regression checkpoint tournament')
    parser.add_argument('--anti-regression-tail-end-frac', type=float, default=None,
                        help='Tail-segment end fraction for anti-regression checkpoint tournament')
    parser.add_argument('--anti-regression-tail-weight', type=float, default=None,
                        help='Penalty weight for negative tail-segment returns in anti-regression tournament')
    parser.add_argument('--anti-regression-base-return-floor', type=float, default=None,
                        help='Soft floor for base-return in anti-regression checkpoint tournament')
    parser.add_argument('--anti-regression-base-penalty-weight', type=float, default=None,
                        help='Penalty weight when base-return falls below the configured floor')
    parser.add_argument('--anti-regression-tiebreak', dest='anti_regression_tiebreak_enabled', action='store_true',
                        help='Enable top-2 checkpoint tie-break probe on a longer validation slice')
    parser.add_argument('--anti-regression-no-tiebreak', dest='anti_regression_tiebreak_enabled', action='store_false',
                        help='Disable top-2 checkpoint tie-break probe')
    parser.set_defaults(anti_regression_tiebreak_enabled=None)
    parser.add_argument('--anti-regression-tiebreak-window-bars', type=int, default=None,
                        help='Window length (bars) for top-2 tie-break probe (default 2400)')
    parser.add_argument('--anti-regression-tiebreak-start-frac', type=float, default=None,
                        help='Validation segment start fraction for top-2 tie-break probe')
    parser.add_argument('--anti-regression-tiebreak-end-frac', type=float, default=None,
                        help='Validation segment end fraction for top-2 tie-break probe')
    parser.add_argument('--anti-regression-tiebreak-return-edge-min', type=float, default=None,
                        help='Minimum return edge required for challenger to win top-2 tie-break')
    parser.add_argument('--anti-regression-tiebreak-pf-edge-min', type=float, default=None,
                        help='Minimum PF edge required for challenger to win top-2 tie-break')
    parser.add_argument('--anti-regression-tiebreak-min-trades', type=float, default=None,
                        help='Minimum median trades required for challenger in top-2 tie-break')
    parser.add_argument('--anti-regression-horizon-rescue', dest='anti_regression_horizon_rescue_enabled', action='store_true',
                        help='Enable longer-horizon rescue probe for weak checkpoint winners')
    parser.add_argument('--anti-regression-no-horizon-rescue', dest='anti_regression_horizon_rescue_enabled', action='store_false',
                        help='Disable longer-horizon rescue probe')
    parser.set_defaults(anti_regression_horizon_rescue_enabled=None)
    parser.add_argument('--anti-regression-horizon-window-bars', type=int, default=None,
                        help='Window length (bars) for horizon-rescue probe')
    parser.add_argument('--anti-regression-horizon-start-frac', type=float, default=None,
                        help='Validation segment start fraction for horizon-rescue probe')
    parser.add_argument('--anti-regression-horizon-end-frac', type=float, default=None,
                        help='Validation segment end fraction for horizon-rescue probe')
    parser.add_argument('--anti-regression-horizon-candidate-limit', type=int, default=None,
                        help='Max number of distinct candidates to evaluate in horizon-rescue probe')
    parser.add_argument('--anti-regression-horizon-incumbent-return-max', type=float, default=None,
                        help='Horizon-rescue trigger: incumbent long-horizon return must be <= this threshold')
    parser.add_argument('--anti-regression-horizon-return-edge-min', type=float, default=None,
                        help='Horizon-rescue trigger: challenger long-horizon return edge minimum')
    parser.add_argument('--anti-regression-horizon-pf-edge-min', type=float, default=None,
                        help='Horizon-rescue trigger: challenger long-horizon PF edge minimum')
    parser.add_argument('--anti-regression-horizon-challenger-base-return-max', type=float, default=None,
                        help='Horizon-rescue trigger: challenger base return must be <= this threshold')
    parser.add_argument('--anti-regression-horizon-challenger-pf-min', type=float, default=None,
                        help='Horizon-rescue trigger: challenger long-horizon PF must be >= this threshold')
    parser.add_argument('--anti-regression-horizon-min-trades', type=float, default=None,
                        help='Horizon-rescue trigger: challenger long-horizon trades must be >= this threshold')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()

    # Broker profile presets (explicit CLI overrides below still take precedence).
    if args.broker_profile == 'hfm-premium':
        config.environment.spread = 0.00014
        config.environment.commission = 0.0
        config.risk.leverage = 1000
        config.environment.swap_type = 'points'
        # Values from MT terminal "Swap type: In points".
        config.environment.swap_by_symbol = {
            'EURUSD': {'type': 'points', 'long': -10.8, 'short': 0.0},
            'USDCHF': {'type': 'points', 'long': 0.0, 'short': -12.1},
            'USDJPY': {'type': 'points', 'long': 0.0, 'short': -21.8},
            'GBPUSD': {'type': 'points', 'long': -3.2, 'short': -3.5},
        }
        # Fallback if primary symbol is not in swap_by_symbol.
        config.environment.swap_long_usd_per_lot_night = 0.0
        config.environment.swap_short_usd_per_lot_night = 0.0
        config.environment.swap_rollover_hour_utc = 22
        config.environment.swap_triple_weekday = 2

    # Data config overrides
    if args.data_mode is not None:
        config.data.data_mode = args.data_mode
    if args.n_bars is not None:
        config.data.n_bars = args.n_bars
    if args.synthetic_drift is not None:
        config.data.synthetic_drift = args.synthetic_drift
    if args.synthetic_volatility is not None:
        config.data.synthetic_volatility = args.synthetic_volatility
    if args.data_dir is not None:
        config.data.data_dir = args.data_dir
    if args.time_col is not None:
        config.data.time_col = args.time_col
    if args.date_col is not None:
        config.data.date_col = args.date_col
    if args.csv_sep is not None:
        config.data.csv_sep = args.csv_sep
    if args.pair_files is not None:
        config.data.pair_files = parse_pair_files_arg(args.pair_files)
        config.data.data_mode = "csv"
    if args.use_symmetry_loss is not None:
        config.agent.use_symmetry_loss = args.use_symmetry_loss
    if args.symmetry_loss_weight is not None:
        config.agent.symmetry_loss_weight = args.symmetry_loss_weight
    if args.use_dual_controller is not None:
        config.agent.use_dual_controller = args.use_dual_controller
    if args.use_strengths is not None:
        config.USE_CURRENCY_STRENGTHS = args.use_strengths
    if args.strengths_all is not None:
        config.INCLUDE_ALL_STRENGTHS = args.strengths_all
    if args.trade_penalty is not None:
        config.environment.trade_penalty = args.trade_penalty
    if args.flip_penalty is not None:
        config.environment.flip_penalty = args.flip_penalty
    if args.min_atr_cost_ratio is not None:
        config.environment.min_atr_cost_ratio = args.min_atr_cost_ratio
    if args.disable_move_sl is not None:
        config.environment.disable_move_sl = args.disable_move_sl
    if args.allow_actions is not None:
        config.environment.allowed_actions = parse_allow_actions(args.allow_actions)
    if args.spread is not None:
        config.environment.spread = args.spread
    if args.commission is not None:
        config.environment.commission = args.commission
    if args.slippage_pips is not None:
        config.environment.slippage_pips = args.slippage_pips
    if args.leverage is not None:
        config.risk.leverage = args.leverage
    if args.swap_long is not None:
        config.environment.swap_long_usd_per_lot_night = args.swap_long
    if args.swap_short is not None:
        config.environment.swap_short_usd_per_lot_night = args.swap_short
    if args.swap_type is not None:
        config.environment.swap_type = args.swap_type
    if args.swap_rollover_hour_utc is not None:
        config.environment.swap_rollover_hour_utc = int(max(0, min(23, args.swap_rollover_hour_utc)))
    if args.swap_triple_weekday is not None:
        config.environment.swap_triple_weekday = int(max(0, min(6, args.swap_triple_weekday)))
    if args.reward_clip is not None:
        config.environment.reward_clip = args.reward_clip
    if args.holding_cost is not None:
        config.environment.holding_cost = args.holding_cost
    if args.use_regime_filter is not None:
        config.environment.use_regime_filter = args.use_regime_filter
    if args.regime_min_vol_z is not None:
        config.environment.regime_min_vol_z = args.regime_min_vol_z
    if args.regime_align_trend is not None:
        config.environment.regime_align_trend = args.regime_align_trend
    if args.regime_require_trending is not None:
        config.environment.regime_require_trending = args.regime_require_trending
    if args.cooldown_bars is not None:
        config.environment.cooldown_bars = args.cooldown_bars
    if args.min_hold_bars is not None:
        config.environment.min_hold_bars = args.min_hold_bars
    if args.max_trades_per_episode is not None:
        config.environment.max_trades_per_episode = args.max_trades_per_episode
    if args.r_multiple_reward_weight is not None:
        config.environment.r_multiple_reward_weight = args.r_multiple_reward_weight
    if args.r_multiple_reward_clip is not None:
        config.environment.r_multiple_reward_clip = args.r_multiple_reward_clip
    if args.hold_tie_tau is not None:
        config.agent.hold_tie_tau = args.hold_tie_tau
    if args.hold_break_after is not None:
        config.agent.hold_break_after = args.hold_break_after
    if args.epsilon_start is not None:
        config.agent.epsilon_start = args.epsilon_start
    if args.epsilon_end is not None:
        config.agent.epsilon_end = args.epsilon_end
    if args.epsilon_decay is not None:
        config.agent.epsilon_decay = args.epsilon_decay
    if args.trade_gate_margin is not None:
        config.agent.trade_gate_margin = args.trade_gate_margin
    if args.trade_gate_z is not None:
        config.agent.trade_gate_z = args.trade_gate_z
    if args.atr_mult_sl is not None:
        config.risk.atr_multiplier = args.atr_mult_sl
    if args.tp_mult is not None:
        config.risk.tp_multiplier = args.tp_mult
    if args.prefill_policy is not None:
        config.training.prefill_policy = args.prefill_policy
    if args.anti_regression_checkpoint_selection is not None:
        config.training.anti_regression_checkpoint_selection = args.anti_regression_checkpoint_selection
    if args.anti_regression_top_k is not None:
        config.training.anti_regression_eval_top_k = max(1, int(args.anti_regression_top_k))
    if args.anti_regression_candidate_keep is not None:
        config.training.anti_regression_candidate_keep = max(1, int(args.anti_regression_candidate_keep))
    if args.anti_regression_min_validations is not None:
        config.training.anti_regression_min_validations = max(1, int(args.anti_regression_min_validations))
    if args.anti_regression_selector_mode is not None:
        config.training.anti_regression_selector_mode = str(args.anti_regression_selector_mode).strip().lower()
    if args.anti_regression_auto_rescue_enabled is not None:
        config.training.anti_regression_auto_rescue_enabled = bool(args.anti_regression_auto_rescue_enabled)
    if args.anti_regression_rescue_winner_forward_return_max is not None:
        config.training.anti_regression_auto_rescue_winner_forward_return_max = float(args.anti_regression_rescue_winner_forward_return_max)
    if args.anti_regression_rescue_forward_return_edge_min is not None:
        config.training.anti_regression_auto_rescue_forward_return_edge_min = float(args.anti_regression_rescue_forward_return_edge_min)
    if args.anti_regression_rescue_forward_pf_edge_min is not None:
        config.training.anti_regression_auto_rescue_forward_pf_edge_min = float(args.anti_regression_rescue_forward_pf_edge_min)
    if args.anti_regression_rescue_challenger_base_return_max is not None:
        config.training.anti_regression_auto_rescue_challenger_base_return_max = float(args.anti_regression_rescue_challenger_base_return_max)
    if args.anti_regression_rescue_challenger_forward_pf_min is not None:
        config.training.anti_regression_auto_rescue_challenger_forward_pf_min = float(args.anti_regression_rescue_challenger_forward_pf_min)
    if args.anti_regression_eval_min_k is not None:
        config.training.anti_regression_eval_min_k = max(1, int(args.anti_regression_eval_min_k))
    if args.anti_regression_eval_max_k is not None:
        config.training.anti_regression_eval_max_k = max(1, int(args.anti_regression_eval_max_k))
    if args.anti_regression_eval_jitter_draws is not None:
        config.training.anti_regression_eval_jitter_draws = max(1, int(args.anti_regression_eval_jitter_draws))
    if args.anti_regression_alt_stride_frac is not None:
        config.training.anti_regression_alt_stride_frac = max(0.01, float(args.anti_regression_alt_stride_frac))
    if args.anti_regression_alt_window_bars is not None:
        config.training.anti_regression_alt_window_bars = max(100, int(args.anti_regression_alt_window_bars))
    if args.anti_regression_tail_start_frac is not None:
        config.training.anti_regression_tail_start_frac = max(0.0, min(0.95, float(args.anti_regression_tail_start_frac)))
    if args.anti_regression_tail_end_frac is not None:
        config.training.anti_regression_tail_end_frac = max(0.05, min(1.0, float(args.anti_regression_tail_end_frac)))
    if args.anti_regression_tail_weight is not None:
        config.training.anti_regression_tail_weight = max(0.0, float(args.anti_regression_tail_weight))
    if args.anti_regression_base_return_floor is not None:
        config.training.anti_regression_base_return_floor = float(args.anti_regression_base_return_floor)
    if args.anti_regression_base_penalty_weight is not None:
        config.training.anti_regression_base_penalty_weight = max(0.0, float(args.anti_regression_base_penalty_weight))
    if args.anti_regression_tiebreak_enabled is not None:
        config.training.anti_regression_tiebreak_enabled = bool(args.anti_regression_tiebreak_enabled)
    if args.anti_regression_tiebreak_window_bars is not None:
        config.training.anti_regression_tiebreak_window_bars = max(200, int(args.anti_regression_tiebreak_window_bars))
    if args.anti_regression_tiebreak_start_frac is not None:
        config.training.anti_regression_tiebreak_start_frac = max(0.0, min(0.95, float(args.anti_regression_tiebreak_start_frac)))
    if args.anti_regression_tiebreak_end_frac is not None:
        config.training.anti_regression_tiebreak_end_frac = max(0.05, min(1.0, float(args.anti_regression_tiebreak_end_frac)))
    if args.anti_regression_tiebreak_return_edge_min is not None:
        config.training.anti_regression_tiebreak_return_edge_min = max(0.0, float(args.anti_regression_tiebreak_return_edge_min))
    if args.anti_regression_tiebreak_pf_edge_min is not None:
        config.training.anti_regression_tiebreak_pf_edge_min = max(0.0, float(args.anti_regression_tiebreak_pf_edge_min))
    if args.anti_regression_tiebreak_min_trades is not None:
        config.training.anti_regression_tiebreak_min_trades = max(0.0, float(args.anti_regression_tiebreak_min_trades))
    if args.anti_regression_horizon_rescue_enabled is not None:
        config.training.anti_regression_horizon_rescue_enabled = bool(args.anti_regression_horizon_rescue_enabled)
    if args.anti_regression_horizon_window_bars is not None:
        config.training.anti_regression_horizon_window_bars = max(200, int(args.anti_regression_horizon_window_bars))
    if args.anti_regression_horizon_start_frac is not None:
        config.training.anti_regression_horizon_start_frac = max(0.0, min(0.95, float(args.anti_regression_horizon_start_frac)))
    if args.anti_regression_horizon_end_frac is not None:
        config.training.anti_regression_horizon_end_frac = max(0.05, min(1.0, float(args.anti_regression_horizon_end_frac)))
    if args.anti_regression_horizon_candidate_limit is not None:
        config.training.anti_regression_horizon_candidate_limit = max(2, int(args.anti_regression_horizon_candidate_limit))
    if args.anti_regression_horizon_incumbent_return_max is not None:
        config.training.anti_regression_horizon_incumbent_return_max = float(args.anti_regression_horizon_incumbent_return_max)
    if args.anti_regression_horizon_return_edge_min is not None:
        config.training.anti_regression_horizon_return_edge_min = max(0.0, float(args.anti_regression_horizon_return_edge_min))
    if args.anti_regression_horizon_pf_edge_min is not None:
        config.training.anti_regression_horizon_pf_edge_min = max(0.0, float(args.anti_regression_horizon_pf_edge_min))
    if args.anti_regression_horizon_challenger_base_return_max is not None:
        config.training.anti_regression_horizon_challenger_base_return_max = float(args.anti_regression_horizon_challenger_base_return_max)
    if args.anti_regression_horizon_challenger_pf_min is not None:
        config.training.anti_regression_horizon_challenger_pf_min = max(0.0, float(args.anti_regression_horizon_challenger_pf_min))
    if args.anti_regression_horizon_min_trades is not None:
        config.training.anti_regression_horizon_min_trades = max(0.0, float(args.anti_regression_horizon_min_trades))
    
    # Override episodes if specified
    if args.episodes is not None:
        config.training.num_episodes = args.episodes
    if args.max_steps_per_episode is not None:
        config.training.max_steps_per_episode = args.max_steps_per_episode
    if args.train_random_start is not None:
        config.training.random_episode_start = args.train_random_start
    if args.episode_timeout_min is not None:
        config.training.episode_timeout_min = args.episode_timeout_min
    if args.heartbeat_secs is not None:
        config.training.heartbeat_secs = args.heartbeat_secs
    if args.heartbeat_steps is not None:
        config.training.heartbeat_steps = args.heartbeat_steps

    if args.validate_every is not None:
        config.training.validate_every = args.validate_every
    if args.val_window_bars is not None:
        config.VAL_WINDOW_BARS = args.val_window_bars
    if args.val_jitter_draws is not None:
        config.VAL_JITTER_DRAWS = args.val_jitter_draws
    if args.val_min_k is not None:
        config.VAL_MIN_K = args.val_min_k
    if args.val_max_k is not None:
        config.VAL_MAX_K = args.val_max_k
    
    # Override seed if specified
    if args.seed is not None:
        config.random_seed = args.seed
    
    # Override output directory if specified
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config.checkpoint_dir = str(output_dir / 'checkpoints')
        config.log_dir = str(output_dir / 'logs')
        config.results_dir = str(output_dir / 'results')
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)

    if args.eval_zero_costs:
        if args.mode != "evaluate":
            print("[WARN] --eval-zero-costs is only applied in mode=evaluate; ignoring.")
        else:
            config.environment.spread = 0.0
            config.environment.commission = 0.0
            config.environment.slippage_pips = 0.0
            config.environment.swap_type = 'usd'
            config.environment.swap_long_usd_per_lot_night = 0.0
            config.environment.swap_short_usd_per_lot_night = 0.0
            # Ensure per-symbol swap overrides do not reintroduce costs.
            config.environment.swap_by_symbol = {}
    
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

    write_run_artifacts(config, args)
    
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

