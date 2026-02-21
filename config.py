"""
Configuration Module
Central configuration for all hyperparameters and settings.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# --- Dev / smoke switches ---
FAST_DEV: bool = False
SMOKE_LEARN: bool = False        # PRODUCTION: Disabled for full runs (was True for testing)
SMOKE_LEARNING_STARTS: int = 1000  # Match prefill amount to reduce early noise
SMOKE_MAX_STEPS_PER_EPISODE: int = 600
SMOKE_BUFFER_CAPACITY: int = 50000
SMOKE_BATCH_SIZE: int = 256
SMOKE_TARGET_UPDATE: int = 250

# --- Validation stability switches ---
FREEZE_VALIDATION_FRICTIONS: bool = False  # PHASE-2.8b Run B: robustness test with friction jitter


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    short_window: int = 5
    medium_window: int = 20
    long_window: int = 50
    atr_period: int = 14
    rsi_period: int = 14
    lr_window: int = 10
    fractal_window: int = 5


@dataclass
class RiskConfig:
    """Risk management configuration."""
    contract_size: float = 100000  # Standard lot
    point: float = 0.00001  # 5-digit broker
    leverage: int = 1000  # HFM Premium target leverage
    risk_per_trade: float = 0.004  # PHASE-2: 0.4% risk per trade (clips tail losses)
    atr_multiplier: float = 1.8  # SL distance in ATR
    max_dd_threshold: float = 0.20  # 20% max drawdown
    margin_safety_factor: float = 0.50  # 50% free margin buffer
    tp_multiplier: float = 2.2  # TP = 2x SL (more realistic)
    cost_budget_pct: float = 0.05  # 5% of balance for spread/commission costs (tighter)
    max_dd_survivability: float = 0.40  # Survive up to 40% DD in position sizing


@dataclass
class EnvironmentConfig:
    """Trading environment configuration."""
    initial_balance: float = 1000.0
    spread: float = 0.00014  # 1.4 pips baseline (HFM Premium-style)
    commission: float = 0.0  # HFM Premium: no commission on Premium account
    slippage_pips: float = 0.8  # Slippage in pips
    swap_type: str = "usd"  # "usd" (USD/lot/night) or "points" (broker swap points)
    swap_long_usd_per_lot_night: float = 0.0  # Swap long value (USD if swap_type=usd, points if swap_type=points)
    swap_short_usd_per_lot_night: float = 0.0  # Swap short value (USD if swap_type=usd, points if swap_type=points)
    swap_rollover_hour_utc: int = 22  # Approximate FX rollover cut at 22:00 UTC
    swap_triple_weekday: int = 2  # Wednesday triple swap for spot FX (Mon=0)
    swap_by_symbol: dict = field(default_factory=dict)  # Optional per-symbol swap overrides
    weekend_close_hours: int = 3  # Flatten N hours before weekend
    cooldown_bars: int = 12  # PHASE-2.8d Fix Pack D2.A: Raise from 11 to 12 (slow re-entries, prevent parking)
    min_hold_bars: int = 6   # PHASE-2.8: Raise from 5 to 6 (reduce flipiness)
    trade_penalty: float = 0.000065  # PHASE-2.8b: Lower from 0.00007 to 0.000065 (allow 2-3 more trades/ep)
    flip_penalty: float = 0.00077    # PHASE-2.8d Fix Pack D1.3: Raise from 0.0005 to 0.00077 (discourage churn)
    min_atr_cost_ratio: float = 0.0  # Gate new trades unless ATR >= ratio * (spread+slip+commission), 0 disables
    use_regime_filter: bool = False  # Gate trades to trending/high-vol regimes
    regime_min_vol_z: float = 0.0  # Minimum realized_vol_24h_z to allow trades (0 disables extra vol gate)
    regime_align_trend: bool = True  # Require trend_96h alignment for LONG/SHORT entries
    regime_require_trending: bool = True  # Require is_trending flag to allow trades
    max_trades_per_episode: int = 100  # STRESS-TEST: Lower from 120 (quality > quantity)
    stack_n: int = 2  # PATCH: Frame stacking - use 2 for SMOKE, 3-4 for long runs
    state_stack_n: int = 2  # Alias for stack_n (observation stacking depth)
    min_trail_buffer_pips: float = 1.0  # PATCH #4: Min pips for meaningful SL tightening
    disable_move_sl: bool = False  # Disable MOVE_SL_CLOSER action for simpler learning
    allowed_actions: Optional[List[int]] = None  # Restrict action set (indices 0-7), None = all
    reward_clip: float = 0.01  # Reward clip for log-return stability
    holding_cost: float = 1e-4  # Per-step holding cost (only while in position)
    r_multiple_reward_weight: float = 0.0  # Reward shaping weight for realized R multiple (0 disables)
    r_multiple_reward_clip: float = 2.0  # Clip for realized R multiple shaping
    
    # PHASE 2.8e: Soft bias steering (no reward penalties, action-selection nudges only)
    entropy_beta: float = 0.014         # PHASE-2.8d Fix Pack D1.1: Keep stable at 0.014 (exploration control)
    directional_bias_beta: float = 0.08  # Soft bias for L/S rebalancing (nudge at action selection)
    hold_bias_gamma: float = 0.05        # Soft bias for HOLD discouragement (nudge at action selection)
    bias_check_interval: int = 10        # Check balance every N steps
    bias_margin_low: float = 0.35        # Trigger SHORT bias if long_ratio < 35%
    bias_margin_high: float = 0.65       # Trigger LONG bias if long_ratio > 65%
    hold_ceiling: float = 0.80           # Discourage HOLD if hold_rate > 80%
    
    # Circuit-breaker (fail-safe for extreme lock-in with hysteresis)
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold_low: float = 0.10   # Trigger if long_ratio < 10%
    circuit_breaker_threshold_high: float = 0.90  # Trigger if long_ratio > 90%
    circuit_breaker_lookback: int = 500           # Must persist 500 steps (hysteresis)
    circuit_breaker_mask_duration: int = 30       # Mask dominant side for 30 steps


@dataclass
class AgentConfig:
    """DQN agent configuration."""
    learning_rate: float = 0.0001
    gamma: float = 0.95  
    epsilon_start: float = 0.12  # PHASE-2.8d Fix Pack D1: Restore production epsilon (was 0.50 Nuclear Fix)
    epsilon_end: float = 0.06    # PHASE-2.8d Fix Pack D1: Restore production floor (was 0.10 Nuclear Fix)
    epsilon_decay: float = 0.997  # Slower decay
    eval_epsilon: float = 0.01   # STABILITY: Minimal eval probing for deterministic checkpoints
    eval_tie_only: bool = True   # QUALITY: Only apply eval_epsilon on Q-value ties
    eval_tie_tau: float = 0.03   # STABILITY: Tighter tie band to avoid noisy probes
    hold_tie_tau: float = 0.030  # PHASE-2.8d Fix Pack D2.A: Lower from 0.038 to 0.030 (reduce HOLD bias that enables parking)
    hold_break_after: int = 8    # PHASE-2.8: Raise from 7 to 8 (less premature breaks)
    trade_gate_margin: float = 0.0  # Minimum Q-gap over HOLD required to allow new trades (0 = disabled)
    trade_gate_z: float = 0.0  # Minimum Q-gap over HOLD in Q std units (0 = disabled)
    buffer_capacity: int = 100000
    batch_size: int = 256  # PATCH #6: 256 for stability
    target_update_freq: int = 450  # STABILITY: Raised from 300 to 450 to smooth Q-drift
    hidden_sizes: List[int] = None
    use_noisy: bool = False  # PHASE-2.8d NUCLEAR FIX: Disable NoisyNet, use epsilon-greedy to force exploration
    noisy_sigma_init: float = 0.02  # PHASE-2.8: Reduce from 0.03 to 0.02 (less parameter noise)
    use_param_ema: bool = True  # PHASE-2: Use EMA model copy for stable evaluation
    ema_decay: float = 0.999    # PHASE-2: EMA decay rate for stable eval model
    grad_clip: float = 1.0  # PATCH #6: Gradient clipping for stability (1.0)
    use_symmetry_loss: bool = True  # Directional symmetry loss (augmentation)
    symmetry_loss_weight: float = 0.5  # Weight for symmetry loss term
    use_dual_controller: bool = True  # Action-balancing controller
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 256, 128]


@dataclass
class FitnessConfig:
    """Fitness metric configuration."""
    # Fitness mode selection
    mode: str = "spr"  # "spr" or "legacy" (Sharpe/CAGR)
    
    # SPR (Sharpe-PF-Recovery) parameters
    spr_pf_cap: float = 5.0                     # PHASE-2.8: Cap PF at 5 (was 6) - stricter outlier control
    spr_target_trades_per_year: float = 120.0   # PHASE-2.8: Raise to 120 (was 100) - match observed cadence
    spr_dd_floor_pct: float = 1.5               # PHASE-2.8: Raise floor to 1.5% (was 1.0) - prevent tiny DD inflation
    spr_use_pandas: bool = True                 # Use pandas for monthly bucketing if available
    
    # Legacy fitness weights (used when mode="legacy")
    sharpe_weight: float = 1.0
    cagr_weight: float = 2.0
    stagnation_weight: float = 1.0
    loss_years_weight: float = 1.0
    ruin_penalty: float = 5.0
    ruin_threshold: float = 0.05  # 5% of initial balance

    # Test walk-forward gating (SPR mode)
    test_walkforward_min_spr: float = 0.0
    test_walkforward_min_pf: float = 1.0
    test_walkforward_min_pos_frac: float = 0.5
    test_walkforward_min_windows: int = 3
    test_walkforward_window_bars: Optional[int] = None
    test_walkforward_stride_frac: Optional[float] = None


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_episodes: int = 500
    validate_every: int = 1  # Validate every episode for better feedback
    save_every: int = 50
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    max_steps_per_episode: int = 1000  # Shorter episodes for faster training
    random_episode_start: bool = True  # Train on random windows instead of always starting at bar 0
    episode_timeout_min: Optional[float] = None  # Optional wall-clock timeout per episode
    heartbeat_secs: float = 60.0  # Print progress heartbeat every N seconds during long episodes
    heartbeat_steps: int = 200  # Also print heartbeat every N environment steps
    prefill_steps: int = 3000  # PATCH #3: Heuristic baseline pre-fill (1000 for smoke, 3000+ for full)
    prefill_policy: str = "baseline"  # baseline | random | none
    disable_early_stop: bool = True  # Set to True for seed sweeps to ensure fixed episode count
    anti_regression_checkpoint_selection: bool = True  # Evaluate top checkpoints at end and pick most robust
    anti_regression_candidate_keep: int = 24  # Keep up to N candidate checkpoints during training
    anti_regression_eval_top_k: int = 6  # Evaluate top-K candidates in end-of-run tournament
    anti_regression_min_validations: int = 4  # Require at least N validations before tournament
    anti_regression_alt_stride_frac: float = 0.20  # Secondary hold-out stride for robustness scoring
    anti_regression_alt_window_bars: Optional[int] = None  # Secondary hold-out window (None uses default)
    anti_regression_eval_min_k: Optional[int] = None  # Optional VAL_MIN_K override during anti-regression tournament
    anti_regression_eval_max_k: Optional[int] = None  # Optional VAL_MAX_K override during anti-regression tournament
    anti_regression_eval_jitter_draws: Optional[int] = None  # Optional VAL_JITTER_DRAWS override during anti-regression tournament
    anti_regression_tail_start_frac: float = 0.50  # Tail-only validation segment start (fraction of val span)
    anti_regression_tail_end_frac: float = 1.00  # Tail-only validation segment end (fraction of val span)
    anti_regression_tail_weight: float = 0.75  # Extra penalty weight when tail slice return is negative
    anti_regression_base_return_floor: float = 0.0  # Soft floor for base return in tournament scoring
    anti_regression_base_penalty_weight: float = 0.15  # Penalty weight when base return falls below floor


@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: str = "./data"
    pairs: List[str] = None
    data_mode: str = "synthetic"  # "synthetic" or "csv"
    pair_files: dict = None  # {PAIR: csv_path} for real data mode
    csv_sep: str = None
    date_col: str = None
    time_col: str = "time"
    parse_dates: bool = True
    n_bars: int = 10000
    start_date: str = "2023-01-01"
    freq: str = "1H"
    synthetic_drift: float = 0.0
    synthetic_volatility: float = 0.0005
    
    def __post_init__(self):
        if self.pairs is None:
            # PHASE-2.8: Add USDCAD, AUDUSD, GBPJPY for diversity (anti-overfit)
            self.pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'AUDUSD', 'GBPJPY']
        if self.pair_files and self.data_mode == "synthetic":
            self.data_mode = "csv"


@dataclass
class CurrencyStrengthConfig:
    """Currency strength configuration."""
    currencies: List[str] = None
    include_all_strengths: bool = True
    strength_window: int = 24  # bars
    strength_lags: int = 3     # _lag1.._lag3
    pairs: List[str] = None
    
    def __post_init__(self):
        if self.currencies is None:
            self.currencies = ["USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF"]
        if self.pairs is None and self.include_all_strengths:
            # Auto-build pairs from currencies
            self.pairs = [f"{cur1}{cur2}" for i, cur1 in enumerate(self.currencies) for cur2 in self.currencies[i+1:]]


@dataclass
class Config:
    """Master configuration."""
    feature: FeatureConfig = None
    risk: RiskConfig = None
    environment: EnvironmentConfig = None
    agent: AgentConfig = None
    fitness: FitnessConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    currency_strength: CurrencyStrengthConfig = None
    
    # Smoke mode switches (module-level for easy access)
    SMOKE_LEARN: bool = SMOKE_LEARN
    SMOKE_LEARNING_STARTS: int = SMOKE_LEARNING_STARTS
    SMOKE_MAX_STEPS_PER_EPISODE: int = SMOKE_MAX_STEPS_PER_EPISODE
    SMOKE_BUFFER_CAPACITY: int = SMOKE_BUFFER_CAPACITY
    SMOKE_BATCH_SIZE: int = SMOKE_BATCH_SIZE
    SMOKE_TARGET_UPDATE: int = SMOKE_TARGET_UPDATE
    
    # Currency strength top-level config
    CURRENCIES: List[str] = None
    USE_CURRENCY_STRENGTHS: bool = True
    INCLUDE_ALL_STRENGTHS: bool = True  # Include all 7 majors
    STRENGTH_WINDOW: int = 24
    STRENGTH_LAGS: int = 3  # 3 lags for each currency
    PAIRS: List[str] = None
    
    # Feature normalization
    USE_FEATURE_SCALER: bool = True  # Enable input normalization
    
    # Validation randomization (frictions jitter) - PHASE-2.8c: Stabilization tweaks
    VAL_SPREAD_JITTER: Tuple[float, float] = (0.90, 1.10)  # ±10% (realistic robustness)
    VAL_COMMISSION_JITTER: Tuple[float, float] = (0.90, 1.10)  # ±10% (realistic robustness)
    VAL_JITTER_DRAWS: int = 3           # PHASE-2.8c: Average over K=3 jitter draws per validation
    
    # Validation robustness (overlapping windows)
    VAL_K: int = 7                      # Target number of validation passes
    VAL_WINDOW_BARS: int = 600          # GATING-FIX: Force 600-bar windows (realistic sizing)
    VAL_WINDOW_FRAC: float = 0.06       # Fallback fraction if window_bars not used
    VAL_STRIDE_FRAC: float = 0.12       # STABILITY: More overlap for steadier median/IQR
    VAL_MIN_K: int = 6                  # Minimum validation passes
    VAL_MAX_K: int = 7                  # Maximum validation passes
    VAL_MIN_BDAYS: int = 60             # Min business days required for stable Sharpe
    VAL_CLIP_SHARPE: float = 5.0        # Cap |Sharpe| in small samples
    VAL_CLIP_CAGR: float = 1.0          # Cap |CAGR| in small samples (100%)
    VAL_IQR_PENALTY: float = 0.6        # SURGICAL: Cap IQR penalty at 0.6 (tighter, was 0.7)
    VAL_TRIM_FRACTION: float = 0.25     # PHASE-2.8d: Trim top/bottom 25% (raised from 0.20, Fix D1.5)
    
    # Gating / expected-trades tuning (PHASE-2.8d: Tighter gating to suppress noise)
    # NOTE: Disabled for now to avoid forcing trades in real-data runs.
    VAL_DISABLE_TRADE_GATING: bool = True
    VAL_EXP_TRADES_SCALE: float = 0.42  # PHASE-2.8d: Raised from 0.38 (tighter min_full threshold, Fix D1.4)
    VAL_EXP_TRADES_CAP: int = 24        # TUNED: Lower from 28 to align with typical 22-27 trades
    VAL_MIN_FULL_TRADES: int = 16       # TUNED: Lower from 18 (achievable for 20-27 trade windows)
    VAL_MIN_HALF_TRADES: int = 8        # TUNED: Lower from 10 (gentler floor)
    VAL_PENALTY_MAX: float = 0.08       # TUNED: Gentler cap from 0.10
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    results_dir: str = "./results"
    
    # Random seed
    random_seed: int = 777
    
    def __post_init__(self):
        if self.feature is None:
            self.feature = FeatureConfig()
        if self.risk is None:
            self.risk = RiskConfig()
        if self.environment is None:
            self.environment = EnvironmentConfig()
        if self.agent is None:
            self.agent = AgentConfig()
        if self.fitness is None:
            self.fitness = FitnessConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.currency_strength is None:
            self.currency_strength = CurrencyStrengthConfig()
            
        # Initialize currency strength defaults (all 7 majors)
        if self.CURRENCIES is None:
            self.CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF"]
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'feature': self.feature.__dict__,
            'risk': self.risk.__dict__,
            'environment': self.environment.__dict__,
            'agent': self.agent.__dict__,
            'fitness': self.fitness.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'currency_strength': self.currency_strength.__dict__,
            'currencies': self.CURRENCIES,
            'use_currency_strengths': self.USE_CURRENCY_STRENGTHS,
            'include_all_strengths': self.INCLUDE_ALL_STRENGTHS,
            'strength_window': self.STRENGTH_WINDOW,
            'strength_lags': self.STRENGTH_LAGS,
            'pairs_override': self.PAIRS,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'results_dir': self.results_dir,
            'random_seed': self.random_seed,
        }


# Default configuration
DEFAULT_CONFIG = Config()


if __name__ == "__main__":
    print("Configuration Module")
    print("=" * 50)
    
    config = Config()
    
    print("\nDefault Configuration:")
    print(f"\nFeature Config:")
    for key, value in config.feature.__dict__.items():
        print(f"  {key}: {value}")
    
    print(f"\nRisk Config:")
    for key, value in config.risk.__dict__.items():
        print(f"  {key}: {value}")
    
    print(f"\nAgent Config:")
    for key, value in config.agent.__dict__.items():
        print(f"  {key}: {value}")
    
    print(f"\nTraining Config:")
    for key, value in config.training.__dict__.items():
        print(f"  {key}: {value}")
    
    print(f"\nCurrency Strength Config:")
    for key, value in config.currency_strength.__dict__.items():
        print(f"  {key}: {value}")

