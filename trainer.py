"""
Training Module
Manages the training loop for the DQN agent.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime
import torch
from contextlib import nullcontext
from torch.utils.tensorboard import SummaryWriter

from agent import DQNAgent, ActionSpace
from environment import ForexTradingEnv
from fitness import FitnessCalculator, calc_sharpe, calc_cagr
from spr_fitness import compute_spr_fitness
from structured_logger import StructuredLogger


# === PATCH D: Metrics Add-On ==============================================
from collections import Counter
import math

def compute_policy_metrics(action_seq, action_names=("HOLD","LONG","SHORT","FLAT")):
    """
    action_seq: list[int] of chosen action ids during validation (all windows)
    action_names: index-aligned names; default matches the env mapping in this repo.
    """
    n = len(action_seq)
    if n == 0:
        return {
            "actions": {},
            "hold_rate": 0.0,
            "action_entropy_bits": 0.0,
            "hold_streak_max": 0,
            "hold_streak_mean": 0.0,
            "avg_hold_length": 0.0,   # alias
            "switch_rate": 0.0,
            "long_short": {"long": 0, "short": 0, "long_ratio": 0.0, "short_ratio": 0.0},
        }

    counts = Counter(action_seq)
    total = float(n)

    # map by name (assumes env indices: 0:HOLD,1:LONG,2:SHORT,3:FLAT; adjust if your env differs)
    name_by_idx = {i: (action_names[i] if i < len(action_names) else f"a{i}") for i in counts.keys()}
    counts_by_name = {name_by_idx[i]: int(c) for i, c in counts.items()}

    # probabilities for entropy (over known actions)
    probs = []
    for i in range(len(action_names)):
        probs.append(counts.get(i, 0) / total)
    action_entropy_bits = -sum(p * math.log2(p) for p in probs if p > 0)

    # indices (fallback to sensible defaults)
    idx_hold  = action_names.index("HOLD")  if "HOLD"  in action_names else 0
    idx_long  = action_names.index("LONG")  if "LONG"  in action_names else 1
    idx_short = action_names.index("SHORT") if "SHORT" in action_names else 2

    hold_rate = counts.get(idx_hold, 0) / total

    # hold streaks
    hold_streaks = []
    cur = 0
    for a in action_seq:
        if a == idx_hold:
            cur += 1
        else:
            if cur > 0:
                hold_streaks.append(cur)
                cur = 0
    if cur > 0:
        hold_streaks.append(cur)

    hold_streak_max  = max(hold_streaks) if hold_streaks else 0
    hold_streak_mean = (sum(hold_streaks) / len(hold_streaks)) if hold_streaks else 0.0

    # switch rate
    switches = sum(1 for i in range(1, n) if action_seq[i] != action_seq[i-1])
    switch_rate = switches / (n - 1) if n > 1 else 0.0

    # long/short split
    long_ct  = counts.get(idx_long, 0)
    short_ct = counts.get(idx_short, 0)
    ls_total = long_ct + short_ct
    long_ratio  = (long_ct  / ls_total) if ls_total else 0.0
    short_ratio = (short_ct / ls_total) if ls_total else 0.0

    return {
        "actions": counts_by_name,
        "hold_rate": hold_rate,
        "action_entropy_bits": action_entropy_bits,
        "hold_streak_max": int(hold_streak_max),
        "hold_streak_mean": float(hold_streak_mean),
        "avg_hold_length": float(hold_streak_mean),  # alias for convenience
        "switch_rate": switch_rate,
        "long_short": {
            "long": int(long_ct),
            "short": int(short_ct),
            "long_ratio": long_ratio,
            "short_ratio": short_ratio,
        },
    }
# ========================================================================


def baseline_policy(obs: np.ndarray, feat_names: List[str], stack_n: int = 3, feature_dim: int = 31) -> int:
    """
    PATCH #3: Simple interpretable baseline for BC warm-start.
    Go long if (strength_base - strength_quote) > +z and RSI<70
    Short if < -z and RSI>30 else hold.
    
    FIX #2: Use the most-recent frame from a stacked state.
    
    Args:
        obs: Observation vector (may include stacked features)
        feat_names: Feature column names
        stack_n: Number of stacked frames (default 3)
        feature_dim: Number of features per frame (default 31)
        
    Returns:
        Action (0=HOLD, 1=LONG, 2=SHORT, 3=MOVE_SL_CLOSER)
    """
    # Map feature names to indices
    fn = {n: i for i, n in enumerate(feat_names)}
    
    # Index offset for the newest frame in the stacked block
    offset = (stack_n - 1) * feature_dim
    
    # Helper to get feature from the most recent frame
    get = lambda name, default_idx=0: obs[offset + fn.get(name, default_idx)]
    
    # Extract features from the most recent frame
    try:
        s_eur = get('strength_EUR')
        s_usd = get('strength_USD')
        rsi = get('rsi')
    except (KeyError, IndexError):
        return 0  # Default to HOLD if features not found
    
    score = s_eur - s_usd
    if score > 0.5 and rsi < 70:
        return 1  # LONG
    if score < -0.5 and rsi > 30:
        return 2  # SHORT
    return 0  # HOLD


def prefill_replay(env: ForexTradingEnv, agent: DQNAgent, steps: int = 5000):
    """
    PATCH #3: Pre-load replay buffer with heuristic baseline transitions.
    Helps DQN avoid starting from white noise.
    
    Args:
        env: Trading environment
        agent: DQN agent with replay buffer
        steps: Number of transitions to collect
    """
    print(f"[PREFILL] Collecting {steps} baseline transitions...")
    s = env.reset()
    collected = 0
    
    for _ in range(steps):
        # Use baseline policy with frame stacking parameters
        a = baseline_policy(s, env.feature_columns, stack_n=env.stack_n, feature_dim=env.feature_dim)
        s2, r, d, _ = env.step(a)
        
        # Store transition
        agent.store_transition(s, a, r, s2, d)
        collected += 1
        
        # Reset if done
        s = env.reset() if d else s2
        
        if collected % 1000 == 0:
            print(f"  Collected {collected}/{steps} transitions")
    
    print(f"[PREFILL] Complete. Buffer size: {agent.replay_size}")


class Trainer:
    """
    Manages training of DQN agent on Forex trading environment.
    """
    
    def __init__(self,
                 agent: DQNAgent,
                 train_env: ForexTradingEnv,
                 val_env: Optional[ForexTradingEnv] = None,
                 fitness_calculator: Optional[FitnessCalculator] = None,
                 checkpoint_dir: str = "./checkpoints",
                 log_dir: str = "./logs",
                 val_spread_jitter: tuple = (0.7, 1.3),
                 val_commission_jitter: tuple = (0.8, 1.2),
                 config=None):
        """
        Initialize trainer.
        
        Args:
            agent: DQN agent
            train_env: Training environment
            val_env: Validation environment (optional)
            fitness_calculator: Fitness calculator
            checkpoint_dir: Directory for saving checkpoints
            log_dir: Directory for saving logs
            val_spread_jitter: Tuple of (min, max) multipliers for spread jitter on validation
            val_commission_jitter: Tuple of (min, max) multipliers for commission jitter on validation
            config: Configuration object for validation parameters
        """
        self.agent = agent
        self.train_env = train_env
        self.val_env = val_env
        self.fitness_calculator = fitness_calculator or FitnessCalculator()
        self.val_spread_jitter = val_spread_jitter
        self.val_commission_jitter = val_commission_jitter
        self.config = config  # Store config for validation parameters
        
        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer for monitoring (optional)
        try:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        except Exception:
            self.writer = None
        
        # Structured logger for detailed events
        self.structured_logger = StructuredLogger(log_dir=str(self.log_dir))

        # Training history
        self.training_history = []
        self.validation_history = []
        
        # Track first learning milestone
        self.first_learn_logged = False
        
        # Grace counter for low-trade penalty (soften first offense)
        self.last_val_was_low_trade = False

        # auto-detect device and configure autocast
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch, 'xpu') and getattr(torch.xpu, 'is_available', lambda: False)():
            self.device = 'xpu'
        elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            self.device = 'mps'
        
        # choose autocast context manager (prefer bf16 on capable devices)
        if self.device == 'cuda':
            self.autocast = torch.cuda.amp.autocast
        elif self.device == 'xpu' and hasattr(torch, 'xpu') and hasattr(torch.xpu, 'amp'):
            try:
                self.autocast = lambda *args, **kwargs: torch.xpu.amp.autocast(dtype=torch.bfloat16)
            except Exception:
                self.autocast = nullcontext
        elif self.device == 'mps':
            # MPS supports autocast in newer PyTorch; fall back to nullcontext if unavailable
            self.autocast = getattr(torch, 'autocast', nullcontext)
        else:
            self.autocast = nullcontext
        
        # set agent device if agent supports it
        try:
            setattr(self.agent, 'device', self.device)
        except Exception:
            pass
        
        # bump replay batch/update schedule when using accelerator
        try:
            if self.device in ('cuda', 'xpu', 'mps'):
                if hasattr(self.agent, 'replay_batch_size'):
                    self.agent.replay_batch_size = max(getattr(self.agent, 'replay_batch_size', 32), 256)
                if hasattr(self.agent, 'update_every'):
                    self.agent.update_every = max(getattr(self.agent, 'update_every', 1), 4)
        except Exception:
            pass
    
    def train_episode(self) -> Dict:
        """
        Train for one episode.
        
        Returns:
            Dict with episode statistics
        """
        # Reset episode tracking for telemetry
        if hasattr(self.agent, 'reset_episode_tracking'):
            self.agent.reset_episode_tracking()
        
        # Wire structured logger into environment for trade logging
        if hasattr(self.train_env, 'structured_logger'):
            self.train_env.structured_logger = self.structured_logger
        
        # Training-time domain randomization: jitter spread/commission slightly
        if hasattr(self.train_env, 'spread'):
            if not hasattr(self.train_env, '_base_spread'):
                self.train_env._base_spread = self.train_env.spread
                self.train_env._base_commission = self.train_env.commission
            # Light jitter (±10% spread, ±10% commission)
            self.train_env.spread = self.train_env._base_spread * np.random.uniform(0.9, 1.1)
            self.train_env.commission = self.train_env._base_commission * np.random.uniform(0.9, 1.1)
        
        state = self.train_env.reset()
        episode_reward = 0
        episode_loss = []
        steps = 0
        update_every = getattr(self.agent, 'update_every', 4)
        grad_steps = getattr(self.agent, 'grad_steps', 2)
        
        # DIVERSITY: Track HOLD streaks during training to enable probe learning
        hold_streak = 0
        HOLD_ACTION = 0
        hold_tie_tau = getattr(self.config.agent, 'hold_tie_tau', 0.06)
        hold_break_after = getattr(self.config.agent, 'hold_break_after', 6)
        
        done = False
        while not done:
            # SURGICAL PATCH: Get legal action mask and select action
            mask = getattr(self.train_env, 'legal_action_mask', lambda: None)()
            action = self.agent.select_action(state, explore=True, mask=mask, env=self.train_env)
            
            # DIVERSITY: Hold-streak breaker (training version - mirrors validation logic)
            if action == HOLD_ACTION:
                hold_streak += 1
                # Probe when streak gets long and Q-values are near-tied
                if hold_streak >= hold_break_after:
                    try:
                        q_values = self.agent.get_q_values(state)
                        non_hold_actions = [1, 2, 3]  # LONG, SHORT, FLAT
                        if mask is not None:
                            non_hold_actions = [a for a in non_hold_actions if mask[a]]
                        
                        if non_hold_actions:
                            best_non_hold_q = max(q_values[a] for a in non_hold_actions)
                            best_non_hold_idx = [a for a in non_hold_actions if q_values[a] == best_non_hold_q][0]
                            hold_q = q_values[HOLD_ACTION]
                            
                            # If near-tie, take best non-HOLD action
                            if best_non_hold_q - hold_q >= -hold_tie_tau:
                                action = best_non_hold_idx
                                hold_streak = 0
                    except Exception:
                        pass  # Fall back to original action on error
            else:
                hold_streak = 0
            
            # Take step
            next_state, reward, done, info = self.train_env.step(action)
            
            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done)

            # Train periodically to cut per-step overhead (only after buffer has enough data)
            learning_starts = getattr(self.agent, 'learning_starts', 0)
            if steps % update_every == 0 and self.agent.replay_size >= learning_starts:
                # Log first learning milestone (one-time sanity check)
                if not self.first_learn_logged:
                    print(f"[FIRST LEARN] Buffer size: {self.agent.replay_size} | Learning starts: {learning_starts}")
                    self.first_learn_logged = True
                
                try:
                    with self.autocast():
                        _ = self.agent.train_step(beta=getattr(self, 'current_beta', 0.4), grad_steps=grad_steps)
                except Exception:
                    _ = self.agent.train_step(beta=getattr(self, 'current_beta', 0.4))
                if _ is not None:
                    episode_loss.append(_)

            # Update state
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Calculate episode statistics
        final_equity = info['equity']
        trade_stats = self.train_env.get_trade_statistics()
        
        episode_stats = {
            'episode_reward': episode_reward,
            'final_equity': final_equity,
            'steps': steps,
            'avg_loss': np.mean(episode_loss) if episode_loss else 0,
            'epsilon': self.agent.epsilon,
            **trade_stats,
        }
        
        # Collect extended telemetry if agent supports it
        if hasattr(self.agent, 'get_episode_telemetry'):
            telemetry = self.agent.get_episode_telemetry()
            episode_stats.update(telemetry)
        
        return episode_stats
    
    def _compute_val_windows(self, val_len: int, max_steps: int, k_target: int, 
                            window_frac: float, stride_frac: float):
        """
        Compute overlapping validation windows with adaptive sizing.
        Guarantees K>1 unless val set is tiny.
        
        Args:
            val_len: Length of validation data
            max_steps: Maximum steps per episode (from training env)
            k_target: Target number of windows
            window_frac: Window size as fraction of val_len
            stride_frac: Stride as fraction of window (controls overlap)
            
        Returns:
            Tuple of (window_len, list of start indices)
        """
        # Choose window length
        if max_steps is None or max_steps <= 0:
            window = int(max(100, val_len * window_frac))
        else:
            window = min(max_steps, int(val_len * window_frac))
            window = max(100, window)  # Guard minimum
        
        # Overlapping stride
        stride = max(1, int(window * stride_frac))
        
        # Build start positions
        starts = []
        pos = 0
        while pos + window <= val_len:
            starts.append(pos)
            pos += stride
        
        if not starts:
            starts = [0]
            window = min(val_len, window)
        
        # If too many windows, thin them to ~k_target evenly spaced
        if len(starts) > k_target:
            idxs = np.linspace(0, len(starts)-1, k_target).round().astype(int).tolist()
            starts = [starts[i] for i in idxs]
        
        return window, starts
    
    def _make_disjoint_windows(self, idx, k: int, min_bars: int = 600):
        """
        DEPRECATED: Use _compute_val_windows for robust overlapping windows.
        Split validation index into k disjoint windows (no overlap).
        Drop windows that are too short.
        
        Args:
            idx: Price index (DatetimeIndex or range)
            k: Number of windows to create
            min_bars: Minimum bars per window
            
        Returns:
            List of (start, end) tuples for each valid window
        """
        n = len(idx)
        step = n // k
        windows = []
        for i in range(k):
            start = i * step
            end = n if i == k - 1 else (i + 1) * step
            if end - start >= min_bars:
                windows.append((start, end))
        return windows
    
    def _run_validation_slice(self, start_idx: int, end_idx: int, 
                              base_spread: float, base_commission: float) -> Dict:
        """
        Run one validation pass on a specific slice of validation data.
        
        Args:
            start_idx: Start index in val_env price data
            end_idx: End index in val_env price data
            base_spread: Base spread to jitter from
            base_commission: Base commission to jitter from
            
        Returns:
            Dict with fitness, trades, and other stats for this slice
        """
        # Apply random jitter using config ranges
        self.val_env.spread = base_spread * np.random.uniform(*self.val_spread_jitter)
        self.val_env.commission = base_commission * np.random.uniform(*self.val_commission_jitter)
        
        # --- NEW: fast-forward to start_idx and clear stats at window start ---
        state = self.val_env.reset()
        # advance to the start of the slice using HOLD actions
        if start_idx > 0:
            steps_to_skip = int(start_idx)
            for _ in range(steps_to_skip):
                state, _, done, _ = self.val_env.step(0)  # 0=HOLD, capture state
                if done:
                    state = self.val_env.reset()
        
        # now zero out histories so metrics cover ONLY this slice
        if hasattr(self.val_env, 'equity_history'):
            self.val_env.equity_history = [getattr(self.val_env, 'equity', 1000.0)]
        if hasattr(self.val_env, 'trades'):
            try:
                self.val_env.trades.clear()
            except Exception:
                self.val_env.trades = []
        
        # State is now at start_idx position from fast-forward loop above
        episode_reward = 0
        steps = 0
        
        # PATCH A+: Track action counts and hold streak for validation diagnostics
        action_counts = np.zeros(4, dtype=int)  # [HOLD, LONG, SHORT, FLAT]
        action_sequence = []  # PATCH D: Track action sequence for metrics
        hold_streak = 0
        HOLD_ACTION = 0  # HOLD is action index 0
        
        # SPR: Track data for SPR fitness computation
        window_timestamps = []  # Bar timestamps
        window_equity = []      # Equity curve per step
        window_trade_pnls = []  # Realized P&L per closed trade
        
        # Get initial timestamp (1H bars starting from 2024-01-01)
        from datetime import timedelta
        base_timestamp = datetime(2024, 1, 1)
        
        # BUGFIX: Get configuration for eval exploration and hold-streak breaker from AGENT config
        # (not training config) to use the tuned values in AgentConfig
        agent_cfg = getattr(self.config, 'agent', None)
        eval_epsilon = getattr(agent_cfg, 'eval_epsilon', 0.05)
        eval_tie_only = getattr(agent_cfg, 'eval_tie_only', True)
        eval_tie_tau = getattr(agent_cfg, 'eval_tie_tau', 0.05)
        hold_tie_tau = getattr(agent_cfg, 'hold_tie_tau', 0.032)
        hold_break_after = getattr(agent_cfg, 'hold_break_after', 7)
        
        done = False
        while not done and steps < (end_idx - start_idx):
            # Get legal action mask
            mask = getattr(self.val_env, 'legal_action_mask', lambda: None)()
            
            # QUALITY: Conditional eval epsilon - only on Q-value ties
            apply_epsilon = False
            if eval_epsilon > 0:
                if eval_tie_only:
                    # Get Q-values to check for ties
                    try:
                        q_values = self.agent.get_q_values(state)
                        # Calculate gap between top two Q-values
                        sorted_q = np.partition(q_values, -2)
                        q_gap = sorted_q[-1] - sorted_q[-2]  # top1 - top2
                        
                        # Only apply epsilon if it's a near-tie
                        if q_gap < eval_tie_tau and np.random.rand() < eval_epsilon:
                            apply_epsilon = True
                    except Exception:
                        pass  # Fall back to greedy on error
                else:
                    # Original behavior: unconditional epsilon
                    if np.random.rand() < eval_epsilon:
                        apply_epsilon = True
            
            if apply_epsilon:
                # Epsilon-greedy: random non-HOLD action to break flatlines
                non_hold_actions = [1, 2, 3]  # LONG, SHORT, FLAT
                if mask is not None:
                    non_hold_actions = [a for a in non_hold_actions if mask[a]]
                
                if non_hold_actions:
                    action = np.random.choice(non_hold_actions)
                else:
                    action = HOLD_ACTION  # Fall back to HOLD if no legal non-HOLD
            else:
                # Normal greedy action with eval_mode=True
                action = self.agent.select_action(state, explore=False, mask=mask, eval_mode=True, env=self.val_env)
                
                # PATCH: Hold-streak breaker (only when not using epsilon)
                if action == HOLD_ACTION:
                    hold_streak += 1
                    # Check if we should probe the market
                    if hold_streak >= hold_break_after:
                        try:
                            # Get Q-values to check if near-tie
                            q_values = self.agent.get_q_values(state)
                            
                            # Find best non-HOLD Q-value
                            non_hold_actions = [1, 2, 3]  # LONG, SHORT, FLAT
                            if mask is not None:
                                # Filter to legal non-HOLD actions
                                non_hold_actions = [a for a in non_hold_actions if mask[a]]
                            
                            if non_hold_actions:
                                non_hold_q_values = [q_values[a] for a in non_hold_actions]
                                best_non_hold_idx = non_hold_actions[np.argmax(non_hold_q_values)]
                                best_non_hold_q = q_values[best_non_hold_idx]
                                hold_q = q_values[HOLD_ACTION]
                                
                                # If near-tie (within tau), take best non-HOLD action
                                if best_non_hold_q - hold_q >= -hold_tie_tau:
                                    action = best_non_hold_idx
                                    hold_streak = 0  # Reset streak
                        except Exception:
                            pass  # Fall back to HOLD on any error
                else:
                    hold_streak = 0
            
            # Track action
            action_counts[action] += 1
            action_sequence.append(action)  # PATCH D: Track for metrics
            
            # Take step
            next_state, reward, done, info = self.val_env.step(action)
            
            # SPR: Track timestamp and equity after each step
            current_timestamp = base_timestamp + timedelta(hours=start_idx + steps)
            window_timestamps.append(current_timestamp)
            current_equity = info.get('equity', getattr(self.val_env, 'equity', 1000.0))
            window_equity.append(current_equity)
            
            # SPR: Track trade P&L if a trade was closed this step
            if hasattr(self.val_env, 'trades') and len(self.val_env.trades) > len(window_trade_pnls):
                # New trade closed - get its P&L
                latest_trade = self.val_env.trades[-1]
                trade_pnl = latest_trade.get('pnl', 0.0)
                window_trade_pnls.append(trade_pnl)
            
            # Update state
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Calculate fitness for this slice
        equity_series = pd.Series(
            self.val_env.equity_history,
            index=pd.date_range(start='2024-01-01', periods=len(self.val_env.equity_history), freq='h')
        )
        
        # Determine fitness mode and compute accordingly
        fitness_mode = getattr(self.config.fitness, 'mode', 'legacy')
        
        # Get trade statistics FIRST (needed for both SPR and legacy)
        trade_stats = self.val_env.get_trade_statistics()
        slice_trades = int(
            trade_stats.get('trades') or
            trade_stats.get('total_trades') or
            trade_stats.get('num_trades') or
            0
        )
        
        if fitness_mode == 'spr':
            # Use SPR (Sharpe-PF-Recovery) fitness
            try:
                # PATCH 1: Compute PF fallback from equity if no trade P&Ls captured
                pf_override = None
                if not window_trade_pnls and len(window_equity) > 1:
                    # Approximate PF from equity curve diffs (gross profit / gross loss)
                    eq_array = np.asarray(window_equity, dtype=float)
                    diffs = np.diff(eq_array)
                    gross_profit = diffs[diffs > 0].sum()
                    gross_loss = -diffs[diffs < 0].sum()
                    
                    if gross_loss <= 0 and gross_profit > 0:
                        pf_override = self.config.fitness.spr_pf_cap  # Cap at configured limit
                    elif gross_loss > 0:
                        pf_override = min(gross_profit / gross_loss, self.config.fitness.spr_pf_cap)
                    else:
                        pf_override = 0.0
                
                fitness_raw, spr_info = compute_spr_fitness(
                    timestamps=window_timestamps,
                    equity_curve=window_equity,
                    trade_pnls=window_trade_pnls if window_trade_pnls else None,
                    pf_override=pf_override,  # Pass equity-based PF if no trades
                    trade_count_override=slice_trades if not window_trade_pnls else None,
                    initial_balance=self.config.environment.initial_balance,
                    seconds_per_bar=3600,  # 1H bars
                    pf_cap=self.config.fitness.spr_pf_cap,
                    dd_floor_pct=self.config.fitness.spr_dd_floor_pct,
                    target_trades_per_year=self.config.fitness.spr_target_trades_per_year,
                    use_pandas=self.config.fitness.spr_use_pandas,
                )
                metrics = {
                    'fitness': fitness_raw,
                    'spr_info': spr_info,
                    # Add legacy keys for compatibility
                    'sharpe': spr_info.get('pf', 0.0),  # Use PF as proxy
                    'cagr': spr_info.get('mmr_pct_mean', 0.0) / 100.0,  # Use MMR as proxy
                }
            except Exception as e:
                # Fallback to legacy on error
                print(f"[SPR WARNING] Error computing SPR fitness: {e}, falling back to legacy")
                metrics = self.fitness_calculator.calculate_all_metrics(equity_series)
        else:
            # Use legacy Sharpe/CAGR fitness
            metrics = self.fitness_calculator.calculate_all_metrics(equity_series)
        
        # Extract raw fitness and trade count (already computed above)
        fitness_raw = float(metrics["fitness"])
        trades = slice_trades
        
        return {
            'fitness': fitness_raw,
            'trades': trades,
            'reward': episode_reward,
            'equity': info.get('equity', self.val_env.equity),
            'steps': steps,
            'metrics': metrics,
            'trade_stats': trade_stats,
            'action_counts': action_counts,  # PATCH C: Return for histogram aggregation
            'action_sequence': action_sequence  # PATCH D: Return for metrics computation
        }
    
    def validate(self) -> Dict:
        """
        Validate agent on validation environment with DISJOINT windows and dispersion penalty.
        SURGICAL PATCH: Runs K disjoint passes (no overlap) with IQR penalty to reduce spiky runs.
        PHASE-2.8c: Jitter-averaged validation (K friction draws per window) for stable readings.
        Uses adaptive trade gating based on window length.
        
        Returns:
            Dict with validation statistics (stability-adjusted median)
        """
        if self.val_env is None:
            return {}
        
        # PHASE-2.8c: Capture CURRENT friction values (base for jitter)
        # Don't restore to a "base" - use whatever was set before this validation!
        current_spread = self.val_env.spread
        current_commission = self.val_env.commission
        
        # PHASE-2.8c: Get jitter-averaging configuration
        jitter_draws = getattr(self.config, "VAL_JITTER_DRAWS", 1)  # Default: no averaging
        freeze_frictions = getattr(self.config, "FREEZE_VALIDATION_FRICTIONS", False)
        spread_jitter = getattr(self.config, "VAL_SPREAD_JITTER", (1.0, 1.0))
        commission_jitter = getattr(self.config, "VAL_COMMISSION_JITTER", (1.0, 1.0))
        
        # Wire structured logger into validation environment
        if hasattr(self.val_env, 'structured_logger'):
            self.val_env.structured_logger = self.structured_logger
        
        # SURGICAL PATCH: Overlapping windows with adaptive sizing
        # Get validation data length
        if hasattr(self.val_env, 'data'):
            val_idx = self.val_env.data.index
            val_length = len(self.val_env.data)
        else:
            val_length = 10000  # Default fallback
            val_idx = range(val_length)
        
        # GATING-FIX: Use fixed window size if specified
        window_bars = getattr(self.config, "VAL_WINDOW_BARS", None)
        if window_bars is not None:
            window_len = min(window_bars, val_length)
        else:
            window_frac = getattr(self.config, "VAL_WINDOW_FRAC", 0.40)
            window_len = max(100, int(val_length * window_frac))
        
        # Cap by env horizon if present
        max_steps = getattr(self.train_env, "max_steps", None)
        if max_steps is not None:
            window_len = min(window_len, max_steps)
        
        # Compute stride and K
        stride_frac = getattr(self.config, "VAL_STRIDE_FRAC", 0.15)
        stride = max(1, int(window_len * stride_frac))
        
        min_k = getattr(self.config, "VAL_MIN_K", 6)
        max_k = getattr(self.config, "VAL_MAX_K", 7)
        
        # Generate window starts
        starts = []
        pos = 0
        while pos + window_len <= val_length and len(starts) < max_k:
            starts.append(pos)
            pos += stride
        
        # Ensure minimum K
        if len(starts) < min_k and val_length >= window_len:
            # Adjust stride to fit min_k windows
            stride = max(1, (val_length - window_len) // (min_k - 1))
            starts = [i * stride for i in range(min_k) if i * stride + window_len <= val_length]
        
        # Build window tuples (start, end)
        windows = [(start, min(start + window_len, val_length)) for start in starts]
        
        # Calculate coverage (how much of val data is seen across all windows)
        if len(windows) > 1:
            stride_actual = starts[1] - starts[0] if len(starts) > 1 else window_len
            coverage = (window_len + (len(windows)-1) * stride_actual) / max(1, val_length)
        else:
            coverage = window_len / max(1, val_length)
        
        # Log validation setup
        jitter_msg = f" | jitter-avg K={jitter_draws}" if jitter_draws > 1 and not freeze_frictions else ""
        print(f"[VAL] {len(windows)} passes | window={window_len} | stride~{int(window_len*stride_frac)} | "
              f"coverage~{coverage:.2f}x{jitter_msg}")
        
        # Log window ranges (first 3 for brevity)
        for i, (lo, hi) in enumerate(windows[:3]):
            if hasattr(val_idx, '__getitem__'):
                try:
                    start_time = val_idx[lo]
                    end_time = val_idx[hi - 1]
                    print(f"[VAL] window {i+1}: {start_time} to {end_time}  ({hi-lo} bars)")
                except:
                    print(f"[VAL] window {i+1}: idx {lo} to {hi}  ({hi-lo} bars)")
            else:
                print(f"[VAL] window {i+1}: idx {lo} to {hi}  ({hi-lo} bars)")
        
        # Run validation on each disjoint window with jitter averaging
        fits, trade_counts = [], []
        all_results = []
        total_action_counts = np.zeros(4, dtype=int)  # PATCH C: Aggregate action histograms
        all_actions = []  # PATCH D: Collect all action sequences
        
        # SPR: Store last SPR info for logging
        self._last_spr_info = None
        
        for (lo, hi) in windows:
            # PHASE-2.8c: Jitter-averaged validation (K friction draws per window)
            if jitter_draws > 1 and not freeze_frictions:
                # Run K draws with different friction values, average the results
                jitter_fits = []
                jitter_trades = []
                jitter_results = []
                jitter_action_counts = np.zeros(4, dtype=int)
                jitter_actions = []
                
                for draw_i in range(jitter_draws):
                    # Randomize frictions for this draw
                    jitter_spread = current_spread * np.random.uniform(*spread_jitter)
                    jitter_commission = current_commission * np.random.uniform(*commission_jitter)
                    
                    # Run validation slice with jittered frictions
                    stats = self._run_validation_slice(lo, hi, jitter_spread, jitter_commission)
                    jitter_fits.append(stats['fitness'])
                    jitter_trades.append(stats['trades'])
                    jitter_results.append({
                        'val_reward': stats['reward'],
                        'val_final_equity': stats['equity'],
                        'val_steps': stats['steps'],
                        **{f'val_{k}': v for k, v in stats['trade_stats'].items()},
                        **{f'val_{k}': v for k, v in stats['metrics'].items() if isinstance(v, (int, float, bool)) and k != 'fitness'},
                    })
                    
                    # Accumulate action counts
                    if 'action_counts' in stats:
                        jitter_action_counts += stats['action_counts']
                    if 'action_sequence' in stats:
                        jitter_actions.extend(stats['action_sequence'])
                    
                    # Capture SPR info from last draw
                    if 'metrics' in stats and 'spr_info' in stats['metrics']:
                        self._last_spr_info = stats['metrics']['spr_info']
                
                # Average over jitter draws
                window_fit = float(np.mean(jitter_fits))
                window_trades = int(np.mean(jitter_trades))
                fits.append(window_fit)
                trade_counts.append(window_trades)
                
                # Average numeric results
                avg_result = {}
                for key in jitter_results[0].keys():
                    values = [r[key] for r in jitter_results if key in r and isinstance(r[key], (int, float))]
                    if values:
                        avg_result[key] = float(np.mean(values))
                all_results.append(avg_result)
                
                # Accumulate action counts
                total_action_counts += jitter_action_counts
                all_actions.extend(jitter_actions)
            else:
                # Single draw (no jitter averaging)
                stats = self._run_validation_slice(lo, hi, current_spread, current_commission)
                fits.append(stats['fitness'])
                trade_counts.append(stats['trades'])
                
                # SPR: Capture SPR info from any window (use last for display)
                if 'metrics' in stats and 'spr_info' in stats['metrics']:
                    self._last_spr_info = stats['metrics']['spr_info']
                
                # PATCH C: Accumulate action counts
                if 'action_counts' in stats:
                    total_action_counts += stats['action_counts']
                
                # PATCH D: Collect action sequence
                if 'action_sequence' in stats:
                    all_actions.extend(stats['action_sequence'])
                
                # Collect results for averaging later
                all_results.append({
                    'val_reward': stats['reward'],
                    'val_final_equity': stats['equity'],
                    'val_steps': stats['steps'],
                    **{f'val_{k}': v for k, v in stats['trade_stats'].items()},
                    **{f'val_{k}': v for k, v in stats['metrics'].items() if isinstance(v, (int, float, bool)) and k != 'fitness'},
                })
        
        # Restore original values after all validation passes
        # PHASE-2.8c: Restore to current values (which may be randomized)
        self.val_env.spread = current_spread
        self.val_env.commission = current_commission
        
        # --- PHASE-2: Robust trimmed aggregation with IQR cap ---
        fits_array = np.asarray(fits)
        trim_frac = getattr(self.config, "VAL_TRIM_FRACTION", 0.2)
        
        # Trimmed median: drop top/bottom 20%, take median of middle 60%
        if len(fits_array) >= 5:  # Need at least 5 samples for meaningful trimming
            k = max(1, int(len(fits_array) * trim_frac))
            fits_sorted = np.sort(fits_array)
            core = fits_sorted[k:len(fits_sorted)-k]
            median = float(np.median(core))
            # IQR on full distribution for stability penalty
            q75, q25 = np.percentile(fits_array, [75, 25])
            iqr = float(q75 - q25)
        else:
            # Fallback: regular median for small K
            median = float(np.median(fits_array)) if len(fits_array) > 0 else 0.0
            q75, q25 = np.percentile(fits_array, [75, 25]) if len(fits_array) >= 2 else (median, median)
            iqr = float(q75 - q25)
        
        # PATCH 3: For SPR mode, 'median' is already the SPR score - no IQR penalty
        fitness_mode = getattr(self.config.fitness, 'mode', 'legacy')
        if fitness_mode == 'spr':
            # SPR mode: median is already the raw SPR score, no IQR adjustment
            stability_adj = median
            iqr_penalty = 0.0  # SPR already includes stagnation penalty
        else:
            # Legacy mode: apply IQR penalty (capped at 0.60 for tighter control)
            iqr_penalty_coef = getattr(self.config, "VAL_IQR_PENALTY", 0.4)
            iqr_penalty = min(iqr_penalty_coef * iqr, 0.6)
            stability_adj = median - iqr_penalty
        
        median_trades = float(np.median(trade_counts)) if trade_counts else 0.0
        
        # --- GATING-FIX: Realistic trade expectations scaled to observed 18-31 range ---
        bars_per_pass = max(1, windows[0][1] - windows[0][0]) if windows else 1
        
        # Compute effective bars per trade from env cadence
        mh = getattr(self.val_env, "min_hold_bars", 5)
        cd = getattr(self.val_env, "cooldown_bars", 10)
        eff = max(1, mh + max(1, cd // 2))  # e.g., 5 + 5 = 10 bars/trade
        
        # Scale and cap expected trades to match observed reality
        raw_expected = bars_per_pass / eff
        scale = getattr(self.config, "VAL_EXP_TRADES_SCALE", 0.40)
        cap = getattr(self.config, "VAL_EXP_TRADES_CAP", 28)
        expected_trades = min(raw_expected * scale, cap)
        
        # Set realistic thresholds with floors
        min_full_floor = getattr(self.config, "VAL_MIN_FULL_TRADES", 18)
        min_half_floor = getattr(self.config, "VAL_MIN_HALF_TRADES", 10)
        
        min_full = max(min_full_floor, int(round(expected_trades)))
        min_half = max(min_half_floor, int(round(expected_trades * 0.6)))
        
        # Apply multiplier based on median trades
        if median_trades >= min_full:
            mult = 1.00
        elif median_trades >= min_half:
            mult = 0.50
        else:
            mult = 0.00
        
        # SURGICAL: Gentle undertrade penalty with grace counter
        # Only penalize if this is the SECOND consecutive low-trade episode
        undertrade_penalty = 0.0
        is_low_trade = median_trades < min_half
        
        if is_low_trade:
            if self.last_val_was_low_trade:
                # Second consecutive low-trade episode - apply penalty
                shortfall = (min_half - median_trades) / max(1, min_half)  # 0..1
                penalty_max = getattr(self.config, "VAL_PENALTY_MAX", 0.10)
                undertrade_penalty = min(penalty_max, round(0.5 * shortfall, 3))
            # else: First offense - grace period, pen=0.00
        
        # Update grace tracker for next validation
        self.last_val_was_low_trade = is_low_trade
        
        # DEBUG: Print gating thresholds (can remove after verification)
        print(f"[GATING] bars={bars_per_pass} eff={eff} raw_exp={raw_expected:.1f} "
              f"scaled_exp={expected_trades:.1f} min_half={min_half} min_full={min_full} "
              f"median_trades={median_trades:.1f} mult={mult:.2f} pen={undertrade_penalty:.3f}")
        
        val_score = stability_adj * mult - undertrade_penalty
        
        # Average other numeric metrics over K passes
        val_stats = {
            'val_reward': 0.0,
            'val_final_equity': 1000.0,
            'val_steps': 0,
        }
        if all_results:
            for key in all_results[0].keys():
                values = [r[key] for r in all_results if key in r]
                if values and all(isinstance(v, (int, float)) for v in values):
                    val_stats[key] = np.mean(values)
                elif values:
                    val_stats[key] = values[0]  # Non-numeric, just take first
        
        # CRITICAL: Wire the computed stability-adjusted score into val_stats
        val_stats['val_fitness'] = val_score
        val_stats['val_trades'] = median_trades
        
        # Quality-of-life print - enhanced for SPR mode
        fitness_mode = getattr(self.config.fitness, 'mode', 'legacy')
        if fitness_mode == 'spr':
            # Extract SPR components from fits list (use median window for display)
            # Note: SPR info is in the stats returned from _run_validation_slice
            # We'll collect it during the window loop and store for display
            spr_components = getattr(self, '_last_spr_info', None)
            
            if spr_components:
                print(f"[VAL] K={len(windows)} overlapping | SPR={val_score:.3f} | "
                      f"PF={spr_components.get('pf', 0):.2f} | "
                      f"MDD={spr_components.get('mdd_pct', 0):.2f}% | "
                      f"MMR={spr_components.get('mmr_pct_mean', 0):.2f}% | "
                      f"TPY={spr_components.get('trades_per_year', 0):.1f} | "
                      f"SIG={spr_components.get('significance', 0):.2f} | "
                      f"trades={median_trades:.1f} | mult={mult:.2f} | pen={undertrade_penalty:.3f}")
            else:
                print(f"[VAL] K={len(windows)} overlapping | SPR={val_score:.3f} | "
                      f"trades={median_trades:.1f} | mult={mult:.2f} | pen={undertrade_penalty:.3f}")
        else:
            # Legacy Sharpe/CAGR display
            print(f"[VAL] K={len(windows)} overlapping | median={median:.3f} (trimmed) | "
                  f"IQR={iqr:.3f} | iqr_pen={iqr_penalty:.3f} | adj={stability_adj:.3f} | "
                  f"trades={median_trades:.1f} | mult={mult:.2f} | pen={undertrade_penalty:.3f} | score={val_score:.3f}")
        
        # --- Write compact JSON summary for analysis tools ---
        import os
        import json
        import datetime as dt
        
        # PATCH C: Calculate action distribution and HOLD rate
        total_actions = max(1, total_action_counts.sum())
        hold_rate = float(total_action_counts[0]) / total_actions
        nonhold_rate = 1.0 - hold_rate
        
        # PATCH D: Compute policy metrics from action sequence
        # FIX: Compute per-window max streaks to avoid artificially long sequences from concatenation
        ACTION_NAMES = ("HOLD", "LONG", "SHORT", "FLAT")
        
        # First, compute metrics on concatenated sequence (for entropy, switch rate)
        policy_metrics = compute_policy_metrics(all_actions, ACTION_NAMES)
        
        # Then, compute per-window max streaks and take the max
        per_window_max_streaks = []
        action_sequences_by_window = []
        start_idx = 0
        for (lo, hi) in windows:
            window_len = hi - lo
            window_actions = all_actions[start_idx:start_idx + window_len]
            if window_actions:
                window_metrics = compute_policy_metrics(window_actions, ACTION_NAMES)
                per_window_max_streaks.append(window_metrics["hold_streak_max"])
            start_idx += window_len
        
        # Override hold_streak_max with per-window max (not concatenated)
        if per_window_max_streaks:
            policy_metrics["hold_streak_max"] = int(max(per_window_max_streaks))
        
        summary = {
            "episode": int(self.episode) if hasattr(self, "episode") else None,
            "k": int(len(windows)),
            "median_fitness": float(median),
            "iqr": float(iqr),
            "adj": float(stability_adj),
            "trades": float(median_trades),
            "mult": float(mult),
            "penalty": float(undertrade_penalty),
            "score": float(val_score),
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "seed": int(getattr(self.config, "random_seed", -1)),
            # PHASE-2.8b: Track friction parameters for robustness verification
            "spread": float(getattr(self.val_env, 'spread', 0)),
            "slippage_pips": float(getattr(self.val_env.risk_manager, 'slippage_pips', 0) if hasattr(self.val_env, 'risk_manager') else 0),
            # PATCH D: Enhanced action metrics (replaces/extends PATCH C)
            "actions": policy_metrics["actions"],
            "hold_rate": policy_metrics["hold_rate"],
            "nonhold_rate": 1.0 - policy_metrics["hold_rate"],
            "action_entropy_bits": policy_metrics["action_entropy_bits"],
            "hold_streak_max": policy_metrics["hold_streak_max"],
            "hold_streak_mean": policy_metrics["hold_streak_mean"],
            "avg_hold_length": policy_metrics["avg_hold_length"],
            "switch_rate": policy_metrics["switch_rate"],
            "long_short": policy_metrics["long_short"],
            # PHASE-2.8d: Add behavioral metrics with expected field names
            "entropy": policy_metrics["action_entropy_bits"],  # Alias for entropy
            "hold_frac": policy_metrics["hold_rate"],  # Alias for hold fraction
            "long_ratio": policy_metrics["long_short"]["long_ratio"],  # Long/(Long+Short) - already computed
        }
        
        # SPR: Add SPR components to summary if available
        if self._last_spr_info:
            summary["spr_components"] = {
                "pf": float(self._last_spr_info.get('pf', 0)),
                "mdd_pct": float(self._last_spr_info.get('mdd_pct', 0)),
                "mmr_pct_mean": float(self._last_spr_info.get('mmr_pct_mean', 0)),
                "trades_per_year": float(self._last_spr_info.get('trades_per_year', 0)),
                "significance": float(self._last_spr_info.get('significance', 0)),
                "stagnation_penalty": float(self._last_spr_info.get('stagnation_penalty', 1)),
            }
        
        out_dir = os.path.join("logs", "validation_summaries")
        os.makedirs(out_dir, exist_ok=True)
        
        # File per episode keeps things simple for the checker
        if summary['episode'] is not None:
            fname = os.path.join(out_dir, f"val_ep{summary['episode']:03d}.json")
            with open(fname, "w", encoding="utf-8") as f:
                # ensure_ascii avoids Windows codepage headaches
                json.dump(summary, f, ensure_ascii=True, indent=2)
        
        return val_stats
    
    def train(self,
             num_episodes: int,
             validate_every: int = 10,
             save_every: int = 50,
             verbose: bool = True,
             telemetry_mode: str = 'standard',
             output_dir: str = None) -> Dict:
        """
        Train agent for multiple episodes.
        
        Args:
            num_episodes: Number of episodes to train
            validate_every: Validate every N episodes
            save_every: Save checkpoint every N episodes
            verbose: Print progress
            telemetry_mode: 'standard' or 'extended' telemetry logging
            output_dir: Output directory for results (optional)
            
        Returns:
            Dict with training history
        """
        best_fitness = -np.inf
        patience = 28  # PATIENCE: Increased from 20 to 28 for K~9-10 validation slices
        bad_count = 0
        
        # epsilon schedule
        # Epsilon schedule: if agent uses NoisyNet, skip epsilon scheduling and keep exploration driven by noisy weights
        if getattr(self.agent, 'use_noisy', False):
            # keep epsilon at current value (likely 0.0) and disable decay
            self.agent.epsilon_start = getattr(self.agent, 'epsilon', 0.0)
            self.agent.epsilon_end = getattr(self.agent, 'epsilon', 0.0)
            self.agent.epsilon_decay_steps = 1
        else:
            if not hasattr(self.agent, 'epsilon_start'):
                self.agent.epsilon_start = 0.2
            if not hasattr(self.agent, 'epsilon_end'):
                # for short smoke runs keep exploration higher
                self.agent.epsilon_end = 0.10
            if not hasattr(self.agent, 'epsilon_decay_steps'):
                # decay to epsilon_end over ~80% of episodes
                self.agent.epsilon_decay_steps = max(1, int(num_episodes * 0.8))

        # PER beta schedule defaults (anneal beta from 0.4 to 1.0 over training)
        # For short runs, anneal faster (70% of episodes) and use more frequent updates
        if not hasattr(self, 'per_beta_start'):
            self.per_beta_start = 0.4
        if not hasattr(self, 'per_beta_end'):
            self.per_beta_end = 1.0
        if not hasattr(self, 'per_beta_anneal_steps'):
            # Anneal over 70% of episodes (faster for short runs)
            self.per_beta_anneal_steps = max(1, int(num_episodes * 0.7))

        # PATCH #3: Heuristic pre-fill before training
        prefill_steps = getattr(self, 'prefill_steps', 3000)  # Default 3000 for full runs
        if num_episodes <= 5:
            prefill_steps = 1000  # Shorter for smoke runs
        
        if prefill_steps > 0 and self.agent.replay_size == 0:
            prefill_replay(self.train_env, self.agent, steps=prefill_steps)
        
        # PATCH #6: Configure agent training parameters
        try:
            self.agent.double_dqn = True
            self.agent.loss_fn = getattr(self.agent, 'loss_fn', None) or 'huber'
            setattr(self.agent, 'grad_clip', getattr(self.agent, 'grad_clip', 1.0))  # PATCH #6: 1.0 for stability
            setattr(self.agent, 'replay_batch_size', getattr(self.agent, 'replay_batch_size', 256))  # PATCH #6: 256
            setattr(self.agent, 'gamma', 0.97)  # PATCH #6: 0.97 for hourly bars with costs
            
            # For smoke runs (≤5 episodes), use more aggressive update schedule
            if num_episodes <= 5:
                setattr(self.agent, 'update_every', 16)  # Update every 16 steps
                setattr(self.agent, 'grad_steps', 1)     # 1 grad step per update
            else:
                setattr(self.agent, 'update_every', getattr(self.agent, 'update_every', 4))

            setattr(self.agent, 'grad_steps', getattr(self.agent, 'grad_steps', 4))
            setattr(self.agent, 'polyak_tau', getattr(self.agent, 'polyak_tau', 0.005))  # PATCH #6: Soft updates

            # Fast path for smoke/short runs (prevents long waits for 1–5 episodes)
            if num_episodes <= 5:
                setattr(self.agent, 'replay_batch_size', min(getattr(self.agent, 'replay_batch_size', 256), 128))
                setattr(self.agent, 'update_every', max(getattr(self.agent, 'update_every', 4), 16))
                setattr(self.agent, 'grad_steps', 1)
        except Exception:
            pass

        for episode in range(1, num_episodes + 1):
            # Store episode number for JSON export
            self.episode = episode
            
            # Log episode start
            episode_start_time = datetime.now()
            self.structured_logger.log_episode_start(episode, episode_start_time)
            
            # compute current PER beta (linear anneal)
            frac_beta = min(1.0, episode / max(1, self.per_beta_anneal_steps))
            self.current_beta = float(self.per_beta_start + (self.per_beta_end - self.per_beta_start) * frac_beta)

            # BUGFIX: Domain-randomize validation frictions only if NOT frozen
            # (frozen by default for cross-episode/cross-seed comparability)
            if self.val_env is not None and not getattr(self.config, 'FREEZE_VALIDATION_FRICTIONS', False):
                try:
                    # Narrower stress band to avoid excessive inactivity during testing
                    s = float(np.random.uniform(0.00013, 0.00020))  # was 0.00012-0.00025
                    sp = float(np.random.uniform(0.6, 1.0))         # was 0.5-1.2
                    self.val_env.spread = s
                    if hasattr(self.val_env.risk_manager, 'slippage_pips'):
                        self.val_env.risk_manager.slippage_pips = sp
                except Exception:
                    pass

            # Train episode
            train_stats = self.train_episode()
            train_stats['episode'] = episode
            self.training_history.append(train_stats)
            
            # Log episode end
            episode_end_time = datetime.now()
            self.structured_logger.log_episode_end(
                episode, episode_end_time,
                reward=train_stats.get('episode_reward', 0.0),
                final_equity=train_stats.get('final_equity', 0.0),
                steps=train_stats.get('steps', 0),
                trades=train_stats.get('total_trades', 0),
                win_rate=train_stats.get('win_rate', 0.0),
                avg_loss=train_stats.get('avg_loss', 0.0),
                epsilon=train_stats.get('epsilon', getattr(self.agent, 'epsilon', 0.0))
            )

            # Log training metrics to TensorBoard if available
            if getattr(self, 'writer', None) is not None:
                try:
                    self.writer.add_scalar('train/episode_reward', float(train_stats.get('episode_reward', 0.0)), episode)
                    self.writer.add_scalar('train/avg_loss', float(train_stats.get('avg_loss', 0.0)), episode)
                    self.writer.add_scalar('train/final_equity', float(train_stats.get('final_equity', 0.0)), episode)
                    self.writer.add_scalar('train/epsilon', float(train_stats.get('epsilon', getattr(self.agent, 'epsilon', 0.0))), episode)
                    # quick Q-value diagnostics from a zero-state probe
                    try:
                        probe_state = np.zeros(getattr(self.agent, 'state_size', self.train_env.state_size), dtype=np.float32)
                        qvals = self.agent.get_q_values(probe_state)
                        self.writer.add_scalar('q/max', float(np.max(qvals)), episode)
                        self.writer.add_scalar('q/mean', float(np.mean(qvals)), episode)
                        self.writer.add_histogram('q/dist', qvals, episode)
                    except Exception:
                        pass
                except Exception:
                    pass

            # Validate
            if episode % validate_every == 0:
                val_stats = self.validate()
                val_stats['episode'] = episode
                self.validation_history.append(val_stats)

                # PATCH: EMA smoothing on stability-adjusted score
                current_fitness = float(val_stats.get('val_fitness', 0.0))
                
                # Initialize EMA on first validation
                if not hasattr(self, 'best_fitness_ema'):
                    self.best_fitness_ema = current_fitness
                    self.best_fitness_ema_saved = -1e9
                
                # Apply exponential smoothing with slower alpha for stability
                alpha = 0.2  # STABILITY: Reduced from 0.3 to 0.2 for less sensitivity to outliers
                self.best_fitness_ema = alpha * current_fitness + (1 - alpha) * self.best_fitness_ema
                metric_for_early_stop = self.best_fitness_ema
                
                # Early stop with min_validations patience floor
                min_validations = 6
                validations_done = len(self.validation_history)
                
                if metric_for_early_stop > best_fitness:
                    best_fitness = metric_for_early_stop
                    bad_count = 0
                    # Save best model based on EMA
                    if self.best_fitness_ema > self.best_fitness_ema_saved:
                        self.save_checkpoint(f"best_model.pt")
                        self.best_fitness_ema_saved = self.best_fitness_ema
                        if verbose:
                            ema_val = self.best_fitness_ema
                            raw_val = current_fitness
                            print(f"  [BEST] New best fitness (EMA): {ema_val:.4f} (raw: {raw_val:.4f})")
                else:
                    bad_count += 1
                    # Check early stop flag (disabled for seed sweeps)
                    disable_early_stop = getattr(self.config.training, 'disable_early_stop', False)
                    if not disable_early_stop and validations_done >= min_validations and bad_count >= patience:
                        if verbose:
                            msg = f"\n[EARLY STOP] Episode {episode} (no fitness improvement for {patience} validations)"
                            print(msg)
                        break

                # Build validation equity series
                if self.val_env is not None:
                    val_idx = pd.date_range(start='2024-01-01', periods=len(self.val_env.equity_history), freq='h')
                    val_equity = pd.Series(self.val_env.equity_history, index=val_idx)

                    fc = FitnessCalculator()
                    metrics = fc.calculate_all_metrics(val_equity)

                    # FIX #1: Wire computed fitness from last validation into val_stats for display
                    # Note: val_stats already has median fitness from validate(), just add sharpe/cagr
                    if 'val_sharpe' not in val_stats:
                        val_stats['val_sharpe'] = float(metrics['sharpe'])
                    if 'val_cagr' not in val_stats:
                        val_stats['val_cagr'] = float(metrics['cagr'])

                    # append metrics to history
                    train_stats.setdefault('val_sharpe', []).append(metrics['sharpe'])
                    train_stats.setdefault('val_cagr', []).append(metrics['cagr'])
                    train_stats.setdefault('val_fitness', []).append(val_stats.get('val_fitness', metrics['fitness']))
                    
                    # Log validation event
                    val_time = datetime.now()
                    self.structured_logger.log_validation(
                        episode, val_time, 
                        fitness=metrics.get('fitness', 0.0),
                        sharpe=metrics.get('sharpe', 0.0),
                        cagr=metrics.get('cagr', 0.0),
                        max_drawdown=metrics.get('max_drawdown_pct', 0.0),
                        total_trades=val_stats.get('val_total_trades', 0)
                    )

                    if verbose:
                        print(f"\nEpisode {episode}/{num_episodes}")
                        print(f"  Train - Reward: {train_stats['episode_reward']:.2f}, "
                              f"Equity: ${train_stats['final_equity']:.2f}, "
                              f"Trades: {train_stats.get('total_trades', 0)}, "
                              f"Win Rate: {train_stats.get('win_rate', 0):.2%}")
                        if val_stats:
                            val_sharpe = metrics['sharpe']
                            val_cagr = metrics['cagr']

                            print(f"  Val   - Reward: {val_stats['val_reward']:.2f}, "
                                  f"Equity: ${val_stats['val_final_equity']:.2f}, "
                                  f"Fitness: {val_stats.get('val_fitness', 0):.4f} | "
                                  f"Sharpe: {val_sharpe:.2f} | CAGR: {val_cagr:.2%}")
                            # write validation metrics
                            if getattr(self, 'writer', None) is not None:
                                try:
                                    self.writer.add_scalar('val/reward', float(val_stats.get('val_reward', 0.0)), episode)
                                    # some fitness components may be present
                                    if 'val_fitness' in val_stats:
                                        self.writer.add_scalar('val/fitness', float(val_stats.get('val_fitness', 0.0)), episode)
                                    if 'val_sharpe' in val_stats:
                                        self.writer.add_scalar('val/sharpe', float(val_stats.get('val_sharpe', val_sharpe)), episode)
                                    if 'val_cagr' in val_stats:
                                        self.writer.add_scalar('val/cagr', float(val_stats.get('val_cagr', val_cagr)), episode)
                                except Exception:
                                    pass
            else:
                if verbose and episode % 5 == 0:
                    print(f"Episode {episode}/{num_episodes} - "
                          f"Reward: {train_stats['episode_reward']:.2f}, "
                          f"Equity: ${train_stats['final_equity']:.2f}, "
                          f"eps: {train_stats['epsilon']:.4f}")

            # linear decay over configured steps, floor at epsilon_end (skip if using NoisyNet)
            if not getattr(self.agent, 'use_noisy', False):
                frac = min(1.0, (episode) / max(1, self.agent.epsilon_decay_steps))
                self.agent.epsilon = self.agent.epsilon_start + (self.agent.epsilon_end - self.agent.epsilon_start) * frac
                if self.agent.epsilon < self.agent.epsilon_end:
                    self.agent.epsilon = self.agent.epsilon_end

            # record epsilon
            train_stats['epsilon'] = self.agent.epsilon

            # Save checkpoint
            if episode % save_every == 0:
                self.save_checkpoint(f"checkpoint_ep{episode}.pt")
        
        # BEST-MODEL-RESTORE: Load best model weights before final save
        # This ensures "Score Final" aligns with "Score Best" from training
        best_model_path = Path(self.checkpoint_dir) / "best_model.pt"
        restored_best_checkpoint = False
        if best_model_path.exists():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(str(best_model_path), map_location=device)
            # Extract the policy_net state dict from checkpoint
            if 'policy_net_state_dict' in checkpoint:
                self.agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
                restored_best_checkpoint = True
                if verbose:
                    print(f"\n[RESTORE] Loaded best model from {best_model_path}")
        
        # POST-RESTORE FINAL EVAL: Run deterministic validation to capture true final score
        if restored_best_checkpoint and self.val_env is not None:
            if verbose:
                print(f"\n[POST-RESTORE] Running final validation with restored best model...")
            
            final_val_stats = self.validate()
            
            # Tag the final summary so comparison tools can recognize it
            import os
            import json
            out_dir = Path("logs") / "validation_summaries"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Load the most recent validation summary to get episode number
            recent_files = sorted(out_dir.glob("val_ep*.json"))
            last_episode = 0
            if recent_files:
                try:
                    last_file = json.load(open(recent_files[-1], "r"))
                    last_episode = last_file.get("episode", 0)
                except:
                    pass
            
            # Create final summary with post-restore tag
            final_summary = {
                "episode": last_episode,
                "episode_index": "final",
                "is_post_restore": True,
                "best_fitness_ema": float(self.best_fitness_ema),
                "score": final_val_stats.get("val_fitness", float("nan")),
                "eval_epsilon": float(getattr(self.config.agent, "eval_epsilon", 0.05)),
                "eval_tie_only": bool(getattr(self.config.agent, "eval_tie_only", True)),
                "k": final_val_stats.get("val_k", 7),
                "median_fitness": final_val_stats.get("val_median_fitness", float("nan")),
                "iqr": final_val_stats.get("val_iqr", float("nan")),
                "trades": final_val_stats.get("val_trades", 0),
                "seed": int(getattr(self.config, "random_seed", -1)),
            }
            
            # Save as val_final.json
            final_path = out_dir / "val_final.json"
            with open(final_path, "w") as f:
                json.dump(final_summary, f, indent=2)
            
            if verbose:
                print(f"[POST-RESTORE] Final score: {final_summary['score']:.3f} (saved to {final_path})")
            
            # --- ALT HOLD-OUT VALIDATION (shifted/strided) ---
            # Check robustness against a different validation regime (not just 600/90)
            if verbose:
                print(f"\n[POST-RESTORE:ALT] Running hold-out validation (shifted/strided)...")
            
            # Temporarily override validation parameters for alt regime
            try:
                # Save original config values
                orig_stride_frac = getattr(self.config, "VAL_STRIDE_FRAC", 0.15)
                orig_window_bars = getattr(self.config, "VAL_WINDOW_BARS", 600)
                
                # Set alt validation parameters (wider stride, shifted)
                self.config.VAL_STRIDE_FRAC = 0.20  # 20% stride (~120 bars)
                self.config.VAL_WINDOW_BARS = 600   # Keep window size same
                
                # Run validation with alt parameters
                alt_val_stats = self.validate()
                alt_score = alt_val_stats.get("val_fitness", 0.0)
                alt_components = alt_val_stats.get("spr_components", {})
                alt_trades = alt_val_stats.get("val_trades", 0)
                alt_k = alt_val_stats.get("val_k", 0)
                
                # Restore original config
                self.config.VAL_STRIDE_FRAC = orig_stride_frac
                self.config.VAL_WINDOW_BARS = orig_window_bars
                
                if verbose:
                    print(f"[POST-RESTORE:ALT] windows={alt_k} | SPR={alt_score:.3f} | "
                          f"PF={alt_components.get('pf', 0):.2f} | MDD={alt_components.get('mdd_pct', 0):.2f}% | "
                          f"MMR={alt_components.get('mmr_pct_mean', 0):.2f}% | TPY={alt_components.get('trades_per_year', 0):.1f} | "
                          f"SIG={alt_components.get('significance', 0):.2f}")
                
                # Save alt validation summary
                alt_summary = {
                    "episode": "final_alt",
                    "score": float(alt_score),
                    "trades": int(alt_trades),
                    "k": int(alt_k),
                    "seed": int(getattr(self.config, "random_seed", -1)),
                    "regime": "alt_600x120_stride20pct",
                    "components": alt_components,
                    "is_alt_holdout": True
                }
                
                alt_path = Path(self.config.log_dir) / "validation_summaries" / "val_final_alt.json"
                alt_path.parent.mkdir(parents=True, exist_ok=True)
                with open(alt_path, "w") as f:
                    json.dump(alt_summary, f, indent=2)
                    
            except Exception as e:
                if verbose:
                    print(f"[POST-RESTORE:ALT] WARNING: Alt validation failed: {e}")
                # Restore config even if validation fails
                try:
                    self.config.VAL_STRIDE_FRAC = orig_stride_frac
                    self.config.VAL_WINDOW_BARS = orig_window_bars
                except:
                    pass
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        
        # Save training history
        self.save_history()
        
        # Export extended telemetry if requested
        print(f"\n[DEBUG] Telemetry mode: {telemetry_mode}, Output dir: {output_dir}")
        if telemetry_mode == 'extended' and output_dir:
            self._export_extended_telemetry(output_dir)
        elif telemetry_mode == 'extended':
            print("[TELEMETRY] Extended telemetry requested but no output directory specified")
        
        # Close writer if present
        try:
            if getattr(self, 'writer', None) is not None:
                self.writer.close()
        except Exception:
            pass
         
        return {
             'training_history': self.training_history,
             'validation_history': self.validation_history,
        }
    
    def save_checkpoint(self, filename: str):
        """
        PATCH 7: Save agent checkpoint with scaler parameters.
        
        Args:
            filename: Checkpoint filename
        """
        filepath = self.checkpoint_dir / filename
        self.agent.save(str(filepath))
        
        # PATCH 7: Save scaler parameters alongside model
        # Extract scaler from training environment if available
        if hasattr(self.train_env, 'scaler_mu') and hasattr(self.train_env, 'scaler_sig'):
            scaler_mu = self.train_env.scaler_mu
            scaler_sig = self.train_env.scaler_sig
            
            if scaler_mu is not None and scaler_sig is not None:
                import json
                scaler_file = filepath.parent / f"{filepath.stem}_scaler.json"
                scaler_data = {
                    'mu': list(scaler_mu) if isinstance(scaler_mu, np.ndarray) else scaler_mu,
                    'sig': list(scaler_sig) if isinstance(scaler_sig, np.ndarray) else scaler_sig,
                    'feature_columns': list(self.train_env.feature_columns) if hasattr(self.train_env, 'feature_columns') else []
                }
                with open(scaler_file, 'w') as f:
                    json.dump(scaler_data, f, indent=2)
    
    def load_checkpoint(self, filename: str):
        """
        PATCH 7: Load agent checkpoint with scaler parameters.
        
        Args:
            filename: Checkpoint filename
        """
        filepath = self.checkpoint_dir / filename
        self.agent.load(str(filepath))
        
        # PATCH 7: Load scaler parameters if available
        scaler_file = filepath.parent / f"{filepath.stem}_scaler.json"
        if scaler_file.exists():
            import json
            with open(scaler_file, 'r') as f:
                scaler_data = json.load(f)
            # Apply to environments if they exist
            if hasattr(self, 'train_env') and self.train_env is not None:
                self.train_env.scaler_mu = np.array(scaler_data['mu'])
                self.train_env.scaler_sig = np.array(scaler_data['sig'])
            if hasattr(self, 'val_env') and self.val_env is not None:
                self.val_env.scaler_mu = np.array(scaler_data['mu'])
                self.val_env.scaler_sig = np.array(scaler_data['sig'])
    
    def save_history(self):
        """Save training and validation history to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save training history
        train_file = self.log_dir / f"training_history_{timestamp}.json"
        with open(train_file, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        # Save validation history
        if self.validation_history:
            val_file = self.log_dir / f"validation_history_{timestamp}.json"
            with open(val_file, 'w') as f:
                json.dump(self.validation_history, f, indent=2, default=str)
        
        print(f"\nHistory saved to {self.log_dir}")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Plot training curves.
        
        Args:
            save_path: Path to save plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.training_history:
                print("No training history to plot")
                return
            
            # Convert to DataFrame
            train_df = pd.DataFrame(self.training_history)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Episode reward
            axes[0, 0].plot(train_df['episode'], train_df['episode_reward'])
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Episode Reward')
            axes[0, 0].set_title('Training Reward')
            axes[0, 0].grid(True)
            
            # Final equity
            axes[0, 1].plot(train_df['episode'], train_df['final_equity'])
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Final Equity ($)')
            axes[0, 1].set_title('Final Equity per Episode')
            axes[0, 1].grid(True)
            
            # Win rate
            axes[1, 0].plot(train_df['episode'], train_df['win_rate'])
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Win Rate')
            axes[1, 0].set_title('Win Rate')
            axes[1, 0].grid(True)
            
            # Epsilon
            axes[1, 1].plot(train_df['episode'], train_df['epsilon'])
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Epsilon')
            axes[1, 1].set_title('Exploration Rate')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")
            else:
                plt.savefig(self.log_dir / "training_curves.png")
                print(f"Plot saved to {self.log_dir / 'training_curves.png'}")
            
            plt.close()
            
        except ImportError:
            print("Matplotlib not available for plotting")
    
    def _export_extended_telemetry(self, output_dir: str):
        """
        Export extended telemetry data for confirmation suite analysis.
        
        This exports episode-level metrics including controller variables
        (lambda_long, lambda_hold, tau, H_bits) and behavioral metrics
        (p_long_smoothed, p_hold_smoothed, run_len_max, switch_rate).
        
        Args:
            output_dir: Directory to save telemetry data
        """
        import json
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Collect extended telemetry from training history
        extended_metrics = []
        
        for train_stats in self.training_history:
            episode = train_stats.get('episode', 0)
            
            # Get controller telemetry (collected during episode)
            p_long = float(train_stats.get('p_long_smoothed', 0.5))
            p_hold = float(train_stats.get('p_hold_smoothed', 0.7))
            lambda_long = float(train_stats.get('lambda_long', 0.0))
            lambda_hold = float(train_stats.get('lambda_hold', 0.0))
            tau = float(train_stats.get('tau', 1.0))
            H_bits = float(train_stats.get('H_bits', 1.0))
            run_len_max = int(train_stats.get('run_len_max', 0))
            switch_rate = float(train_stats.get('switch_rate', 0.0))
            
            # Get trades from episode stats
            trades = int(train_stats.get('total_trades', train_stats.get('trades', 0)))
            
            # Calculate SPR from episode data
            final_equity = float(train_stats.get('final_equity', 1000.0))
            initial_equity = 1000.0  # Default from config
            
            # Simple return-based score
            SPR = (final_equity - initial_equity) / initial_equity if initial_equity > 0 else 0.0
            
            metric = {
                'episode': int(episode),
                'p_long_smoothed': p_long,
                'p_hold_smoothed': p_hold,
                'lambda_long': lambda_long,
                'lambda_hold': lambda_hold,
                'tau': tau,
                'H_bits': H_bits,
                'run_len_max': run_len_max,
                'trades': trades,
                'SPR': float(SPR),
                'switch_rate': switch_rate,
            }
            
            extended_metrics.append(metric)
        
        # Save to JSON in format expected by analyzer
        metrics_data = {
            'episodes': extended_metrics,
            'config': {
                'random_seed': int(getattr(self.config, 'random_seed', -1)),
                'num_episodes': len(extended_metrics),
            }
        }
        
        telemetry_file = output_path / 'episode_metrics.json'
        with open(telemetry_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"\n[TELEMETRY] Extended telemetry exported to: {telemetry_file}")
        print(f"[TELEMETRY] {len(extended_metrics)} episode metrics saved")


if __name__ == "__main__":
    print("Trainer Module")
    print("=" * 50)
    
    # This is a placeholder - actual training would be done in main.py
    print("\nTrainer module loaded successfully.")
    print("Use main.py to run training.")

