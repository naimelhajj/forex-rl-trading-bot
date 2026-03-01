"""
Training Module
Manages the training loop for the DQN agent.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import json
import random
from datetime import datetime
import time
import hashlib
import torch
import shutil
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

def compute_policy_metrics(action_seq, action_names=("HOLD","LONG","SHORT","CLOSE_POSITION","MOVE_SL_CLOSER","MOVE_SL_CLOSER_AGGRESSIVE","REVERSE_TO_LONG","REVERSE_TO_SHORT")):
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

    # map by name (assumes env index alignment with action_names)
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
        Action (0=HOLD, 1=LONG, 2=SHORT)
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


def prefill_replay(env: ForexTradingEnv, agent: DQNAgent, steps: int = 5000, policy: str = "baseline"):
    """
    PATCH #3: Pre-load replay buffer with heuristic baseline transitions.
    Helps DQN avoid starting from white noise.
    
    Args:
        env: Trading environment
        agent: DQN agent with replay buffer
        steps: Number of transitions to collect
    """
    if policy in (None, "", "none"):
        print("[PREFILL] Skipped (policy=none)")
        return
    policy = str(policy).lower()
    print(f"[PREFILL] Collecting {steps} transitions (policy={policy})...")
    s = env.reset()
    collected = 0
    
    for _ in range(steps):
        a = None
        if policy == "baseline":
            # Use baseline policy with frame stacking parameters
            a = baseline_policy(s, env.feature_columns, stack_n=env.stack_n, feature_dim=env.feature_dim)
            if getattr(env, "allowed_actions", None) is not None and a not in env.allowed_actions:
                a = None
        elif policy == "random":
            a = None
        else:
            raise ValueError(f"Unknown prefill policy: {policy}")

        if a is None:
            mask = getattr(env, "legal_action_mask", lambda: None)()
            if mask is not None:
                valid_actions = [i for i, ok in enumerate(mask) if ok]
            else:
                valid_actions = list(range(getattr(env, "action_space_size", ActionSpace.get_action_size())))
            if not valid_actions:
                a = 0
            else:
                non_hold = [i for i in valid_actions if i != 0]
                if non_hold and np.random.rand() < 0.7:
                    a = int(np.random.choice(non_hold))
                else:
                    a = int(np.random.choice(valid_actions))
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
        # Candidate checkpoints used for anti-regression end-of-run selection.
        self.checkpoint_candidates = []

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
        info = {'equity': getattr(self.train_env, 'equity', 0.0)}
        update_every = getattr(self.agent, 'update_every', 4)
        grad_steps = getattr(self.agent, 'grad_steps', 2)
        timeout_min = getattr(getattr(self, 'config', None), 'training', None)
        timeout_min = getattr(timeout_min, 'episode_timeout_min', None)
        timeout_seconds = None if timeout_min is None else max(0.0, float(timeout_min) * 60.0)
        episode_start = time.monotonic()
        heartbeat_secs = float(getattr(self.config.training, 'heartbeat_secs', 60.0) or 0.0)
        heartbeat_steps = int(getattr(self.config.training, 'heartbeat_steps', 200) or 0)
        next_heartbeat = (episode_start + heartbeat_secs) if heartbeat_secs > 0 else None
        
        # DIVERSITY: Track HOLD streaks during training to enable probe learning
        hold_streak = 0
        HOLD_ACTION = 0
        hold_tie_tau = getattr(self.config.agent, 'hold_tie_tau', 0.06)
        hold_break_after = getattr(self.config.agent, 'hold_break_after', 6)
        trade_gate_margin = getattr(self.config.agent, 'trade_gate_margin', 0.0)
        trade_gate_z = getattr(self.config.agent, 'trade_gate_z', 0.0)
        
        done = False
        while not done:
            if timeout_seconds is not None and (time.monotonic() - episode_start) > timeout_seconds:
                print(f"[TIMEOUT] Episode aborted after {timeout_min:.2f} minutes at step {steps}")
                done = True
                break

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
                        action_dim = int(getattr(self.train_env, "action_space_size", len(q_values)))
                        non_hold_actions = [a for a in range(action_dim) if a != HOLD_ACTION]
                        if mask is not None:
                            non_hold_actions = [a for a in non_hold_actions if mask[a]]
                        
                        if non_hold_actions:
                            best_non_hold_q = max(q_values[a] for a in non_hold_actions)
                            best_non_hold_idx = [a for a in non_hold_actions if q_values[a] == best_non_hold_q][0]
                            hold_q = q_values[HOLD_ACTION]
                            
                            # If near-tie, take best non-HOLD action (or require trade gate threshold)
                            required_gap = 0.0
                            if trade_gate_margin > 0.0:
                                required_gap = trade_gate_margin
                            if trade_gate_z > 0.0:
                                q_std = float(np.std(q_values))
                                if q_std < 1e-8:
                                    q_std = 1e-8
                                required_gap = max(required_gap, trade_gate_z * q_std)
                            if required_gap <= 0.0:
                                required_gap = -hold_tie_tau
                            if best_non_hold_q - hold_q >= required_gap:
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

            # Progress heartbeat so long episodes do not look hung.
            now = time.monotonic()
            heartbeat_due = False
            if heartbeat_steps > 0 and steps > 0 and (steps % heartbeat_steps == 0):
                heartbeat_due = True
            if next_heartbeat is not None and now >= next_heartbeat:
                heartbeat_due = True
            if heartbeat_due:
                print(
                    f"[HB] step={steps} env_bar={getattr(self.train_env, 'current_step', -1)} "
                    f"trades={getattr(self.train_env, 'trades_this_ep', 0)} "
                    f"equity={info.get('equity', 0.0):.2f} eps={getattr(self.agent, 'epsilon', 0.0):.4f} "
                    f"replay={getattr(self.agent, 'replay_size', 0)} elapsed={(now - episode_start):.1f}s"
                )
                if next_heartbeat is not None:
                    while next_heartbeat <= now:
                        next_heartbeat += heartbeat_secs
        
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
        # Apply exact frictions provided by caller.
        # Validation jitter/freeze policy is handled in validate().
        self.val_env.spread = float(base_spread)
        self.val_env.commission = float(base_commission)
        
        # Start directly at the target bar (avoids expensive HOLD fast-forward).
        state = self.val_env.reset(start_idx=int(start_idx))
        
        # now zero out histories so metrics cover ONLY this slice
        if hasattr(self.val_env, 'equity_history'):
            self.val_env.equity_history = [getattr(self.val_env, 'equity', 1000.0)]
        if hasattr(self.val_env, 'trade_history'):
            try:
                self.val_env.trade_history.clear()
            except Exception:
                self.val_env.trade_history = []
        
        # State is now at start_idx position from fast-forward loop above
        episode_reward = 0
        steps = 0
        
        # PATCH A+: Track action counts for validation diagnostics
        action_dim = int(getattr(self.val_env, "action_space_size", ActionSpace.get_action_size()))
        action_counts = np.zeros(action_dim, dtype=int)
        action_sequence = []  # PATCH D: Track action sequence for metrics
        
        # SPR: Track data for SPR fitness computation
        window_timestamps = []  # Bar timestamps
        window_equity = []      # Equity curve per step
        window_trade_pnls = []  # Realized P&L per closed trade
        
        # Get initial timestamp (1H bars starting from 2024-01-01)
        from datetime import timedelta
        base_timestamp = datetime(2024, 1, 1)
        
        done = False
        while not done and steps < (end_idx - start_idx):
            # Get legal action mask
            mask = getattr(self.val_env, 'legal_action_mask', lambda: None)()
            # Keep validation action policy identical to test evaluation policy:
            # deterministic greedy in eval mode with legal-action masking.
            action = self.agent.select_action(state, explore=False, mask=mask, eval_mode=True, env=self.val_env)
            
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
            if hasattr(self.val_env, 'trade_history') and len(self.val_env.trade_history) > len(window_trade_pnls):
                # New trade closed - get its P&L
                latest_trade = self.val_env.trade_history[-1]
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
        
        final_equity = float(info.get('equity', self.val_env.equity))
        init_balance = float(getattr(self.config.environment, "initial_balance", 1000.0))
        return_pct = ((final_equity - init_balance) / max(1e-9, init_balance)) * 100.0

        return {
            'fitness': fitness_raw,
            'trades': trades,
            'reward': episode_reward,
            'equity': final_equity,
            'return_pct': float(return_pct),
            'steps': steps,
            'metrics': metrics,
            'trade_stats': trade_stats,
            'action_counts': action_counts,  # PATCH C: Return for histogram aggregation
            'action_sequence': action_sequence  # PATCH D: Return for metrics computation
        }
    
    def validate(self,
                 stride_frac_override: Optional[float] = None,
                 window_bars_override: Optional[int] = None,
                 start_frac_override: Optional[float] = None,
                 end_frac_override: Optional[float] = None,
                 use_all_windows_override: Optional[bool] = None,
                 persist_summary: bool = True,
                 quiet: bool = False) -> Dict:
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
        current_slippage = float(getattr(self.val_env, "slippage_pips", 0.0))
        
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
        
        if val_length <= 0:
            return {}

        # Optional validation segment (used by anti-regression tournament to
        # score a tail hold-out slice separately from the full validation span).
        start_frac = 0.0 if start_frac_override is None else float(start_frac_override)
        end_frac = 1.0 if end_frac_override is None else float(end_frac_override)
        start_frac = max(0.0, min(1.0, start_frac))
        end_frac = max(start_frac + 1e-6, min(1.0, end_frac))
        seg_lo = int(round(val_length * start_frac))
        seg_hi = int(round(val_length * end_frac))
        seg_lo = max(0, min(seg_lo, val_length - 1))
        seg_hi = max(seg_lo + 1, min(seg_hi, val_length))
        segment_length = max(1, seg_hi - seg_lo)

        # GATING-FIX: Use fixed window size if specified
        window_bars = window_bars_override if window_bars_override is not None else getattr(self.config, "VAL_WINDOW_BARS", None)
        if window_bars is not None:
            window_len = min(window_bars, segment_length)
        else:
            window_frac = getattr(self.config, "VAL_WINDOW_FRAC", 0.40)
            window_len = max(100, int(segment_length * window_frac))
        
        # Compute stride and K
        stride_frac = stride_frac_override if stride_frac_override is not None else getattr(self.config, "VAL_STRIDE_FRAC", 0.15)
        stride = max(1, int(window_len * stride_frac))
        
        min_k = getattr(self.config, "VAL_MIN_K", 6)
        max_k = getattr(self.config, "VAL_MAX_K", 7)

        def _pick_even_spread(candidates, k):
            if not candidates:
                return []
            if k >= len(candidates):
                return list(candidates)
            idx = np.linspace(0, len(candidates) - 1, num=k, dtype=int)
            out = []
            seen = set()
            for i in idx:
                s = int(candidates[int(i)])
                if s not in seen:
                    seen.add(s)
                    out.append(s)
            return out

        # Generate candidate starts over the entire validation span, then pick evenly.
        max_start = max(0, segment_length - window_len)
        if max_start == 0:
            candidate_starts = [0]
        else:
            candidate_starts = list(range(0, max_start + 1, stride))
            if candidate_starts[-1] != max_start:
                candidate_starts.append(max_start)

        use_all_windows = bool(use_all_windows_override) if use_all_windows_override is not None else False
        if use_all_windows:
            starts = list(candidate_starts)
        else:
            starts = _pick_even_spread(candidate_starts, min(max_k, len(candidate_starts)))

            # Ensure minimum K windows by reducing stride if needed.
            if len(starts) < min_k and segment_length >= window_len:
                stride_min = max(1, (segment_length - window_len) // max(1, (min_k - 1)))
                if max_start == 0:
                    dense_candidates = [0]
                else:
                    dense_candidates = list(range(0, max_start + 1, stride_min))
                    if dense_candidates[-1] != max_start:
                        dense_candidates.append(max_start)
                starts = _pick_even_spread(dense_candidates, min(min_k, len(dense_candidates)))
        starts = sorted(starts)
        
        # Build window tuples (start, end)
        windows = [
            (seg_lo + start, min(seg_lo + start + window_len, seg_hi))
            for start in starts
        ]

        # Validation may need a longer horizon than training episodes (e.g., tournament alt windows).
        # Temporarily raise val_env.max_steps so each slice can run for the full selected window length.
        orig_val_max_steps = getattr(self.val_env, "max_steps", None)
        if orig_val_max_steps is not None and int(orig_val_max_steps) < int(window_len):
            self.val_env.max_steps = int(window_len)
        
        # Calculate coverage (how much of val data is seen across all windows)
        if len(windows) > 1:
            stride_actual = starts[1] - starts[0] if len(starts) > 1 else window_len
            coverage = (window_len + (len(windows)-1) * stride_actual) / max(1, segment_length)
        else:
            coverage = window_len / max(1, segment_length)
        
        # Log validation setup
        jitter_msg = f" | jitter-avg K={jitter_draws}" if jitter_draws > 1 and not freeze_frictions else ""
        if not quiet:
            print(f"[VAL] {len(windows)} passes | window={window_len} | stride~{int(window_len*stride_frac)} | "
                  f"coverage~{coverage:.2f}x{jitter_msg} | all_windows={use_all_windows}")
            if start_frac_override is not None or end_frac_override is not None:
                print(
                    f"[VAL] segment={start_frac:.2f}-{end_frac:.2f} "
                    f"(idx {seg_lo}:{seg_hi}, bars={segment_length})"
                )
        
        # Log window ranges (first 3 for brevity)
        if not quiet:
            for i, (lo, hi) in enumerate(windows[:3]):
                if hasattr(val_idx, '__getitem__'):
                    try:
                        start_time = val_idx[lo]
                        end_time = val_idx[hi - 1]
                        print(f"[VAL] window {i+1}: {start_time} to {end_time}  ({hi-lo} bars)")
                    except Exception:
                        print(f"[VAL] window {i+1}: idx {lo} to {hi}  ({hi-lo} bars)")
                else:
                    print(f"[VAL] window {i+1}: idx {lo} to {hi}  ({hi-lo} bars)")
        
        # Run validation on each disjoint window with jitter averaging
        fits, trade_counts = [], []
        window_return_pcts, window_pfs = [], []
        all_results = []
        action_dim = int(getattr(self.val_env, "action_space_size", ActionSpace.get_action_size()))
        total_action_counts = np.zeros(action_dim, dtype=int)
        all_actions = []  # PATCH D: Collect all action sequences
        
        # SPR: Store last SPR info for logging
        self._last_spr_info = None
        
        for (lo, hi) in windows:
            # PHASE-2.8c: Jitter-averaged validation (K friction draws per window)
            if jitter_draws > 1 and not freeze_frictions:
                # Run K draws with different friction values, average the results
                jitter_fits = []
                jitter_trades = []
                jitter_returns = []
                jitter_pfs = []
                jitter_results = []
                jitter_action_counts = np.zeros(action_dim, dtype=int)
                jitter_actions = []
                
                for draw_i in range(jitter_draws):
                    # Randomize frictions for this draw
                    jitter_spread = current_spread * np.random.uniform(*spread_jitter)
                    jitter_commission = current_commission * np.random.uniform(*commission_jitter)
                    
                    # Run validation slice with jittered frictions
                    stats = self._run_validation_slice(lo, hi, jitter_spread, jitter_commission)
                    jitter_fits.append(stats['fitness'])
                    jitter_trades.append(stats['trades'])
                    jitter_returns.append(float(stats.get('return_pct', 0.0)))
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
                        jitter_pfs.append(float(stats['metrics']['spr_info'].get('pf', 0.0)))
                    elif 'metrics' in stats and 'profit_factor' in stats['metrics']:
                        jitter_pfs.append(float(stats['metrics'].get('profit_factor', 0.0)))
                
                # Average over jitter draws
                window_fit = float(np.mean(jitter_fits))
                window_trades = int(np.mean(jitter_trades))
                fits.append(window_fit)
                trade_counts.append(window_trades)
                window_return_pcts.append(float(np.mean(jitter_returns)) if jitter_returns else 0.0)
                if jitter_pfs:
                    window_pfs.append(float(np.mean(jitter_pfs)))
                
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
                window_return_pcts.append(float(stats.get('return_pct', 0.0)))
                
                # SPR: Capture SPR info from any window (use last for display)
                if 'metrics' in stats and 'spr_info' in stats['metrics']:
                    self._last_spr_info = stats['metrics']['spr_info']
                    window_pfs.append(float(stats['metrics']['spr_info'].get('pf', 0.0)))
                elif 'metrics' in stats and 'profit_factor' in stats['metrics']:
                    window_pfs.append(float(stats['metrics'].get('profit_factor', 0.0)))
                
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
        self.val_env.slippage_pips = current_slippage
        self.val_env.max_steps = orig_val_max_steps
        if hasattr(self.val_env, "risk_manager"):
            self.val_env.risk_manager.slippage_pips = current_slippage
        
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
        
        disable_trade_gating = getattr(self.config, "VAL_DISABLE_TRADE_GATING", False)

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
        
        if not disable_trade_gating:
            if is_low_trade:
                if self.last_val_was_low_trade:
                    # Second consecutive low-trade episode - apply penalty
                    shortfall = (min_half - median_trades) / max(1, min_half)  # 0..1
                    penalty_max = getattr(self.config, "VAL_PENALTY_MAX", 0.10)
                    undertrade_penalty = min(penalty_max, round(0.5 * shortfall, 3))
                # else: First offense - grace period, pen=0.00
        else:
            mult = 1.00
            undertrade_penalty = 0.0
            is_low_trade = False
        
        # Update grace tracker for next validation
        self.last_val_was_low_trade = is_low_trade
        
        # DEBUG: Print gating thresholds (can remove after verification)
        if not quiet:
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
        # Expose robust-validation internals to callers (training loop/post-restore ALT).
        val_stats['val_k'] = int(len(windows))
        val_stats['val_window_bars'] = int(window_len)
        val_stats['val_stride_bars'] = int(stride)
        val_stats['val_segment_bars'] = int(segment_length)
        val_stats['val_use_all_windows'] = bool(use_all_windows)
        val_stats['val_median_fitness'] = float(median)
        val_stats['val_iqr'] = float(iqr)
        val_stats['val_stability_adj'] = float(stability_adj)
        val_stats['val_mult'] = float(mult)
        val_stats['val_undertrade_penalty'] = float(undertrade_penalty)
        val_stats['val_return_pct'] = float(np.mean(window_return_pcts)) if window_return_pcts else 0.0
        val_stats['val_median_return_pct'] = float(np.median(window_return_pcts)) if window_return_pcts else 0.0
        val_stats['val_return_q25_pct'] = float(np.percentile(window_return_pcts, 25)) if window_return_pcts else 0.0
        val_stats['val_return_q10_pct'] = float(np.percentile(window_return_pcts, 10)) if window_return_pcts else 0.0
        val_stats['val_median_pf'] = float(np.median(window_pfs)) if window_pfs else 0.0
        val_stats['val_pf_q25'] = float(np.percentile(window_pfs, 25)) if window_pfs else 0.0
        val_stats['val_pf_q10'] = float(np.percentile(window_pfs, 10)) if window_pfs else 0.0
        if window_return_pcts:
            return_arr = np.asarray(window_return_pcts, dtype=float)
            val_stats['val_positive_frac'] = float(np.mean(return_arr > 0.0))
        else:
            val_stats['val_positive_frac'] = 0.0
        if window_pfs:
            pf_arr = np.asarray(window_pfs, dtype=float)
            val_stats['val_pf_ge_1_frac'] = float(np.mean(pf_arr >= 1.0))
        else:
            val_stats['val_pf_ge_1_frac'] = 0.0
        
        # Quality-of-life print - enhanced for SPR mode
        fitness_mode = getattr(self.config.fitness, 'mode', 'legacy')
        if fitness_mode == 'spr':
            # Extract SPR components from fits list (use median window for display)
            # Note: SPR info is in the stats returned from _run_validation_slice
            # We'll collect it during the window loop and store for display
            spr_components = getattr(self, '_last_spr_info', None)
            
            if not quiet:
                if spr_components:
                    print(f"[VAL] K={len(windows)} overlapping | SPR={val_score:.3f} | "
                          f"PF={spr_components.get('pf', 0):.2f} | "
                          f"MDD={spr_components.get('mdd_pct', 0):.2f}% | "
                          f"MMR={spr_components.get('mmr_pct_mean', 0):.2f}% | "
                          f"TPY={spr_components.get('trades_per_year', 0):.1f} | "
                          f"SIG={spr_components.get('significance', 0):.2f} | "
                          f"ret_med={val_stats['val_median_return_pct']:.2f}% | "
                          f"trades={median_trades:.1f} | mult={mult:.2f} | pen={undertrade_penalty:.3f}")
                else:
                    print(f"[VAL] K={len(windows)} overlapping | SPR={val_score:.3f} | "
                          f"ret_med={val_stats['val_median_return_pct']:.2f}% | "
                          f"trades={median_trades:.1f} | mult={mult:.2f} | pen={undertrade_penalty:.3f}")
        else:
            # Legacy Sharpe/CAGR display
            if not quiet:
                print(f"[VAL] K={len(windows)} overlapping | median={median:.3f} (trimmed) | "
                      f"IQR={iqr:.3f} | iqr_pen={iqr_penalty:.3f} | adj={stability_adj:.3f} | "
                      f"ret_med={val_stats['val_median_return_pct']:.2f}% | "
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
        ACTION_NAMES = tuple(
            ActionSpace.get_action_name(i)
            for i in range(int(getattr(self.val_env, "action_space_size", ActionSpace.get_action_size())))
        )
        
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
            "slippage_pips": float(getattr(self.val_env, "slippage_pips", 0.0)),
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
            "return_pct_mean": val_stats["val_return_pct"],
            "return_pct_median": val_stats["val_median_return_pct"],
            "return_pct_q25": val_stats["val_return_q25_pct"],
            "return_pct_q10": val_stats["val_return_q10_pct"],
            "pf_median": val_stats["val_median_pf"],
            "pf_q25": val_stats["val_pf_q25"],
            "pf_q10": val_stats["val_pf_q10"],
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
            # Mirror SPR components in returned stats for post-restore/ALT logging.
            val_stats["spr_components"] = dict(summary["spr_components"])
        
        if persist_summary:
            out_dir = self.log_dir / "validation_summaries"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # File per episode keeps things simple for the checker
            if summary['episode'] is not None:
                fname = out_dir / f"val_ep{summary['episode']:03d}.json"
                with open(fname, "w", encoding="utf-8") as f:
                    # ensure_ascii avoids Windows codepage headaches
                    json.dump(summary, f, ensure_ascii=True, indent=2)
        
        return val_stats

    def _register_checkpoint_candidate(self, episode: int, val_stats: Dict, ema_score: float) -> None:
        """Persist a validation-time candidate checkpoint for end-of-run tournament."""
        keep_limit = int(getattr(self.config.training, "anti_regression_candidate_keep", 24))
        filename = f"candidate_ep{int(episode):03d}.pt"
        self.save_checkpoint(filename)

        entry = {
            "episode": int(episode),
            "filename": filename,
            "val_fitness": float(val_stats.get("val_fitness", 0.0)),
            "val_median_fitness": float(val_stats.get("val_median_fitness", 0.0)),
            "val_iqr": float(val_stats.get("val_iqr", 0.0)),
            "val_trades": float(val_stats.get("val_trades", 0.0)),
            "ema_fitness": float(ema_score),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.checkpoint_candidates.append(entry)

        # Keep top-N by EMA (plus newest), remove pruned files to avoid checkpoint bloat.
        if len(self.checkpoint_candidates) > keep_limit:
            newest = max(self.checkpoint_candidates, key=lambda x: x.get("episode", -1))
            ranked = sorted(
                self.checkpoint_candidates,
                key=lambda x: (x.get("ema_fitness", -1e9), x.get("val_fitness", -1e9)),
                reverse=True,
            )
            survivors = ranked[:keep_limit]
            if newest not in survivors:
                survivors.append(newest)

            survivor_names = {item["filename"] for item in survivors}
            for candidate in self.checkpoint_candidates:
                name = candidate.get("filename")
                if name and name not in survivor_names:
                    path = self.checkpoint_dir / name
                    scaler_path = self.checkpoint_dir / f"{Path(name).stem}_scaler.json"
                    try:
                        if path.exists():
                            path.unlink()
                    except OSError:
                        pass
                    try:
                        if scaler_path.exists():
                            scaler_path.unlink()
                    except OSError:
                        pass
            self.checkpoint_candidates = survivors

    def _run_anti_regression_tournament(self, verbose: bool = True) -> Optional[str]:
        """
        Evaluate shortlisted checkpoints on base + alternate validation regimes and
        return the best robust candidate filename.
        """
        training_cfg = getattr(self.config, "training", None)
        if training_cfg is None or not getattr(training_cfg, "anti_regression_checkpoint_selection", True):
            return None
        if self.val_env is None:
            return None

        min_validations = int(getattr(training_cfg, "anti_regression_min_validations", 4))
        if len(self.validation_history) < min_validations:
            return None

        # Build shortlist from tracked candidates and current best model.
        ranked = sorted(
            self.checkpoint_candidates,
            key=lambda x: (x.get("ema_fitness", -1e9), x.get("val_fitness", -1e9)),
            reverse=True,
        )
        top_k = max(1, int(getattr(training_cfg, "anti_regression_eval_top_k", 6)))
        shortlist = ranked[:top_k]
        if ranked:
            newest = max(ranked, key=lambda x: x.get("episode", -1))
            if newest not in shortlist:
                shortlist.append(newest)

        best_model_name = "best_model.pt"
        best_model_path = self.checkpoint_dir / best_model_name
        if best_model_path.exists() and not any(c.get("filename") == best_model_name for c in shortlist):
            shortlist.append(
                {
                    "episode": -1,
                    "filename": best_model_name,
                    "val_fitness": -1e9,
                    "val_median_fitness": -1e9,
                    "val_iqr": 0.0,
                    "val_trades": 0.0,
                    "ema_fitness": -1e9,
                    "saved_at": "best_model_fallback",
                }
            )

        # De-duplicate while preserving order.
        dedup = []
        seen = set()
        for item in shortlist:
            name = item.get("filename")
            if not name or name in seen:
                continue
            seen.add(name)
            dedup.append(item)
        shortlist = dedup

        if not shortlist:
            return None

        alt_stride = float(getattr(training_cfg, "anti_regression_alt_stride_frac", 0.20))
        alt_window = getattr(training_cfg, "anti_regression_alt_window_bars", None)
        if alt_window is not None:
            alt_window = int(alt_window)
        tail_start_frac = float(getattr(training_cfg, "anti_regression_tail_start_frac", 0.50))
        tail_end_frac = float(getattr(training_cfg, "anti_regression_tail_end_frac", 1.0))
        tail_weight = float(getattr(training_cfg, "anti_regression_tail_weight", 0.75))
        tail_start_frac = max(0.0, min(0.95, tail_start_frac))
        tail_end_frac = max(tail_start_frac + 0.05, min(1.0, tail_end_frac))
        trade_floor = float(getattr(self.config, "VAL_MIN_HALF_TRADES", 0))
        min_positive_frac = float(getattr(training_cfg, "anti_regression_min_positive_frac", 0.50))
        selector_mode = str(getattr(training_cfg, "anti_regression_selector_mode", "tail_holdout")).strip().lower()
        if selector_mode not in ("tail_holdout", "future_first", "auto_rescue", "base_first"):
            selector_mode = "tail_holdout"
        ranking_selector_mode = "tail_holdout" if selector_mode == "auto_rescue" else selector_mode
        base_return_floor = float(getattr(training_cfg, "anti_regression_base_return_floor", 0.0))
        base_penalty_weight = float(getattr(training_cfg, "anti_regression_base_penalty_weight", 0.15))
        auto_rescue_enabled = bool(getattr(training_cfg, "anti_regression_auto_rescue_enabled", True))
        auto_rescue_winner_forward_return_max = float(
            getattr(training_cfg, "anti_regression_auto_rescue_winner_forward_return_max", 0.65)
        )
        auto_rescue_forward_return_edge_min = float(
            getattr(training_cfg, "anti_regression_auto_rescue_forward_return_edge_min", 0.10)
        )
        auto_rescue_forward_pf_edge_min = float(
            getattr(training_cfg, "anti_regression_auto_rescue_forward_pf_edge_min", 0.10)
        )
        auto_rescue_challenger_base_return_max = float(
            getattr(training_cfg, "anti_regression_auto_rescue_challenger_base_return_max", 0.0)
        )
        auto_rescue_challenger_forward_pf_min = float(
            getattr(training_cfg, "anti_regression_auto_rescue_challenger_forward_pf_min", 1.35)
        )
        tiebreak_enabled = bool(getattr(training_cfg, "anti_regression_tiebreak_enabled", False))
        tiebreak_window_bars = max(
            200,
            int(getattr(training_cfg, "anti_regression_tiebreak_window_bars", 2400))
        )
        tiebreak_start_frac = float(getattr(training_cfg, "anti_regression_tiebreak_start_frac", 0.20))
        tiebreak_end_frac = float(getattr(training_cfg, "anti_regression_tiebreak_end_frac", 1.00))
        tiebreak_start_frac = max(0.0, min(0.95, tiebreak_start_frac))
        tiebreak_end_frac = max(tiebreak_start_frac + 0.01, min(1.0, tiebreak_end_frac))
        tiebreak_return_edge_min = float(
            getattr(training_cfg, "anti_regression_tiebreak_return_edge_min", 0.15)
        )
        tiebreak_pf_edge_min = float(
            getattr(training_cfg, "anti_regression_tiebreak_pf_edge_min", 0.10)
        )
        tiebreak_min_trades = float(
            getattr(training_cfg, "anti_regression_tiebreak_min_trades", 10.0)
        )
        horizon_rescue_enabled = bool(
            getattr(training_cfg, "anti_regression_horizon_rescue_enabled", False)
        )
        horizon_window_bars = max(
            200,
            int(getattr(training_cfg, "anti_regression_horizon_window_bars", 2400)),
        )
        horizon_start_frac = float(
            getattr(training_cfg, "anti_regression_horizon_start_frac", 0.20)
        )
        horizon_end_frac = float(
            getattr(training_cfg, "anti_regression_horizon_end_frac", 1.00)
        )
        horizon_start_frac = max(0.0, min(0.95, horizon_start_frac))
        horizon_end_frac = max(horizon_start_frac + 0.01, min(1.0, horizon_end_frac))
        horizon_candidate_limit = max(
            2,
            int(getattr(training_cfg, "anti_regression_horizon_candidate_limit", 8)),
        )
        horizon_incumbent_return_max = float(
            getattr(training_cfg, "anti_regression_horizon_incumbent_return_max", 0.40)
        )
        horizon_return_edge_min = float(
            getattr(training_cfg, "anti_regression_horizon_return_edge_min", 0.25)
        )
        horizon_pf_edge_min = float(
            getattr(training_cfg, "anti_regression_horizon_pf_edge_min", 0.10)
        )
        horizon_challenger_base_return_max = float(
            getattr(
                training_cfg,
                "anti_regression_horizon_challenger_base_return_max",
                1.0,
            )
        )
        horizon_challenger_robust_return_min = float(
            getattr(
                training_cfg,
                "anti_regression_horizon_challenger_robust_return_min",
                0.0,
            )
        )
        horizon_challenger_pf_min = float(
            getattr(training_cfg, "anti_regression_horizon_challenger_pf_min", 1.35)
        )
        horizon_min_trades = float(
            getattr(training_cfg, "anti_regression_horizon_min_trades", 10.0)
        )
        alignment_probe_enabled = bool(
            getattr(training_cfg, "anti_regression_alignment_probe_enabled", False)
        )
        alignment_probe_top_k = max(
            2,
            int(getattr(training_cfg, "anti_regression_alignment_probe_top_k", 2)),
        )
        alignment_probe_window_bars = getattr(
            training_cfg, "anti_regression_alignment_probe_window_bars", None
        )
        if alignment_probe_window_bars is not None:
            alignment_probe_window_bars = max(100, int(alignment_probe_window_bars))
        alignment_probe_stride_frac = getattr(
            training_cfg, "anti_regression_alignment_probe_stride_frac", None
        )
        if alignment_probe_stride_frac is not None:
            alignment_probe_stride_frac = max(0.01, float(alignment_probe_stride_frac))
        alignment_probe_use_all_windows = bool(
            getattr(training_cfg, "anti_regression_alignment_probe_use_all_windows", True)
        )
        alignment_probe_return_edge_min = float(
            getattr(training_cfg, "anti_regression_alignment_probe_return_edge_min", 0.10)
        )
        alignment_probe_pf_edge_min = float(
            getattr(training_cfg, "anti_regression_alignment_probe_pf_edge_min", 0.10)
        )
        alignment_probe_min_trades = float(
            getattr(training_cfg, "anti_regression_alignment_probe_min_trades", 10.0)
        )
        alignment_probe_require_pass = bool(
            getattr(training_cfg, "anti_regression_alignment_probe_require_pass", True)
        )
        tournament_min_k = getattr(training_cfg, "anti_regression_eval_min_k", None)
        tournament_max_k = getattr(training_cfg, "anti_regression_eval_max_k", None)
        tournament_jitter_draws = getattr(training_cfg, "anti_regression_eval_jitter_draws", None)
        if tournament_min_k is not None:
            tournament_min_k = max(1, int(tournament_min_k))
        if tournament_max_k is not None:
            tournament_max_k = max(1, int(tournament_max_k))
        if tournament_min_k is not None and tournament_max_k is not None and tournament_max_k < tournament_min_k:
            tournament_max_k = tournament_min_k
        if tournament_jitter_draws is not None:
            tournament_jitter_draws = max(1, int(tournament_jitter_draws))

        rng_state = np.random.get_state()
        py_rng_state = random.getstate()
        orig_val_min_k = getattr(self.config, "VAL_MIN_K", None)
        orig_val_max_k = getattr(self.config, "VAL_MAX_K", None)
        orig_val_jitter_draws = getattr(self.config, "VAL_JITTER_DRAWS", None)
        if tournament_min_k is not None:
            setattr(self.config, "VAL_MIN_K", tournament_min_k)
        if tournament_max_k is not None:
            setattr(self.config, "VAL_MAX_K", tournament_max_k)
        if tournament_jitter_draws is not None:
            setattr(self.config, "VAL_JITTER_DRAWS", tournament_jitter_draws)
        base_seed = int(getattr(self.config, "random_seed", 777))
        tournament = []
        try:
            for item in shortlist:
                filename = item["filename"]
                ckpt_path = self.checkpoint_dir / filename
                if not ckpt_path.exists():
                    continue

                try:
                    self.load_checkpoint(filename)
                except Exception:
                    continue

                # Use deterministic seeds so candidates are compared on identical jitter draws.
                random.seed(base_seed + 17)
                np.random.seed(base_seed + 17)
                base_stats = self.validate(persist_summary=False, quiet=True)
                random.seed(base_seed + 101)
                np.random.seed(base_seed + 101)
                alt_stats = self.validate(
                    stride_frac_override=alt_stride,
                    window_bars_override=alt_window,
                    persist_summary=False,
                    quiet=True,
                )
                random.seed(base_seed + 149)
                np.random.seed(base_seed + 149)
                tail_stats = self.validate(
                    stride_frac_override=alt_stride,
                    window_bars_override=alt_window,
                    start_frac_override=tail_start_frac,
                    end_frac_override=tail_end_frac,
                    persist_summary=False,
                    quiet=True,
                )

                base_score = float(base_stats.get("val_fitness", 0.0))
                alt_score = float(alt_stats.get("val_fitness", 0.0))
                tail_score = float(tail_stats.get("val_fitness", 0.0))
                base_return_median = float(base_stats.get("val_median_return_pct", base_stats.get("val_return_pct", 0.0)))
                alt_return_median = float(alt_stats.get("val_median_return_pct", alt_stats.get("val_return_pct", 0.0)))
                tail_return_median = float(tail_stats.get("val_median_return_pct", tail_stats.get("val_return_pct", 0.0)))
                base_pf_median = float(base_stats.get("val_median_pf", 0.0))
                alt_pf_median = float(alt_stats.get("val_median_pf", 0.0))
                tail_pf_median = float(tail_stats.get("val_median_pf", 0.0))
                base_return = base_return_median
                alt_return = alt_return_median
                tail_return = tail_return_median
                base_pf = base_pf_median
                alt_pf = alt_pf_median
                tail_pf = tail_pf_median
                base_pos_frac = float(base_stats.get("val_positive_frac", 0.0))
                alt_pos_frac = float(alt_stats.get("val_positive_frac", 0.0))
                tail_pos_frac = float(tail_stats.get("val_positive_frac", 0.0))
                base_pf_ge_1_frac = float(base_stats.get("val_pf_ge_1_frac", 0.0))
                alt_pf_ge_1_frac = float(alt_stats.get("val_pf_ge_1_frac", 0.0))
                tail_pf_ge_1_frac = float(tail_stats.get("val_pf_ge_1_frac", 0.0))
                base_iqr = float(base_stats.get("val_iqr", 0.0))
                alt_iqr = float(alt_stats.get("val_iqr", 0.0))
                tail_iqr = float(tail_stats.get("val_iqr", 0.0))
                base_trades = float(base_stats.get("val_trades", 0.0))
                alt_trades = float(alt_stats.get("val_trades", 0.0))
                tail_trades = float(tail_stats.get("val_trades", 0.0))

                # Selector modes:
                # - tail_holdout (default): strict robustness across base+alt+tail.
                # - future_first: prioritize alt+tail and use base as a soft penalty.
                # - base_first: prioritize base-window robustness only.
                # - auto_rescue: scored as tail_holdout, with optional post-score rescue.
                if ranking_selector_mode == "future_first":
                    robust_score = min(alt_score, tail_score)
                    robust_return = min(alt_return, tail_return)
                    robust_pf = min(alt_pf, tail_pf)
                    robust_pos_frac = min(alt_pos_frac, tail_pos_frac)
                    robust_pf_ge_1_frac = min(alt_pf_ge_1_frac, tail_pf_ge_1_frac)
                    dispersion_penalty = 0.10 * max(alt_iqr, tail_iqr)
                    trade_min = min(alt_trades, tail_trades)
                elif ranking_selector_mode == "base_first":
                    robust_score = base_score
                    robust_return = base_return
                    robust_pf = base_pf
                    robust_pos_frac = base_pos_frac
                    robust_pf_ge_1_frac = base_pf_ge_1_frac
                    dispersion_penalty = 0.10 * max(0.0, base_iqr)
                    trade_min = base_trades
                else:
                    robust_score = min(base_score, alt_score, tail_score)
                    robust_return = min(base_return, alt_return, tail_return)
                    robust_pf = min(base_pf, alt_pf, tail_pf)
                    robust_pos_frac = min(base_pos_frac, alt_pos_frac, tail_pos_frac)
                    robust_pf_ge_1_frac = min(base_pf_ge_1_frac, alt_pf_ge_1_frac, tail_pf_ge_1_frac)
                    dispersion_penalty = 0.10 * max(base_iqr, alt_iqr, tail_iqr)
                    trade_min = min(base_trades, alt_trades, tail_trades)
                low_trade_penalty = 0.0
                if trade_floor > 0:
                    if trade_min < trade_floor:
                        low_trade_penalty = 0.02 * ((trade_floor - trade_min) / max(1.0, trade_floor))

                # Profitability-first ranking.
                pf_bonus = 0.30 * max(0.0, robust_pf - 1.0)
                spr_bonus = 0.15 * max(0.0, robust_score)
                pf_shortfall_penalty = 0.50 * max(0.0, 1.0 - robust_pf)
                consistency_bonus = 0.25 * max(0.0, robust_pos_frac - min_positive_frac)
                consistency_penalty = 1.25 * max(0.0, min_positive_frac - robust_pos_frac)
                tail_penalty = 0.0
                if ranking_selector_mode != "base_first":
                    tail_penalty = tail_weight * max(0.0, -tail_return)
                base_return_penalty = 0.0
                if ranking_selector_mode == "future_first":
                    base_return_penalty = base_penalty_weight * max(0.0, base_return_floor - base_return)
                negative_return_penalty = 0.0
                if robust_return <= 0.0:
                    negative_return_penalty = 1.0 + 0.25 * abs(robust_return)

                composite = (
                    robust_return
                    + pf_bonus
                    + spr_bonus
                    + consistency_bonus
                    - dispersion_penalty
                    - low_trade_penalty
                    - pf_shortfall_penalty
                    - consistency_penalty
                    - tail_penalty
                    - base_return_penalty
                    - negative_return_penalty
                )
                if ranking_selector_mode == "future_first":
                    profit_feasible = bool(
                        alt_return > 0.0
                        and tail_return > 0.0
                        and alt_pf >= 1.0
                        and tail_pf >= 1.0
                    )
                    consistency_feasible = bool(
                        profit_feasible
                        and alt_pos_frac >= min_positive_frac
                        and tail_pos_frac >= min_positive_frac
                        and alt_pf_ge_1_frac >= min_positive_frac
                        and tail_pf_ge_1_frac >= min_positive_frac
                    )
                elif ranking_selector_mode == "base_first":
                    profit_feasible = bool(
                        base_return > 0.0
                        and base_pf >= 1.0
                    )
                    consistency_feasible = bool(
                        profit_feasible
                        and base_pos_frac >= min_positive_frac
                        and base_pf_ge_1_frac >= min_positive_frac
                    )
                else:
                    profit_feasible = bool(
                        base_return > 0.0
                        and alt_return > 0.0
                        and tail_return > 0.0
                        and base_pf >= 1.0
                        and alt_pf >= 1.0
                        and tail_pf >= 1.0
                    )
                    consistency_feasible = bool(
                        profit_feasible
                        and base_pos_frac >= min_positive_frac
                        and alt_pos_frac >= min_positive_frac
                        and tail_pos_frac >= min_positive_frac
                        and base_pf_ge_1_frac >= min_positive_frac
                        and alt_pf_ge_1_frac >= min_positive_frac
                        and tail_pf_ge_1_frac >= min_positive_frac
                    )

                # Always compute a future-first view for diagnostics/auto-rescue.
                future_robust_score = min(alt_score, tail_score)
                future_robust_return = min(alt_return, tail_return)
                future_robust_pf = min(alt_pf, tail_pf)
                future_robust_pos_frac = min(alt_pos_frac, tail_pos_frac)
                future_robust_pf_ge_1_frac = min(alt_pf_ge_1_frac, tail_pf_ge_1_frac)
                future_dispersion_penalty = 0.10 * max(alt_iqr, tail_iqr)
                future_trade_min = min(alt_trades, tail_trades)
                future_low_trade_penalty = 0.0
                if trade_floor > 0 and future_trade_min < trade_floor:
                    future_low_trade_penalty = 0.02 * ((trade_floor - future_trade_min) / max(1.0, trade_floor))
                future_pf_bonus = 0.30 * max(0.0, future_robust_pf - 1.0)
                future_spr_bonus = 0.15 * max(0.0, future_robust_score)
                future_pf_shortfall_penalty = 0.50 * max(0.0, 1.0 - future_robust_pf)
                future_consistency_bonus = 0.25 * max(0.0, future_robust_pos_frac - min_positive_frac)
                future_consistency_penalty = 1.25 * max(0.0, min_positive_frac - future_robust_pos_frac)
                future_tail_penalty = tail_weight * max(0.0, -tail_return)
                future_base_return_penalty = base_penalty_weight * max(0.0, base_return_floor - base_return)
                future_negative_return_penalty = 0.0
                if future_robust_return <= 0.0:
                    future_negative_return_penalty = 1.0 + 0.25 * abs(future_robust_return)
                future_composite = (
                    future_robust_return
                    + future_pf_bonus
                    + future_spr_bonus
                    + future_consistency_bonus
                    - future_dispersion_penalty
                    - future_low_trade_penalty
                    - future_pf_shortfall_penalty
                    - future_consistency_penalty
                    - future_tail_penalty
                    - future_base_return_penalty
                    - future_negative_return_penalty
                )
                future_profit_feasible = bool(
                    alt_return > 0.0
                    and tail_return > 0.0
                    and alt_pf >= 1.0
                    and tail_pf >= 1.0
                )
                future_consistency_feasible = bool(
                    future_profit_feasible
                    and alt_pos_frac >= min_positive_frac
                    and tail_pos_frac >= min_positive_frac
                    and alt_pf_ge_1_frac >= min_positive_frac
                    and tail_pf_ge_1_frac >= min_positive_frac
                )

                tournament.append(
                    {
                        "episode": int(item.get("episode", -1)),
                        "filename": filename,
                        "base_score": base_score,
                        "alt_score": alt_score,
                        "tail_score": tail_score,
                        "robust_score": robust_score,
                        "base_return_pct": base_return,
                        "alt_return_pct": alt_return,
                        "tail_return_pct": tail_return,
                        "base_return_median_pct": base_return_median,
                        "alt_return_median_pct": alt_return_median,
                        "tail_return_median_pct": tail_return_median,
                        "robust_return_pct": robust_return,
                        "base_pf": base_pf,
                        "alt_pf": alt_pf,
                        "tail_pf": tail_pf,
                        "base_pf_median": base_pf_median,
                        "alt_pf_median": alt_pf_median,
                        "tail_pf_median": tail_pf_median,
                        "robust_pf": robust_pf,
                        "base_pos_frac": base_pos_frac,
                        "alt_pos_frac": alt_pos_frac,
                        "tail_pos_frac": tail_pos_frac,
                        "robust_pos_frac": robust_pos_frac,
                        "base_pf_ge_1_frac": base_pf_ge_1_frac,
                        "alt_pf_ge_1_frac": alt_pf_ge_1_frac,
                        "tail_pf_ge_1_frac": tail_pf_ge_1_frac,
                        "robust_pf_ge_1_frac": robust_pf_ge_1_frac,
                        "base_iqr": base_iqr,
                        "alt_iqr": alt_iqr,
                        "tail_iqr": tail_iqr,
                        "base_trades": base_trades,
                        "alt_trades": alt_trades,
                        "tail_trades": tail_trades,
                        "dispersion_penalty": dispersion_penalty,
                        "low_trade_penalty": low_trade_penalty,
                        "pf_shortfall_penalty": pf_shortfall_penalty,
                        "consistency_bonus": consistency_bonus,
                        "consistency_penalty": consistency_penalty,
                        "tail_penalty": tail_penalty,
                        "base_return_penalty": base_return_penalty,
                        "negative_return_penalty": negative_return_penalty,
                        "profit_feasible": profit_feasible,
                        "consistency_feasible": consistency_feasible,
                        "composite_score": composite,
                        "forward_return_min": future_robust_return,
                        "forward_pf_min": future_robust_pf,
                        "future_profit_feasible": future_profit_feasible,
                        "future_consistency_feasible": future_consistency_feasible,
                        "future_composite_score": future_composite,
                    }
                )
        finally:
            np.random.set_state(rng_state)
            random.setstate(py_rng_state)
            setattr(self.config, "VAL_MIN_K", orig_val_min_k)
            setattr(self.config, "VAL_MAX_K", orig_val_max_k)
            setattr(self.config, "VAL_JITTER_DRAWS", orig_val_jitter_draws)

        if not tournament:
            return None

        pool_mode = "consistency_feasible"
        ranking_pool = [x for x in tournament if x.get("consistency_feasible")]
        if not ranking_pool:
            pool_mode = "profit_feasible"
            ranking_pool = [x for x in tournament if x.get("profit_feasible")]
        if not ranking_pool:
            pool_mode = "all"
            ranking_pool = list(tournament)

        ranking_pool.sort(
            key=lambda x: (
                x["composite_score"],
                x.get("robust_pos_frac", -1e9),
                x.get("robust_return_pct", -1e9),
                x.get("robust_pf", -1e9),
                x["robust_score"],
            ),
            reverse=True,
        )
        winner = ranking_pool[0]
        selected_mode = ranking_selector_mode
        selection_pool_profit_only = bool(any(x.get("profit_feasible") for x in tournament))
        auto_rescue_summary = None

        if selector_mode == "auto_rescue":
            future_pool_mode = "future_consistency_feasible"
            future_pool = [x for x in tournament if x.get("future_consistency_feasible")]
            if not future_pool:
                future_pool_mode = "future_profit_feasible"
                future_pool = [x for x in tournament if x.get("future_profit_feasible")]
            if not future_pool:
                future_pool_mode = "all"
                future_pool = list(tournament)

            future_pool.sort(
                key=lambda x: (
                    x.get("future_composite_score", -1e9),
                    x.get("forward_return_min", -1e9),
                    x.get("forward_pf_min", -1e9),
                    x.get("alt_score", -1e9),
                    x.get("tail_score", -1e9),
                ),
                reverse=True,
            )
            future_winner = future_pool[0]

            winner_forward_return = float(winner.get("forward_return_min", 0.0))
            winner_forward_pf = float(winner.get("forward_pf_min", 0.0))
            future_forward_return = float(future_winner.get("forward_return_min", 0.0))
            future_forward_pf = float(future_winner.get("forward_pf_min", 0.0))
            challenger_base_return = float(future_winner.get("base_return_pct", 0.0))
            forward_return_edge = future_forward_return - winner_forward_return
            forward_pf_edge = future_forward_pf - winner_forward_pf

            rescue_triggered = bool(
                auto_rescue_enabled
                and future_winner.get("filename") != winner.get("filename")
                and future_winner.get("future_consistency_feasible", False)
                and winner_forward_return <= auto_rescue_winner_forward_return_max
                and forward_return_edge >= auto_rescue_forward_return_edge_min
                and forward_pf_edge >= auto_rescue_forward_pf_edge_min
                and challenger_base_return <= auto_rescue_challenger_base_return_max
                and future_forward_pf >= auto_rescue_challenger_forward_pf_min
            )

            auto_rescue_summary = {
                "enabled": bool(auto_rescue_enabled),
                "triggered": bool(rescue_triggered),
                "thresholds": {
                    "winner_forward_return_max": float(auto_rescue_winner_forward_return_max),
                    "forward_return_edge_min": float(auto_rescue_forward_return_edge_min),
                    "forward_pf_edge_min": float(auto_rescue_forward_pf_edge_min),
                    "challenger_base_return_max": float(auto_rescue_challenger_base_return_max),
                    "challenger_forward_pf_min": float(auto_rescue_challenger_forward_pf_min),
                },
                "tail_winner_filename": winner.get("filename"),
                "future_winner_filename": future_winner.get("filename"),
                "winner_forward_return": float(winner_forward_return),
                "winner_forward_pf": float(winner_forward_pf),
                "future_forward_return": float(future_forward_return),
                "future_forward_pf": float(future_forward_pf),
                "forward_return_edge": float(forward_return_edge),
                "forward_pf_edge": float(forward_pf_edge),
                "challenger_base_return": float(challenger_base_return),
                "future_pool_mode": future_pool_mode,
            }

            if rescue_triggered:
                winner = future_winner
                ranking_pool = future_pool
                pool_mode = f"auto_rescue:{future_pool_mode}"
                selected_mode = "future_first"
                selection_pool_profit_only = bool(any(x.get("future_profit_feasible") for x in tournament))
            else:
                selected_mode = "tail_holdout"

        tiebreak_summary = {"enabled": bool(tiebreak_enabled), "evaluated": False}
        checkpoint_fingerprint_cache = {}

        def _checkpoint_fingerprint(filename: Optional[str]) -> Optional[str]:
            """Return a stable content hash so aliases (e.g. best_model.pt) dedupe cleanly."""
            if not filename:
                return None
            cached = checkpoint_fingerprint_cache.get(filename)
            if cached is not None:
                return cached
            ckpt_path = self.checkpoint_dir / filename
            if not ckpt_path.exists():
                checkpoint_fingerprint_cache[filename] = None
                return None
            try:
                hasher = hashlib.sha1()
                with open(ckpt_path, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        hasher.update(chunk)
                digest = hasher.hexdigest()
            except OSError:
                digest = None
            checkpoint_fingerprint_cache[filename] = digest
            return digest

        dedupe_metric_fields = (
            "base_return_pct",
            "alt_return_pct",
            "tail_return_pct",
            "base_pf",
            "alt_pf",
            "tail_pf",
            "base_trades",
            "alt_trades",
            "tail_trades",
            "future_composite_score",
            "composite_score",
        )

        def _candidate_identity_key(candidate: Dict[str, object]) -> Optional[tuple]:
            name = candidate.get("filename")
            if not name:
                return None
            metric_values = []
            has_metric = False
            for field in dedupe_metric_fields:
                value = candidate.get(field)
                if isinstance(value, (int, float)):
                    has_metric = True
                    metric_values.append(round(float(value), 10))
                else:
                    metric_values.append(None)
            if has_metric:
                return ("metrics", tuple(metric_values))
            fp = _checkpoint_fingerprint(name)
            if fp is not None:
                return ("fp", fp)
            return ("name", name)

        distinct_pool_filenames = []
        distinct_keys = set()
        for item in ranking_pool:
            name = item.get("filename")
            key = _candidate_identity_key(item)
            if not name or key is None:
                continue
            if key in distinct_keys:
                continue
            distinct_keys.add(key)
            distinct_pool_filenames.append(name)

        tiebreak_summary["distinct_pool_size"] = len(distinct_pool_filenames)
        tiebreak_summary["distinct_pool_filenames"] = distinct_pool_filenames

        if tiebreak_enabled and len(ranking_pool) >= 2 and len(distinct_pool_filenames) >= 2:
            incumbent_name = winner.get("filename")
            incumbent_key = _candidate_identity_key(winner)
            runner_up = next(
                (
                    x
                    for x in ranking_pool
                    if x.get("filename")
                    and x.get("filename") != incumbent_name
                    and _candidate_identity_key(x) != incumbent_key
                ),
                None,
            )
            if runner_up is not None:
                challenger_name = runner_up.get("filename")
                tiebreak_summary["evaluated"] = True

                probe_results = {}
                tb_rng_state = np.random.get_state()
                tb_py_rng_state = random.getstate()
                tb_orig_val_min_k = getattr(self.config, "VAL_MIN_K", None)
                tb_orig_val_max_k = getattr(self.config, "VAL_MAX_K", None)
                tb_orig_val_jitter_draws = getattr(self.config, "VAL_JITTER_DRAWS", None)
                try:
                    # Force a single-slice deterministic probe for a cheap top-2 tie-break.
                    setattr(self.config, "VAL_MIN_K", 1)
                    setattr(self.config, "VAL_MAX_K", 1)
                    setattr(self.config, "VAL_JITTER_DRAWS", 1)

                    for idx, candidate in enumerate((winner, runner_up)):
                        filename = candidate.get("filename")
                        if not filename:
                            continue
                        ckpt_path = self.checkpoint_dir / filename
                        if not ckpt_path.exists():
                            continue
                        try:
                            self.load_checkpoint(filename)
                        except Exception:
                            continue

                        seed_offset = 311 + 37 * idx
                        random.seed(base_seed + seed_offset)
                        np.random.seed(base_seed + seed_offset)
                        probe_stats = self.validate(
                            window_bars_override=tiebreak_window_bars,
                            start_frac_override=tiebreak_start_frac,
                            end_frac_override=tiebreak_end_frac,
                            persist_summary=False,
                            quiet=True,
                        )
                        probe_results[filename] = {
                            "return_pct": float(
                                probe_stats.get(
                                    "val_median_return_pct",
                                    probe_stats.get("val_return_pct", 0.0),
                                )
                            ),
                            "pf": float(probe_stats.get("val_median_pf", 0.0)),
                            "trades": float(probe_stats.get("val_trades", 0.0)),
                            "fitness": float(probe_stats.get("val_fitness", 0.0)),
                        }
                finally:
                    np.random.set_state(tb_rng_state)
                    random.setstate(tb_py_rng_state)
                    setattr(self.config, "VAL_MIN_K", tb_orig_val_min_k)
                    setattr(self.config, "VAL_MAX_K", tb_orig_val_max_k)
                    setattr(self.config, "VAL_JITTER_DRAWS", tb_orig_val_jitter_draws)

                incumbent_probe = probe_results.get(incumbent_name)
                challenger_probe = probe_results.get(challenger_name)
                return_edge = 0.0
                pf_edge = 0.0
                switched = False
                if incumbent_probe is not None and challenger_probe is not None:
                    return_edge = float(challenger_probe["return_pct"] - incumbent_probe["return_pct"])
                    pf_edge = float(challenger_probe["pf"] - incumbent_probe["pf"])
                    switched = bool(
                        return_edge >= tiebreak_return_edge_min
                        and pf_edge >= tiebreak_pf_edge_min
                        and challenger_probe["trades"] >= tiebreak_min_trades
                        and challenger_probe["return_pct"] > 0.0
                        and challenger_probe["pf"] >= 1.0
                    )
                    if switched:
                        winner = runner_up
                        selected_mode = f"{selected_mode}+tiebreak"

                tiebreak_summary.update(
                    {
                        "window_bars": int(tiebreak_window_bars),
                        "start_frac": float(tiebreak_start_frac),
                        "end_frac": float(tiebreak_end_frac),
                        "thresholds": {
                            "return_edge_min": float(tiebreak_return_edge_min),
                            "pf_edge_min": float(tiebreak_pf_edge_min),
                            "min_trades": float(tiebreak_min_trades),
                        },
                        "incumbent_filename": incumbent_name,
                        "challenger_filename": challenger_name,
                        "incumbent_probe": incumbent_probe,
                        "challenger_probe": challenger_probe,
                        "return_edge": float(return_edge),
                        "pf_edge": float(pf_edge),
                        "switched": bool(switched),
                        "winner_after_tiebreak": winner.get("filename"),
                    }
                )

        horizon_summary = {"enabled": bool(horizon_rescue_enabled), "evaluated": False}
        if horizon_rescue_enabled and len(tournament) >= 2:
            candidate_by_name = {}
            for item in tournament:
                name = item.get("filename")
                if name:
                    candidate_by_name[name] = item

            probe_candidates = sorted(
                candidate_by_name.values(),
                key=lambda x: (
                    x.get("composite_score", -1e9),
                    x.get("robust_return_pct", -1e9),
                    x.get("robust_pf", -1e9),
                    x.get("episode", -1),
                ),
                reverse=True,
            )[:horizon_candidate_limit]

            incumbent_name = winner.get("filename")
            incumbent_candidate = candidate_by_name.get(incumbent_name)
            if incumbent_candidate is not None and all(
                x.get("filename") != incumbent_name for x in probe_candidates
            ):
                probe_candidates.append(incumbent_candidate)

            if candidate_by_name:
                latest_candidate = max(
                    candidate_by_name.values(), key=lambda x: x.get("episode", -1)
                )
                latest_name = latest_candidate.get("filename")
                if latest_name and all(
                    x.get("filename") != latest_name for x in probe_candidates
                ):
                    probe_candidates.append(latest_candidate)

            distinct_probe_candidates = []
            seen_probe_keys = set()
            for item in probe_candidates:
                key = _candidate_identity_key(item)
                if key is None or key in seen_probe_keys:
                    continue
                seen_probe_keys.add(key)
                distinct_probe_candidates.append(item)

            horizon_summary["probe_pool_size"] = len(distinct_probe_candidates)
            horizon_summary["probe_pool_filenames"] = [
                x.get("filename") for x in distinct_probe_candidates
            ]

            if len(distinct_probe_candidates) >= 2 and incumbent_name:
                horizon_summary["evaluated"] = True
                probe_results = {}
                hr_rng_state = np.random.get_state()
                hr_py_rng_state = random.getstate()
                try:
                    # Reuse active validation robustness settings so horizon rescue
                    # is not decided by a single slice artifact.

                    for idx, candidate in enumerate(distinct_probe_candidates):
                        filename = candidate.get("filename")
                        if not filename:
                            continue
                        ckpt_path = self.checkpoint_dir / filename
                        if not ckpt_path.exists():
                            continue
                        try:
                            self.load_checkpoint(filename)
                        except Exception:
                            continue

                        seed_offset = 611 + 53 * idx
                        random.seed(base_seed + seed_offset)
                        np.random.seed(base_seed + seed_offset)
                        probe_stats = self.validate(
                            window_bars_override=horizon_window_bars,
                            start_frac_override=horizon_start_frac,
                            end_frac_override=horizon_end_frac,
                            persist_summary=False,
                            quiet=True,
                        )
                        probe_results[filename] = {
                            "return_pct": float(
                                probe_stats.get(
                                    "val_median_return_pct",
                                    probe_stats.get("val_return_pct", 0.0),
                                )
                            ),
                            "pf": float(probe_stats.get("val_median_pf", 0.0)),
                            "trades": float(probe_stats.get("val_trades", 0.0)),
                            "fitness": float(probe_stats.get("val_fitness", 0.0)),
                        }
                finally:
                    np.random.set_state(hr_rng_state)
                    random.setstate(hr_py_rng_state)

                ranked_probe_names = sorted(
                    probe_results.keys(),
                    key=lambda name: (
                        probe_results[name]["return_pct"],
                        probe_results[name]["pf"],
                        probe_results[name]["fitness"],
                    ),
                    reverse=True,
                )
                fallback_challenger_name = next(
                    (name for name in ranked_probe_names if name != incumbent_name),
                    None,
                )
                challenger_name = None
                for name in ranked_probe_names:
                    if name == incumbent_name:
                        continue
                    probe = probe_results.get(name)
                    candidate = candidate_by_name.get(name)
                    if probe is None or candidate is None:
                        continue
                    candidate_base_return = float(candidate.get("base_return_pct", 0.0))
                    candidate_robust_return = float(candidate.get("robust_return_pct", 0.0))
                    if (
                        probe["return_pct"] > 0.0
                        and probe["pf"] >= horizon_challenger_pf_min
                        and probe["trades"] >= horizon_min_trades
                        and candidate_base_return <= horizon_challenger_base_return_max
                        and candidate_robust_return >= horizon_challenger_robust_return_min
                    ):
                        challenger_name = name
                        break
                if challenger_name is None:
                    challenger_name = fallback_challenger_name
                incumbent_probe = probe_results.get(incumbent_name)
                challenger_probe = probe_results.get(challenger_name) if challenger_name else None
                challenger_candidate = candidate_by_name.get(challenger_name) if challenger_name else None

                incumbent_robust_return = float(winner.get("robust_return_pct", 0.0))
                incumbent_return = float(incumbent_probe["return_pct"]) if incumbent_probe else 0.0
                incumbent_pf = float(incumbent_probe["pf"]) if incumbent_probe else 0.0
                return_edge = 0.0
                pf_edge = 0.0
                challenger_base_return = 0.0
                challenger_robust_return = 0.0
                switched = False
                if incumbent_probe is not None and challenger_probe is not None and challenger_candidate is not None:
                    challenger_base_return = float(
                        challenger_candidate.get("base_return_pct", 0.0)
                    )
                    challenger_robust_return = float(
                        challenger_candidate.get("robust_return_pct", 0.0)
                    )
                    return_edge = float(challenger_probe["return_pct"] - incumbent_probe["return_pct"])
                    pf_edge = float(challenger_probe["pf"] - incumbent_probe["pf"])
                    switched = bool(
                        incumbent_robust_return <= horizon_incumbent_return_max
                        and return_edge >= horizon_return_edge_min
                        and pf_edge >= horizon_pf_edge_min
                        and challenger_probe["return_pct"] > 0.0
                        and challenger_probe["pf"] >= horizon_challenger_pf_min
                        and challenger_probe["trades"] >= horizon_min_trades
                        and challenger_base_return <= horizon_challenger_base_return_max
                        and challenger_robust_return >= horizon_challenger_robust_return_min
                    )
                    if switched:
                        winner = challenger_candidate
                        selected_mode = f"{selected_mode}+horizon"
                        if all(
                            x.get("filename") != challenger_name for x in ranking_pool
                        ):
                            ranking_pool = list(ranking_pool) + [winner]
                            pool_mode = f"{pool_mode}+horizon"

                horizon_summary.update(
                    {
                        "window_bars": int(horizon_window_bars),
                        "start_frac": float(horizon_start_frac),
                        "end_frac": float(horizon_end_frac),
                        "thresholds": {
                            "incumbent_return_max": float(horizon_incumbent_return_max),
                            "return_edge_min": float(horizon_return_edge_min),
                            "pf_edge_min": float(horizon_pf_edge_min),
                            "challenger_base_return_max": float(
                                horizon_challenger_base_return_max
                            ),
                            "challenger_robust_return_min": float(
                                horizon_challenger_robust_return_min
                            ),
                            "challenger_pf_min": float(horizon_challenger_pf_min),
                            "min_trades": float(horizon_min_trades),
                        },
                        "incumbent_filename": incumbent_name,
                        "challenger_filename": challenger_name,
                        "incumbent_probe": incumbent_probe,
                        "challenger_probe": challenger_probe,
                        "incumbent_robust_return": float(incumbent_robust_return),
                        "incumbent_return": float(incumbent_return),
                        "incumbent_pf": float(incumbent_pf),
                        "challenger_base_return": float(challenger_base_return),
                        "challenger_robust_return": float(challenger_robust_return),
                        "return_edge": float(return_edge),
                        "pf_edge": float(pf_edge),
                        "switched": bool(switched),
                        "winner_after_horizon": winner.get("filename"),
                        "probe_results": probe_results,
                    }
                )

        alignment_summary = {"enabled": bool(alignment_probe_enabled), "evaluated": False}
        if alignment_probe_enabled and len(tournament) >= 2:
            tournament_by_name = {
                item.get("filename"): item for item in tournament if item.get("filename")
            }
            incumbent_name = winner.get("filename")
            probe_candidates = []
            seen_probe_keys = set()
            # Prefer feasible ranking-pool candidates, but backfill from the full
            # tournament so alignment probe can still run when the feasible pool
            # collapses to a single candidate.
            tournament_ranked = sorted(
                tournament,
                key=lambda x: (
                    x.get("composite_score", -1e9),
                    x.get("robust_return_pct", -1e9),
                    x.get("robust_pf", -1e9),
                    x.get("episode", -1),
                ),
                reverse=True,
            )
            probe_source = list(ranking_pool) + tournament_ranked
            for item in probe_source:
                key = _candidate_identity_key(item)
                if key is None or key in seen_probe_keys:
                    continue
                seen_probe_keys.add(key)
                probe_candidates.append(item)
                if len(probe_candidates) >= alignment_probe_top_k:
                    break

            if (
                incumbent_name
                and all(x.get("filename") != incumbent_name for x in probe_candidates)
            ):
                incumbent_item = tournament_by_name.get(incumbent_name)
                if incumbent_item is not None:
                    probe_candidates.append(incumbent_item)

            probe_candidates = [x for x in probe_candidates if x.get("filename")]
            alignment_summary["ranking_pool_size"] = len(ranking_pool)
            alignment_summary["tournament_size"] = len(tournament)
            alignment_summary["probe_pool_size"] = len(probe_candidates)
            alignment_summary["probe_pool_filenames"] = [
                x.get("filename") for x in probe_candidates
            ]

            if len(probe_candidates) >= 2 and incumbent_name:
                alignment_summary["evaluated"] = True
                probe_results = {}
                al_rng_state = np.random.get_state()
                al_py_rng_state = random.getstate()
                al_orig_val_jitter_draws = getattr(self.config, "VAL_JITTER_DRAWS", None)
                try:
                    # Keep probes deterministic and cheap; test evaluation has no jitter averaging.
                    setattr(self.config, "VAL_JITTER_DRAWS", 1)

                    for idx, candidate in enumerate(probe_candidates):
                        filename = candidate.get("filename")
                        if not filename:
                            continue
                        ckpt_path = self.checkpoint_dir / filename
                        if not ckpt_path.exists():
                            continue
                        try:
                            self.load_checkpoint(filename)
                        except Exception:
                            continue

                        seed_offset = 877 + 29 * idx
                        random.seed(base_seed + seed_offset)
                        np.random.seed(base_seed + seed_offset)
                        probe_stats = self.validate(
                            stride_frac_override=alignment_probe_stride_frac,
                            window_bars_override=alignment_probe_window_bars,
                            use_all_windows_override=alignment_probe_use_all_windows,
                            persist_summary=False,
                            quiet=True,
                        )
                        probe_windows = int(probe_stats.get("val_k", 0))
                        probe_spr = float(
                            probe_stats.get(
                                "val_median_fitness",
                                probe_stats.get("val_fitness", 0.0),
                            )
                        )
                        probe_pf = float(probe_stats.get("val_median_pf", 0.0))
                        probe_pos_frac = float(probe_stats.get("val_positive_frac", 0.0))
                        probe_return = float(
                            probe_stats.get(
                                "val_median_return_pct",
                                probe_stats.get("val_return_pct", 0.0),
                            )
                        )
                        probe_trades = float(probe_stats.get("val_trades", 0.0))
                        probe_pass = bool(
                            probe_windows >= int(self.config.fitness.test_walkforward_min_windows)
                            and probe_spr >= float(self.config.fitness.test_walkforward_min_spr)
                            and probe_pf >= float(self.config.fitness.test_walkforward_min_pf)
                            and probe_pos_frac >= float(self.config.fitness.test_walkforward_min_pos_frac)
                        )
                        probe_results[filename] = {
                            "windows": int(probe_windows),
                            "spr": float(probe_spr),
                            "return_pct": float(probe_return),
                            "pf": float(probe_pf),
                            "positive_frac": float(probe_pos_frac),
                            "trades": float(probe_trades),
                            "pass": bool(probe_pass),
                        }
                finally:
                    np.random.set_state(al_rng_state)
                    random.setstate(al_py_rng_state)
                    setattr(self.config, "VAL_JITTER_DRAWS", al_orig_val_jitter_draws)

                ranked_probe_names = sorted(
                    probe_results.keys(),
                    key=lambda name: (
                        int(bool(probe_results[name]["pass"])),
                        probe_results[name]["return_pct"],
                        probe_results[name]["pf"],
                        probe_results[name]["positive_frac"],
                        probe_results[name]["spr"],
                        probe_results[name]["trades"],
                    ),
                    reverse=True,
                )
                challenger_name = next(
                    (name for name in ranked_probe_names if name != incumbent_name),
                    None,
                )
                incumbent_probe = probe_results.get(incumbent_name)
                challenger_probe = probe_results.get(challenger_name) if challenger_name else None
                return_edge = 0.0
                pf_edge = 0.0
                switched = False
                if incumbent_probe is not None and challenger_probe is not None:
                    return_edge = float(challenger_probe["return_pct"] - incumbent_probe["return_pct"])
                    pf_edge = float(challenger_probe["pf"] - incumbent_probe["pf"])
                    challenger_passes = bool(challenger_probe.get("pass", False))
                    if not alignment_probe_require_pass:
                        challenger_passes = True

                    switched = bool(
                        challenger_probe["trades"] >= alignment_probe_min_trades
                        and challenger_passes
                        and (
                            (
                                challenger_probe.get("pass", False)
                                and not incumbent_probe.get("pass", False)
                                and challenger_probe["return_pct"] > 0.0
                                and challenger_probe["pf"] >= 1.0
                            )
                            or (
                                return_edge >= alignment_probe_return_edge_min
                                and pf_edge >= alignment_probe_pf_edge_min
                                and challenger_probe["return_pct"] > 0.0
                                and challenger_probe["pf"] >= 1.0
                            )
                        )
                    )

                    if switched and challenger_name in tournament_by_name:
                        winner = tournament_by_name[challenger_name]
                        selected_mode = f"{selected_mode}+wfalign"
                        if all(
                            x.get("filename") != challenger_name for x in ranking_pool
                        ):
                            ranking_pool = list(ranking_pool) + [winner]
                            pool_mode = f"{pool_mode}+wfalign"

                alignment_summary.update(
                    {
                        "window_bars": int(alignment_probe_window_bars) if alignment_probe_window_bars is not None else None,
                        "stride_frac": float(alignment_probe_stride_frac) if alignment_probe_stride_frac is not None else None,
                        "use_all_windows": bool(alignment_probe_use_all_windows),
                        "thresholds": {
                            "return_edge_min": float(alignment_probe_return_edge_min),
                            "pf_edge_min": float(alignment_probe_pf_edge_min),
                            "min_trades": float(alignment_probe_min_trades),
                            "require_pass": bool(alignment_probe_require_pass),
                            "walkforward_min_windows": int(self.config.fitness.test_walkforward_min_windows),
                            "walkforward_min_spr": float(self.config.fitness.test_walkforward_min_spr),
                            "walkforward_min_pf": float(self.config.fitness.test_walkforward_min_pf),
                            "walkforward_min_positive_frac": float(self.config.fitness.test_walkforward_min_pos_frac),
                        },
                        "incumbent_filename": incumbent_name,
                        "challenger_filename": challenger_name,
                        "incumbent_probe": incumbent_probe,
                        "challenger_probe": challenger_probe,
                        "return_edge": float(return_edge),
                        "pf_edge": float(pf_edge),
                        "switched": bool(switched),
                        "winner_after_alignment": winner.get("filename"),
                        "probe_results": probe_results,
                    }
                )

        chosen = winner["filename"]

        summary_payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "alt_stride_frac": alt_stride,
            "alt_window_bars": alt_window,
            "tail_start_frac": tail_start_frac,
            "tail_end_frac": tail_end_frac,
            "tail_weight": tail_weight,
            "selector_mode": selected_mode,
            "selector_mode_requested": selector_mode,
            "base_return_floor": base_return_floor,
            "base_penalty_weight": base_penalty_weight,
            "trade_floor": trade_floor,
            "winner": winner,
            "selection_pool_size": len(ranking_pool),
            "selection_pool_mode": pool_mode,
            "selection_pool_profit_only": selection_pool_profit_only,
            "selection_pool_filenames": [x.get("filename") for x in ranking_pool],
            "candidates": tournament,
        }
        if auto_rescue_summary is not None:
            summary_payload["auto_rescue"] = auto_rescue_summary
        if tiebreak_summary is not None:
            summary_payload["tiebreak"] = tiebreak_summary
        if horizon_summary is not None:
            summary_payload["horizon_rescue"] = horizon_summary
        if alignment_summary is not None:
            summary_payload["alignment_probe"] = alignment_summary
        tournament_path = self.log_dir / "checkpoint_tournament.json"
        try:
            with open(tournament_path, "w", encoding="utf-8") as f:
                json.dump(summary_payload, f, indent=2)
        except OSError:
            pass

        if verbose:
            use_future_metrics = str(selected_mode).startswith("future_first")
            display_composite = float(
                winner.get("future_composite_score", winner.get("composite_score", 0.0))
                if use_future_metrics
                else winner.get("composite_score", 0.0)
            )
            display_return = float(
                winner.get("forward_return_min", winner.get("robust_return_pct", 0.0))
                if use_future_metrics
                else winner.get("robust_return_pct", 0.0)
            )
            display_pf = float(
                winner.get("forward_pf_min", winner.get("robust_pf", 0.0))
                if use_future_metrics
                else winner.get("robust_pf", 0.0)
            )
            print(
                f"[ANTI-REG] mode={selected_mode} (requested={selector_mode}) | "
                f"winner={chosen} | composite={display_composite:.4f} | "
                f"ret={display_return:.2f}% | pf={display_pf:.2f} | "
                f"spr_base={winner['base_score']:.4f} | spr_alt={winner['alt_score']:.4f}"
            )

        return chosen
    
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
            prefill_policy = getattr(self.config.training, "prefill_policy", "baseline")
            prefill_replay(self.train_env, self.agent, steps=prefill_steps, policy=prefill_policy)
        
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
                    self.val_env.slippage_pips = sp
                    if hasattr(self.val_env, "risk_manager"):
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

                if getattr(self.config.training, "anti_regression_checkpoint_selection", True):
                    self._register_checkpoint_candidate(episode, val_stats, metric_for_early_stop)
                
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
        
        # BEST-MODEL-RESTORE: Optionally run anti-regression checkpoint tournament,
        # then load selected best model before final save.
        selected_best = self._run_anti_regression_tournament(verbose=verbose)
        best_model_path = Path(self.checkpoint_dir) / "best_model.pt"
        if selected_best and selected_best != "best_model.pt":
            selected_path = Path(self.checkpoint_dir) / selected_best
            if selected_path.exists():
                try:
                    shutil.copy2(selected_path, best_model_path)
                    selected_scaler = selected_path.parent / f"{selected_path.stem}_scaler.json"
                    best_scaler = best_model_path.parent / "best_model_scaler.json"
                    if selected_scaler.exists():
                        shutil.copy2(selected_scaler, best_scaler)
                except OSError:
                    pass

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
            out_dir = self.log_dir / "validation_summaries"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Load the most recent validation summary to get episode number
            recent_files = sorted(out_dir.glob("val_ep*.json"))
            last_episode = 0
            if recent_files:
                try:
                    with open(recent_files[-1], "r", encoding="utf-8") as f:
                        last_file = json.load(f)
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

