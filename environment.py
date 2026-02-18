"""
Trading Environment Module
Gym-compatible environment for Forex trading with RL agent.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta
from risk_manager import RiskManager


# helpers
def pip_size(symbol: str) -> float:
    """Get pip size for any pair."""
    return 0.01 if symbol.endswith('JPY') else 0.0001


def _usd_conv_factor(ts, quote_ccy: str, fx_lookup: dict) -> float:
    """
    Return 1 unit of quote_ccy priced in USD at time ts.
    fx_lookup: dict like {"EURUSD": pd.Series, "USDJPY": pd.Series, ...} (close prices)
    """
    if quote_ccy == "USD":
        return 1.0
    # Prefer XUSD if it exists; else USDX (invert).
    pair1 = f"{quote_ccy}USD"   # e.g., GBPUSD
    pair2 = f"USD{quote_ccy}"   # e.g., USDJPY
    if pair1 in fx_lookup:
        px = float(fx_lookup[pair1].get(ts, np.nan))
        if np.isnan(px): 
            # Fallback to approximate conversion
            return _static_usd_conv(quote_ccy)
        return px  # 1 quote_ccy = px USD
    if pair2 in fx_lookup:
        px = float(fx_lookup[pair2].get(ts, np.nan))
        if np.isnan(px):
            return _static_usd_conv(quote_ccy)
        return 1.0 / px  # 1 quote_ccy = (1/px) USD
    # Fallback (rare): try routing via EUR ➔ USD
    if quote_ccy != "EUR" and "EURUSD" in fx_lookup and f"{quote_ccy}EUR" in fx_lookup:
        eur_rate = float(fx_lookup[quote_ccy+"EUR"].get(ts, np.nan))
        eurusd_rate = float(fx_lookup["EURUSD"].get(ts, np.nan))
        if not np.isnan(eur_rate) and not np.isnan(eurusd_rate):
            return eur_rate * eurusd_rate
    # Static fallback
    return _static_usd_conv(quote_ccy)


def _static_usd_conv(quote_ccy: str) -> float:
    """Static fallback conversion rates."""
    rates = {
        'JPY': 0.0067, 'EUR': 1.10, 'GBP': 1.27, 'CHF': 1.12,
        'CAD': 0.74, 'AUD': 0.65, 'NZD': 0.60
    }
    return rates.get(quote_ccy, 1.0)


def pip_value_usd_ts(symbol: str, price: float, lots: float, ts, fx_lookup: dict, contract_size: int = 100_000) -> float:
    """
    Time-varying pip value in USD using current timestamp and available FX series.
    
    Args:
        symbol: Currency pair (e.g., 'EURUSD', 'EURJPY', 'EURGBP')
        price: Current price
        lots: Position size in lots
        ts: Timestamp for FX lookup
        fx_lookup: Dict of {pair: pd.Series} with close prices
        contract_size: Contract size (default 100,000)
        
    Returns:
        USD value of 1 pip move
    """
    base, quote = symbol[:3], symbol[3:]
    ps = pip_size(symbol)
    # Pip value in quote ccy per 1 lot
    pip_in_quote = (ps / price) * contract_size * lots
    # Convert quote ➔ USD at timestamp
    q2usd = _usd_conv_factor(ts, quote, fx_lookup)
    return pip_in_quote * q2usd


# Legacy function for backward compatibility
def pip_value_usd(symbol: str, price: float, lots: float, contract_size: int = 100_000) -> float:
    """
    Calculate USD value of one pip for any forex pair (static approximation).
    Deprecated: Use pip_value_usd_ts for time-accurate conversion.
    """
    ps = pip_size(symbol)
    base, quote = symbol[:3], symbol[3:]
    pip_in_quote = (ps / price) * contract_size * lots
    return pip_in_quote * _static_usd_conv(quote)


class ForexTradingEnv:
    """
    Forex trading environment for reinforcement learning.
    
    State space includes:
    - Market features (OHLC, indicators, etc.)
    - Position information (type, size, entry, SL, TP)
    - Account information (balance, equity, margin)
    
    Action space:
    - 0: HOLD
    - 1: LONG (open-only)
    - 2: SHORT (open-only)
    - 3: CLOSE_POSITION
    - 4: MOVE_SL_CLOSER
    - 5: MOVE_SL_CLOSER_AGGRESSIVE
    - 6: REVERSE_TO_LONG
    - 7: REVERSE_TO_SHORT
    """
    
    def __init__(self,
                 data: pd.DataFrame,
                 feature_columns: List[str],
                 initial_balance: float = 1000.0,
                 risk_manager: RiskManager = None,
                 spread: float = 0.00020,  # 2 pips spread
                 commission: float = 0.0,  # Commission per lot per side
                 slippage_pips: float = 0.8,  # Slippage in pips
                 swap_type: str = "usd",  # "usd" or "points"
                 swap_long_usd_per_lot_night: float = 0.0,
                 swap_short_usd_per_lot_night: float = 0.0,
                 swap_rollover_hour_utc: int = 22,
                 swap_triple_weekday: int = 2,
                 weekend_close_hours: int = 3,  # Flatten positions N hours before weekend
                 max_steps: Optional[int] = None,
                 symbol: str = 'EURUSD',
                 fx_lookup: Optional[Dict[str, pd.Series]] = None,  # NEW: FX price series
                 scaler_mu: Optional[Dict[str, float]] = None,
                 scaler_sig: Optional[Dict[str, float]] = None,
                 cooldown_bars: int = 16,
                 min_hold_bars: int = 6,
                 trade_penalty: float = 0.0005,
                 flip_penalty: float = 0.002,
                 max_trades_per_episode: int = 120,
                 risk_per_trade: float = 0.0075,
                 atr_mult_sl: float = 2.5,
                 tp_mult: float = 2.0,
                 min_trail_buffer_pips: float = 1.0,
                 disable_move_sl: bool = False,
                 allowed_actions: Optional[List[int]] = None,
                 reward_clip: float = 0.01,
                 holding_cost: float = 1e-4,
                 r_multiple_reward_weight: float = 0.0,
                 r_multiple_reward_clip: float = 2.0,
                 min_atr_cost_ratio: float = 0.0,
                 use_regime_filter: bool = False,
                 regime_min_vol_z: float = 0.0,
                 regime_align_trend: bool = True,
                 regime_require_trending: bool = True,
                 random_episode_start: bool = False):
        """
        Initialize trading environment.
        
        Args:
            fx_lookup: Dict of {pair: pd.Series} with close prices for USD conversion
            scaler_mu: Feature means for normalization (from training data)
            scaler_sig: Feature stds for normalization (from training data)
            slippage_pips: Slippage in pips
            swap_type: "usd" (USD/lot/night) or "points" (broker swap points)
            swap_long_usd_per_lot_night: Overnight swap for LONG positions (USD per lot per night)
            swap_short_usd_per_lot_night: Overnight swap for SHORT positions (USD per lot per night)
            swap_rollover_hour_utc: UTC hour when daily swap rollover is charged
            swap_triple_weekday: Weekday index for triple swap (0=Mon..6=Sun; FX usually Wed=2)
            cooldown_bars: Bars to wait before allowing new position
            min_hold_bars: Minimum bars to hold a position
            trade_penalty: Penalty per trade (normalized)
            flip_penalty: Additional penalty for immediate reversals
            max_trades_per_episode: Maximum trades allowed per episode
            risk_per_trade: Risk per trade as fraction of equity (0.0075 = 0.75%)
            atr_mult_sl: ATR multiplier for stop loss (2.5)
            tp_mult: TP multiplier relative to SL distance (2.0)
            min_trail_buffer_pips: PATCH #4: Minimum pips required for meaningful SL tightening (1.0)
            allowed_actions: Optional list of action indices to allow (0-7); None allows all
            r_multiple_reward_weight: Reward weight for realized R-multiple shaping (0 disables)
            r_multiple_reward_clip: Clip for realized R-multiple shaping
            min_atr_cost_ratio: Gate new trades unless ATR >= ratio * (spread+slip+commission), 0 disables
            use_regime_filter: Gate trades to trending/high-vol regimes
            regime_min_vol_z: Minimum realized_vol_24h_z to allow trades (0 disables extra vol gate)
            regime_align_trend: Require trend_96h alignment for LONG/SHORT entries
            regime_require_trending: Require is_trending flag to allow trades
            random_episode_start: Sample random start bars on reset (train-only usage)
        """
        self.data = data.copy()
        self.feature_columns = feature_columns
        self.initial_balance = initial_balance
        self.fx_lookup = fx_lookup if fx_lookup is not None else {}  # NEW
        self.spread = spread
        self.commission = commission
        self.slippage_pips = float(slippage_pips)
        self.swap_type = "points" if str(swap_type).lower().startswith("point") else "usd"
        self.swap_long_usd_per_lot_night = float(swap_long_usd_per_lot_night)
        self.swap_short_usd_per_lot_night = float(swap_short_usd_per_lot_night)
        self.swap_rollover_hour_utc = int(max(0, min(23, swap_rollover_hour_utc)))
        self.swap_triple_weekday = int(max(0, min(6, swap_triple_weekday)))
        self.use_swap_charging = (
            abs(self.swap_long_usd_per_lot_night) > 1e-12 or
            abs(self.swap_short_usd_per_lot_night) > 1e-12
        )
        self.weekend_close_hours = weekend_close_hours
        self.max_steps = max_steps if max_steps else len(data)
        self.symbol = symbol
        self.allowed_actions = sorted(set(allowed_actions)) if allowed_actions else None
        self.random_episode_start = bool(random_episode_start)
        
        # Feature normalization
        self.scaler_mu = np.array([scaler_mu.get(col, 0.0) for col in feature_columns]) if scaler_mu else None
        self.scaler_sig = np.array([scaler_sig.get(col, 1.0) for col in feature_columns]) if scaler_sig else None
        
        # PATCH #2: Frame stacking for temporal memory (balance-invariant)
        from collections import deque
        self.stack_n = 3  # Stack last 3 observations
        self._frame_stack = None
        
        # Trading behavior parameters
        self.cooldown_bars = cooldown_bars
        self.min_hold_bars = min_hold_bars
        self.trade_penalty = trade_penalty
        self.flip_penalty = flip_penalty
        self.min_atr_cost_ratio = min_atr_cost_ratio
        self.max_trades_per_episode = max_trades_per_episode
        self.risk_per_trade = risk_per_trade
        self.atr_mult_sl = atr_mult_sl
        self.tp_mult = tp_mult
        self.min_trail_buffer_pips = min_trail_buffer_pips  # PATCH #4
        self.disable_move_sl = disable_move_sl
        self.reward_clip = reward_clip
        self.holding_cost = holding_cost
        self.trades_this_ep = 0
        self._accum_entry_cost = 0.0
        self.swap_costs_this_ep = 0.0
        self.r_multiple_reward_weight = r_multiple_reward_weight
        self.r_multiple_reward_clip = r_multiple_reward_clip
        self._last_trade_r_multiple = 0.0
        self._apply_r_multiple_reward = False
        self.use_regime_filter = use_regime_filter
        self.regime_min_vol_z = regime_min_vol_z
        self.regime_align_trend = regime_align_trend
        self.regime_require_trending = regime_require_trending
        
        # Cost accounting: use EITHER extra_entry_penalty OR equity-based costs, not both
        self.extra_entry_penalty = False  # Set to True to add normalized entry costs to reward
        
        # simple holiday set (dates to force flatten) - extend as needed
        self.holidays = set([
            pd.Timestamp('2024-01-01').date(),
            pd.Timestamp('2024-12-25').date(),
        ])

        # Risk manager
        self.risk_manager = risk_manager if risk_manager else RiskManager()
        # Keep a single canonical slippage source on the environment and mirror to risk manager.
        self.risk_manager.slippage_pips = float(self.slippage_pips)
        
        # Structured logger (optional - set by trainer)
        self.structured_logger = None

        # State variables
        self.current_step = 0
        self.episode_step = 0
        self.episode_start_step = 0
        self.balance = initial_balance
        self.equity = initial_balance
        self.prev_equity = initial_balance
        self.position = None  # Dict with position info
        self.trade_history = []
        self.equity_history = [initial_balance]
        
        # Action space
        self.action_space_size = 8
        if self.allowed_actions is not None:
            self.allowed_actions = [a for a in self.allowed_actions if 0 <= int(a) < self.action_space_size]
            if not self.allowed_actions:
                self.allowed_actions = None
        
        # PATCH #2: State space size with frame stacking
        # State = stacked market features + portfolio features (includes action one-hot)
        self.feature_dim = len(feature_columns)
        self.context_dim = 19 + self.action_space_size
        self.state_size = self.feature_dim * self.stack_n + self.context_dim
        
        # PHASE 2.8e: Soft bias parameters (loaded from config in reset)
        self.directional_bias_beta = 0.08
        self.hold_bias_gamma = 0.05
        self.bias_check_interval = 10
        self.bias_margin_low = 0.35
        self.bias_margin_high = 0.65
        self.hold_ceiling = 0.80
        self.circuit_breaker_enabled = True
        self.circuit_breaker_threshold_low = 0.10
        self.circuit_breaker_threshold_high = 0.90
        self.circuit_breaker_lookback = 500
        self.circuit_breaker_mask_duration = 30
        
        # Soft bias tracking (episode-level)
        self.long_trades = 0
        self.short_trades = 0
        self.action_counts = [0] * self.action_space_size
        self.circuit_breaker_active = False
        self.circuit_breaker_counter = 0
        self.circuit_breaker_side = None  # 'long' or 'short'
        self._action_history = []  # Rolling window for circuit-breaker
        
    def reset(self, start_idx: Optional[int] = None) -> np.ndarray:
        """
        Reset environment to initial state.
        PATCH #2: Initialize frame stack.

        Args:
            start_idx: Optional explicit starting bar index for this episode.
        
        Returns:
            Initial state
        """
        max_data_idx = max(0, len(self.data) - 1)
        max_episode_steps = int(max(1, self.max_steps)) if self.max_steps is not None else max_data_idx
        if start_idx is not None:
            start_step = int(np.clip(start_idx, 0, max_data_idx))
        elif self.random_episode_start:
            latest_start = max(0, max_data_idx - max_episode_steps)
            start_step = int(np.random.randint(0, latest_start + 1)) if latest_start > 0 else 0
        else:
            start_step = 0

        self.current_step = start_step
        self.episode_start_step = start_step
        self.episode_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = None
        self.trade_history = []
        self.equity_history = [self.initial_balance]
        # ensure prev_equity reset so first step reward is sane
        self.prev_equity = self.equity
        # reset per-episode trade counter
        self.trades_this_ep = 0
        self._accum_entry_cost = 0.0
        self.swap_costs_this_ep = 0.0
        # SURGICAL PATCH: Reset cost budget tracking
        self.costs_this_ep = 0.0
        self.trading_locked = False
        self.cost_budget_pct = 0.05  # 5% of initial balance
        # reset churn control state
        self.bars_in_position = 0
        self.bars_since_close = 0
        self.last_action = [0] * self.action_space_size
        self._last_trade_r_multiple = 0.0
        self._apply_r_multiple_reward = False
        self._peak_equity = self.initial_balance
        
        # PHASE 2.8e: Reset soft bias tracking
        self.long_trades = 0
        self.short_trades = 0
        self.action_counts = [0] * self.action_space_size
        self.circuit_breaker_active = False
        self.circuit_breaker_counter = 0
        self.circuit_breaker_side = None
        self._action_history = []
        
        # PATCH #2: Reset frame stack
        self._frame_stack = None

        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE, 4=MOVE_SL, 5=MOVE_SL_AGGR, 6=REV_LONG, 7=REV_SHORT)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Get current market data
        current_data = self.data.iloc[self.current_step]
        current_price = current_data['close']
        current_atr = current_data.get('atr', 0.001)
        
        # Check if we need to flatten before weekend
        should_flatten = self._should_flatten_for_weekend(current_data)
        
        # Initialize reward and track if this action changed the position
        reward = 0.0
        did_trade = False
        did_flip = False
        swap_pnl_step = 0.0
        
        atr_cost_ok = self._atr_cost_ratio_ok(current_price, current_atr)
        regime_ok_long = self._regime_allows(1, current_data)
        regime_ok_short = self._regime_allows(-1, current_data)

        # Update existing position (check SL/TP)
        if self.position is not None:
            pnl, hit_sl_tp = self._update_position(current_price)
            if hit_sl_tp or should_flatten:
                reward += pnl
                self.position = None
        
        # Execute action (with cooldown gating)
        if action == 0:  # HOLD
            pass

        elif action == 1:  # LONG (open-only)
            # Check max trades per episode limit
            if self.trades_this_ep >= self.max_trades_per_episode:
                pass  # Ignore action if max trades reached
            elif self.position is None and self.trades_this_ep < self.max_trades_per_episode and atr_cost_ok and regime_ok_long:
                self._open_position('long', current_price, current_atr)
                did_trade = True
                self.trades_this_ep += 1
                if self.extra_entry_penalty:
                    try:
                        lots = self.position.get('lots', 0.0)
                        rtc = (self.spread * pip_value_usd(self.symbol, current_price, lots) + 2.0 * self.commission * lots)
                        self._accum_entry_cost += rtc / max(self.equity, 1e-9)
                    except Exception:
                        pass

        elif action == 2:  # SHORT (open-only)
            # Check max trades per episode limit
            if self.trades_this_ep >= self.max_trades_per_episode:
                pass  # Ignore action if max trades reached
            elif self.position is None and self.trades_this_ep < self.max_trades_per_episode and atr_cost_ok and regime_ok_short:
                self._open_position('short', current_price, current_atr)
                did_trade = True
                self.trades_this_ep += 1
                if self.extra_entry_penalty:
                    try:
                        lots = self.position.get('lots', 0.0)
                        rtc = (self.spread * pip_value_usd(self.symbol, current_price, lots) + 2.0 * self.commission * lots)
                        self._accum_entry_cost += rtc / max(self.equity, 1e-9)
                    except Exception:
                        pass
        elif action == 3:  # CLOSE_POSITION
            if self.position is not None and self._can_modify():
                pnl, _ = self._close_position(current_price)
                reward += pnl
                did_trade = True
                self.trades_this_ep += 1
        elif action in (4, 5):  # MOVE_SL actions (never close)
            if not self.disable_move_sl and self.position is not None:
                recent_data = self.data.iloc[max(0, self.current_step - 40):self.current_step + 1]
                self._move_sl_closer(recent_data, aggressive=(action == 5))
        elif action == 6:  # REVERSE_TO_LONG
            can_reverse = (
                self.position is not None and
                self.position.get('type') == 'short' and
                self._can_modify() and
                (not getattr(self, 'trading_locked', False)) and
                (not should_flatten) and
                atr_cost_ok and
                regime_ok_long and
                self.trades_this_ep <= (self.max_trades_per_episode - 2)
            )
            if can_reverse:
                pnl, _ = self._close_position(current_price)
                reward += pnl
                did_trade = True
                did_flip = True
                self.trades_this_ep += 1
                self._open_position('long', current_price, current_atr)
                did_trade = True
                self.trades_this_ep += 1
                if self.extra_entry_penalty:
                    try:
                        lots = self.position.get('lots', 0.0)
                        rtc = (self.spread * pip_value_usd(self.symbol, current_price, lots) + 2.0 * self.commission * lots)
                        self._accum_entry_cost += rtc / max(self.equity, 1e-9)
                    except Exception:
                        pass
        elif action == 7:  # REVERSE_TO_SHORT
            can_reverse = (
                self.position is not None and
                self.position.get('type') == 'long' and
                self._can_modify() and
                (not getattr(self, 'trading_locked', False)) and
                (not should_flatten) and
                atr_cost_ok and
                regime_ok_short and
                self.trades_this_ep <= (self.max_trades_per_episode - 2)
            )
            if can_reverse:
                pnl, _ = self._close_position(current_price)
                reward += pnl
                did_trade = True
                did_flip = True
                self.trades_this_ep += 1
                self._open_position('short', current_price, current_atr)
                did_trade = True
                self.trades_this_ep += 1
                if self.extra_entry_penalty:
                    try:
                        lots = self.position.get('lots', 0.0)
                        rtc = (self.spread * pip_value_usd(self.symbol, current_price, lots) + 2.0 * self.commission * lots)
                        self._accum_entry_cost += rtc / max(self.equity, 1e-9)
                    except Exception:
                        pass
        
        # Enforce weekend rules (flatten positions if weekend approaching)
        weekend_pnl = self._enforce_weekend_rules(current_price)
        reward += weekend_pnl

        # Apply overnight swap when crossing rollover while position stays open.
        if self.position is not None:
            swap_pnl_step = self._apply_rollover_swap_if_due()
        
        # Update equity
        # _calculate_equity updates self.equity internally and returns it; don't overwrite with None
        self._calculate_equity(current_price)
        self.equity_history.append(self.equity)
        
        # Add small penalty for holding losing positions
        if self.position is not None:
            unrealized_pnl = self._calculate_unrealized_pnl(current_price)
            if unrealized_pnl < 0:
                reward += unrealized_pnl * 0.001  # Small penalty
        
        # Move to next step
        self.current_step += 1
        self.episode_step += 1

        # increment position age for cooldown enforcement
        if self.position is not None:
            self.position['age'] = self.position.get('age', 0) + 1

        # ensure equity and balance are initialized to avoid None comparisons
        if getattr(self, 'equity', None) is None:
            self.balance = getattr(self, 'balance', getattr(self, 'initial_balance', 1000.0))
            self.equity = self.balance
            self.prev_equity = self.equity
            if not hasattr(self, 'equity_history') or self.equity_history is None:
                self.equity_history = []

        # Check if episode is done
        reached_data_end = self.current_step >= (len(self.data) - 1)
        reached_episode_limit = self.max_steps is not None and self.episode_step >= int(self.max_steps)
        done = reached_data_end or reached_episode_limit or self.equity <= 0.05 * self.initial_balance
        
        # SURGICAL PATCH: Cost budget kill-switch
        costs = getattr(self, 'costs_this_ep', 0.0)
        budget = getattr(self, 'cost_budget_pct', 0.05) * self.initial_balance
        
        # PATCH 10: Assert cost budget (catch frictions drift)
        # Allow up to 5% of initial balance in spread+commission+swap costs per episode
        if costs > budget * 1.5:  # Warning threshold at 1.5x budget
            import warnings
            warnings.warn(f"Cost budget exceeded: ${costs:.2f} > ${budget:.2f} (150% threshold)")
        
        if costs > budget and not getattr(self, 'trading_locked', False):
            self.trading_locked = True  # Stop opening new positions this episode
        
        # Get next state
        next_state = self._get_state()
        
        # Info dict
        info = {
            'equity': self.equity,
            'balance': self.balance,
            'position': self.position,
            'step': self.current_step,
            'swap_pnl_step': swap_pnl_step,
            'swap_costs_this_ep': self.swap_costs_this_ep,
        }
        
        # after price update and potential open/close, compute reward as log-return
        prev = max(self.prev_equity, 1e-9)
        curr = max(self.equity, 1e-9)
        reward = np.log(curr / prev)
        # clip reward to ±0.01 for tighter Q-value stability
        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))
        
        # PATCH 5: Tiny holding cost to discourage churn (only while in position)
        # ~2.5 bps per day on H1 bars (24 bars/day -> 1e-4/bar = 0.0024/day)
        if self.position is not None:
            reward -= float(self.holding_cost)
        
        # apply small trade penalty if the action changed position
        if did_trade:
            reward -= float(self.trade_penalty)
        if did_flip:
            reward -= float(self.flip_penalty)
        # subtract accumulated entry cost (normalized) - only if extra_entry_penalty is True
        if self.extra_entry_penalty and getattr(self, '_accum_entry_cost', 0.0) > 0.0:
            reward -= float(self._accum_entry_cost)
            self._accum_entry_cost = 0.0
        if self._apply_r_multiple_reward:
            if self.r_multiple_reward_weight != 0.0:
                clip_value = max(self.r_multiple_reward_clip, 1e-9)
                r_multiple = float(np.clip(self._last_trade_r_multiple, -clip_value, clip_value))
                reward += self.r_multiple_reward_weight * r_multiple
            self._apply_r_multiple_reward = False
        # cap trades per episode to avoid runaway churn
        if getattr(self, 'trades_this_ep', 0) > max(1, int(getattr(self, 'max_trades_per_episode', 120))):
            # force flatten and signal done by setting current_step to max
            if self.position is not None:
                self._close_position(current_price)
                self._calculate_equity(current_price)
            done = True
        self.prev_equity = curr
        
        # PHASE 2.8e: Track action counts for soft bias
        if 0 <= int(action) < len(self.action_counts):
            self.action_counts[int(action)] += 1
        
        # Update churn control state tracking
        if self.position is not None:
            self.bars_in_position += 1
            self.bars_since_close = 0  # Reset cooldown when in position
        else:
            self.bars_in_position = 0
            self.bars_since_close += 1  # Increment cooldown when flat
        
        # SURGICAL PATCH #5: Gate reward for low-activity episodes
        # If episode is ending with very few trades, downweight reward to discourage passivity
        if done and getattr(self, 'trades_this_ep', 0) < 3:
            reward *= 0.25
        
        # Update last action one-hot
        self.last_action = [0] * self.action_space_size
        if 0 <= int(action) < self.action_space_size:
            self.last_action[int(action)] = 1

        return next_state, reward, done, info
    
    def get_action_bias(self) -> np.ndarray:
        """
        PHASE 2.8e: Compute soft biases for action selection (steering, not penalties).
        
        This method applies symmetric, soft nudges to Q-values at action-selection time
        to encourage balanced trading without corrupting the reward signal.
        
        Returns:
            np.ndarray: Bias vector aligned to action space indices
        """
        bias = np.zeros(self.action_space_size, dtype=np.float32)
        
        # Only apply bias periodically (every bias_check_interval steps)
        if self.current_step % self.bias_check_interval != 0:
            return bias
        
        # 1. Directional bias (L/S balance)
        total_trades = self.long_trades + self.short_trades
        if total_trades >= 10:  # Need minimum history
            long_ratio = self.long_trades / total_trades
            
            # Check if circuit-breaker should trigger (extreme lock-in with hysteresis)
            if self.circuit_breaker_enabled:
                # Add current state to history
                self._action_history.append(('long' if self.position and self.position['type'] == 'long' else 
                                            'short' if self.position and self.position['type'] == 'short' else 'flat'))
                
                # Keep only lookback window
                if len(self._action_history) > self.circuit_breaker_lookback:
                    self._action_history = self._action_history[-self.circuit_breaker_lookback:]
                
                # Check if we're in extreme territory for sustained period
                if len(self._action_history) >= self.circuit_breaker_lookback:
                    long_count = sum(1 for a in self._action_history if a == 'long')
                    lookback_long_ratio = long_count / len(self._action_history)
                    
                    # Trigger circuit-breaker if sustained extreme
                    if lookback_long_ratio > self.circuit_breaker_threshold_high and not self.circuit_breaker_active:
                        self.circuit_breaker_active = True
                        self.circuit_breaker_side = 'long'
                        self.circuit_breaker_counter = self.circuit_breaker_mask_duration
                    elif lookback_long_ratio < self.circuit_breaker_threshold_low and not self.circuit_breaker_active:
                        self.circuit_breaker_active = True
                        self.circuit_breaker_side = 'short'
                        self.circuit_breaker_counter = self.circuit_breaker_mask_duration
            
            # Apply circuit-breaker if active (hard mask for short duration)
            if self.circuit_breaker_active and self.circuit_breaker_counter > 0:
                if self.circuit_breaker_side == 'long':
                    # Mask LONG heavily, encourage SHORT
                    bias[1] -= 10.0  # Strong discouragement
                    bias[2] += 0.5   # Mild encouragement
                elif self.circuit_breaker_side == 'short':
                    # Mask SHORT heavily, encourage LONG
                    bias[2] -= 10.0
                    bias[1] += 0.5
                
                self.circuit_breaker_counter -= 1
                if self.circuit_breaker_counter <= 0:
                    self.circuit_breaker_active = False
                    self.circuit_breaker_side = None
            
            # Apply soft directional bias (if not in circuit-breaker mode)
            elif long_ratio > self.bias_margin_high:  # >65% long
                # Discourage LONG, encourage SHORT
                bias[1] -= self.directional_bias_beta
                bias[2] += self.directional_bias_beta
            elif long_ratio < self.bias_margin_low:  # <35% long (too many shorts)
                # Discourage SHORT, encourage LONG
                bias[2] -= self.directional_bias_beta
                bias[1] += self.directional_bias_beta
        
        # 2. Hold bias (prevent passivity)
        total_actions = sum(self.action_counts)
        if total_actions > 50:  # Need minimum history
            hold_rate = self.action_counts[0] / total_actions
            if hold_rate > self.hold_ceiling:  # >80% holding
                # Discourage HOLD
                bias[0] -= self.hold_bias_gamma
        
        return bias
    
    def _portfolio_features(self, current_data: pd.Series) -> np.ndarray:
        """
        Compute balance-invariant portfolio features.
        All features are ratios, flags, or normalized values independent of account size.
        """
        current_price = current_data['close']
        current_atr = max(current_data.get('atr', 0.001), 1e-6)
        
        # Position flags and direction
        long_on = 1.0 if (self.position and self.position['type'] == 'long') else 0.0
        short_on = 1.0 if (self.position and self.position['type'] == 'short') else 0.0
        pos_dir = long_on - short_on  # [-1, 0, 1]
        
        if self.position is None:
            lots_norm = 0.0
            size_frac = 0.0
            entry_diff_atr = 0.0
            sl_dist_atr = 0.0
            tp_dist_atr = 0.0
            unreal_R = 0.0
            unrealized_pct = 0.0
            sl_risk_R = 0.0
            leverage_used = 0.0
            margin_used_pct = 0.0
            sl_dist_norm = 0.0
            tp_dist_norm = 0.0
        else:
            # Normalize lots as "risk utilization" - balance-invariant
            target_risk_dollars = self.initial_balance * self.risk_per_trade
            actual_risk_dollars = abs(self.position['entry'] - self.position['sl']) * self.position['lots'] * self.risk_manager.contract_size
            lots_norm = actual_risk_dollars / max(target_risk_dollars, 1e-6)
            
            # Lot size fraction relative to hard cap
            size_frac = self.position['lots'] / max(getattr(self.risk_manager, 'hard_max_lots', 1.0), 1e-9)
            
            # Distance metrics in ATR units
            entry_diff_atr = (current_price - self.position['entry']) / current_atr
            sl_dist_atr = abs(current_price - self.position['sl']) / current_atr
            tp_dist_atr = abs(self.position['tp'] - current_price) / current_atr
            
            # Unrealized PnL as R-multiples and % of equity
            unrealized_pnl = self._calculate_unrealized_pnl(current_price)
            sl_risk_dollars = abs(self.position['entry'] - self.position['sl']) * self.position['lots'] * self.risk_manager.contract_size
            unreal_R = unrealized_pnl / max(sl_risk_dollars, 1e-6)
            unrealized_pct = unrealized_pnl / max(self.equity, 1e-9)
            sl_risk_R = 1.0
            
            # Leverage and margin (normalized by initial balance)
            notional = self.position['lots'] * self.risk_manager.contract_size * current_price
            leverage_used = notional / max(self.initial_balance, 1e-6)
            margin_required = notional / self.risk_manager.leverage
            margin_used_pct = margin_required / max(self.initial_balance, 1e-6)
            
            # Distance to SL/TP in pips, normalized by target pips (scale-free)
            sl_pips = max(1.0, getattr(self, 'current_sl_pips', sl_dist_atr * (pip_size(self.symbol) / current_atr)))
            tp_pips = max(1.0, getattr(self, 'current_tp_pips', tp_dist_atr * (pip_size(self.symbol) / current_atr)))
            sl_dist_norm = min(3.0, sl_dist_atr / max(sl_pips, 1.0))  # cap to avoid outliers
            tp_dist_norm = min(3.0, tp_dist_atr / max(tp_pips, 1.0))
        
        # Equity metrics (balance-invariant)
        equity_log_rel = np.log(max(self.equity, 1e-6) / self.initial_balance)
        
        # Drawdown from running peak (%)
        if not hasattr(self, '_peak_equity'):
            self._peak_equity = self.initial_balance
        self._peak_equity = max(self._peak_equity, self.equity)
        dd_pct = (self.equity - self._peak_equity) / max(self._peak_equity, 1e-9)
        
        # Margin and constraint metrics
        free_margin_pct = 1.0 - margin_used_pct
        
        # SURGICAL PATCH #6: Clip portfolio features to prevent outliers
        sl_dist_atr = np.clip(sl_dist_atr, 0.0, 5.0)
        tp_dist_atr = np.clip(tp_dist_atr, 0.0, 10.0)
        dd_pct = np.clip(dd_pct, -0.90, 0.0)
        unrealized_pct = np.clip(unrealized_pct, -0.20, 0.20)
        
        # Action/cooldown fractions
        hold_left = max(0, getattr(self, 'bars_in_position', 0) - self.min_hold_bars) / max(1, self.min_hold_bars)
        cool_left = max(0, self.cooldown_bars - self.bars_since_close) / max(1, self.cooldown_bars)
        trades_frac = self.trades_this_ep / max(1, self.max_trades_per_episode)
        
        # Weekend flag
        weekend_flag = 1.0 if self._should_flatten_for_weekend(current_data) else 0.0
        
        # Last action one-hot (aligned to action space size)
        last_action_onehot = np.zeros(self.action_space_size, dtype=np.float32)
        if hasattr(self, 'last_action') and len(self.last_action) == self.action_space_size:
            last_action_onehot = np.array(self.last_action, dtype=np.float32)
        
        # SURGICAL PATCH: Build portfolio feature array with global clipping
        pf = np.array([
            pos_dir, long_on, short_on, size_frac,
            unrealized_pct, dd_pct,
            hold_left, cool_left, trades_frac,
            sl_dist_norm, tp_dist_norm,
            equity_log_rel, leverage_used, margin_used_pct, free_margin_pct,
            entry_diff_atr, sl_dist_atr, tp_dist_atr,
            weekend_flag,
            *last_action_onehot
        ], dtype=np.float32)
        
        # Light global clip on the portfolio block (broad safety rails)
        pf = np.clip(pf, -5.0, 5.0)
        
        return pf
    
    def _stack_obs(self, x):
        """PATCH #2: Stack observations for temporal memory."""
        from collections import deque
        if self._frame_stack is None:
            self._frame_stack = deque([x]*self.stack_n, maxlen=self.stack_n)
        else:
            self._frame_stack.append(x)
        return np.concatenate(list(self._frame_stack), axis=0)
    
    def _get_state(self) -> np.ndarray:
        """
        PATCH #2: Get current state representation with frame stacking.
        
        Returns:
            State vector: stacked normalized market features + portfolio features
        """
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
        
        # Market features (normalized)
        current_data = self.data.iloc[self.current_step]
        market_features = current_data[self.feature_columns].values.astype(np.float32)
        
        # Apply normalization if scaler is available
        if self.scaler_mu is not None and self.scaler_sig is not None:
            market_features = (market_features - self.scaler_mu) / self.scaler_sig
        
        # PATCH #2: Stack market features for temporal context
        stacked_market = self._stack_obs(market_features)
        
        # Portfolio features
        portfolio_features = self._portfolio_features(current_data)
        
        # Combine stacked market features + portfolio context
        state = np.concatenate([stacked_market, portfolio_features])
        
        # Handle NaN/Inf
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return state
    
    def _open_position(self, side: str, price: float, atr: float):
        # no stacking
        if getattr(self, 'position', None) is not None:
            return

        # Get current timestamp for dynamic pip value calculation
        ts = self.data.index[self.current_step] if hasattr(self.data.index[self.current_step], 'to_pydatetime') else self.current_step
        
        ps = pip_size(self.symbol)
        atr_pips = max(1.0, atr / ps)
        stop_pips = max(10.0, 2.0 * atr_pips)

        # Helper for pip value at current timestamp
        pv = lambda lots: pip_value_usd_ts(self.symbol, price, lots, ts, self.fx_lookup) if self.fx_lookup else pip_value_usd(self.symbol, price, lots)

        # Ask risk manager for enhanced position sizing if available
        desired_lots = None
        sizing_info = {}
        try:
            # Try enhanced method with cost budget first
            if hasattr(self.risk_manager, 'compute_lots_enhanced'):
                sizing = self.risk_manager.compute_lots_enhanced(
                    balance=self.balance,
                    free_margin=getattr(self, 'balance', 0.0),
                    price=price,
                    atr=atr,
                    spread=self.spread,
                    commission=self.commission,
                    symbol=self.symbol
                )
            else:
                # Fallback to basic method
                sizing = self.risk_manager.calculate_position_size(
                    balance=self.balance,
                    free_margin=getattr(self, 'balance', 0.0),
                    price=price,
                    atr=atr,
                    symbol=self.symbol
                )
            
            desired_lots = float(sizing.get('lots', 0.0))
            sizing_info = sizing
            
            # Reject if budget or survivability constraints violated
            if sizing.get('budget_rejected', False) or sizing.get('survivability_rejected', False):
                desired_lots = 0.0
                
        except Exception:
            desired_lots = None

        # fallback simple risk-based sizing
        if desired_lots is None or desired_lots <= 0:
            risk_usd = self.equity * getattr(self.risk_manager, 'risk_per_trade', 0.01)
            pv_1lot = pv(1.0)
            desired_lots = max(0.01, risk_usd / (stop_pips * pv_1lot + 1e-9))

        # enforce hard max lots and affordability via margin cap
        desired_lots = min(desired_lots, getattr(self.risk_manager, 'hard_max_lots', 1.0))

        # PATCH 3: Volatility-aware slippage (tie to ATR)
        # Get current ATR in pips
        current_row = self.data.iloc[self.current_step]
        atr_val = current_row.get('atr', np.nan)
        if np.isnan(atr_val):
            atr_val = 0.0015 * price  # fallback ~15 pips
        atr_pips = max(1.0, atr_val / ps)
        
        # Dynamic slippage: base + volatility component
        slippage_pips_base = float(getattr(self, 'slippage_pips', 0.8))
        slippage_pips_eff = slippage_pips_base + 0.10 * np.sqrt(atr_pips)
        
        # calculate exec price with dynamic slippage and half-spread
        half_spread = self.spread / 2.0
        exec_price = (
            price
            + (slippage_pips_eff * ps) * (1.0 if side == 'long' else -1.0)
            + (half_spread) * (1.0 if side == 'long' else -1.0)
        )

        if side == 'long':
            sl_price = exec_price - stop_pips * ps
            tp_price = exec_price + stop_pips * ps * 2.0
        else:
            sl_price = exec_price + stop_pips * ps
            tp_price = exec_price - stop_pips * ps * 2.0

        self.position = {
            'type': side,
            'entry': exec_price,
            'lots': desired_lots,
            'stop_pips': stop_pips,
            'sl': sl_price,
            'tp': tp_price,
            'age': 0,
            'open_time': self.current_step,
        }

        # charge commission at entry (per side)
        entry_cost = self.commission * desired_lots
        self.balance -= entry_cost
        
        # SURGICAL PATCH: Track costs for budget enforcement
        self.costs_this_ep = getattr(self, 'costs_this_ep', 0.0) + entry_cost
        
        self.prev_equity = self.equity
        
        # Log trade opening if structured logger available
        if getattr(self, 'structured_logger', None) is not None:
            try:
                timestamp = datetime.now()
                margin_used = desired_lots * getattr(self.risk_manager, 'contract_size', 100_000) * exec_price / max(getattr(self.risk_manager, 'leverage', 100), 1)
                
                # Ensure all values are valid floats, no NaNs
                safe_price = float(exec_price) if not np.isnan(exec_price) else 0.0
                safe_lots = float(desired_lots) if not np.isnan(desired_lots) else 0.0
                safe_sl_price = float(sl_price) if sl_price is not None and not np.isnan(sl_price) else None
                safe_tp_price = float(tp_price) if tp_price is not None and not np.isnan(tp_price) else None
                safe_equity = float(self.prev_equity) if not np.isnan(self.prev_equity) else 0.0
                safe_margin = float(margin_used) if not np.isnan(margin_used) else 0.0
                safe_stop_pips = float(stop_pips) if stop_pips is not None and not np.isnan(stop_pips) else None
                
                self.structured_logger.log_trade_open(
                    timestamp=timestamp,
                    action=str(side).upper(),
                    price=safe_price,
                    lots=safe_lots,
                    sl_price=safe_sl_price,
                    tp_price=safe_tp_price,
                    equity_before=safe_equity,
                    margin_used=safe_margin,
                    step=self.current_step,
                    stop_pips=safe_stop_pips
                )
            except Exception:
                pass
        
        # PHASE 2.8e: Track long/short trades for soft bias
        if side == 'long':
            self.long_trades += 1
        elif side == 'short':
            self.short_trades += 1

        # simple survivability/margin check (reduce lots if insufficient margin)
        try:
            contract = getattr(self.risk_manager, 'contract_size', 100_000)
            leverage = getattr(self.risk_manager, 'leverage', 100)
            margin_needed = desired_lots * contract * exec_price / max(leverage, 1)
            free_margin = max(self.balance, 0.0)
            if margin_needed > free_margin and margin_needed > 0:
                affordable_lots = (free_margin * leverage) / (contract * exec_price + 1e-9)
                affordable_lots = max(0.0, affordable_lots)
                # apply a safety cap
                affordable_lots = min(affordable_lots, getattr(self.risk_manager, 'hard_max_lots', 1.0))
                # update position lots and adjust commission already charged
                # refund difference if we overcharged
                if affordable_lots < self.position['lots']:
                    # refund proportional commission for reduced lots
                    refund = (self.commission * (self.position['lots'] - affordable_lots))
                    self.balance += refund
                    self.position['lots'] = affordable_lots
        except Exception:
            pass
    
    def _close_position(self, price: float) -> Tuple[float, bool]:
        if self.position is None:
            return 0.0, False

        # Get current timestamp for dynamic pip value calculation
        ts = self.data.index[self.current_step] if hasattr(self.data.index[self.current_step], 'to_pydatetime') else self.current_step
        
        # Helper for pip value at current timestamp
        pv = lambda lots: pip_value_usd_ts(self.symbol, price, lots, ts, self.fx_lookup) if self.fx_lookup else pip_value_usd(self.symbol, price, lots)
        
        half_spread = self.spread / 2.0
        if self.position['type'] == 'long':
            exit_price = price - half_spread
        else:
            exit_price = price + half_spread

        # Calculate PnL using dynamic pip values
        ps = pip_size(self.symbol)
        pips_move = (exit_price - self.position['entry']) / ps
        pips_move *= (1 if self.position['type'] == 'long' else -1)
        pnl = pips_move * pv(self.position['lots'])

        # commission per lot per side at CLOSE
        exit_cost = self.commission * self.position['lots']
        pnl -= exit_cost
        
        # SURGICAL PATCH: Track exit costs
        self.costs_this_ep = getattr(self, 'costs_this_ep', 0.0) + exit_cost
        
        self.balance += pnl

        self.trade_history.append({
            'type': self.position['type'],
            'entry': self.position['entry'],
            'exit': exit_price,
            'lots': self.position['lots'],
            'pnl': pnl,
            'open_time': self.position['open_time'],
            'close_time': self.current_step,
        })

        # Track realized R multiple for reward shaping (pnl vs SL risk).
        self._apply_r_multiple_reward = False
        sl_price = self.position.get('sl')
        if sl_price is not None:
            sl_risk_pips = abs(self.position['entry'] - sl_price) / ps
            sl_risk_dollars = sl_risk_pips * pv(self.position['lots'])
            if sl_risk_dollars > 0:
                r_multiple = pnl / sl_risk_dollars
                if np.isfinite(r_multiple):
                    self._last_trade_r_multiple = float(r_multiple)
                    self._apply_r_multiple_reward = True
        
        # Log trade closing if structured logger available
        if getattr(self, 'structured_logger', None) is not None:
            try:
                timestamp = datetime.now()
                duration_bars = self.current_step - self.position['open_time']
                # Ensure all values are valid floats, no NaNs
                safe_exit_price = float(exit_price) if not np.isnan(exit_price) else 0.0
                safe_pnl = float(pnl) if not np.isnan(pnl) else 0.0
                safe_equity = float(self.balance) if not np.isnan(self.balance) else 0.0
                
                self.structured_logger.log_trade_close(
                    timestamp=timestamp,
                    reason="MANUAL_CLOSE",
                    exit_price=safe_exit_price,
                    pnl=safe_pnl,
                    equity_after=safe_equity,
                    duration_bars=int(duration_bars),
                    step=self.current_step,
                    trade_type=str(self.position['type']),
                    action="CLOSE_" + str(self.position['type']).upper()  # Add action field
                )
            except Exception:
                pass

        # clear position
        self.position = None
        return pnl, False
    
    def _update_position(self, price: float) -> Tuple[float, bool]:
        """
        Update position and check if SL or TP is hit.
        
        Args:
            price: Current price
            
        Returns:
            Tuple of (pnl, hit_sl_tp)
        """
        if self.position is None:
            return 0.0, False
        
        hit_sl_tp = False
        pnl = 0.0
        
        # Check stop loss and take profit
        if self.position['type'] == 'long':
            if price <= self.position['sl']:
                pnl, _ = self._close_position(self.position['sl'])
                hit_sl_tp = True
            elif price >= self.position['tp']:
                pnl, _ = self._close_position(self.position['tp'])
                hit_sl_tp = True
        else:  # short
            if price >= self.position['sl']:
                pnl, _ = self._close_position(self.position['sl'])
                hit_sl_tp = True
            elif price <= self.position['tp']:
                pnl, _ = self._close_position(self.position['tp'])
                hit_sl_tp = True
        
        return pnl, hit_sl_tp
    
    def _compute_sl_target(self, price_data: pd.DataFrame, aggressive: bool = False) -> Optional[float]:
        """
        Compute a structure-aware SL target for tightening.
        The model selects HOW much tightening to apply via discrete action:
        MOVE_SL_CLOSER (softer) or MOVE_SL_CLOSER_AGGRESSIVE (faster).
        """
        if self.position is None or price_data is None or len(price_data) == 0:
            return None

        current_price = float(price_data['close'].iloc[-1])
        current_sl = float(self.position['sl'])
        position_type = self.position['type']
        current_atr = float(price_data.get('atr', pd.Series([0.001])).iloc[-1])
        current_atr = max(current_atr, 1e-6)
        ps = pip_size(self.symbol)

        # Keep a minimum distance from price to avoid self-triggering moves.
        min_gap = max(float(self.min_trail_buffer_pips) * ps, 0.08 * current_atr)

        # Build structure candidates from confirmed fractals and recent bar extremes.
        top_frac = price_data.get('top_fractal_confirmed', pd.Series([np.nan]))
        bottom_frac = price_data.get('bottom_fractal_confirmed', pd.Series([np.nan]))
        highs = price_data['high'].tail(8)
        lows = price_data['low'].tail(8)
        frac_pad = (0.20 if aggressive else 0.30) * current_atr
        trail_gap = min_gap * (0.9 if aggressive else 1.5)
        min_step = max(ps * 0.25, ps * 0.25 * float(self.min_trail_buffer_pips))

        if position_type == 'long':
            max_sl = current_price - min_gap
            if max_sl <= current_sl + min_step:
                return None

            candidates = [current_price - trail_gap]
            if len(bottom_frac) > 0:
                b = bottom_frac.iloc[-1]
                if not np.isnan(b):
                    candidates.append(float(b) - frac_pad)
            if len(lows) > 0:
                candidates.append(float(lows.min()) - 0.15 * current_atr)

            raw_target = max(candidates)
            target = min(raw_target, max_sl)
            if target <= current_sl + min_step:
                return None
            return float(target)

        # SHORT position
        min_sl = current_price + min_gap
        if min_sl >= current_sl - min_step:
            return None

        candidates = [current_price + trail_gap]
        if len(top_frac) > 0:
            t = top_frac.iloc[-1]
            if not np.isnan(t):
                candidates.append(float(t) + frac_pad)
        if len(highs) > 0:
            candidates.append(float(highs.max()) + 0.15 * current_atr)

        raw_target = min(candidates)
        target = max(raw_target, min_sl)
        if target >= current_sl - min_step:
            return None
        return float(target)

    def _move_sl_closer(self, price_data: pd.DataFrame, aggressive: bool = False) -> bool:
        """Tighten stop-loss only; never closes a position."""
        if self.position is None:
            return False

        current_sl = float(self.position['sl'])
        target = self._compute_sl_target(price_data, aggressive=aggressive)
        if target is None:
            return False

        # Action controls tightening speed. Aggressive action moves farther in one step.
        alpha = 0.80 if aggressive else 0.45
        new_sl = current_sl + alpha * (target - current_sl)
        pos_type = self.position['type']

        if pos_type == 'long':
            if new_sl <= current_sl:
                return False
            self.position['sl'] = float(new_sl)
        else:
            if new_sl >= current_sl:
                return False
            self.position['sl'] = float(new_sl)

        if getattr(self, 'structured_logger', None) is not None:
            try:
                self.structured_logger.log_trade_sl_move(
                    timestamp=datetime.now(),
                    old_sl=current_sl,
                    new_sl=float(self.position['sl']),
                    current_price=float(price_data['close'].iloc[-1]),
                    step=self.current_step,
                    method="adaptive_structure_aggr" if aggressive else "adaptive_structure",
                )
            except Exception:
                pass
        return True
    
    def _calculate_unrealized_pnl(self, price: float) -> float:
        """
        Calculate unrealized PnL for current position using dynamic pip values.
        
        Args:
            price: Current price
            
        Returns:
            Unrealized PnL
        """
        if self.position is None:
            return 0.0
        
        # Get current timestamp for dynamic pip value calculation
        ts = self.data.index[self.current_step] if hasattr(self.data.index[self.current_step], 'to_pydatetime') else self.current_step
        
        # Helper for pip value at current timestamp
        pv = lambda lots: pip_value_usd_ts(self.symbol, price, lots, ts, self.fx_lookup) if self.fx_lookup else pip_value_usd(self.symbol, price, lots)
        
        # Calculate PnL using dynamic pip values
        ps = pip_size(self.symbol)
        pips_move = (price - self.position['entry']) / ps
        pips_move *= (1 if self.position['type'] == 'long' else -1)
        return pips_move * pv(self.position['lots'])
    
    def _calculate_equity(self, price: float):
        # mark-to-market PnL based on pips moved * pip value
        if getattr(self, 'position', None) is None:
            self.equity = self.balance
            return self.equity

        # Get current timestamp for dynamic pip value calculation
        ts = self.data.index[self.current_step] if hasattr(self.data.index[self.current_step], 'to_pydatetime') else self.current_step
        
        # Helper for pip value at current timestamp
        pv = lambda lots: pip_value_usd_ts(self.symbol, price, lots, ts, self.fx_lookup) if self.fx_lookup else pip_value_usd(self.symbol, price, lots)
        
        ps = pip_size(self.symbol)
        dprice = price - self.position['entry']
        pips_move = (dprice / ps) * (1.0 if self.position['type'] == 'long' else -1.0)
        unrealized = pips_move * pv(self.position['lots'])

        # unrealized PnL + balance (which already had commission deducted on open)
        self.equity = self.balance + unrealized
        return self.equity

    def _get_bar_timestamp(self, step: int) -> Optional[pd.Timestamp]:
        """Best-effort timestamp lookup for a bar index; returns None if unavailable."""
        if step < 0 or step >= len(self.data):
            return None

        # Prefer explicit time column when present.
        if 'time' in self.data.columns:
            try:
                ts = pd.to_datetime(self.data.iloc[step]['time'])
                if not pd.isna(ts):
                    return pd.Timestamp(ts).tz_localize(None) if getattr(ts, 'tzinfo', None) else pd.Timestamp(ts)
            except Exception:
                pass

        # Fallback to DatetimeIndex.
        if isinstance(self.data.index, pd.DatetimeIndex):
            try:
                ts = self.data.index[step]
                return pd.Timestamp(ts).tz_localize(None) if getattr(ts, 'tzinfo', None) else pd.Timestamp(ts)
            except Exception:
                pass

        return None

    def _count_rollovers_between(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> int:
        """
        Count rollover events crossed from (start_ts, end_ts].
        Uses UTC-hour cutoff and weekday triple-swap multiplier.
        """
        try:
            s = pd.Timestamp(start_ts)
            e = pd.Timestamp(end_ts)
        except Exception:
            return 0
        if e <= s:
            return 0

        total_rollovers = 0
        start_date = (s - pd.Timedelta(days=1)).normalize()
        end_date = e.normalize()
        for day in pd.date_range(start=start_date, end=end_date, freq='D'):
            cutoff = day + pd.Timedelta(hours=self.swap_rollover_hour_utc)
            if s < cutoff <= e:
                wd = int(cutoff.weekday())
                if wd >= 5:
                    continue  # No weekend rollover charge.
                mult = 3 if wd == self.swap_triple_weekday else 1
                total_rollovers += mult
        return int(total_rollovers)

    def _apply_rollover_swap_if_due(self) -> float:
        """
        Apply swap credit/debit when crossing rollover cutoff while a position is open.
        Returns applied USD amount (negative = cost, positive = credit).
        """
        if not self.use_swap_charging or self.position is None:
            return 0.0
        if self.current_step + 1 >= len(self.data):
            return 0.0

        start_ts = self._get_bar_timestamp(self.current_step)
        end_ts = self._get_bar_timestamp(self.current_step + 1)
        if start_ts is None or end_ts is None:
            return 0.0

        rollovers = self._count_rollovers_between(start_ts, end_ts)
        if rollovers <= 0:
            return 0.0

        lots = float(self.position.get('lots', 0.0))
        if lots <= 0:
            return 0.0

        side = self.position.get('type')
        rate = self.swap_long_usd_per_lot_night if side == 'long' else self.swap_short_usd_per_lot_night
        if self.swap_type == "points":
            # Broker "swap in points": convert points -> price delta -> quote PnL -> USD.
            symbol = str(getattr(self, "symbol", "EURUSD"))
            quote_ccy = symbol[3:6] if len(symbol) >= 6 else "USD"
            point_size = 0.001 if symbol.endswith("JPY") else 0.00001
            contract_size = float(getattr(self.risk_manager, "contract_size", 100000.0))
            quote_pnl = float(rate) * point_size * contract_size * lots * float(rollovers)
            q2usd = _usd_conv_factor(end_ts, quote_ccy, self.fx_lookup) if self.fx_lookup else _static_usd_conv(quote_ccy)
            swap_pnl = float(quote_pnl) * float(q2usd)
        else:
            # USD per lot per night.
            swap_pnl = float(rate) * lots * float(rollovers)
        if abs(swap_pnl) <= 1e-12:
            return 0.0

        self.balance += swap_pnl
        self.swap_costs_this_ep += swap_pnl
        if swap_pnl < 0.0:
            self.costs_this_ep = getattr(self, 'costs_this_ep', 0.0) + abs(swap_pnl)
        return swap_pnl
    
    def _should_flatten_for_weekend(self, current_data: pd.Series) -> bool:
        current_time = None
        # prefer explicit time column if present
        if 'time' in self.data.columns:
            try:
                current_time = pd.to_datetime(self.data.iloc[self.current_step]['time'])
            except Exception:
                current_time = None
        # fallback to DatetimeIndex
        if current_time is None and isinstance(self.data.index, pd.DatetimeIndex):
            try:
                current_time = self.data.index[self.current_step]
            except Exception:
                current_time = None

        if current_time is None:
            return False

        # holiday flatten
        if current_time.date() in getattr(self, 'holidays', set()):
            return True

        # Friday close guard
        if current_time.dayofweek == 4:
            hours_to_close = 17 - current_time.hour
            if hours_to_close <= self.weekend_close_hours:
                return True

        return False
    
    def _is_weekend_approaching(self) -> bool:
        """
        Check if weekend is approaching and positions should be flattened.
        
        Returns:
            True if within weekend_close_hours of market close
        """
        try:
            # Get current timestamp from data index
            if len(self.data) > self.current_step:
                current_time = self.data.index[self.current_step]
                
                # Check if it's a pandas datetime
                if hasattr(current_time, 'weekday'):
                    weekday = current_time.weekday()  # Monday=0, Sunday=6
                    hour = current_time.hour if hasattr(current_time, 'hour') else 0
                    
                    # Friday after a certain hour (market typically closes Friday 22:00 UTC)
                    if weekday == 4 and hour >= (22 - self.weekend_close_hours):  # Friday
                        return True
                        
                    # Saturday or Sunday
                    if weekday >= 5:  # Saturday=5, Sunday=6
                        return True
                        
        except Exception:
            pass
            
        return False
        
    def _enforce_weekend_rules(self, current_price: float) -> float:
        """
        Enforce weekend trading rules - flatten positions if weekend approaching.
        
        Args:
            current_price: Current market price
            
        Returns:
            PnL from any forced closures
        """
        weekend_pnl = 0.0
        
        if self.position is not None and self._is_weekend_approaching():
            try:
                # Force close position before weekend
                pnl, _ = self._close_position(current_price)
                weekend_pnl += pnl
                
                # Log weekend closure
                if getattr(self, 'structured_logger', None) is not None:
                    try:
                        timestamp = datetime.now()
                        self.structured_logger.log_trade_close(
                            timestamp=timestamp,
                            reason="WEEKEND_CLOSE",
                            exit_price=current_price,
                            pnl=pnl,
                            equity_after=self.balance,
                            duration_bars=self.current_step - self.position.get('open_time', 0),
                            step=self.current_step,
                            trade_type=self.position.get('type', 'UNKNOWN'),
                            action="WEEKEND_FLATTEN"
                        )
                    except Exception:
                        pass
                        
            except Exception:
                pass
                
        return weekend_pnl
    
    def get_trade_statistics(self) -> Dict:
        """
        Calculate trading statistics.
        
        Returns:
            Dict with statistics
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
            }
        
        pnls = [t['pnl'] for t in self.trade_history]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        total_trades = len(pnls)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(pnls)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'total_trades': total_trades,
            'trades': total_trades,  # Alias for compatibility
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
        }
    
    def legal_action_mask(self) -> list:
        """
        SURGICAL PATCH: Legal action masking (stricter version).
        Returns a boolean mask [HOLD, LONG, SHORT, CLOSE, MOVE_SL, MOVE_SL_AGGR, REV_LONG, REV_SHORT]
        True = allowed, False = illegal right now.
        """
        hold_ok = True
        current_price = None
        current_atr = None
        current_data = None
        if self.current_step < len(self.data):
            current_data = self.data.iloc[self.current_step]
            current_price = float(current_data['close'])
            current_atr = float(current_data.get('atr', 0.001))
        atr_cost_ok = True
        if current_price is not None:
            atr_cost_ok = self._atr_cost_ratio_ok(current_price, current_atr)
        regime_ok_long = self._regime_allows(1, current_data)
        regime_ok_short = self._regime_allows(-1, current_data)
        
        # Disallow trading under episode/global locks OR max trades reached
        trading_blocked = False
        if self.bars_since_close < self.cooldown_bars:
            trading_blocked = True
        if getattr(self, 'trading_locked', False):
            trading_blocked = True
        if self.trades_this_ep >= self.max_trades_per_episode:
            trading_blocked = True  # Max trades reached
        if self._should_flatten_for_weekend(self.data.iloc[self.current_step] if self.current_step < len(self.data) else None):
            trading_blocked = True
        
        long_ok = short_ok = not trading_blocked
        close_ok = False
        move_sl_ok = False
        move_sl_aggr_ok = False
        reverse_long_ok = False
        reverse_short_ok = False
        
        if self.position is None:
            # Flat: only entries are legal (subject to gates).
            move_sl_ok = False
            move_sl_aggr_ok = False
            close_ok = False
            reverse_long_ok = False
            reverse_short_ok = False
        else:
            # In-position: disallow open actions; use explicit CLOSE/MOVE_SL actions.
            long_ok = False
            short_ok = False
            close_ok = self._can_modify()
            can_tighten = self._can_tighten_sl()
            move_sl_ok = can_tighten
            move_sl_aggr_ok = can_tighten
            can_reverse_common = (
                self._can_modify() and
                (not getattr(self, 'trading_locked', False)) and
                (not self._should_flatten_for_weekend(self.data.iloc[self.current_step] if self.current_step < len(self.data) else None)) and
                (self.trades_this_ep <= (self.max_trades_per_episode - 2))
            )
            if self.position.get('type') == 'short':
                reverse_long_ok = can_reverse_common and atr_cost_ok and regime_ok_long
            elif self.position.get('type') == 'long':
                reverse_short_ok = can_reverse_common and atr_cost_ok and regime_ok_short

        if self.disable_move_sl:
            move_sl_ok = False
            move_sl_aggr_ok = False

        if not atr_cost_ok:
            long_ok = False
            short_ok = False

        if not regime_ok_long:
            long_ok = False
        if not regime_ok_short:
            short_ok = False

        mask = [
            hold_ok,
            long_ok,
            short_ok,
            close_ok,
            move_sl_ok,
            move_sl_aggr_ok,
            reverse_long_ok,
            reverse_short_ok,
        ]
        if self.allowed_actions is not None:
            mask = [mask[i] and (i in self.allowed_actions) for i in range(len(mask))]
        
        return mask
    
    def _can_tighten_sl(self) -> bool:
        """
        Check if stop-loss can be tightened meaningfully.
        """
        if self.position is None or self.position.get('sl') is None:
            return False
        if self.current_step >= len(self.data):
            return False
        recent_data = self.data.iloc[max(0, self.current_step - 40):self.current_step + 1]
        return self._compute_sl_target(recent_data, aggressive=False) is not None
    
    def _tighten_sl(self):
        """
        Legacy helper retained for compatibility.
        """
        if self.current_step >= len(self.data):
            return
        recent_data = self.data.iloc[max(0, self.current_step - 40):self.current_step + 1]
        self._move_sl_closer(recent_data, aggressive=False)
    
    def _can_modify(self) -> bool:
        """Return True if we can close/flip the current position (cooldown)."""
        if self.position is None:
            return True
        age = self.position.get('age', 0)
        required = max(getattr(self, 'cooldown_bars', 12), getattr(self, 'min_hold_bars', 6))
        return age >= required

    def _regime_allows(self, action_dir: int, current_data: pd.Series | None) -> bool:
        if not getattr(self, 'use_regime_filter', False):
            return True
        if current_data is None:
            return True
        if getattr(self, 'regime_require_trending', True):
            try:
                is_trending = float(current_data.get('is_trending', 0.0))
            except Exception:
                is_trending = 0.0
            if is_trending < 0.5:
                return False
        try:
            vol_z = float(current_data.get('realized_vol_24h_z', 0.0))
        except Exception:
            vol_z = 0.0
        if vol_z < float(getattr(self, 'regime_min_vol_z', 0.0)):
            return False
        if getattr(self, 'regime_align_trend', True):
            try:
                trend = float(current_data.get('trend_96h', 0.0))
            except Exception:
                trend = 0.0
            if action_dir > 0 and trend <= 0.0:
                return False
            if action_dir < 0 and trend >= 0.0:
                return False
        return True

    def _atr_cost_ratio_ok(self, price: float, atr: float) -> bool:
        """
        Gate new trades unless ATR sufficiently exceeds expected costs.
        """
        min_ratio = getattr(self, 'min_atr_cost_ratio', 0.0)
        if min_ratio <= 0:
            return True
        ps = pip_size(self.symbol)
        if ps <= 0:
            return True
        atr_pips = atr / ps
        if atr_pips <= 0:
            return False

        ts = self.data.index[self.current_step] if self.current_step < len(self.data) else None
        try:
            if self.fx_lookup and ts is not None:
                pv_1lot = pip_value_usd_ts(self.symbol, price, 1.0, ts, self.fx_lookup)
            else:
                pv_1lot = pip_value_usd(self.symbol, price, 1.0)
        except Exception:
            pv_1lot = 0.0

        if pv_1lot <= 0:
            return True

        commission_pips = (2.0 * self.commission) / pv_1lot
        slippage_pips_base = float(getattr(self, 'slippage_pips', 0.8))
        slippage_pips_eff = slippage_pips_base + 0.10 * np.sqrt(max(atr_pips, 0.0))
        spread_pips = self.spread / ps
        cost_pips = spread_pips + slippage_pips_eff + commission_pips
        if cost_pips <= 0:
            return True
        return (atr_pips / cost_pips) >= min_ratio
    
    def _find_fractals(self, data: pd.DataFrame, window: int = 3) -> tuple:
        """
        Find fractal highs and lows in price data.
        
        Args:
            data: DataFrame with 'high' and 'low' columns
            window: Window size for fractal detection
            
        Returns:
            Tuple of (fractal_highs, fractal_lows) arrays
        """
        if len(data) < 2 * window + 1:
            return np.array([]), np.array([])
            
        highs = data['high'].values
        lows = data['low'].values
        
        fractal_highs = []
        fractal_lows = []
        
        for i in range(window, len(data) - window):
            # Check for fractal high (peak)
            is_high_fractal = True
            for j in range(i - window, i + window + 1):
                if j != i and highs[j] >= highs[i]:
                    is_high_fractal = False
                    break
            if is_high_fractal:
                fractal_highs.append(highs[i])
                
            # Check for fractal low (trough)  
            is_low_fractal = True
            for j in range(i - window, i + window + 1):
                if j != i and lows[j] <= lows[i]:
                    is_low_fractal = False
                    break
            if is_low_fractal:
                fractal_lows.append(lows[i])
                
        return np.array(fractal_highs), np.array(fractal_lows)

    def _enhanced_move_sl_closer(self, price_data: pd.DataFrame) -> bool:
        """
        Backward-compatible alias for the aggressive SL tighten action.
        """
        return self._move_sl_closer(price_data, aggressive=True)


if __name__ == "__main__":
    print("Trading Environment Module")
    print("=" * 50)
    
    # Realistic synthetic FX prices for the demo
    n_bars = 1000
    rng = np.random.default_rng(42)
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='1h')

    # Hourly log-returns ~ N(0, 0.0002^2) ≈ 2 pips std
    ret = rng.normal(0.0, 2e-4, size=n_bars)
    price = 1.1000 * np.exp(np.cumsum(ret))

    close = price
    open_ = np.roll(close, 1); open_[0] = close[0]
    span = np.abs(rng.normal(0.0, 5e-5, size=n_bars))  # ~0.5 pip
    high = np.maximum(open_, close) + span
    low  = np.minimum(open_, close) - span

    # simple ATR proxy (mean absolute change) - FAST vectorized version
    abs_changes = np.abs(np.diff(close))
    atr = np.zeros(n_bars)
    for i in range(14, n_bars):
        atr[i] = np.mean(abs_changes[i-14:i])
    atr[:14] = atr[14]  # backfill first 14 bars

    data = pd.DataFrame({
        'time': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'atr': atr,
        'rsi': rng.uniform(0, 100, size=n_bars),
        'feature1': rng.standard_normal(n_bars),
        'feature2': rng.standard_normal(n_bars),
    }).set_index('time')

    feature_columns = ['open','high','low','close','atr','rsi','feature1','feature2']

    # sensible risk manager defaults for demo/smoke runs
    rm = RiskManager()
    try:
        rm.risk_per_trade = 0.005  # 0.5% per trade
        rm.hard_max_lots = 0.10    # cap at 0.10 lots for safety
    except Exception:
        pass

    env = ForexTradingEnv(
        data=data,
        feature_columns=feature_columns,
        initial_balance=1000.0,
        risk_manager=rm,
        spread=0.00014,
        commission=0.0,
        symbol='EURUSD'
    )

    print(f"\nEnvironment created:")
    print(f"  State size: {env.state_size}")
    print(f"  Action space: {env.action_space_size}")
    print(f"  Data length: {len(data)}")

    # Test episode (random actions)
    state = env.reset()
    print(f"\nInitial state shape: {state.shape}")

    total_reward = 0.0
    for i in range(300):
        action = np.random.randint(0, env.action_space_size)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    # Force flat so final equity is clean
    if env.position is not None:
        env._close_position(float(env.data['close'].iloc[env.current_step]))
        env._calculate_equity(float(env.data['close'].iloc[env.current_step]))

    print(f"\nEpisode completed:\n  Steps: {i+1}\n  Total reward: {total_reward:.2f}\n  Final equity: ${env.equity:.2f}")

    stats = env.get_trade_statistics()
    print("\nTrade statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

