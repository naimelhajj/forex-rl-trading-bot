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
    - 1: LONG
    - 2: SHORT
    - 3: MOVE_SL_CLOSER
    """
    
    def __init__(self,
                 data: pd.DataFrame,
                 feature_columns: List[str],
                 initial_balance: float = 1000.0,
                 risk_manager: RiskManager = None,
                 spread: float = 0.00020,  # 2 pips spread
                 commission: float = 0.0,  # Commission per lot per side
                 slippage_pips: float = 0.8,  # Slippage in pips
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
                 min_trail_buffer_pips: float = 1.0):
        """
        Initialize trading environment.
        
        Args:
            fx_lookup: Dict of {pair: pd.Series} with close prices for USD conversion
            scaler_mu: Feature means for normalization (from training data)
            scaler_sig: Feature stds for normalization (from training data)
            slippage_pips: Slippage in pips
            cooldown_bars: Bars to wait before allowing new position
            min_hold_bars: Minimum bars to hold a position
            trade_penalty: Penalty per trade (normalized)
            flip_penalty: Additional penalty for immediate reversals
            max_trades_per_episode: Maximum trades allowed per episode
            risk_per_trade: Risk per trade as fraction of equity (0.0075 = 0.75%)
            atr_mult_sl: ATR multiplier for stop loss (2.5)
            tp_mult: TP multiplier relative to SL distance (2.0)
            min_trail_buffer_pips: PATCH #4: Minimum pips required for meaningful SL tightening (1.0)
        """
        self.data = data.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.initial_balance = initial_balance
        self.fx_lookup = fx_lookup if fx_lookup is not None else {}  # NEW
        self.spread = spread
        self.commission = commission
        self.slippage_pips = slippage_pips
        self.weekend_close_hours = weekend_close_hours
        self.max_steps = max_steps if max_steps else len(data)
        self.symbol = symbol
        
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
        self.max_trades_per_episode = max_trades_per_episode
        self.risk_per_trade = risk_per_trade
        self.atr_mult_sl = atr_mult_sl
        self.tp_mult = tp_mult
        self.min_trail_buffer_pips = min_trail_buffer_pips  # PATCH #4
        self.trades_this_ep = 0
        self._accum_entry_cost = 0.0
        
        # Cost accounting: use EITHER extra_entry_penalty OR equity-based costs, not both
        self.extra_entry_penalty = False  # Set to True to add normalized entry costs to reward
        
        # simple holiday set (dates to force flatten) - extend as needed
        self.holidays = set([
            pd.Timestamp('2024-01-01').date(),
            pd.Timestamp('2024-12-25').date(),
        ])

        # Risk manager
        self.risk_manager = risk_manager if risk_manager else RiskManager()
        
        # Structured logger (optional - set by trainer)
        self.structured_logger = None

        # State variables
        self.current_step = 0
        self.balance = initial_balance
        self.equity = initial_balance
        self.prev_equity = initial_balance
        self.position = None  # Dict with position info
        self.trade_history = []
        self.equity_history = [initial_balance]
        
        # Action space
        self.action_space_size = 4
        
        # PATCH #2: State space size with frame stacking
        # State = stacked market features (stack_n * feature_dim) + 23 portfolio features
        self.feature_dim = len(feature_columns)
        self.context_dim = 23  # Portfolio features
        self.state_size = self.feature_dim * self.stack_n + self.context_dim
        
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        PATCH #2: Initialize frame stack.
        
        Returns:
            Initial state
        """
        self.current_step = 0
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
        # SURGICAL PATCH: Reset cost budget tracking
        self.costs_this_ep = 0.0
        self.trading_locked = False
        self.cost_budget_pct = 0.05  # 5% of initial balance
        # reset churn control state
        self.bars_in_position = 0
        self.bars_since_close = 0
        self.last_action = [0, 0, 0, 0]
        
        # PATCH #2: Reset frame stack
        self._frame_stack = None

        return self._get_state()
        self.trading_locked = False
        self.cost_budget_pct = 0.05  # 5% of initial balance
        # reset churn control state
        self.bars_in_position = 0
        self.bars_since_close = 0
        self.last_action = [0, 0, 0, 0]

        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0=HOLD, 1=LONG, 2=SHORT, 3=MOVE_SL_CLOSER)
            
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
        
        # Update existing position (check SL/TP)
        if self.position is not None:
            pnl, hit_sl_tp = self._update_position(current_price)
            if hit_sl_tp or should_flatten:
                reward += pnl
                self.position = None
        
        # Execute action (with cooldown gating)
        if action == 0:  # HOLD
            pass

        elif action == 1:  # LONG
            # Check max trades per episode limit
            if self.trades_this_ep >= self.max_trades_per_episode:
                pass  # Ignore action if max trades reached
            # flip only if cooldown allows
            elif self.position is not None and self.position['type'] == 'short':
                if self._can_modify():
                    pnl, _ = self._close_position(current_price)
                    reward += pnl
                    did_trade = True
                    # flip penalty for immediate reversal
                    reward -= float(self.flip_penalty)
                    self.trades_this_ep += 1
                    if self.trades_this_ep < self.max_trades_per_episode:
                        self._open_position('long', current_price, current_atr)
                        did_trade = True
                        self.trades_this_ep += 1
                    # account for expected round-trip cost on entry (normalized) - ONLY if extra_entry_penalty enabled
                    if self.extra_entry_penalty:
                        try:
                            lots = self.position.get('lots', 0.0)
                            rtc = (self.spread * pip_value_usd(self.symbol, current_price, lots) + 2.0 * self.commission * lots)
                            self._accum_entry_cost += rtc / max(self.equity, 1e-9)
                        except Exception:
                            pass
            elif self.position is None and self.trades_this_ep < self.max_trades_per_episode:
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

        elif action == 2:  # SHORT
            # Check max trades per episode limit
            if self.trades_this_ep >= self.max_trades_per_episode:
                pass  # Ignore action if max trades reached
            elif self.position is not None and self.position['type'] == 'long':
                if self._can_modify():
                    pnl, _ = self._close_position(current_price)
                    reward += pnl
                    did_trade = True
                    # flip penalty
                    reward -= float(self.flip_penalty)
                    self.trades_this_ep += 1
                    if self.trades_this_ep < self.max_trades_per_episode:
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
            elif self.position is None and self.trades_this_ep < self.max_trades_per_episode:
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
        elif action == 3:  # MOVE_SL_CLOSER / FLATTEN (if cooldown allows)
            if self.position is not None:
                # if cooldown allows, permit closure; else move SL closer
                if self._can_modify():
                    pnl, _ = self._close_position(current_price)
                    reward += pnl
                    did_trade = True
                else:
                    # Use enhanced trailing stop method with recent price data
                    try:
                        recent_data = self.data.iloc[max(0, self.current_step-20):self.current_step+1]
                        if len(recent_data) > 5:  # Ensure sufficient data
                            self._enhanced_move_sl_closer(recent_data)
                        else:
                            # Fallback to simple method
                            self._move_sl_closer(current_price, current_atr)
                    except Exception:
                        # Fallback to simple method on any error
                        self._move_sl_closer(current_price, current_atr)
        
        # Enforce weekend rules (flatten positions if weekend approaching)
        weekend_pnl = self._enforce_weekend_rules(current_price)
        reward += weekend_pnl
        
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
        done = (self.current_step >= min(len(self.data) - 1, self.max_steps) or 
                self.equity <= 0.05 * self.initial_balance)  # Ruin condition
        
        # SURGICAL PATCH: Cost budget kill-switch
        costs = getattr(self, 'costs_this_ep', 0.0)
        budget = getattr(self, 'cost_budget_pct', 0.05) * self.initial_balance
        
        # PATCH 10: Assert cost budget (catch frictions drift)
        # Allow up to 5% of initial balance in spread+commission costs per episode
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
        }
        
        # after price update and potential open/close, compute reward as log-return
        prev = max(self.prev_equity, 1e-9)
        curr = max(self.equity, 1e-9)
        reward = np.log(curr / prev)
        # clip reward to ±0.01 for tighter Q-value stability
        reward = float(np.clip(reward, -0.01, 0.01))
        
        # PATCH 5: Tiny holding cost to discourage churn (only while in position)
        # ~2.5 bps per day on H1 bars (24 bars/day -> 1e-4/bar = 0.0024/day)
        if self.position is not None:
            reward -= 1e-4
        
        # apply small trade penalty if the action changed position
        if did_trade:
            reward -= float(self.trade_penalty)
        # subtract accumulated entry cost (normalized) - only if extra_entry_penalty is True
        if self.extra_entry_penalty and getattr(self, '_accum_entry_cost', 0.0) > 0.0:
            reward -= float(self._accum_entry_cost)
            self._accum_entry_cost = 0.0
        # cap trades per episode to avoid runaway churn
        if getattr(self, 'trades_this_ep', 0) > 120:
            # force flatten and signal done by setting current_step to max
            if self.position is not None:
                self._close_position(current_price)
                self._calculate_equity(current_price)
            done = True
        self.prev_equity = curr
        
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
        self.last_action = [0, 0, 0, 0]
        self.last_action[action] = 1

        return next_state, reward, done, info
    
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
        
        # Last action one-hot (4 actions: HOLD, LONG, SHORT, MOVE_SL_CLOSER)
        last_action_onehot = np.zeros(4, dtype=np.float32)
        if hasattr(self, 'last_action') and len(self.last_action) == 4:
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
            *last_action_onehot  # 4 elements
        ], dtype=np.float32)  # Total: 23 features
        
        # Light global clip on the 23-d portfolio block (broad safety rails)
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
            State vector: stacked normalized market features + 23 portfolio features
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
        
        # Portfolio features (23 balance-invariant features)
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
        slippage_pips_base = getattr(self.risk_manager, 'slippage_pips', 0.5)
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
    
    def _move_sl_closer(self, price: float, atr: float):
        """
        PATCH 4: Enhanced MOVE_SL_CLOSER with breakeven + ATR trailing.
        Two-stage approach:
        1. If RR >= 1.0 and SL not at entry -> move to breakeven (entry - 0.2 pip)
        2. Else if in profit -> trail at (close - k*ATR) with k=0.6
        
        Args:
            price: Current price
            atr: Current ATR
        """
        if self.position is None:
            return
        
        ps = pip_size(self.symbol)
        entry = self.position['entry']
        current_sl = self.position['sl']
        pos_type = self.position['type']
        
        # Calculate current RR (risk-reward ratio)
        if pos_type == 'long':
            risk_pips = (entry - current_sl) / ps
            reward_pips = (price - entry) / ps
        else:  # short
            risk_pips = (current_sl - entry) / ps
            reward_pips = (entry - price) / ps
        
        rr_ratio = reward_pips / (risk_pips + 1e-9)
        
        # Stage 1: Move to breakeven if RR >= 1.0
        if rr_ratio >= 1.0:
            if pos_type == 'long' and current_sl < entry:
                # Move SL to entry - 0.2 pip (avoid immediate stop)
                new_sl = entry - 0.2 * ps
                if new_sl > current_sl:  # Only move closer (tighter)
                    self.position['sl'] = new_sl
                    return
            elif pos_type == 'short' and current_sl > entry:
                new_sl = entry + 0.2 * ps
                if new_sl < current_sl:
                    self.position['sl'] = new_sl
                    return
        
        # Stage 2: ATR-based trailing if in profit
        if reward_pips > 0:
            atr_pips = max(1.0, atr / ps)
            k = 0.6  # ATR multiplier for trail distance
            
            if pos_type == 'long':
                # Trail: SL = price - k*ATR
                new_sl = price - k * atr_pips * ps
                if new_sl > current_sl:  # Only move closer
                    self.position['sl'] = new_sl
            else:  # short
                # Trail: SL = price + k*ATR
                new_sl = price + k * atr_pips * ps
                if new_sl < current_sl:  # Only move closer
                    self.position['sl'] = new_sl
    
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
    
    def _should_flatten_for_weekend(self, current_data: pd.Series) -> bool:
        current_time = None
        # prefer explicit time column if present
        if 'time' in self.data.columns:
            try:
                current_time = pd.to_datetime(self.data.loc[self.current_step, 'time'])
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
        Returns a boolean mask [HOLD, LONG, SHORT, MOVE_SL_CLOSER]
        True = allowed, False = illegal right now.
        """
        hold_ok = True
        
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
        move_sl_ok = False
        
        if self.position is None:
            # No position → can open long/short, cannot move SL
            move_sl_ok = False
        else:
            # Enforce min-hold: cannot flip while in cooldown or min-hold period
            if self.bars_in_position < self.min_hold_bars or not self._can_modify():
                # Allow MOVE_SL_CLOSER but disallow flipping direction
                if self.position['type'] == 'long':
                    short_ok = False  # Cannot flip to short
                    long_ok = True    # Can stay long (hold)
                else:
                    long_ok = False   # Cannot flip to long
                    short_ok = True   # Can stay short (hold)
            # Can tighten SL only if SL exists and still > min step away
            move_sl_ok = self._can_tighten_sl()
            
            # Don't allow opening same side again (redundant with logic above, but explicit)
            if self.position['type'] == 'long':
                long_ok = False
            else:
                short_ok = False
        
        return [hold_ok, long_ok, short_ok, move_sl_ok]
    
    def _can_tighten_sl(self) -> bool:
        """
        PATCH #4: Check if SL can be tightened meaningfully.
        Requires at least min_trail_buffer_pips of tightening room.
        """
        if self.position is None or self.position.get('sl') is None:
            return False
        
        # Get current price
        if self.current_step >= len(self.data):
            return False
        current_price = float(self.data.iloc[self.current_step]['close'])
        
        # PATCH #4: Require at least min_trail_buffer_pips of meaningful tightening
        min_buffer = getattr(self, 'min_trail_buffer_pips', 1.0)  # Default 1 pip
        ps = pip_size(self.symbol)
        min_step = ps * min_buffer
        
        if self.position['type'] == 'long':
            # For long, new SL must be at least min_buffer pips closer than current SL
            proposed_sl = current_price - 2 * ps  # Leave 2 pips breathing room
            return (proposed_sl - self.position['sl']) > min_step
        else:
            # For short, new SL must be at least min_buffer pips closer than current SL
            proposed_sl = current_price + 2 * ps  # Leave 2 pips breathing room
            return (self.position['sl'] - proposed_sl) > min_step
    
    def _tighten_sl(self):
        """
        SURGICAL PATCH: Tighten SL by 33% of remaining distance toward price.
        Always leaves 2 pips breathing room.
        """
        if not self._can_tighten_sl():
            return
        
        if self.current_step >= len(self.data):
            return
            
        current_price = float(self.data.iloc[self.current_step]['close'])
        k = 0.33  # Tighten 33% of remaining distance toward price
        ps = pip_size(self.symbol)
        
        if self.position['type'] == 'long':
            target = current_price - 2 * ps  # Keep 2 pips breathing room
            new_sl = max(self.position['sl'], self.position['sl'] + k * (target - self.position['sl']))
        else:
            target = current_price + 2 * ps
            new_sl = min(self.position['sl'], self.position['sl'] + k * (target - self.position['sl']))
        
        self.position['sl'] = float(new_sl)
    
    def _can_modify(self) -> bool:
        """Return True if we can close/flip the current position (cooldown)."""
        if self.position is None:
            return True
        age = self.position.get('age', 0)
        required = max(getattr(self, 'cooldown_bars', 12), getattr(self, 'min_hold_bars', 6))
        return age >= required
    
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
        Enhanced MOVE_SL_CLOSER with CAUSAL (confirmed) fractals and ATR-based trailing stops.
        
        Args:
            price_data: Recent price data DataFrame (must include *_confirmed fractal columns)
            
        Returns:
            True if stop loss was moved, False otherwise
        """
        if not self.position:
            return False
            
        current_price = price_data['close'].iloc[-1]
        current_sl = self.position['sl']
        position_type = self.position['type']
        current_atr = price_data.get('atr', pd.Series([0.001])).iloc[-1]
        current_atr = max(current_atr, 1e-6)
        
        # Get CAUSAL (confirmed) fractals from data
        top_frac_confirmed = price_data.get('top_fractal_confirmed', pd.Series([np.nan]))
        bottom_frac_confirmed = price_data.get('bottom_fractal_confirmed', pd.Series([np.nan]))
        
        new_sl = None
        atr_padding = 0.5 * current_atr
        
        if position_type == 'long':
            # For long positions, use bottom_fractal_confirmed - atr_padding as SL
            recent_bottom = bottom_frac_confirmed.iloc[-1]
            if not np.isnan(recent_bottom):
                fractal_sl = recent_bottom - atr_padding
                new_sl = fractal_sl
            else:
                # Fallback: simple ATR trailing
                new_sl = current_price - 2.0 * current_atr
            
            # Only move SL up, never down
            if new_sl > current_sl:
                self.position['sl'] = new_sl
                
                # Log SL movement
                if getattr(self, 'structured_logger', None) is not None:
                    try:
                        timestamp = datetime.now()
                        self.structured_logger.log_trade_sl_move(
                            timestamp=timestamp,
                            old_sl=current_sl,
                            new_sl=new_sl,
                            current_price=current_price,
                            step=self.current_step,
                            method="confirmed_fractal_atr"
                        )
                    except Exception:
                        pass
                        
                return True
                
        elif position_type == 'short':
            # For short positions, use top_fractal_confirmed + atr_padding as SL
            recent_top = top_frac_confirmed.iloc[-1]
            if not np.isnan(recent_top):
                fractal_sl = recent_top + atr_padding
                new_sl = fractal_sl
            else:
                # Fallback: simple ATR trailing
                new_sl = current_price - self.atr_mult_sl * current_atr
        
        # For SHORT: only move SL down, never up  
        if new_sl < current_sl:
            self.position['sl'] = float(new_sl)
            
            # Log SL movement
            if getattr(self, 'structured_logger', None) is not None:
                try:
                    timestamp = datetime.now()
                    self.structured_logger.log_trade_sl_move(
                        timestamp=timestamp,
                        old_sl=current_sl,
                        new_sl=new_sl,
                        current_price=current_price,
                        step=self.current_step,
                        method="confirmed_fractal_atr"
                    )
                except Exception:
                    pass
            
            return True
        
        return False


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
        spread=0.00015,
        commission=7.0,
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
        action = np.random.randint(0, 4)
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

