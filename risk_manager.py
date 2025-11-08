"""
Risk Management Module
Handles position sizing, margin calculations, and risk constraints.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def round_step(x: float, step: float = 0.01, min_lot: float = 0.01, max_lot: float = 1.0) -> float:
    """
    Round lot size to broker step size and clamp to [min_lot, max_lot].
    
    Args:
        x: Raw lot size
        step: Rounding step (e.g., 0.01)
        min_lot: Minimum allowed lot size
        max_lot: Maximum allowed lot size
        
    Returns:
        Rounded lot size within bounds
    """
    if x <= 0:
        return 0.0
    rounded = round(x / step) * step
    return float(np.clip(rounded, min_lot, max_lot))


class RiskManager:
    """
    Manages position sizing, margin, and risk constraints for live trading.
    """
    
    def __init__(self,
                 contract_size: float = 100000,  # Standard lot size
                 point: float = 0.00001,  # Pip size for 5-digit broker
                 leverage: int = 100,
                 risk_per_trade: float = 0.02,  # 2% risk per trade
                 atr_multiplier: float = 2.0,  # SL distance in ATR units
                 max_dd_threshold: float = 0.20,  # 20% max drawdown
                 margin_safety_factor: float = 0.5,  # Keep 50% free margin
                 tp_multiplier: float = 3.0):  # TP = 3x SL (risk:reward)
        """
        Initialize risk manager.
        
        Args:
            contract_size: Contract size (100,000 for standard lot)
            point: Point size (0.00001 for 5-digit, 0.0001 for 4-digit)
            leverage: Account leverage
            risk_per_trade: Maximum risk per trade as fraction of balance
            atr_multiplier: Stop loss distance in ATR multiples
            max_dd_threshold: Maximum drawdown threshold for survivability
            margin_safety_factor: Minimum free margin to maintain
            tp_multiplier: Take profit multiplier relative to stop loss
        """
        self.contract_size = contract_size
        self.point = point
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier
        self.max_dd_threshold = max_dd_threshold
        self.margin_safety_factor = margin_safety_factor
        self.tp_multiplier = tp_multiplier
        
        # Pip is typically 10x point for 5-digit brokers
        self.pip = 10 * point

    def pip_size(self, symbol: str) -> float:
        """
        Typical pip size: 0.0001 for most pairs, 0.01 for JPY pairs.
        
        Args:
            symbol: Currency pair symbol
            
        Returns:
            Pip size in price units
        """
        return 0.01 if str(symbol).upper().endswith('JPY') else 0.0001

    def pip_value_usd(self, symbol: str, price: float, lots: float, contract_size: int = 100_000) -> float:
        """
        Approximate pip value in USD for pairs quoted vs USD.
        
        Args:
            symbol: Currency pair symbol
            price: Current market price
            lots: Position size in lots
            contract_size: Contract size (default: 100,000)
            
        Returns:
            Pip value in USD
        """
        ps = self.pip_size(symbol)
        # For pairs like EURUSD, pip value per lot ≈ (ps / price) * contract_size
        return (ps / max(price, 1e-9)) * contract_size * lots

    def calculate_position_size(self,
                               balance: float,
                               free_margin: float,
                               price: float,
                               atr: float,
                               symbol: str = 'EURUSD',
                               volume_min: float = 0.01,
                               volume_max: float = 100.0,
                               volume_step: float = 0.01) -> Dict:
        """
        Calculate optimal position size considering risk, margin, and survivability.
        
        Args:
            balance: Account balance
            free_margin: Available free margin
            price: Current market price
            atr: Current ATR value (in price units)
            symbol: Currency pair symbol
            volume_min: Minimum volume allowed by broker
            volume_max: Maximum volume allowed by broker
            volume_step: Volume step size
            
        Returns:
            Dict with position sizing details
        """
        # symbol-specific pip size (price units)
        ps = self.pip_size(symbol)

        # Ensure ATR and price are sane
        atr = max(float(atr), 0.0)
        price = max(float(price), 1e-9)

        # Stop distance in price units (use ATR multiple but enforce at least one pip)
        stop_distance_price = max(atr * self.atr_multiplier, ps)
        # Stop distance in pips
        stop_pips = stop_distance_price / ps

        # Risk amount in account currency
        risk_amount = balance * self.risk_per_trade

        # Pip value per lot in account currency (per 1 pip)
        pip_value_per_lot = self.pip_value_usd(symbol, price, lots=1.0, contract_size=self.contract_size)

        # Risk per lot = pip_value_per_lot * stop_pips
        risk_per_lot = pip_value_per_lot * stop_pips if pip_value_per_lot > 0 else float('inf')

        # Risk-based lot calculation (how many lots such that risk_amount is at most risk_per_lot * lots)
        lots_risk = (risk_amount / risk_per_lot) if risk_per_lot > 0 else 0.0

        # Margin-based lot calculation
        # margin_required_per_lot = (contract_size * price) / leverage
        margin_per_lot = (self.contract_size * price) / max(1, self.leverage)
        lots_margin = free_margin / margin_per_lot if margin_per_lot > 0 else 0.0

        # Drawdown survivability check
        balance_post_dd = balance * (1 - self.max_dd_threshold)
        required_free_margin = self.margin_safety_factor * balance_post_dd
        available_for_position = max(0.0, free_margin - required_free_margin)
        lots_dd_cap = available_for_position / margin_per_lot if margin_per_lot > 0 else 0.0

        # Take minimum of all constraints
        raw_lots = min(lots_risk, lots_margin, lots_dd_cap, volume_max)

        # If the computed lots are below broker minimum, do NOT forcibly bump to min; instead reject (0.0)
        rejected_due_to_min = False
        if raw_lots < volume_min:
            final_lots = 0.0
            rejected_due_to_min = True
        else:
            # Round to volume step
            final_lots = self._round_to_step(raw_lots, volume_step, volume_min)

        # Calculate actual margin required
        margin_required = final_lots * margin_per_lot

        # Take profit and stop loss distances (in price units and pips)
        sl_distance = stop_distance_price
        tp_distance = sl_distance * self.tp_multiplier

        # SAFETY: ensure lots is affordable and capped by account leverage and any hard caps
        try:
            max_affordable_lots = (free_margin * max(1, self.leverage)) / (self.contract_size * max(price, 1e-9))
        except Exception:
            max_affordable_lots = 0.0
        final_lots = min(final_lots, max_affordable_lots, volume_max)

        HARD_MAX_LOTS = getattr(self, 'hard_max_lots', volume_max)
        final_lots = min(final_lots, HARD_MAX_LOTS)
        final_lots = max(0.0, float(final_lots))

        # Pip-value for the returned lot size
        pip_value = pip_value_per_lot * final_lots

        return {
            'lots': final_lots,
            'rejected_due_to_min': rejected_due_to_min,
            'lots_risk': lots_risk,
            'lots_margin': lots_margin,
            'lots_dd_cap': lots_dd_cap,
            'margin_required': margin_required,
            'sl_pips': sl_distance / ps,
            'tp_pips': tp_distance / ps,
            'sl_distance': sl_distance,
            'tp_distance': tp_distance,
            'risk_amount': risk_amount,
            'pip_value': pip_value,
        }
    
    def calculate_sl_tp_prices(self,
                              entry_price: float,
                              position_type: str,
                              sl_distance: float,
                              tp_distance: float) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit prices.
        
        Args:
            entry_price: Entry price
            position_type: 'long' or 'short'
            sl_distance: Stop loss distance in price units
            tp_distance: Take profit distance in price units
            
        Returns:
            Tuple of (sl_price, tp_price)
        """
        if position_type.lower() == 'long':
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:  # short
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
        
        return sl_price, tp_price
    
    def calculate_new_sl_closer(self,
                               current_price: float,
                               current_sl: float,
                               position_type: str,
                               atr: float,
                               min_distance_factor: float = 0.5) -> Optional[float]:
        """
        Calculate new stop loss when moving it closer to current price.
        
        Args:
            current_price: Current market price
            current_sl: Current stop loss price
            position_type: 'long' or 'short'
            atr: Current ATR value
            min_distance_factor: Minimum distance as factor of ATR
            
        Returns:
            New stop loss price, or None if cannot move closer
        """
        min_distance = atr * self.atr_multiplier * min_distance_factor
        
        if position_type.lower() == 'long':
            # For long, SL is below price
            # Move SL up (closer to price) but maintain minimum distance
            new_sl = current_price - min_distance
            if new_sl > current_sl:  # Only if moving up
                return new_sl
        else:  # short
            # For short, SL is above price
            # Move SL down (closer to price) but maintain minimum distance
            new_sl = current_price + min_distance
            if new_sl < current_sl:  # Only if moving down
                return new_sl
        
        return None
    
    def check_margin_call_risk(self,
                              balance: float,
                              equity: float,
                              margin_used: float,
                              margin_call_level: float = 0.5) -> bool:
        """
        Check if account is at risk of margin call.
        
        Args:
            balance: Account balance
            equity: Account equity
            margin_used: Margin currently used
            margin_call_level: Margin level threshold (e.g., 0.5 = 50%)
            
        Returns:
            True if at risk of margin call
        """
        if margin_used == 0:
            return False
        
        margin_level = equity / margin_used
        return margin_level < margin_call_level
    
    def _round_to_step(self, value: float, step: float, minimum: float) -> float:
        """
        Round value to nearest step size, ensuring it's >= minimum and <= max.
        Uses global round_step for broker-realistic lot sizing.
        
        Args:
            value: Value to round
            step: Step size
            minimum: Minimum allowed value
            
        Returns:
            Rounded value
        """
        # Use global round_step helper with typical broker caps (min=0.01, max=1.0)
        return round_step(value, step=step, min_lot=minimum, max_lot=1.0)
    
    def calculate_pip_value(self, lots: float, pair: str = "EURUSD") -> float:
        """
        Calculate pip value for a given position size.
        
        Args:
            lots: Position size in lots
            pair: Currency pair (for future enhancement)
            
        Returns:
            Pip value in account currency
        """
        return self.contract_size * self.pip * lots
    
    def calculate_pnl(self,
                     entry_price: float,
                     exit_price: float,
                     lots: float,
                     position_type: str) -> float:
        """
        Calculate profit/loss for a closed position.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            lots: Position size in lots
            position_type: 'long' or 'short'
            
        Returns:
            PnL in account currency
        """
        price_diff = exit_price - entry_price
        
        if position_type.lower() == 'short':
            price_diff = -price_diff
        
        pnl = price_diff * self.contract_size * lots
        return pnl

    def _expected_rt_cost(self, lots: float, price: float, spread: float, commission: float) -> float:
        """
        Calculate expected round-trip cost for a position.
        
        Args:
            lots: Position size in lots
            price: Current market price  
            spread: Broker spread (in price units)
            commission: Commission per lot per side
            
        Returns:
            Expected total cost (spread + commission for open + close)
        """
        if lots <= 0:
            return 0.0
            
        # Spread cost = lots * contract_size * spread
        spread_cost = lots * self.contract_size * spread
        
        # Commission cost = 2 sides * lots * commission per lot per side
        commission_cost = 2 * lots * commission
        
        return spread_cost + commission_cost

    def _maybe_end_on_budget(self, balance: float, lots: float, price: float, 
                            spread: float, commission: float) -> bool:
        """
        Check if position would exceed cost budget and should be rejected.
        
        Args:
            balance: Current account balance
            lots: Proposed position size
            price: Current market price
            spread: Broker spread
            commission: Commission rate
            
        Returns:
            True if position should be rejected due to budget constraints
        """
        if lots <= 0:
            return False
            
        cost_budget = balance * self.cost_budget_pct
        expected_cost = self._expected_rt_cost(lots, price, spread, commission)
        
        return expected_cost > cost_budget

    def compute_lots_enhanced(self, balance: float, free_margin: float, price: float,
                            atr: float, spread: float, commission: float,
                            symbol: str = 'EURUSD', volume_min: float = 0.01,
                            volume_max: float = 100.0, volume_step: float = 0.01) -> Dict:
        """
        Enhanced position sizing with cost budget and survivability checks.
        
        Args:
            balance: Account balance
            free_margin: Available free margin
            price: Current market price
            atr: Current ATR value
            spread: Broker spread
            commission: Commission per lot per side
            symbol: Currency pair symbol
            volume_min: Minimum volume
            volume_max: Maximum volume  
            volume_step: Volume step
            
        Returns:
            Enhanced position sizing details with budget checks
        """
        # Start with base calculation
        base_result = self.calculate_position_size(
            balance, free_margin, price, atr, symbol, volume_min, volume_max, volume_step
        )
        
        lots = base_result['lots']
        
        # Apply cost budget constraint
        budget_rejected = False
        if lots > 0:
            if self._maybe_end_on_budget(balance, lots, price, spread, commission):
                # Try to reduce lots to fit budget
                cost_budget = balance * self.cost_budget_pct
                
                # Binary search for maximum affordable lots within budget
                low, high = 0.0, lots
                affordable_lots = 0.0
                
                for _ in range(20):  # Max iterations
                    mid = (low + high) / 2
                    cost = self._expected_rt_cost(mid, price, spread, commission)
                    
                    if cost <= cost_budget:
                        affordable_lots = mid
                        low = mid
                    else:
                        high = mid
                        
                    if (high - low) < volume_step:
                        break
                        
                # Round to step and ensure minimum
                if affordable_lots >= volume_min:
                    lots = self._round_to_step(affordable_lots, volume_step, volume_min)
                else:
                    lots = 0.0
                    budget_rejected = True
        
        # Enhanced survivability check with DD tolerance
        survivability_rejected = False
        if lots > 0:
            # Check if position can survive max_dd_survivability drawdown
            worst_case_balance = balance * (1 - self.max_dd_survivability)
            margin_per_lot = (self.contract_size * price) / max(1, self.leverage)
            total_margin_required = lots * margin_per_lot
            
            # Ensure sufficient margin would remain even in worst case
            required_free_margin_buffer = worst_case_balance * self.margin_safety_factor
            
            if (worst_case_balance - total_margin_required) < required_free_margin_buffer:
                # Position too large for survivability
                max_survivable_lots = max(0.0, 
                    (worst_case_balance - required_free_margin_buffer) / margin_per_lot)
                
                if max_survivable_lots >= volume_min:
                    lots = min(lots, self._round_to_step(max_survivable_lots, volume_step, volume_min))
                else:
                    lots = 0.0
                    survivability_rejected = True
        
        # Update result with enhanced checks
        enhanced_result = base_result.copy()
        enhanced_result.update({
            'lots': lots,
            'budget_rejected': budget_rejected,
            'survivability_rejected': survivability_rejected,
            'expected_cost': self._expected_rt_cost(lots, price, spread, commission),
            'cost_budget': balance * self.cost_budget_pct,
            'cost_budget_pct': self.cost_budget_pct
        })
        
        return enhanced_result

    # Add cost_budget_pct and max_dd_survivability as class attributes with defaults
    cost_budget_pct: float = 0.15  # Default if not set in config
    max_dd_survivability: float = 0.40  # Default if not set in config

