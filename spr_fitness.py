"""
SPR (Sharpe-PF-Recovery) Fitness Module
Composite fitness metric for trading strategies that avoids over-reliance on Sharpe ratio.

Components:
- Profit Factor (PF): Gross profit / Gross loss
- Max Drawdown (MDD): Largest peak-to-trough decline
- Monthly Mean Return (MMR): Average monthly gain as % of initial balance
- Significance: Scales with trade frequency (guards against low-sample luck)
- Stagnation Penalty: Penalizes long periods without new equity highs

Formula: SPR = (PF / MDD%) × MMR% × Significance × Stagnation_Penalty
"""

import math
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Any, Union, Optional


def _to_datetime64(ts: List) -> np.ndarray:
    """
    Convert various timestamp formats to numpy datetime64[ns].
    
    Accepts:
    - pandas Timestamps
    - python datetime objects
    - numpy datetime64
    - POSIX seconds (float/int)
    """
    if len(ts) == 0:
        return np.array([], dtype="datetime64[ns]")
    
    # pandas Timestamp
    if hasattr(ts[0], "to_datetime64"):
        return np.array([t.to_datetime64() for t in ts], dtype="datetime64[ns]")
    
    # numpy datetime64
    if isinstance(ts[0], np.datetime64):
        return np.array(ts, dtype="datetime64[ns]")
    
    # python datetime
    if isinstance(ts[0], datetime):
        return np.array(ts, dtype="datetime64[ns]")
    
    # assume POSIX seconds (float/int)
    return (np.array(ts, dtype="float64") * 1e9).astype("datetime64[ns]")


def _max_drawdown_pct(equity: np.ndarray) -> float:
    """
    Calculate maximum drawdown as percentage of running peak.
    
    MDD = max((Peak - Trough) / Peak)
    
    Args:
        equity: Equity curve array
        
    Returns:
        Maximum drawdown as percentage (0-100)
    """
    if len(equity) == 0:
        return 0.0
    
    equity = np.asarray(equity, dtype=float)
    peaks = np.maximum.accumulate(equity)
    drawdowns = (peaks - equity) / np.maximum(peaks, 1e-9)
    
    return float(drawdowns.max() * 100.0)


def _profit_factor(trade_pnls: np.ndarray, cap: float = 10.0) -> float:
    """
    Calculate Profit Factor with optional cap.
    
    PF = Gross Profit / Gross Loss
    
    Args:
        trade_pnls: Array of individual trade P&Ls
        cap: Maximum PF value to prevent outliers
        
    Returns:
        Profit Factor (capped)
    """
    if len(trade_pnls) == 0:
        return 0.0
    
    pnls = np.asarray(trade_pnls, dtype=float)
    
    gross_profit = pnls[pnls > 0].sum()
    gross_loss = -pnls[pnls < 0].sum()
    
    # No profit, no loss -> neutral
    if gross_profit <= 0 and gross_loss <= 0:
        return 0.0
    
    # Profit but no loss -> cap at maximum
    if gross_loss <= 0 and gross_profit > 0:
        return float(cap)
    
    # Normal case: profit / loss
    pf = gross_profit / gross_loss
    return float(min(cap, pf))


def _monthly_return_pct_mean(
    timestamps: np.ndarray,
    equity: np.ndarray,
    initial_balance: float,
    use_pandas: bool = True
) -> float:
    """
    Calculate mean monthly return as % of initial balance.
    
    Uses pandas for accurate monthly resampling if available,
    falls back to manual bucketing otherwise.
    
    Args:
        timestamps: Array of datetime64 timestamps
        equity: Equity curve array
        initial_balance: Starting balance
        use_pandas: Whether to use pandas for resampling
        
    Returns:
        Mean monthly return as percentage of initial balance
    """
    if len(timestamps) == 0 or len(equity) == 0:
        return 0.0
    
    ts = _to_datetime64(timestamps)
    eq = np.asarray(equity, dtype=float)
    
    # Try pandas approach first (most accurate)
    if use_pandas:
        try:
            import pandas as pd
            
            idx = pd.to_datetime(ts)
            series = pd.Series(eq, index=idx)
            
            # Resample to month-end, take last value (ME = Month-End, M deprecated)
            month_end = series.resample("ME").last()
            
            # Calculate month-to-month changes
            delta = month_end.diff().dropna()
            
            if delta.empty:
                return 0.0
            
            # Return mean monthly change as % of initial balance
            return float((delta / float(initial_balance) * 100.0).mean())
            
        except Exception:
            pass  # Fall through to manual approach
    
    # Manual bucketing by (year, month)
    ts_dates = ts.astype("datetime64[D]")
    
    # Extract year-month tuples
    year_months = []
    for d in ts_dates:
        date_str = str(d)
        year = int(date_str[:4])
        month = int(date_str[5:7])
        year_months.append((year, month))
    
    # Get last equity value for each month
    month_last = {}
    for ym, value in zip(year_months, eq):
        month_last[ym] = value
    
    # Sort months chronologically
    months = sorted(month_last.keys())
    
    if len(months) < 2:
        return 0.0
    
    # Get equity values in order
    month_values = [month_last[m] for m in months]
    
    # Calculate month-to-month deltas
    deltas = np.diff(month_values)
    
    # Return mean as % of initial balance
    return float((deltas / float(initial_balance) * 100.0).mean())


def _days_between(
    t_start: Optional[np.datetime64],
    t_end: Optional[np.datetime64],
    n_bars: int,
    seconds_per_bar: float
) -> float:
    """
    Calculate days between timestamps with fallback.
    
    Args:
        t_start: Start timestamp (or None)
        t_end: End timestamp (or None)
        n_bars: Number of bars (fallback)
        seconds_per_bar: Bar duration in seconds (fallback)
        
    Returns:
        Number of days
    """
    if t_start is not None and t_end is not None:
        try:
            days = (t_end - t_start) / np.timedelta64(1, "D")
            if np.isfinite(days) and days > 0:
                return float(days)
        except Exception:
            pass
    
    # Fallback: estimate from bar count
    return float(n_bars * seconds_per_bar / 86400.0)


def _stagnation_days(timestamps: np.ndarray, equity: np.ndarray) -> float:
    """
    Calculate days since last equity peak.
    
    Args:
        timestamps: Array of datetime64 timestamps
        equity: Equity curve array
        
    Returns:
        Days since last peak (0 if currently at peak)
    """
    if len(equity) == 0:
        return 0.0
    
    ts = _to_datetime64(timestamps)
    eq = np.asarray(equity, dtype=float)
    
    # Find last occurrence of peak equity
    peak_value = eq.max()
    last_peak_idx = np.where(eq == peak_value)[0][-1]
    
    # Days from last peak to end
    last_peak_time = ts[last_peak_idx]
    end_time = ts[-1]
    
    days = (end_time - last_peak_time) / np.timedelta64(1, "D")
    
    return float(max(0.0, days))


def compute_spr_fitness(
    timestamps: List,
    equity_curve: List[float],
    trade_pnls: Optional[List[float]] = None,
    initial_balance: float = 1000.0,
    seconds_per_bar: float = 3600.0,
    *,
    pf_override: Optional[float] = None,
    trade_count_override: Optional[int] = None,
    pf_cap: float = 10.0,
    dd_floor_pct: float = 0.05,
    target_trades_per_year: float = 100.0,
    use_pandas: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute SPR (Sharpe-PF-Recovery) composite fitness score.
    
    SPR = (PF / MDD%) × MMR% × Significance × Stagnation_Penalty
    
    Where:
    - PF = Profit Factor (gross profit / gross loss, capped)
    - MDD% = Maximum Drawdown percentage (floored to avoid div/0)
    - MMR% = Mean Monthly Return as % of initial balance
    - Significance = (min(1, trades_per_year / target)) ^ 2
    - Stagnation_Penalty = 1 - (stagnation_days / test_days)
    
    Args:
        timestamps: List of bar timestamps (datetime, pd.Timestamp, or epoch seconds)
        equity_curve: List of equity values over time
        trade_pnls: List of realized P&L per closed trade (optional)
        initial_balance: Starting account balance
        seconds_per_bar: Duration of each bar in seconds
        pf_override: Override PF with equity-based fallback (optional)
        trade_count_override: Override trade count when trade_pnls unavailable (optional)
        pf_cap: Maximum Profit Factor (prevents outliers)
        dd_floor_pct: Minimum drawdown % (prevents div/0)
        target_trades_per_year: Target annual trade count for significance
        use_pandas: Use pandas for monthly resampling if available
        
    Returns:
        Tuple of (score, extras_dict)
        - score: SPR fitness value
        - extras: Dict with component breakdowns
    """
    # Convert to numpy arrays
    eq = np.asarray(equity_curve, dtype=float)
    ts = _to_datetime64(timestamps)
    pnls = np.asarray(trade_pnls, dtype=float) if trade_pnls else np.array([], dtype=float)
    
    # Core components
    # PATCH 2: Use pf_override if provided (equity-based PF when no trade P&Ls)
    if pf_override is not None:
        pf = float(pf_override)
    else:
        pf = _profit_factor(pnls, cap=pf_cap)
    mdd_pct = max(dd_floor_pct, _max_drawdown_pct(eq))
    mmr_pct_mean = _monthly_return_pct_mean(ts, eq, initial_balance, use_pandas)
    
    # Base score: (PF / MDD%) × MMR%
    base = (pf / mdd_pct) * mmr_pct_mean
    
    # Significance factor (trade frequency guard)
    test_days = _days_between(
        ts[0] if len(ts) > 0 else None,
        ts[-1] if len(ts) > 0 else None,
        len(eq),
        seconds_per_bar
    )
    
    # PATCH 2: Use trade_count_override if provided when pnls list is empty
    actual_trades = trade_count_override if trade_count_override is not None else len(pnls)
    trades_per_day = (actual_trades / test_days) if test_days > 0 else 0.0
    trades_per_year = trades_per_day * 252.0  # Trading days
    
    significance = (min(1.0, trades_per_year / target_trades_per_year) ** 2) if actual_trades > 1 else 0.0
    
    # Stagnation penalty (favors consistent growth)
    stagnation_days_val = _stagnation_days(ts, eq)
    stagnation_penalty = max(0.0, 1.0 - (stagnation_days_val / test_days)) if test_days > 0 else 0.0
    
    # Final SPR score
    score = base * significance * stagnation_penalty
    
    # Component breakdown for diagnostics
    extras = {
        "pf": float(pf),
        "mdd_pct": float(mdd_pct),
        "mmr_pct_mean": float(mmr_pct_mean),
        "trades_per_year": float(trades_per_year),
        "significance": float(significance),
        "stagnation_days": float(stagnation_days_val),
        "stagnation_penalty": float(stagnation_penalty),
        "spr_base": float(base),
        "test_days": float(test_days),
    }
    
    return float(score), extras


if __name__ == "__main__":
    """Quick test of SPR computation."""
    
    # Synthetic test data
    timestamps = [datetime(2023, 1, 1) + np.timedelta64(i, "D") for i in range(100)]
    equity = [1000.0 + i * 2.5 for i in range(100)]  # Steady growth
    trade_pnls = [10.0, -5.0, 15.0, -3.0, 20.0] * 10  # Mixed trades
    
    score, extras = compute_spr_fitness(
        timestamps=timestamps,
        equity_curve=equity,
        trade_pnls=trade_pnls,
        initial_balance=1000.0,
        seconds_per_bar=86400.0,  # Daily bars
        pf_cap=10.0,
        dd_floor_pct=0.05,
        target_trades_per_year=100.0,
        use_pandas=True,
    )
    
    print("SPR Fitness Test")
    print("=" * 50)
    print(f"Score: {score:.6f}")
    print("\nComponents:")
    for key, value in extras.items():
        print(f"  {key}: {value:.4f}")
