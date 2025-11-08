"""
Fitness Metric Module
Implements stability-adjusted fitness score for evaluating trading performance.
PATCH #1: Compute fitness on same equity series after ruin-clamp with business-day resample.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

def _years(idx: pd.DatetimeIndex) -> float:
    """Time-based CAGR calculation using actual timestamps."""
    s, e = pd.Timestamp(idx[0]), pd.Timestamp(idx[-1])
    return max((e - s).total_seconds() / (365.2425*24*3600), 1e-9)

def _ruin_clamp(equity: pd.Series, ruin_frac=0.05) -> Tuple[pd.Series, bool]:
    """Truncate equity at ruin threshold."""
    thr = ruin_frac * float(equity.iloc[0])
    ruined = (equity <= thr).any()
    if ruined:
        cut = equity.index[(equity <= thr)][0]
        equity = equity.loc[:cut]
    return equity, ruined

def stability_fitness(equity: pd.Series, w=(1.0, 2.0, -1.0, -1.0), ruin_penalty=-5.0,
                     clip_sharpe=5.0, clip_cagr=1.0, min_bdays=60):
    """
    PATCH #1: Bug-proof fitness computation with small-sample robustness.
    Compute all metrics on SAME equity series AFTER ruin-clamp with business-day resample.
    Caps Sharpe/CAGR on small samples and applies coverage scaling.
    
    Args:
        equity: Equity series with DatetimeIndex
        w: Weights (sharpe, cagr, stagnation, loss_years)
        ruin_penalty: Penalty if ruined
        clip_sharpe: Cap |Sharpe| to prevent explosion on small samples (default 5.0)
        clip_cagr: Cap |CAGR| to prevent explosion on small samples (default 1.0 = 100%)
        min_bdays: Minimum business days for stable Sharpe (default 60)
        
    Returns:
        (fitness_score, metrics_dict)
    """
    equity = equity.dropna()
    equity, ruined = _ruin_clamp(equity)

    # Daily sampling BEFORE metrics
    daily = equity.resample('1B').last().ffill()
    r = daily.pct_change().dropna()
    bdays = len(r)
    
    if bdays < 3: 
        return 0.0, {'Sharpe':0.0,'CAGR':0.0,'Stagnation':1.0,'LossYears':1,'Ruined':ruined}

    # Compute raw Sharpe and CAGR
    sharpe = float((r.mean() / (r.std(ddof=1) + 1e-12)) * np.sqrt(252))
    years  = _years(daily.index)
    cagr   = float((float(daily.iloc[-1])/float(daily.iloc[0]))**(1.0/years) - 1.0)
    
    # Apply caps for small samples
    sharpe = float(np.clip(sharpe, -clip_sharpe, clip_sharpe))
    cagr = float(np.clip(cagr, -clip_cagr, clip_cagr))

    underwater = (daily < daily.cummax())
    stagnation = float(underwater.sum() / len(daily))

    yearly_last = daily.resample('YE').last().pct_change().dropna()
    loss_years  = int((yearly_last < 0).sum())

    fit = (w[0]*sharpe + w[1]*cagr + w[2]*stagnation + w[3]*loss_years)
    if ruined: fit += ruin_penalty
    
    # Apply coverage scaling if too few business days
    if bdays < min_bdays:
        coverage = bdays / max(1, min_bdays)
        fit *= coverage
    
    return float(fit), {'Sharpe':sharpe,'CAGR':cagr,'Stagnation':stagnation,'LossYears':loss_years,'Ruined':ruined}

# Legacy functions for backward compatibility
def _years_from_index(idx: pd.DatetimeIndex) -> float:
    return _years(idx)

def calc_sharpe(equity: pd.Series, rf: float = 0.0) -> float:
    # resample to business days to avoid intraday over-counting
    try:
        daily = equity.resample('1B').last().pct_change().dropna()
    except Exception:
        daily = equity.pct_change().dropna()
    if len(daily) < 3 or daily.std(ddof=1) == 0:
        return 0.0
    excess = daily - (rf / 252.0)
    return float((excess.mean() / (daily.std(ddof=1) + 1e-12)) * np.sqrt(252))

def calc_cagr(equity: pd.Series) -> float:
    if not hasattr(equity.index, 'dtype'):
        return 0.0
    years = _years_from_index(equity.index)
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    if start <= 0 or years <= 0:
        return 0.0
    return float((end / start) ** (1.0 / years) - 1.0)

def _stagnation_fraction(equity: pd.Series) -> float:
    # fraction of days below running max (after resample to business days)
    try:
        daily = equity.resample('1B').last().ffill()
    except Exception:
        daily = equity.fillna(method='ffill')
    if len(daily) == 0:
        return 0.0
    cummax = daily.cummax()
    frac = (daily < cummax).sum() / float(len(daily))
    return float(frac)

def _loss_years(equity: pd.Series) -> int:
    # resample yearly, count negative yearly returns
    try:
        yearly = equity.resample('YE').last()
    except Exception:
        # fallback simple grouping by year
        yearly = equity.groupby(equity.index.year).last()
    if len(yearly) < 2:
        return 0
    returns = yearly.pct_change().dropna()
    return int((returns < 0).sum())

def _apply_ruin_clamp(equity: pd.Series, ruin_threshold: float = 0.05) -> Tuple[pd.Series, bool]:
    return _ruin_clamp(equity, ruin_threshold)


class FitnessCalculator:
    """PATCH #1: Use stability_fitness for consistent metrics computation with small-sample caps."""
    def __init__(self, weights: Dict = None, sharpe_weight: float = None, cagr_weight: float = None,
                 stagnation_weight: float = None, loss_years_weight: float = None,
                 ruin_penalty: float = None, ruin_threshold: float = 0.05,
                 clip_sharpe: float = 5.0, clip_cagr: float = 1.0, min_bdays: int = 60):
        # default weights
        base = {
            'sharpe': 1.0,
            'cagr': 2.0,
            'stagnation': -1.0,
            'loss_years': -1.0,
            'ruin_penalty': -5.0,
        }
        # start from provided dict or defaults
        w = base.copy()
        if weights:
            w.update(weights)
        # override with explicit keyword args if provided
        if sharpe_weight is not None:
            w['sharpe'] = sharpe_weight
        if cagr_weight is not None:
            w['cagr'] = cagr_weight
        if stagnation_weight is not None:
            w['stagnation'] = stagnation_weight
        if loss_years_weight is not None:
            w['loss_years'] = loss_years_weight
        if ruin_penalty is not None:
            w['ruin_penalty'] = ruin_penalty

        self.weights = w
        self.ruin_threshold = ruin_threshold
        self.clip_sharpe = clip_sharpe
        self.clip_cagr = clip_cagr
        self.min_bdays = min_bdays
        self.ruin_threshold = ruin_threshold

    def calculate_all_metrics(self, equity_series: pd.Series) -> Dict:
        """PATCH #1: Use new stability_fitness for consistent computation."""
        # ensure datetime index
        if not isinstance(equity_series.index, pd.DatetimeIndex):
            try:
                equity_series.index = pd.to_datetime(equity_series.index)
            except Exception:
                # create hourly index fallback
                equity_series.index = pd.date_range('2024-01-01', periods=len(equity_series), freq='h')

        # Use new stability_fitness function
        weights_tuple = (
            self.weights['sharpe'],
            self.weights['cagr'],
            self.weights['stagnation'],
            self.weights['loss_years']
        )
        fitness, metrics = stability_fitness(
            equity_series,
            w=weights_tuple,
            ruin_penalty=self.weights.get('ruin_penalty', -5.0),
            clip_sharpe=self.clip_sharpe,
            clip_cagr=self.clip_cagr,
            min_bdays=self.min_bdays
        )

        # Add equity start/end for compatibility
        all_metrics = {
            'sharpe': metrics['Sharpe'],
            'cagr': metrics['CAGR'],
            'stagnation': metrics['Stagnation'],
            'loss_years': metrics['LossYears'],
            'ruined': metrics['Ruined'],
            'fitness': fitness,
            'equity_start': float(equity_series.iloc[0]) if len(equity_series) else 0.0,
            'equity_end': float(equity_series.iloc[-1]) if len(equity_series) else 0.0,
        }
        return all_metrics


def compare_strategies(equity_series_dict: Dict[str, pd.Series],
                      fitness_calculator: FitnessCalculator = None) -> pd.DataFrame:
    """
    Compare multiple strategies using fitness metrics.
    
    Args:
        equity_series_dict: Dict of {strategy_name: equity_series}
        fitness_calculator: FitnessCalculator instance
        
    Returns:
        DataFrame with comparison metrics
    """
    if fitness_calculator is None:
        fitness_calculator = FitnessCalculator()
    
    results = []
    
    for strategy_name, equity_series in equity_series_dict.items():
        metrics = fitness_calculator.calculate_all_metrics(equity_series)
        metrics['strategy'] = strategy_name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # Sort by fitness
    df = df.sort_values('fitness', ascending=False)
    
    return df


if __name__ == "__main__":
    print("Fitness Metric Module")
    print("=" * 50)
    
    # Create sample equity curve
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    
    # Strategy 1: Steady growth with small drawdowns
    equity1 = pd.Series(
        1000 * (1 + np.cumsum(np.random.normal(0.001, 0.01, 365))),
        index=dates
    )
    equity1 = equity1.clip(lower=900)  # Prevent going too low
    
    # Strategy 2: High growth but volatile
    equity2 = pd.Series(
        1000 * (1 + np.cumsum(np.random.normal(0.002, 0.03, 365))),
        index=dates
    )
    equity2 = equity2.clip(lower=800)
    
    # Strategy 3: Low growth, very stable
    equity3 = pd.Series(
        1000 * (1 + np.cumsum(np.random.normal(0.0005, 0.005, 365))),
        index=dates
    )
    
    # Initialize fitness calculator
    fc = FitnessCalculator()
    
    # Calculate fitness for each strategy
    print("\nStrategy 1 (Steady Growth):")
    metrics1 = fc.calculate_all_metrics(equity1)
    for key, value in metrics1.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nStrategy 2 (High Growth, Volatile):")
    metrics2 = fc.calculate_all_metrics(equity2)
    for key, value in metrics2.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    print("\nStrategy 3 (Low Growth, Stable):")
    metrics3 = fc.calculate_all_metrics(equity3)
    for key, value in metrics3.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    # Compare strategies
    print("\n\nStrategy Comparison:")
    comparison = compare_strategies({
        'Steady Growth': equity1,
        'High Growth Volatile': equity2,
        'Low Growth Stable': equity3,
    }, fc)
    
    print(comparison[['strategy', 'fitness', 'sharpe', 'cagr', 'max_drawdown_pct', 'return']].to_string(index=False))

