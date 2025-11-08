"""
Robust Feature Scaling Utilities
Provides winsorization and robust scaling to handle outliers.
"""

import numpy as np
import pandas as pd
from typing import Dict


def robust_fit(df: pd.DataFrame, cols) -> Dict:
    """
    Fit robust scaler statistics with winsorization.
    
    Args:
        df: DataFrame with features
        cols: List of column names to scale
        
    Returns:
        Dict with statistics: {'med', 'mad', 'lo', 'hi'}
    """
    X = df[cols].copy()
    # Winsorize to 1st–99th percentile to kill spikes
    q1 = X.quantile(0.01)
    q99 = X.quantile(0.99)
    Xc = X.clip(lower=q1, upper=q99, axis=1)
    med = Xc.median()
    mad = (Xc - med).abs().median() + 1e-9
    return {
        'med': med.to_dict(),
        'mad': mad.to_dict(),
        'lo': q1.to_dict(),
        'hi': q99.to_dict()
    }


def robust_transform(df: pd.DataFrame, cols, stats: Dict) -> pd.DataFrame:
    """
    Transform features using robust scaler statistics.
    
    Args:
        df: DataFrame with features
        cols: List of column names to scale
        stats: Statistics dict from robust_fit
        
    Returns:
        Scaled DataFrame
    """
    X = df[cols].copy()
    Xc = X.clip(lower=pd.Series(stats['lo']), upper=pd.Series(stats['hi']), axis=1)
    med = pd.Series(stats['med'])
    mad = pd.Series(stats['mad'])
    Z = (Xc - med) / mad
    return Z
