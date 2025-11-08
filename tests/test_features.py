import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from features import FeatureEngineer


def test_atr_rsi_percentile_lr():
    # Create simple synthetic OHLC data with clear trend and volatility
    n = 50
    base = 1.1000
    # prices slowly increase with some noise
    closes = base + np.linspace(0, 0.01, n) + 0.0005 * np.random.randn(n)
    highs = closes + 0.0005 + 0.0002 * np.random.rand(n)
    lows = closes - 0.0005 - 0.0002 * np.random.rand(n)
    opens = closes + 0.0001 * np.random.randn(n)
    times = pd.date_range('2024-01-01', periods=n, freq='H')

    df = pd.DataFrame({'time': times, 'open': opens, 'high': highs, 'low': lows, 'close': closes}, index=times)
    fe = FeatureEngineer(atr_period=5, rsi_period=5, lr_window=5, fractal_window=5)
    fdf = fe.compute_all_features(df)

    # ATR should be non-negative and have no NaNs after fill
    assert 'atr' in fdf.columns
    assert fdf['atr'].notna().all()
    assert (fdf['atr'] >= 0).all()

    # RSI between 0 and 100
    assert 'rsi' in fdf.columns
    assert (fdf['rsi'] >= 0).all() and (fdf['rsi'] <= 100).all()

    # Percentiles in 0-100
    for col in ['percentile_short', 'percentile_medium', 'percentile_long']:
        assert col in fdf.columns
        assert (fdf[col] >= 0).all() and (fdf[col] <= 100).all()

    # LR slope should be a finite number
    assert 'lr_slope' in fdf.columns
    assert np.isfinite(fdf['lr_slope'].iloc[-1])

    print('ATR/RSI/Percentile/LR tests passed')


def test_fractals_detection():
    # Build a series with a clear top fractal at index 5 and bottom at index 10
    n = 30
    prices = np.linspace(1.0, 1.0, n)
    prices = prices + 0.001 * np.random.randn(n)
    # create a pronounced top at 10 and bottom at 20
    prices[10] = prices.max() + 0.01
    prices[20] = prices.min() - 0.01

    highs = prices + 0.0005
    lows = prices - 0.0005
    opens = prices + 0.0001
    closes = prices
    times = pd.date_range('2024-01-01', periods=n, freq='H')

    df = pd.DataFrame({'time': times, 'open': opens, 'high': highs, 'low': lows, 'close': closes}, index=times)
    fe = FeatureEngineer(fractal_window=5)
    fdf = fe.compute_all_features(df)

    # There should be some non-null fractal values
    assert fdf['top_fractal'].notna().any()
    assert fdf['bottom_fractal'].notna().any()

    print('Fractal detection tests passed')


def test_currency_strength():
    # Create two pair dfs: EURUSD and EURJPY where EUR is strengthening
    n = 40
    times = pd.date_range('2024-01-01', periods=n, freq='H')

    # EURUSD rises
    eurusd_close = 1.10 + np.linspace(0, 0.01, n)
    df_eurusd = pd.DataFrame({'time': times, 'open': eurusd_close, 'high': eurusd_close, 'low': eurusd_close, 'close': eurusd_close}, index=times)

    # EURJPY rises as well (EUR strengthening)
    eurjpy_close = 130.0 + np.linspace(0, 1.0, n)
    df_eurjpy = pd.DataFrame({'time': times, 'open': eurjpy_close, 'high': eurjpy_close, 'low': eurjpy_close, 'close': eurjpy_close}, index=times)

    currency_data = {'EURUSD': df_eurusd, 'EURJPY': df_eurjpy}
    fe = FeatureEngineer()
    strength_df = fe.compute_currency_strength(currency_data)

    # Expect strength_EUR column present and contains finite values
    assert 'strength_EUR' in strength_df.columns
    recent_strength = strength_df['strength_EUR'].dropna()
    assert len(recent_strength) > 0
    # Ensure values are finite and not all zeros (robust to z-scoring windowing)
    assert np.isfinite(recent_strength).all()
    assert recent_strength.abs().max() > 0

    print('Currency strength tests passed')


if __name__ == '__main__':
    test_atr_rsi_percentile_lr()
    test_fractals_detection()
    test_currency_strength()
    print('All feature tests passed')
