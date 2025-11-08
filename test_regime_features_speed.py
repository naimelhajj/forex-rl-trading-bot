"""Test regime features speed"""
import pandas as pd
import numpy as np
import time
from features import FeatureEngineer

print("Generating test data (10000 bars)...")
dates = pd.date_range('2023-01-01', periods=10000, freq='h')
data = pd.DataFrame({
    'open': 1.08 + np.random.randn(10000) * 0.01,
    'high': 1.08 + np.random.randn(10000) * 0.01 + 0.001,
    'low': 1.08 + np.random.randn(10000) * 0.01 - 0.001,
    'close': 1.08 + np.random.randn(10000) * 0.01,
}, index=dates)

print("Computing features...")
fe = FeatureEngineer()

start = time.time()
result = fe.compute_all_features(data)
elapsed = time.time() - start

print(f"✅ Features computed in {elapsed:.2f}s")
print(f"Result shape: {result.shape}")
print(f"Regime features: {[c for c in result.columns if 'vol' in c or 'trend' in c]}")
print(f"\nSample regime values:")
print(result[['realized_vol_24h_z', 'realized_vol_96h_z', 'trend_24h', 'trend_96h', 'is_trending']].tail())

if elapsed < 5:
    print("\n✅ Performance acceptable (<5s)")
else:
    print(f"\n⚠ Performance slow ({elapsed:.2f}s)")
