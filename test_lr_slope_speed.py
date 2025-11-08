"""Test _rolling_lr_slope performance"""
import numpy as np
import pandas as pd
import time
from features import _rolling_lr_slope

# Create 10k data points like real data
np.random.seed(42)
data = np.cumsum(np.random.randn(10000)) + 1.10
series = pd.Series(data)

print("Testing _rolling_lr_slope performance on 10k points...")
print(f"Data shape: {series.shape}")

# Test with window=10 (normal lr_slope)
start = time.time()
result_10 = _rolling_lr_slope(series, 10)
elapsed_10 = time.time() - start
print(f"Window=10:  {elapsed_10:.2f}s")

# Test with window=24 (regime features)
start = time.time()
result_24 = _rolling_lr_slope(series, 24)
elapsed_24 = time.time() - start
print(f"Window=24:  {elapsed_24:.2f}s")

# Test with window=96 (regime features)
start = time.time()
result_96 = _rolling_lr_slope(series, 96)
elapsed_96 = time.time() - start
print(f"Window=96:  {elapsed_96:.2f}s")

print(f"\nTotal time: {elapsed_10 + elapsed_24 + elapsed_96:.2f}s")
print("If this is slow (>10s total), _rolling_lr_slope is the bottleneck!")
