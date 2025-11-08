"""
SIMPLEST test - just regime features timing.
"""
import time
import pandas as pd
import numpy as np
from features import _rolling_lr_slope

# Create test data
n = 10000
data = np.random.randn(n).cumsum()
series = pd.Series(data)

print(f"Testing _rolling_lr_slope on {n} points...")
print("=" * 60)

# Test window=10
print("\n[1/3] Window=10...")
t0 = time.time()
result = _rolling_lr_slope(series, 10)
elapsed = time.time() - t0
print(f"  ✓ Completed in {elapsed:.4f}s")
if elapsed > 0.5:
    print(f"    ⚠️  SLOW! (expected <0.1s)")

# Test window=24
print("\n[2/3] Window=24...")
t0 = time.time()
result = _rolling_lr_slope(series, 24)
elapsed = time.time() - t0
print(f"  ✓ Completed in {elapsed:.4f}s")
if elapsed > 0.5:
    print(f"    ⚠️  SLOW! (expected <0.1s)")

# Test window=96  
print("\n[3/3] Window=96...")
t0 = time.time()
result = _rolling_lr_slope(series, 96)
elapsed = time.time() - t0
print(f"  ✓ Completed in {elapsed:.4f}s")
if elapsed > 1.0:
    print(f"    ⚠️  SLOW! (expected <0.5s)")

print("\n" + "=" * 60)
print("If all three completed quickly, _rolling_lr_slope is optimized.")
print("If any hung or were >1s, there's still an issue.")
