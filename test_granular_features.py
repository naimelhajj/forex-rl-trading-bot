"""
Test each feature computation step individually with timing.
"""
import time
import sys
import numpy as np
import pandas as pd

def p(msg):
    print(msg, flush=True)

p("Loading data...")
from config import Config
from data_loader import DataLoader

config = Config()
loader = DataLoader(config.data.data_dir)
train_df, _, _ = loader.load_split_data(0.7, 0.15)
p(f"Data loaded: {len(train_df)} bars\n")

from features import FeatureEngineer, rolling_percentile, _rolling_lr_slope
eng = FeatureEngineer()

df = train_df.copy()

# Test each feature individually
p("=" * 60)
p("INDIVIDUAL FEATURE TIMING TEST")
p("=" * 60)

# 1. ATR
p("\n[1/11] Computing ATR...")
t0 = time.time()
df['atr'] = eng.compute_atr(df)
p(f"  ✓ ATR: {time.time()-t0:.3f}s")

# 2. RSI
p("[2/11] Computing RSI...")
t0 = time.time()
df['rsi'] = eng.compute_rsi(df)
p(f"  ✓ RSI: {time.time()-t0:.3f}s")

# 3. Percentile short
p("[3/11] Computing percentile_short (window=5)...")
t0 = time.time()
df['percentile_short'] = rolling_percentile(df['close'], eng.short_window)
elapsed = time.time()-t0
p(f"  ✓ percentile_short: {elapsed:.3f}s")
if elapsed > 1.0:
    p(f"    ⚠️  SLOW! Expected <0.1s")

# 4. Percentile medium
p("[4/11] Computing percentile_medium (window=20)...")
t0 = time.time()
df['percentile_medium'] = rolling_percentile(df['close'], eng.medium_window)
elapsed = time.time()-t0
p(f"  ✓ percentile_medium: {elapsed:.3f}s")
if elapsed > 1.0:
    p(f"    ⚠️  SLOW! Expected <0.1s")

# 5. Percentile long
p("[5/11] Computing percentile_long (window=50)...")
t0 = time.time()
df['percentile_long'] = rolling_percentile(df['close'], eng.long_window)
elapsed = time.time()-t0
p(f"  ✓ percentile_long: {elapsed:.3f}s")
if elapsed > 1.0:
    p(f"    ⚠️  SLOW! Expected <0.1s")

# 6. Temporal features
p("[6/11] Computing temporal features...")
t0 = time.time()
df = eng.add_temporal_features(df)
df = eng.add_cyclical_time(df)
p(f"  ✓ Temporal: {time.time()-t0:.3f}s")

# 7. Fractals
p("[7/11] Computing fractals...")
t0 = time.time()
df = eng._compute_fractals_safe(df, w=eng.fractal_window)
p(f"  ✓ Fractals: {time.time()-t0:.3f}s")

# 8. LR slope (short window)
p("[8/11] Computing lr_slope (window=10)...")
t0 = time.time()
hlc = (df['high'] + df['low'] + df['close']) / 3.0
df['lr_slope'] = _rolling_lr_slope(hlc.astype(float), eng.lr_window)
elapsed = time.time()-t0
p(f"  ✓ lr_slope: {elapsed:.3f}s")
if elapsed > 0.5:
    p(f"    ⚠️  SLOW! Expected <0.1s")

# 9. Regime features (THE BIG ONE - multiple lr_slope calls)
p("[9/11] Computing regime features (THIS IS THE CRITICAL TEST)...")
p("  This includes lr_slope_24h and lr_slope_96h...")
t0 = time.time()
df = eng.add_regime_features(df)
elapsed = time.time()-t0
p(f"  ✓ Regime features: {elapsed:.3f}s")
if elapsed > 2.0:
    p(f"    ⚠️  SLOW! Expected <1s")
    p(f"    This suggests _rolling_lr_slope with windows 24/96 is still slow!")

# 10. Final cleanup
p("[10/11] Final fillna...")
t0 = time.time()
df = df.ffill().bfill().fillna(0)
p(f"  ✓ Fillna: {time.time()-t0:.3f}s")

p("\n" + "=" * 60)
p("FEATURE COMPUTATION COMPLETE")
p("=" * 60)
p(f"\nFinal dataframe: {df.shape}")
p(f"Columns: {len(df.columns)}")
