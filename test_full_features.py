"""Test full feature computation timing"""
import time
import numpy as np
from data_loader import generate_sample_data
from features import FeatureEngineer, compute_currency_strengths

print("Generating sample data...")
start = time.time()
currency_data = generate_sample_data(num_points=10000, seed=42)
print(f"Data generation: {time.time()-start:.2f}s")

print("\nComputing features for EURUSD...")
feature_engineer = FeatureEngineer()
df = currency_data['EURUSD'].copy()

start = time.time()
df_features = feature_engineer.compute_all_features(df, currency_data=currency_data)
elapsed = time.time() - start

print(f"Feature computation: {elapsed:.2f}s")
print(f"Features shape: {df_features.shape}")
print(f"Feature columns: {df_features.columns.tolist()[:10]}...")

if elapsed < 5.0:
    print("\n✅ Feature computation is FAST!")
else:
    print(f"\n⚠️  Feature computation is SLOW ({elapsed:.1f}s)")
