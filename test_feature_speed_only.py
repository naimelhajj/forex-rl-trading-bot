"""
JUST test feature computation speed on real data.
"""
import time
import sys

print("=" * 70)
print("FEATURE COMPUTATION SPEED TEST")
print("=" * 70)

print("\n[1/4] Loading dependencies...")
t0 = time.time()
from config import Config
from data_loader import DataLoader
from features import FeatureEngineer
print(f"✓ Loaded in {time.time()-t0:.3f}s")

print("\n[2/4] Loading data...")
t0 = time.time()
config = Config()
loader = DataLoader(config.data_dir, config.pair)
train_df, val_df, _ = loader.load_split_data(
    train_size=config.train_size,
    val_size=config.val_size
)
print(f"✓ Data loaded in {time.time()-t0:.3f}s")
print(f"  Train shape: {train_df.shape}")
print(f"  Val shape: {val_df.shape}")

print("\n[3/4] Computing TRAIN features (MAIN BOTTLENECK TEST)...")
print(f"  Processing {len(train_df)} bars...")
sys.stdout.flush()

t0 = time.time()
engineer = FeatureEngineer()
train_df_feat = engineer.compute_all_features(train_df)
elapsed = time.time() - t0

print(f"✓ Train features computed in {elapsed:.3f}s")
print(f"  Features: {train_df_feat.shape[1]} columns")

if elapsed > 3.0:
    print(f"\n⚠️  HANG DETECTED: Feature computation took {elapsed:.3f}s")
    print(f"  Expected: <1.0s for ~10k bars")
    print(f"  There is still a slow operation in features.py!")
    sys.exit(1)
elif elapsed > 1.0:
    print(f"\n⚠️  SLOW: Feature computation took {elapsed:.3f}s (expected <1s)")
else:
    print(f"\n✓ FAST: Feature computation is optimized ({elapsed:.3f}s)")

print("\n[4/4] Computing VAL features...")
sys.stdout.flush()

t0 = time.time()
val_df_feat = engineer.compute_all_features(val_df)
elapsed = time.time() - t0
print(f"✓ Val features computed in {elapsed:.3f}s")

print("\n" + "=" * 70)
print("FEATURE COMPUTATION TEST COMPLETE")
print("=" * 70)
print(f"\nIf this test passed quickly (<1s per dataset), then:")
print("  - Features are optimized ✓")
print("  - The hang is elsewhere (Trainer, TensorBoard, validation, etc.)")
