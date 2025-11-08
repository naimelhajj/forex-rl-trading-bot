"""
Ultra-minimal test - literally just main.py startup sequence with timing.
"""
import sys
import time

def tprint(msg):
    """Print with immediate flush."""
    print(msg, flush=True)

tprint("=" * 70)
tprint("ULTRA-MINIMAL HANG TEST - Tracing main.py startup")
tprint("=" * 70)

# Step 1
tprint("\n[1] Importing config...")
t0 = time.time()
from config import Config
tprint(f"✓ config imported in {time.time()-t0:.3f}s")

# Step 2
tprint("[2] Creating Config()...")
t0 = time.time()
config = Config()
config.num_episodes = 5  # Override
tprint(f"✓ Config() created in {time.time()-t0:.3f}s")

# Step 3
tprint("[3] Importing DataLoader...")
t0 = time.time()
from data_loader import DataLoader
tprint(f"✓ DataLoader imported in {time.time()-t0:.3f}s")

# Step 4
tprint("[4] Creating DataLoader and loading data...")
t0 = time.time()
loader = DataLoader(config.data_dir, config.pair)
tprint(f"  DataLoader created in {time.time()-t0:.3f}s")

t0 = time.time()
train_df, val_df, test_df = loader.load_split_data(
    train_size=config.train_size,
    val_size=config.val_size
)
tprint(f"✓ Data loaded in {time.time()-t0:.3f}s ({len(train_df)} train bars)")

# Step 5 - THE CRITICAL ONE
tprint("[5] Importing FeatureEngineer...")
t0 = time.time()
from features import FeatureEngineer
tprint(f"✓ FeatureEngineer imported in {time.time()-t0:.3f}s")

tprint("[6] Creating FeatureEngineer and computing features...")
t0 = time.time()
engineer = FeatureEngineer()
tprint(f"  FeatureEngineer created in {time.time()-t0:.3f}s")

tprint("  ⏳ Computing train features (THIS IS WHERE IT MIGHT HANG)...")
t0 = time.time()
train_df = engineer.compute_all_features(train_df)
elapsed = time.time() - t0
tprint(f"✓ Train features computed in {elapsed:.3f}s")
if elapsed > 2.0:
    tprint(f"  ⚠️  WARNING: Took {elapsed:.3f}s (expected <1s)")
    tprint("  This suggests rolling functions are still slow!")

t0 = time.time()
tprint("  Computing val features...")
val_df = engineer.compute_all_features(val_df)
tprint(f"✓ Val features computed in {time.time()-t0:.3f}s")

# Step 6
tprint("[7] Importing environment and agent...")
t0 = time.time()
from environment import TradingEnv
from agent import DQNAgent
tprint(f"✓ Modules imported in {time.time()-t0:.3f}s")

# Step 7
tprint("[8] Creating environments...")
t0 = time.time()
train_env = TradingEnv(
    df=train_df,
    initial_balance=config.initial_balance,
    max_position_size=config.max_position_size,
    leverage=config.leverage,
    spread=config.spread,
    commission=config.commission
)
tprint(f"  Train env created in {time.time()-t0:.3f}s")

t0 = time.time()
val_env = TradingEnv(
    df=val_df,
    initial_balance=config.initial_balance,
    max_position_size=config.max_position_size,
    leverage=config.leverage,
    spread=config.spread,
    commission=config.commission
)
tprint(f"✓ Val env created in {time.time()-t0:.3f}s")

# Step 8
tprint("[9] Creating agent...")
t0 = time.time()
agent = DQNAgent(
    state_dim=train_env.state_size,
    action_dim=train_env.action_space,
    config=config
)
tprint(f"✓ Agent created in {time.time()-t0:.3f}s")

# Step 9 - Test NoisyNet
tprint("[10] Testing NoisyNet reset_noise() (potential recursion issue)...")
t0 = time.time()
for i in range(50):
    agent.reset_noise()
    if i % 10 == 0:
        tprint(f"  Completed {i}/50 calls...")
elapsed = time.time() - t0
tprint(f"✓ 50 reset_noise() calls in {elapsed:.4f}s")
if elapsed > 0.5:
    tprint(f"  ⚠️  WARNING: Slow reset_noise ({elapsed:.4f}s for 50 calls)")
    tprint("  This suggests infinite recursion or slow module iteration!")

# Step 10 - THE BIG ONE: Trainer
tprint("[11] Importing Trainer...")
t0 = time.time()
from trainer import Trainer
tprint(f"✓ Trainer imported in {time.time()-t0:.3f}s")

tprint("[12] Creating Trainer (TensorBoard, logger, etc.)...")
t0 = time.time()
trainer = Trainer(
    agent=agent,
    train_env=train_env,
    val_env=val_env,
    config=config
)
elapsed = time.time() - t0
tprint(f"✓ Trainer created in {elapsed:.3f}s")
if elapsed > 5.0:
    tprint(f"  ⚠️  WARNING: Trainer init took {elapsed:.3f}s")
    tprint("  This might indicate TensorBoard or logger issues!")

tprint("\n" + "=" * 70)
tprint("SUCCESS - All initialization complete without hanging!")
tprint("=" * 70)
tprint("\nIf main.py still hangs, the issue is in trainer.train() method itself.")
tprint("Most likely culprits:")
tprint("  - First episode reset()")
tprint("  - Validation loop")
tprint("  - Checkpoint saving")
tprint("  - TensorBoard logging during training")
