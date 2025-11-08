"""
Check if the issue is eval_mode freezing exploration during validation.
"""
import json
import glob

print("=" * 60)
print("VALIDATION vs TRAINING BEHAVIOR CHECK")
print("=" * 60)

# Check validation files
val_files = sorted(glob.glob("logs/validation_summaries/val_ep*.json"))
print(f"\nFound {len(val_files)} validation files")

if len(val_files) >= 8:
    # Check episodes 1-8
    print("\nValidation Results (Episodes 1-8):")
    for i in range(1, 9):
        filepath = f"logs/validation_summaries/val_ep{i:03d}.json"
        try:
            with open(filepath) as f:
                data = json.load(f)
            trades = data.get('trades', 0)
            entropy = data.get('entropy', data.get('action_entropy_bits', 0))
            hold_rate = data.get('hold_frac', data.get('hold_rate', 0))
            print(f"  Ep {i}: trades={trades:.1f}  entropy={entropy:.2f}  hold={hold_rate:.2f}")
        except Exception as e:
            print(f"  Ep {i}: Error reading file - {e}")

# Now we need to check if there's ANY evidence of trading during training
# Look for tensorboard events or any training logs
import os

print("\n" + "=" * 60)
print("LOOKING FOR TRAINING EVIDENCE")
print("=" * 60)

# Check for tensorboard events
tb_files = glob.glob("logs/events.out.tfevents.*")
if tb_files:
    print(f"\n✓ Found {len(tb_files)} TensorBoard event file(s)")
    print("  → Training is logging, but need to check if agent trades DURING training")
else:
    print("\n✗ No TensorBoard events found")

# Check for any training action logs
if os.path.exists("logs/training.log"):
    print("\n✓ Found training.log")
    with open("logs/training.log") as f:
        lines = f.readlines()
    print(f"  → {len(lines)} lines in training log")
else:
    print("\n✗ No training.log found")

print("\n" + "=" * 60)
print("DIAGNOSIS")
print("=" * 60)

print("""
KEY FINDING: Agent shows 100% HOLD in ALL validation episodes (1-8)

This could be because:
1. Agent NEVER trades (even during training) → Exploration failure
2. Agent trades during training but NOT during validation → eval_mode issue

HYPOTHESIS: NoisyNet exploration is DISABLED during validation
- During training: NoisyNet adds noise to weights → some exploration
- During validation: eval_mode=True freezes noise → deterministic HOLD policy

TEST: We need to check if agent TRADES during training episodes

NUCLEAR FIX: Force epsilon-greedy exploration
- Disable NoisyNet (use_noisy=False)  
- Enable epsilon-greedy (epsilon_start=0.5)
- This will force random exploration during both training AND validation

IMMEDIATE ACTION:
1. Stop current training (Ctrl+C)
2. Apply Nuclear Fix (disable NoisyNet, add epsilon-greedy)
3. Clear checkpoints again
4. Restart with forced exploration
""")
