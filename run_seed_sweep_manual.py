"""
Simple seed sweep - run training 3 times with different seeds
Just manually edit config.py between runs.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

SEEDS = [7, 77, 777]
EPISODES = 50
LOG_DIR = Path("./seed_sweep_results")

def run_training(seed: int):
    """Run one training session."""
    print("\n" + "="*60)
    print(f"SEED: {seed}")
    print("="*60)
    print(f"\nBEFORE STARTING:")
    print(f"  1. Edit config.py")
    print(f"  2. Change: random_seed: int = {seed}")
    print(f"  3. Save the file")
    print("")
    
    response = input(f"Ready to train with seed {seed}? (y/n): ").strip().lower()
    if response != 'y':
        print("Skipped.")
        return False
    
    # Create log dir
    seed_log_dir = LOG_DIR / f"seed_{seed}"
    seed_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Run training
    print(f"\nStarting training with seed {seed}...")
    result = subprocess.run(
        [sys.executable, "main.py", "--episodes", str(EPISODES)],
        capture_output=False,
        text=True
    )
    
    if result.returncode == 0:
        print(f"\n[OK] Seed {seed} completed successfully")
        return True
    else:
        print(f"\n[ERROR] Seed {seed} failed with code {result.returncode}")
        return False

if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("FOREX RL BOT - MANUAL SEED SWEEP")
    print("="*60)
    print(f"Seeds: {SEEDS}")
    print(f"Episodes per seed: {EPISODES}")
    print("\nYou will be prompted to edit config.py before each seed.")
    print("")
    
    results = {}
    for i, seed in enumerate(SEEDS, 1):
        print(f"\n{'='*60}")
        print(f"SEED {i}/{len(SEEDS)}: {seed}")
        print(f"{'='*60}")
        
        success = run_training(seed)
        results[seed] = success
        
        if not success:
            print(f"\nSeed {seed} did not complete successfully.")
            cont = input("Continue with remaining seeds? (y/n): ").strip().lower()
            if cont != 'y':
                break
    
    # Summary
    print("\n" + "="*60)
    print("SEED SWEEP SUMMARY")
    print("="*60)
    for seed, success in results.items():
        status = "[OK]" if success else "[FAILED]"
        print(f"  Seed {seed}: {status}")
    
    successful = sum(1 for s in results.values() if s)
    print(f"\nCompleted: {successful}/{len(results)} seeds")
    print("="*60)
