"""
Simple Seed Sweep - Run training with different seeds sequentially
Just updates config and runs, no output capture complexity.
"""

import subprocess
import os
import sys
import argparse
from pathlib import Path
import re

SEEDS = [7, 77, 777]
EPISODES = 25  # Changed from 50 to 25 for faster sweep
CONFIG_FILE = Path("config.py")

def update_config_value(pattern: str, value: str, description: str) -> bool:
    """Generic config updater"""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not re.search(pattern, content):
            print(f"[ERROR] Could not find pattern '{description}' in config.py")
            return False
        
        new_content = re.sub(pattern, value, content)
        
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"[OK] Updated config.py: {description}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to update config: {e}")
        return False

def update_config_seed(seed: int) -> bool:
    """Update random_seed in config.py"""
    pattern = r'(random_seed:\s*int\s*=\s*)\d+'
    return update_config_value(pattern, fr'\g<1>{seed}', f"random_seed = {seed}")

def disable_early_stop() -> bool:
    """Disable early stopping for seed sweeps"""
    pattern = r'(disable_early_stop:\s*bool\s*=\s*)(True|False)'
    return update_config_value(pattern, r'\g<1>True', "disable_early_stop = True")

def enable_early_stop() -> bool:
    """Re-enable early stopping after sweep"""
    pattern = r'(disable_early_stop:\s*bool\s*=\s*)(True|False)'
    return update_config_value(pattern, r'\g<1>False', "disable_early_stop = False")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run seed sweep for RL training")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="Seeds to run (e.g. --seeds 7 77 777 or --seeds 7)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=EPISODES,
        help=f"Episodes per seed (default: {EPISODES})"
    )
    args = parser.parse_args()
    
    seeds = args.seeds
    episodes = args.episodes
    
    print("="*60)
    print("FOREX RL BOT - SIMPLE SEED SWEEP")
    print("="*60)
    print(f"Seeds: {seeds}")
    print(f"Episodes per seed: {episodes}")
    print()
    
    # Disable early stopping for seed sweeps
    print("[CONFIG] Disabling early stop for seed sweep...")
    if not disable_early_stop():
        print("[WARN] Could not disable early stop - sweep may have inconsistent episode counts")
    print()
    
    # Set unbuffered environment
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    for i, seed in enumerate(seeds, 1):
        print()
        print("="*60)
        print(f"SEED {i}/{len(seeds)}: {seed}")
        print("="*60)
        print()
        
        # Update config
        if not update_config_seed(seed):
            print(f"[ERROR] Skipping seed {seed}")
            continue
        
        print(f"[RUN] Starting training with seed {seed}...")
        print()
        
        # Run training - output goes directly to console
        result = subprocess.run(
            ["python", "-u", "main.py", "--episodes", str(episodes)],
            env=env
        )
        
        if result.returncode == 0:
            print()
            print(f"[OK] Seed {seed} completed successfully")
        else:
            print()
            print(f"[ERROR] Seed {seed} failed with exit code {result.returncode}")
            # Continue to next seed anyway (don't abort the sweep)
    
    print()
    print("="*60)
    print("SEED SWEEP COMPLETE")
    print("="*60)
    print()
    
    # Re-enable early stopping
    print("[CONFIG] Re-enabling early stop...")
    enable_early_stop()
    print()

if __name__ == "__main__":
    main()
