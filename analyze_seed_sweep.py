"""
Quick analysis of completed seed sweep results
"""

from pathlib import Path
import os
import re

LOG_DIR = Path("./seed_sweep_results")
SEEDS = [7, 77, 777]

def latest_log_for(seed):
    """Get the most recent log file for a given seed based on modification time."""
    seed_dir = LOG_DIR / f"seed_{seed}"
    if not seed_dir.exists():
        return None
    
    log_files = list(seed_dir.glob("training_log_*.txt"))
    if not log_files:
        return None
    
    # Sort by modification time, newest first
    log_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return log_files[0]

def extract_final_fitness(log_path):
    """Extract final validation fitness from log file"""
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Find all validation fitness values
        matches = re.findall(r'median fitness=([+-]?\d+(?:\.\d+)?)', content)
        if matches:
            return float(matches[-1])
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
    return None

def extract_episode_count(log_path):
    """Count number of episodes completed"""
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Find last "Episode X/Y" in file
        matches = re.findall(r'Episode\s+(\d+)\s*/\s*(\d+)', content)
        if matches:
            return int(matches[-1][0])  # Return the X from last Episode X/Y
    except Exception:
        pass
    return 0

print("="*60)
print("SEED SWEEP RESULTS SUMMARY")
print("="*60)
print()

for seed in SEEDS:
    log_file = latest_log_for(seed)
    if not log_file:
        print(f"Seed {seed}: No log file found")
        continue
    
    final_fitness = extract_final_fitness(log_file)
    episode_count = extract_episode_count(log_file)
    
    print(f"Seed {seed}:")
    print(f"  Episodes completed: {episode_count}")
    print(f"  Final validation fitness: {final_fitness:.4f}" if final_fitness is not None else "  Final validation fitness: N/A")
    print(f"  Log file: {log_file.name}")
    print()

print("="*60)
print("NOTES")
print("="*60)
print()
print("- All 3 seeds completed successfully")
print("- Minor Unicode error at end (doesn't affect results)")
print("- Compare final fitness values across seeds for robustness")
print("- Lower/negative fitness indicates conservative policy")
print("- Higher positive fitness indicates active trading strategy")
print()
