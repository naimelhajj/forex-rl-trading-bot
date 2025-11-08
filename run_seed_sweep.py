"""
Seed Sweep Script for Forex RL Bot
Trains the agent with multiple random seeds to assess stability and variance.
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime

# Seeds to test
SEEDS = [7, 77, 777]
EPISODES = 50
LOG_DIR = Path("./seed_sweep_results")

def run_seed_training(seed: int):
    """Run training with a specific seed."""
    print(f"\n{'='*60}")
    print(f"TRAINING WITH SEED {seed}")
    print(f"{'='*60}\n")
    
    # Create log directory for this seed
    seed_log_dir = LOG_DIR / f"seed_{seed}"
    seed_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Run training (modify config.py first to use this seed)
    # For simplicity, we'll run with command line and capture output
    cmd = f"python main.py --episodes {EPISODES}"
    
    # Log file for this run
    log_file = seed_log_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(log_file, 'w') as f:
        f.write(f"Seed: {seed}\n")
        f.write(f"Episodes: {EPISODES}\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")
        f.flush()
        
        # Run subprocess
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output to both console and file
        for line in process.stdout:
            print(line, end='')
            f.write(line)
        
        process.wait()
    
    print(f"\n✓ Seed {seed} complete. Log saved to: {log_file}")
    return log_file

def parse_validation_stats(log_file: Path):
    """Extract validation statistics from log file."""
    stats = {
        'seed': None,
        'episodes': [],
        'val_fitness': [],
        'val_iqr': [],
        'val_median': [],
        'val_adj': [],
        'val_k': [],
    }
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
        for i, line in enumerate(lines):
            # Extract seed from header
            if 'Seed:' in line:
                stats['seed'] = int(line.split(':')[1].strip())
            
            # Parse validation lines
            if '[VAL] K=' in line and 'overlapping' in line:
                try:
                    # Extract K, median, IQR, adj
                    parts = line.split('|')
                    k_part = [p for p in parts if 'K=' in p][0]
                    k = int(k_part.split('K=')[1].split()[0])
                    
                    median_part = [p for p in parts if 'median fitness=' in p][0]
                    median = float(median_part.split('=')[1].strip())
                    
                    iqr_part = [p for p in parts if 'IQR=' in p][0]
                    iqr = float(iqr_part.split('=')[1].strip())
                    
                    adj_part = [p for p in parts if 'adj=' in p][0]
                    adj = float(adj_part.split('=')[1].strip())
                    
                    stats['val_k'].append(k)
                    stats['val_median'].append(median)
                    stats['val_iqr'].append(iqr)
                    stats['val_adj'].append(adj)
                except:
                    pass
            
            # Extract episode number and fitness from summary
            if 'Episode' in line and '/' in line and 'Val   - Reward:' in lines[i+1]:
                try:
                    ep = int(line.split('/')[0].split()[-1])
                    val_line = lines[i+1]
                    fitness = float(val_line.split('Fitness:')[1].split('|')[0].strip())
                    stats['episodes'].append(ep)
                    stats['val_fitness'].append(fitness)
                except:
                    pass
    
    return stats

def summarize_sweep():
    """Summarize results from all seeds."""
    print(f"\n{'='*60}")
    print("SEED SWEEP SUMMARY")
    print(f"{'='*60}\n")
    
    all_stats = []
    for seed in SEEDS:
        seed_log_dir = LOG_DIR / f"seed_{seed}"
        log_files = list(seed_log_dir.glob("training_log_*.txt"))
        if log_files:
            stats = parse_validation_stats(log_files[-1])  # Use most recent
            all_stats.append(stats)
            
            print(f"Seed {seed}:")
            if stats['val_fitness']:
                print(f"  Final Fitness: {stats['val_fitness'][-1]:.4f}")
                print(f"  Mean Fitness: {sum(stats['val_fitness'])/len(stats['val_fitness']):.4f}")
                print(f"  Median IQR: {sum(stats['val_iqr'])/len(stats['val_iqr']):.4f}")
                print(f"  Mean K: {sum(stats['val_k'])/len(stats['val_k']):.1f}")
            print()
    
    # Save combined results
    summary_file = LOG_DIR / "seed_sweep_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"✓ Summary saved to: {summary_file}")

if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("FOREX RL BOT - SEED SWEEP")
    print("="*60)
    print(f"Seeds: {SEEDS}")
    print(f"Episodes: {EPISODES}")
    print(f"Results directory: {LOG_DIR}")
    print()
    
    # Note: You need to manually update config.py with each seed
    print("⚠️  MANUAL STEP REQUIRED:")
    print("Before running each seed, update config.py:")
    print("  random_seed: int = <SEED>")
    print()
    input("Press Enter when ready to start first seed training...")
    
    for i, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"SEED {i+1}/{len(SEEDS)}: {seed}")
        print(f"{'='*60}")
        print(f"⚠️  Update config.py: random_seed = {seed}")
        input(f"Press Enter when config.py updated with seed {seed}...")
        
        run_seed_training(seed)
        
        if i < len(SEEDS) - 1:
            print(f"\n✓ Seed {seed} complete. Next seed: {SEEDS[i+1]}")
    
    # Generate summary
    summarize_sweep()
    
    print("\n✓ Seed sweep complete!")
