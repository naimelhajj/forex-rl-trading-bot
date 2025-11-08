"""
Automated Seed Sweep Script for Forex RL Bot
Automatically updates config.py and trains with multiple seeds.
"""

import subprocess
import json
import os
from pathlib import Path
from datetime import datetime
import re

# Seeds to test
SEEDS = [7, 77, 777]
EPISODES = 50
LOG_DIR = Path("./seed_sweep_results")
CONFIG_FILE = Path("./config.py")

def update_config_seed(seed: int):
    """Update random_seed in config.py."""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace random_seed value - match the exact format
        pattern = r'(random_seed:\s*int\s*=\s*)\d+'
        if not re.search(pattern, content):
            print(f"ERROR: Could not find 'random_seed: int = <number>' in config.py")
            return False
        
        new_content = re.sub(pattern, f'\\g<1>{seed}', content)
        
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"[OK] Updated config.py: random_seed = {seed}")
        return True
    except Exception as e:
        print(f"ERROR updating config: {e}")
        return False

def run_seed_training(seed: int):
    """Run training with a specific seed."""
    import sys
    
    print(f"\n{'='*60}")
    print(f"TRAINING WITH SEED {seed}")
    print(f"{'='*60}\n")
    
    # Update config
    success = update_config_seed(seed)
    if not success:
        print(f"[ERROR] Failed to update config for seed {seed}, skipping...")
        return None
    
    # Create log directory for this seed
    seed_log_dir = LOG_DIR / f"seed_{seed}"
    seed_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Log file for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = seed_log_dir / f"training_log_{timestamp}.txt"
    
    # Write header to log file
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Seed: {seed}\n")
        f.write(f"Episodes: {EPISODES}\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")
    
    # Set environment for unbuffered output
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    
    print("[CMD] Running main.py with live tee to console + file")
    print(f"[LOG] Output will be saved to: {log_file}\n")
    
    # Native Python tee: stream stdout to both console and file
    proc = subprocess.Popen(
        [sys.executable, "-u", "main.py", "--episodes", str(EPISODES)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )
    
    with open(log_file, 'a', encoding='utf-8', buffering=1) as lf:
        for line in proc.stdout:
            print(line, end='')   # console
            lf.write(line)        # file
    
    rc = proc.wait()
    if rc != 0:
        print(f"\n[ERROR] Training failed with exit code {rc}")
        return None
    
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
            
            # Extract episode fitness
            if 'Episode' in line and '/' in line:
                for j in range(i+1, min(i+5, len(lines))):
                    if 'Val   - Reward:' in lines[j]:
                        try:
                            ep = int(line.split('/')[0].split()[-1])
                            fitness = float(lines[j].split('Fitness:')[1].split('|')[0].strip())
                            stats['episodes'].append(ep)
                            stats['val_fitness'].append(fitness)
                        except:
                            pass
                        break
    
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
            stats = parse_validation_stats(log_files[-1])
            all_stats.append(stats)
            
            print(f"Seed {seed}:")
            if stats['val_fitness']:
                final = stats['val_fitness'][-1]
                mean = sum(stats['val_fitness'])/len(stats['val_fitness'])
                print(f"  Final Fitness: {final:.4f}")
                print(f"  Mean Fitness: {mean:.4f}")
                if stats['val_iqr']:
                    mean_iqr = sum(stats['val_iqr'])/len(stats['val_iqr'])
                    print(f"  Mean IQR: {mean_iqr:.4f}")
                if stats['val_k']:
                    mean_k = sum(stats['val_k'])/len(stats['val_k'])
                    print(f"  Mean K: {mean_k:.1f}")
            else:
                print("  No validation data found")
            print()
    
    # Calculate dispersion across seeds
    if all_stats:
        final_fitness = [s['val_fitness'][-1] for s in all_stats if s['val_fitness']]
        if final_fitness:
            print("Across Seeds:")
            print(f"  Final Fitness Range: {min(final_fitness):.4f} to {max(final_fitness):.4f}")
            print(f"  Final Fitness Std: {(sum((f-sum(final_fitness)/len(final_fitness))**2 for f in final_fitness)/len(final_fitness))**0.5:.4f}")
    
    # Save combined results
    summary_file = LOG_DIR / "seed_sweep_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\n✓ Summary saved to: {summary_file}")

if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("FOREX RL BOT - AUTOMATED SEED SWEEP")
    print("="*60)
    print(f"Seeds: {SEEDS}")
    print(f"Episodes: {EPISODES}")
    print(f"Results directory: {LOG_DIR}")
    print()
    
    for i, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"SEED {i+1}/{len(SEEDS)}: {seed}")
        print(f"{'='*60}")
        
        run_seed_training(seed)
        
        if i < len(SEEDS) - 1:
            print(f"\n✓ Seed {seed} complete. Moving to seed {SEEDS[i+1]}...")
    
    # Generate summary
    summarize_sweep()
    
    print("\n✓ Seed sweep complete!")
