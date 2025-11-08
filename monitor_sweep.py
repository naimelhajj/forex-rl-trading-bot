"""
Quick progress monitor for seed sweep
Shows last few lines of current training log
"""

from pathlib import Path
import time
import sys

LOG_DIR = Path("./seed_sweep_results")

def find_latest_log():
    """Find the most recently modified log file"""
    log_files = list(LOG_DIR.glob("seed_*/training_log_*.txt"))
    if not log_files:
        return None
    return max(log_files, key=lambda p: p.stat().st_mtime)

def main():
    print("Monitoring seed sweep progress...")
    print("Press Ctrl+C to stop monitoring\n")
    
    last_size = 0
    
    try:
        while True:
            log_file = find_latest_log()
            
            if not log_file:
                print("Waiting for log file...")
                time.sleep(2)
                continue
            
            current_size = log_file.stat().st_size
            
            if current_size != last_size:
                # Read last 15 lines
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    last_lines = lines[-15:] if len(lines) > 15 else lines
                
                # Clear screen and show
                print("\033[2J\033[H")  # Clear screen
                print(f"Current log: {log_file.name}")
                print(f"File size: {current_size:,} bytes")
                print("="*60)
                print(''.join(last_lines))
                
                last_size = current_size
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    main()
