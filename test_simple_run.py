"""
Simple test - just run main.py once to see where it hangs
"""

import subprocess
import sys
import time

print("="*60)
print("SIMPLE MAIN.PY TEST")
print("="*60)
print("\nThis will run: python main.py --episodes 3")
print("Watch for where it hangs...\n")

start_time = time.time()

# Run with real-time output
process = subprocess.Popen(
    [sys.executable, "main.py", "--episodes", "3"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    universal_newlines=True
)

try:
    for line in process.stdout:
        elapsed = time.time() - start_time
        print(f"[{elapsed:6.1f}s] {line}", end='')
        
    process.wait()
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Completed in {total_time:.1f} seconds")
    print(f"Exit code: {process.returncode}")
    print("="*60)
    
except KeyboardInterrupt:
    print(f"\n\n{'='*60}")
    print("INTERRUPTED BY USER")
    print(f"{'='*60}")
    elapsed = time.time() - start_time
    print(f"Ran for {elapsed:.1f} seconds before interruption")
    print("Process was hung or running slowly.")
    process.kill()
    sys.exit(1)
