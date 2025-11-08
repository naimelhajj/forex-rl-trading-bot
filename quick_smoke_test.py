"""Quick smoke test - 3 episodes to verify no hangs or Unicode errors"""
import subprocess
import sys

print("="*60)
print("QUICK SMOKE TEST - 3 EPISODES")
print("="*60)
print("This will verify:")
print("  1. No Unicode encoding errors")
print("  2. No hangs during training")
print("  3. System runs end-to-end")
print("\nStarting...\n")

# Run 3-episode smoke test
result = subprocess.run(
    [sys.executable, "main.py", "--episodes", "3"],
    capture_output=False,
    text=True
)

print("\n" + "="*60)
if result.returncode == 0:
    print("SMOKE TEST PASSED!")
    print("System is ready for production runs")
else:
    print(f"SMOKE TEST FAILED with exit code {result.returncode}")
    print("Check the output above for errors")
print("="*60)

sys.exit(result.returncode)
