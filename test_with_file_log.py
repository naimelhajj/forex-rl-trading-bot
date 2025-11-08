"""
Write timing results to file so we can see them even if terminal hangs.
"""
import time
import sys

with open("hang_test_results.txt", "w") as f:
    def log(msg):
        print(msg, flush=True)
        f.write(msg + "\n")
        f.flush()
    
    log("=" * 60)
    log("HANG TEST WITH FILE LOGGING")
    log("=" * 60)
    
    log("\n[1] Importing features module...")
    t0 = time.time()
    from features import _rolling_lr_slope
    import pandas as pd
    import numpy as np
    log(f"  ✓ Import: {time.time()-t0:.3f}s")
    
    log("\n[2] Creating test data (10k points)...")
    t0 = time.time()
    data = np.random.randn(10000).cumsum()
    series = pd.Series(data)
    log(f"  ✓ Data created: {time.time()-t0:.3f}s")
    
    log("\n[3] Testing _rolling_lr_slope(window=10)...")
    t0 = time.time()
    result10 = _rolling_lr_slope(series, 10)
    elapsed = time.time() - t0
    log(f"  ✓ Window=10: {elapsed:.4f}s")
    if elapsed > 0.5:
        log(f"    ⚠️  WARNING: SLOW!")
    
    log("\n[4] Testing _rolling_lr_slope(window=24)...")
    t0 = time.time()
    result24 = _rolling_lr_slope(series, 24)
    elapsed = time.time() - t0
    log(f"  ✓ Window=24: {elapsed:.4f}s")
    if elapsed > 0.5:
        log(f"    ⚠️  WARNING: SLOW!")
    
    log("\n[5] Testing _rolling_lr_slope(window=96)...")
    t0 = time.time()
    result96 = _rolling_lr_slope(series, 96)
    elapsed = time.time() - t0
    log(f"  ✓ Window=96: {elapsed:.4f}s")
    if elapsed > 1.0:
        log(f"    ⚠️  WARNING: SLOW!")
    
    log("\n" + "=" * 60)
    log("TEST COMPLETE")
    log("=" * 60)
    log(f"\nIf you see this, all tests passed!")
    log(f"Check timing above - should all be <1s")

print("\n\nResults written to hang_test_results.txt")
print("If test hung, the file will show where it stopped.")
