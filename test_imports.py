"""
Test which import is slow.
"""
import time

with open("import_test.txt", "w") as f:
    def log(msg):
        print(msg, flush=True)
        f.write(msg + "\n")
        f.flush()
    
    log("Testing imports...")
    
    log("[1] numpy...")
    t0 = time.time()
    import numpy as np
    log(f"  ✓ {time.time()-t0:.3f}s")
    
    log("[2] pandas...")
    t0 = time.time()
    import pandas as pd
    log(f"  ✓ {time.time()-t0:.3f}s")
    
    log("[3] scipy.stats.percentileofscore...")
    t0 = time.time()
    from scipy.stats import percentileofscore
    log(f"  ✓ {time.time()-t0:.3f}s")
    
    log("\nAll imports successful!")

print("Check import_test.txt for results")
