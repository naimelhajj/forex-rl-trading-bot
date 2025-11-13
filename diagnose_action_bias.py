import json
import numpy as np
from collections import Counter

# Load telemetry
with open('confirmation_results/seed_42/episode_metrics.json', encoding='utf-8') as f:
    data = json.load(f)

print("=== COMPREHENSIVE ACTION ANALYSIS ===\n")

# Analyze all episodes to reconstruct action distributions
print("Analyzing episode metrics...")
print(f"Total episodes: {len(data['episodes'])}\n")

# Get aggregated stats
p_longs = [ep['p_long_smoothed'] for ep in data['episodes']]
p_holds = [ep['p_hold_smoothed'] for ep in data['episodes']]
lambda_longs = [ep['lambda_long'] for ep in data['episodes']]
switch_rates = [ep['switch_rate'] for ep in data['episodes']]

print("ACTION FREQUENCY ESTIMATES:")
print(f"  HOLD (action 0): ~{np.mean(p_holds)*100:.1f}% ± {np.std(p_holds)*100:.1f}%")
print(f"  LONG (action 1): ~{np.mean(p_longs)*100:.1f}% ± {np.std(p_longs)*100:.1f}%")
print(f"  SHORT (action 2): ~{(1-np.mean(p_holds)-np.mean(p_longs))*100:.1f}% (inferred)")
print()

print("CONTROLLER STATE:")
print(f"  lambda_long saturated at -1.2: {100 * sum(1 for x in lambda_longs if x == -1.2) / len(lambda_longs):.0f}% of episodes")
print(f"  Mean switch rate: {np.mean(switch_rates):.3f} (target: 0.15-0.19)")
print()

print("DIAGNOSIS:")
print(f"  1. Agent is taking LONG action only {np.mean(p_longs)*100:.1f}% of the time")
print(f"  2. Agent is taking HOLD action ~{np.mean(p_holds)*100:.1f}% of the time")
print(f"  3. Inferred SHORT action: ~{(1-np.mean(p_holds)-np.mean(p_longs))*100:.1f}% of the time")
print()

# The issue
if np.mean(p_longs) < 0.10:
    print("❌ CRITICAL ISSUE:")
    print("   The agent has learned a policy that almost never goes LONG.")
    print("   This is NOT a controller tuning issue - the Q-network itself")
    print("   has learned that action 1 (LONG) has low Q-values.")
    print()
    print("   Possible causes:")
    print("   1. Reward structure penalizes LONG trades more than SHORT")
    print("   2. Training data has bearish bias (prices trend down)")
    print("   3. Feature extraction gives better signals for SHORT")
    print("   4. Initial exploration found SHORT trades more profitable")
    print()
    print("   The dual controller is working correctly - lambda_long = -1.2")
    print("   means it's applying maximum bias to encourage LONG, but the")
    print("   Q-network's learned weights are too strong to overcome.")
