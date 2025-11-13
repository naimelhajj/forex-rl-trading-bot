import json
import numpy as np

# Load telemetry
with open('confirmation_results/seed_42/episode_metrics.json', encoding='utf-8') as f:
    data = json.load(f)

p_longs = [ep['p_long_smoothed'] for ep in data['episodes']]
lambda_longs = [ep['lambda_long'] for ep in data['episodes']]

print("=== LONG RATIO DIAGNOSIS ===")
print(f"\np_long statistics (target: 0.40-0.60):")
print(f"  Min: {min(p_longs):.4f}")
print(f"  Max: {max(p_longs):.4f}")
print(f"  Mean: {np.mean(p_longs):.4f}")
print(f"  Median: {np.median(p_longs):.4f}")
print(f"  Std: {np.std(p_longs):.4f}")

print(f"\nlambda_long statistics (range: -1.2 to +1.2):")
print(f"  Min: {min(lambda_longs):.4f}")
print(f"  Max: {max(lambda_longs):.4f}")
print(f"  Mean: {np.mean(lambda_longs):.4f}")
print(f"  % at -1.2 (saturated): {100 * sum(1 for x in lambda_longs if x == -1.2) / len(lambda_longs):.1f}%")

print(f"\nEpisodes in target band [0.40, 0.60]: {sum(1 for x in p_longs if 0.40 <= x <= 0.60)}/{len(p_longs)}")
