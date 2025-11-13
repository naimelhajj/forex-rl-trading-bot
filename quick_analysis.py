import json
from pathlib import Path

data = json.loads(Path('fresh_antibias_test/episode_metrics.json').read_text())
eps = data['episodes']

print("=" * 60)
print("AGGRESSIVE FLOOR TEST RESULTS")
print("=" * 60)
print(f"Episodes completed: {len(eps)}")
print(f"Seed: {data.get('seed', 'unknown')}")
print()

# p_long analysis
p_longs = [e['p_long_smoothed'] for e in eps]
print("LONG ACTION FREQUENCY (p_long):")
print(f"  Min: {min(p_longs):.4f}")
print(f"  Max: {max(p_longs):.4f}")
print(f"  Mean: {sum(p_longs)/len(p_longs):.4f}")
print(f"  Median: {sorted(p_longs)[len(p_longs)//2]:.4f}")
print()

# Long entries
long_entries = [e.get('long_entries', 0) for e in eps]
total_entries = sum(long_entries)
print("LONG ENTRIES PER EPISODE:")
print(f"  Total LONG entries: {total_entries}")
print(f"  Episodes with LONG>0: {sum(1 for x in long_entries if x > 0)}/{len(eps)}")
print(f"  Mean per episode: {total_entries/len(eps):.1f}")
if total_entries > 0:
    print(f"  Max in one episode: {max(long_entries)}")
print()

# Lambda saturation
lambda_longs = [e.get('lambda_long', 0) for e in eps]
saturated = sum(1 for x in lambda_longs if abs(x + 2.0) < 0.01)
print("LAMBDA_LONG (controller bias):")
print(f"  Saturated at -2.0: {saturated}/{len(eps)}")
if saturated < len(eps):
    non_sat = [x for x in lambda_longs if abs(x + 2.0) >= 0.01]
    print(f"  Non-saturated range: {min(non_sat):.2f} - {max(non_sat):.2f}")
print()

# Returns
sprs = [e['SPR'] for e in eps]
positive = sum(1 for x in sprs if x > 0)
print("RETURNS (SPR):")
print(f"  Mean SPR: {sum(sprs)/len(sprs):.4f}")
print(f"  Positive episodes: {positive}/{len(eps)}")
print()

# Assessment
print("=" * 60)
print("ASSESSMENT:")
if total_entries > 50:
    print("  ✅ FLOOR IS TRIGGERING (>50 LONG entries)")
else:
    print(f"  ❌ FLOOR STILL NOT TRIGGERING ({total_entries} entries)")

mean_p_long = sum(p_longs)/len(p_longs)
if mean_p_long > 0.10:
    print(f"  ✅ p_long IMPROVED (mean {mean_p_long:.1%})")
else:
    print(f"  ⚠️  p_long still low (mean {mean_p_long:.1%})")

print("=" * 60)
