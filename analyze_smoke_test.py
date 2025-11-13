"""
Analyze smoke test results for anti-bias fixes
"""
import json
import numpy as np
from pathlib import Path
import sys

# Allow specifying directory as argument, default to fresh test
results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('fresh_antibias_test')
metrics_file = results_dir / 'episode_metrics.json'

if not metrics_file.exists():
    print("❌ Metrics file not found!")
    exit(1)

with open(metrics_file, encoding='utf-8') as f:
    data = json.load(f)

episodes = data['episodes']

print("=" * 70)
print("SMOKE TEST RESULTS - Anti-Bias Fixes (A1+A2)")
print("=" * 70)
print(f"Episodes completed: {len(episodes)}")
print(f"Seed: {data['config']['random_seed']}")
print()

# Extract metrics
p_longs = [ep['p_long_smoothed'] for ep in episodes]
lambda_longs = [ep['lambda_long'] for ep in episodes]
long_entries = [ep.get('long_entries', 0) for ep in episodes]
sprs = [ep['SPR'] for ep in episodes]

print("LONG ACTION FREQUENCY (p_long):")
print(f"  Min: {min(p_longs):.4f}")
print(f"  Max: {max(p_longs):.4f}")
print(f"  Mean: {np.mean(p_longs):.4f} (baseline was 0.0349)")
print(f"  Median: {np.median(p_longs):.4f}")
print()

print("LONG ENTRIES PER EPISODE:")
print(f"  Total LONG entries: {sum(long_entries)}")
print(f"  Mean per episode: {np.mean(long_entries):.1f}")
print(f"  Episodes with LONG>0: {sum(1 for e in long_entries if e > 0)}/{len(long_entries)}")
print()

print("LAMBDA_LONG (controller bias):")
print(f"  Min: {min(lambda_longs):.2f}")
print(f"  Max: {max(lambda_longs):.2f}")
print(f"  Mean: {np.mean(lambda_longs):.2f}")
print(f"  Saturated at -1.2: {sum(1 for x in lambda_longs if x == -1.2)}/{len(lambda_longs)}")
print(f"  Saturated at -2.0: {sum(1 for x in lambda_longs if x == -2.0)}/{len(lambda_longs)}")
print()

print("RETURNS (SPR):")
print(f"  Mean SPR: {np.mean(sprs):.4f} (baseline was -0.0236)")
print(f"  Positive episodes: {sum(1 for s in sprs if s > 0)}/{len(sprs)}")
print()

print("EPISODE-BY-EPISODE:")
print("Ep | p_long | λ_long | LONG entries | SPR")
print("-" * 60)
for ep in episodes:
    ep_num = ep['episode']
    p_long = ep['p_long_smoothed']
    lambda_long = ep['lambda_long']
    entries = ep.get('long_entries', 0)
    spr = ep['SPR']
    status = "✅" if p_long > 0.10 else "❌"
    floor = "🔧" if ep_num <= 60 else "  "
    print(f"{status}{floor} {ep_num:2d} | {p_long:.4f} | {lambda_long:+5.2f} | {entries:12d} | {spr:+.4f}")

print()
print("=" * 70)
print("ASSESSMENT:")
print()

# Compare to baseline
baseline_p_long = 0.0349
improvement = np.mean(p_longs) - baseline_p_long
pct_improvement = (improvement / baseline_p_long) * 100 if baseline_p_long > 0 else 0

if np.mean(p_longs) > baseline_p_long * 1.5:
    print("✅ SIGNIFICANT IMPROVEMENT in p_long detected!")
    print(f"   Baseline: {baseline_p_long:.4f}")
    print(f"   Current: {np.mean(p_longs):.4f}")
    print(f"   Improvement: {improvement:.4f} ({pct_improvement:+.1f}%)")
elif np.mean(p_longs) > baseline_p_long:
    print("⚠️  MINOR IMPROVEMENT in p_long")
    print(f"   Baseline: {baseline_p_long:.4f}")
    print(f"   Current: {np.mean(p_longs):.4f}")
    print(f"   Improvement: {improvement:.4f} ({pct_improvement:+.1f}%)")
else:
    print("❌ NO IMPROVEMENT - p_long still too low")
    print(f"   Current: {np.mean(p_longs):.4f} (target: >0.10)")

print()

if sum(long_entries) > 0:
    print(f"✅ LONG entries detected: {sum(long_entries)} total")
else:
    print("❌ NO LONG entries - exploration floor may not be triggering")

print()

# Check if lambda is now hitting the new ceiling
if any(x <= -1.5 for x in lambda_longs):
    print("✅ Controller using extended range (λ < -1.5)")
    print(f"   Min λ_long: {min(lambda_longs):.2f} (ceiling is -2.0)")
else:
    print("⚠️  Controller still in old range (λ > -1.5)")
    print(f"   Min λ_long: {min(lambda_longs):.2f}")

print()
print("NEXT STEPS:")
if np.mean(p_longs) < 0.10 and sum(long_entries) == 0:
    print("❌ Quick guards not effective enough")
    print("   Recommend: Implement B3 (sign-flip) + B4 (balanced sampling)")
    print("   These are the durable fixes that address root cause")
elif np.mean(p_longs) < 0.20:
    print("⚠️  Modest improvement but still below target")
    print("   Recommend: Proceed with B3+B4 implementation")
    print("   Then run 200-episode probe")
else:
    print("✅ Encouraging results!")
    print("   Recommend: Implement B3+B4 for durability")
    print("   Then run 200-episode probe to verify stability")

print("=" * 70)
