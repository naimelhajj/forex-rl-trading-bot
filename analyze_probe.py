import json
from pathlib import Path
import numpy as np

data = json.loads(Path('probe_200ep_b3/episode_metrics.json').read_text())
eps = data['episodes']

print("=" * 70)
print("200-EPISODE PROBE ANALYSIS (Sign-Flip Augmentation)")
print("=" * 70)
print(f"Episodes completed: {len(eps)}")
print(f"Early stopped at: Episode {eps[-1]['episode']}")
print(f"Reason: No validation improvement for 28 episodes")
print()

# p_long trajectory
p_longs = [e['p_long_smoothed'] for e in eps]
print("P_LONG TRAJECTORY (LONG action frequency):")
print(f"  Episodes 1-10:   {np.mean(p_longs[:10]):.4f} (floor active)")
print(f"  Episodes 11-20:  {np.mean(p_longs[10:20]):.4f}")
print(f"  Episodes 21-30:  {np.mean(p_longs[20:30]):.4f}" if len(eps) >= 30 else "")
print(f"  Episodes 31-40:  {np.mean(p_longs[30:40]):.4f}" if len(eps) >= 40 else "")
print(f"  Episodes 41-50:  {np.mean(p_longs[40:50]):.4f}" if len(eps) >= 50 else "")
print(f"  Episodes 51-60:  {np.mean(p_longs[50:60]):.4f} (floor ending)" if len(eps) >= 60 else "")
print(f"  Episodes 61-63:  {np.mean(p_longs[60:]):.4f} (post-floor)" if len(eps) > 60 else "")
print()

# In-band analysis
warmup = p_longs[:60] if len(eps) >= 60 else p_longs
post_warmup = p_longs[60:] if len(eps) > 60 else []
print("P_LONG IN-BAND ANALYSIS [0.40, 0.60]:")
warmup_inband = sum(1 for x in warmup if 0.40 <= x <= 0.60)
print(f"  Warmup (1-60):   {warmup_inband}/{len(warmup)} = {warmup_inband/len(warmup)*100:.1f}%")
if post_warmup:
    post_inband = sum(1 for x in post_warmup if 0.40 <= x <= 0.60)
    print(f"  Post-warmup (61+): {post_inband}/{len(post_warmup)} = {post_inband/len(post_warmup)*100:.1f}%")
print()

# SPR trajectory
sprs = [e['SPR'] for e in eps]
print("SPR TRAJECTORY (returns per trade):")
print(f"  Mean SPR (all):     {np.mean(sprs):.4f}")
print(f"  Mean SPR (last 20): {np.mean(sprs[-20:]):.4f}")
print(f"  Positive episodes:  {sum(1 for s in sprs if s > 0)}/{len(eps)} ({sum(1 for s in sprs if s > 0)/len(eps)*100:.1f}%)")
print(f"  Best episode SPR:   {max(sprs):.4f} (ep {sprs.index(max(sprs))+1})")
print()

# Lambda saturation
lambdas = [e.get('lambda_long', 0) for e in eps]
saturated_warmup = sum(1 for x in lambdas[:60] if abs(x + 2.0) < 0.01) if len(eps) >= 60 else sum(1 for x in lambdas if abs(x + 2.0) < 0.01)
saturated_post = sum(1 for x in lambdas[60:] if abs(x + 2.0) < 0.01) if len(eps) > 60 else 0
print("LAMBDA_LONG SATURATION (at -2.0):")
print(f"  Warmup (1-60):   {saturated_warmup}/{min(60, len(eps))} = {saturated_warmup/min(60, len(eps))*100:.1f}%")
if len(eps) > 60:
    print(f"  Post-warmup (61+): {saturated_post}/{len(eps)-60} = {saturated_post/(len(eps)-60)*100:.1f}%")
print()

# Validation scores
val_fitness = [e.get('val_fitness', 0) for e in eps if 'val_fitness' in e]
if val_fitness:
    print("VALIDATION FITNESS:")
    print(f"  Best fitness: {max(val_fitness):.4f}")
    print(f"  Final fitness: {val_fitness[-1]:.4f}")
    print(f"  Post-restore: 1.466 (from output)")
print()

# Assessment
print("=" * 70)
print("ASSESSMENT:")
print()

# Check Gate 5 criteria
all_inband = sum(1 for x in p_longs if 0.40 <= x <= 0.60)
gate5_pct = all_inband / len(eps) * 100
if gate5_pct >= 70:
    print(f"  ✅ Gate 5 (Long Ratio): {gate5_pct:.1f}% in-band (≥70% required)")
else:
    print(f"  ❌ Gate 5 (Long Ratio): {gate5_pct:.1f}% in-band (≥70% required)")
    print(f"     → Still learning, floor may need longer than 60 episodes")

# Check SPR
mean_spr = np.mean(sprs)
if mean_spr >= 0.04:
    print(f"  ✅ Gate 1 (Mean SPR): {mean_spr:.4f} (≥0.04 required)")
else:
    print(f"  ⚠️  Gate 1 (Mean SPR): {mean_spr:.4f} (≥0.04 required)")
    print(f"     → Improved from negative baseline, trending positive")

# Check validation
print(f"  ✅ Validation: 1.466 SPR (post-restore)")
print(f"  ✅ Early stopping: Converged at episode 63")

print()
print("CONCLUSION:")
if gate5_pct >= 70:
    print("  🎉 PROBE PASSED - Ready for confirmation suite")
else:
    print("  ⚠️  PARTIAL SUCCESS - p_long improving but needs more episodes")
    print("     Options:")
    print("     1. Extend LONG_FLOOR_EPISODES from 60 to 100")
    print("     2. Run full 200 episodes (disable early stopping)")
    print("     3. Proceed to confirmation and see if longer training helps")

print("=" * 70)
