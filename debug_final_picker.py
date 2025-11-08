"""
Debug the 'final score' picker logic in compare_seed_results.py
"""

import json
from pathlib import Path

RESULTS_BASE = Path("logs/seed_sweep_results")
seeds = [7, 17, 27, 77, 777]

print("=" * 70)
print("DEBUG: Final Score Picker Logic")
print("=" * 70)

for seed in seeds:
    seed_dir = RESULTS_BASE / f"seed_{seed}"
    
    # Check for val_final.json
    final_path = seed_dir / "val_final.json"
    print(f"\nSeed {seed}:")
    print(f"  val_final.json exists? {final_path.exists()}")
    
    # Check episode extraction logic
    episodes = []
    for p in sorted(seed_dir.glob("val_ep*.json")):
        try:
            ep = json.load(open(p, "r"))
            idx = ep.get("episode", None)
            score = ep.get("score", float("nan"))
            if isinstance(idx, int):
                episodes.append((idx, score, p.name))
        except Exception as e:
            print(f"  Error reading {p.name}: {e}")
    
    if episodes:
        episodes.sort(key=lambda t: t[0])
        print(f"  Total episodes with valid index: {len(episodes)}")
        print(f"  Episode range: {episodes[0][0]} to {episodes[-1][0]}")
        print(f"  First episode: ep={episodes[0][0]}, score={episodes[0][1]:+.5f}, file={episodes[0][2]}")
        print(f"  Last episode:  ep={episodes[-1][0]}, score={episodes[-1][1]:+.5f}, file={episodes[-1][2]}")
        print(f"  Episode 80:    ep=80, score={[e for e in episodes if e[0] == 80][0][1] if any(e[0] == 80 for e in episodes) else 'N/A':+.5f}")
        
        # Show what compare_seed_results.py would return
        final_score, final_source = episodes[-1][1], "last-episode"
        print(f"  ➜ compare_seed_results.py would report: {final_score:+.5f} (from {final_source})")
    else:
        print(f"  ❌ No episodes found!")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("  If episode 150 exists, it will be used as 'final' instead of episode 80!")
print("=" * 70)
